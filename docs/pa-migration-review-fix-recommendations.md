# PA 迁移验收问题修复建议

> 适用分支：`dev_pa_mig`
>
> 直接依据：`docs/pa-migration-plan.md`
>
> 上位设计：`docs/pa-watch-system-design.md`
>
> 文档目的：给出迁移验收中发现问题的修复方案、实施顺序和验收标准。本文只提出建议，不包含代码修改。

## 1. 结论与修复原则

PA 核心资产本身已经基本正确迁移：prompt、特征计算、决策树、两阶段编排和校验栈与 PA_Agent `1090a5b` 基线保持一致。当前问题主要集中在 freqtrade 集成边界，而不是 PA 决策核心。

建议按以下原则修复：

1. **先保证不会错误交易，再扩展 trade 能力。** 在交易计划执行链完整之前，`watch_only=false` 必须 fail closed，不能把不完整的二元信号当作可用实盘能力。
2. **live 与 backtest 的副作用必须隔离。** 回测默认不得写正式 `analysis_record` / `signal` 表，也不得推送 Telegram。
3. **一根已收盘 K 线最多产生一个有效交易意图。** 应同时用策略层跳过和数据库唯一约束防止重复。
4. **影响数据预加载的配置必须在策略解析阶段确定。** 不能等到 `bot_start()` 再修改 `startup_candle_count`。
5. **知识资产属于运行时依赖。** wheel、Docker 和 editable install 三种安装方式都必须包含 prompt 文件。

## 2. 建议实施顺序

| 顺序 | 工作项 | 原因 |
|---|---|---|
| 1 | trade 模式安全闸门 | 立即阻止不完整执行链进入实盘 |
| 2 | 回测副作用隔离 | 避免继续污染正式交易意图账本 |
| 3 | 重启幂等与数据库唯一约束 | 避免重复信号和重复通知 |
| 4 | `startup_candle_count` 提前计算 | 保证 live/回测输入数据正确 |
| 5 | wheel 知识资产打包 | 保证所有部署方式可运行 |
| 6 | 完整交易计划执行链 | 最后单独开发、回测和小额验证 |

前四项属于迁移收尾，可以独立完成。第 5、6 项涉及真实订单语义，建议单独提交和验收。

## 3. 问题一：trade 模式没有执行完整 PA 交易计划

### 3.1 当前问题

当前策略只把 PA 决策转换成 `enter_long` / `enter_short`：

- `限价单`、`突破单`、`市价单`没有不同执行语义；
- `entry_price` 没有用于实际委托价格；
- `stop_loss_price`、`take_profit_price`、`take_profit_price_2` 没有用于持仓管理；
- 策略默认 `can_short=False`，做空信号不会被 freqtrade 执行；
- `stoploss=-0.99` 且没有退出信号，实盘持仓缺少与 PA 计划一致的退出机制；
- `signal.executed_trade_id` 没有回填链路。

因此当前系统的 watch 模式是安全的，但文档所述“把 `watch_only` 改为 false 即切换实盘”尚不成立。

### 3.2 第一阶段：立即增加实盘安全闸门

在完整交易执行链实现前，建议在 `bot_start()` 中加入显式能力开关：

```json
"pa_llm": {
  "watch_only": true,
  "trade_execution_enabled": false
}
```

规则：

- `watch_only=true`：正常盯盘；
- `watch_only=false` 且 `trade_execution_enabled=false`：启动失败并给出明确错误；
- `watch_only=false` 且 `trade_execution_enabled=true`：仅在完整执行链和配置校验全部通过后允许启动。

启动校验至少应覆盖：

- 不允许 `max_open_trades=0`；
- `dry_run=false` 时不允许空交易所凭据；`dry_run=true` 可用于 paper trade 验证；
- 做空启用时必须是 futures/margin 模式；
- signal 持久化不可用时禁止交易，不能回退 SQLite 后继续下单；
- 不支持的 PA `order_type` 必须拒绝，不能静默降级。

这一步的目标不是实现实盘，而是消除“配置一改就能安全交易”的错误预期。使用说明也应暂时把 trade 模式标为未完成。

### 3.3 第二阶段：建立交易计划关联键

每个有效 PA 计划必须先成功写入 `signal`，再允许写入场信号。建议：

1. 为 `signal` 增加可查询的唯一身份；使用现有自增 `id` 即可。
2. `_apply_decision()` 在落库后取得 `signal_id`。
3. DataFrame 写入：

   ```python
   dataframe.loc[index, "enter_tag"] = f"pa_signal:{signal_id}"
   ```

4. 后续 `custom_entry_price`、`confirm_trade_entry`、`order_filled` 根据 `enter_tag` 读取同一条计划。
5. 如果 signal 落库失败，必须不写 `enter_long` / `enter_short`，即 **ledger-before-order**。

为避免现在 `TwoStageOrchestrator.submit()` 内部保存、策略却拿不到 signal ID，建议调整持久化返回契约：

```python
@dataclass(frozen=True)
class SaveResult:
    analysis_record_id: int
    signal_id: int | None
```

`PgRecordStore.save_full()` 返回 `SaveResult`；或者把 signal 写入从 orchestrator 中拆到策略层，并提供一个原子事务方法 `save_analysis_and_signal(record)`。推荐后者，因为“是否产生交易副作用”属于运行模式/策略层职责，不属于两阶段推理本身。

### 3.4 第三阶段：实现订单计划状态机

建议不要在得到任何 order plan 后立刻统一写入场信号，而是维护明确状态：

```text
planned → armed → submitted → filled
                    └→ cancelled / expired / rejected
```

可以给 `signal` 增加：

- `status`
- `submitted_at`
- `filled_at`
- `cancelled_at`
- `cancel_reason`
- `execution_order_type`
- `executed_trade_id`

订单类型建议这样处理：

| PA 计划 | 推荐执行语义 |
|---|---|
| `限价单` | 当前 K 线产生入场信号；`custom_entry_price()` 返回计划 `entry_price` |
| `突破单` | 先保持 `armed`；后续已收盘 K 线达到触发条件时才产生入场信号 |
| `市价单` | 仅在明确支持逐信号订单类型后启用；否则拒绝，不要伪装成限价单 |
| `不下单` | 只保存分析记录，不创建 signal |

freqtrade 当前 `order_types["entry"]` 是策略级配置，而 PA `order_type` 是逐信号配置。完整支持“同一策略同时产生 market/limit”有两个选择：

1. **推荐，保持框架不变：** 第一版 trade 模式只支持一种真实订单类型，其余 PA 类型 fail closed；突破单通过策略状态机延迟触发。
2. **后续扩展：** 给 freqtrade 增加类似 `custom_entry_order_type()` 的小型策略回调，使订单类型可按 `entry_tag` 决定。该方案会增加 fork 维护成本，应另行设计和测试。

### 3.5 止损、止盈与成交对账

建议使用 freqtrade 原生策略回调：

- `custom_entry_price()`：返回 signal 中的计划入场价；
- `confirm_trade_entry()`：再次校验计划未过期、方向一致、signal 状态可提交；
- `order_filled()`：首次入场成交后回填 `signal.executed_trade_id`，并把止损/止盈快照写入 `trade.set_custom_data()`；
- `custom_stoploss()`：用 `stoploss_from_absolute()` 将计划绝对止损价转换为 long/short 均正确的比例；
- `custom_exit()`：处理 TP1/TP2、计划失效或新的反向决策；
- 如需分批止盈，再启用 `position_adjustment_enable` 并实现 `adjust_trade_position()`。

不要只从内存字典读取计划，因为进程重启后会丢失。交易回调应以 `enter_tag → signal` 或 trade custom data 为真相源。

### 3.6 trade 模式验收标准

- 限价做多、限价做空的实际委托价与 signal 计划价一致；
- spot 模式不会尝试做空，futures 模式做空可正常触发；
- 计划止损在 long/short 两个方向计算正确；
- TP/失效退出有自动化测试；
- signal 写入失败时不会产生订单；
- 成交后 `executed_trade_id` 必须回填；
- 重启后已有持仓仍可恢复止损、止盈与计划关联；
- 不支持的订单类型明确拒绝并记录原因。

## 4. 问题二：回测污染正式 PG 账本

### 4.1 推荐方案：持久化端口按运行模式注入

给 orchestrator 使用的 writer 定义最小协议：

```python
class RecordWriter(Protocol):
    def save_full(self, record: AnalysisRecord): ...
    def save_partial(self, record: AnalysisRecord, reason: str): ...
```

实现三种 writer：

| Writer | 用途 | 行为 |
|---|---|---|
| `PgRecordStore` | LIVE / DRY_RUN | 写 analysis_record；有效计划写 signal |
| `NullRecordWriter` | BACKTEST 默认 | 所有写操作 no-op，不连接 PG |
| `BacktestRecordStore` | 显式调试 | 写独立数据库/独立 schema，并标记 `run_id` |

`bot_start()` 根据 `self.dp.runmode` 选择 writer：

```python
if runmode == RunMode.BACKTEST:
    writer = NullRecordWriter()
else:
    writer = PgRecordStore(pa_db_url)
```

同时：

- BACKTEST 不执行 `find_latest_successful()`，每个 pair 从 `prev=None` 开始；
- 回测循环只用局部变量 `prev` 做增量链；
- BACKTEST 不调用 `dp.send_msg()`；
- 如果需要保存回测分析，必须配置单独的 `pa_backtest_db_url`，不能复用 `pa_db_url`；
- 回测记录建议增加 `backtest_run_id`，防止不同回测批次混在一起。

### 4.2 配套修改

- 修改 `tools/bt_offline_driver.py`：默认断言数据库没有新增正式 signal；
- 更新运行手册中“289 条 partial 落库”的描述，改为默认 no-op 或独立 backtest DB；
- `PgRecordStore` 的“写 signal”最好从通用 `save_full` 中拆出，避免任何调用者保存分析记录时意外创建交易意图。

### 4.3 验收标准

- 用正式 `pa_db_url` 执行回测前后，正式库两张 PA 表行数不变；
- 回测仍能在内存中串行传递 previous record；
- LIVE/DRY_RUN 行为不变；
- 显式设置 `pa_backtest_db_url` 时，记录只出现在独立库并带唯一 `run_id`。

## 5. 问题三：重启会重复分析同一根 K 线

### 5.1 策略层修复

`count_new_bars_since_record()` 的结果应按三种情况处理：

```python
new_bars = count_new_bars_since_record(frame, prev)

if new_bars == 0:
    return None                    # 当前 K1 已分析，直接跳过
if new_bars is None:
    previous_record = None         # 找不到锚点，执行完整分析
elif new_bars > incremental_max_new_bars:
    previous_record = None         # 停机过久，执行完整分析
else:
    previous_record = prev         # 正常增量分析
```

同时应真正接入 `GeneralSettings.incremental_max_new_bars`。当前设置项已定义，但策略没有使用。

### 5.2 数据库防重

策略层跳过不能替代数据库约束。建议给 signal 建立唯一索引：

```sql
CREATE UNIQUE INDEX uq_signal_intent_candle
ON signal (symbol, timeframe, candle_time);
```

如果未来允许同一根 K 线多策略并存，应先增加 `strategy_name` 或 `strategy_version`，唯一键改为：

```text
(strategy_name, symbol, timeframe, candle_time)
```

写入时应使用幂等语义：重复键返回现有 `signal_id`，而不是再推送一次。

`analysis_record` 可以保留失败重试记录，不必与 signal 使用相同的唯一约束；但建议增加 `analysis_key` / `attempt` 字段，便于区分重复尝试和有效结果。

### 5.3 数据迁移注意事项

`SQLAlchemy.metadata.create_all()` 不会给已有表补唯一索引。需要显式迁移脚本：

1. 查询现有重复 `(symbol, timeframe, candle_time)`；
2. 保留最早或最后一条有效 signal，并修正外部引用；
3. 删除/归档重复行；
4. 创建唯一索引；
5. 在空库和已有库各跑一次迁移测试。

### 5.4 验收标准

- bot 重启后面对同一根最新 K 线时，LLM 调用次数为 0；
- analysis_record、signal 和 TG 通知均不新增；
- 停机 1～N 根 K 线后正确进入增量模式；
- 超过 `incremental_max_new_bars` 后正确退化为完整分析；
- 并发或重复调用也只能得到一条 signal。

## 6. 问题四：动态 startup_candle_count 设置过晚

### 6.1 推荐修复

在策略 `__init__()` 中读取 `pa_llm.analysis_bar_count` 并设置实例属性，而不是等到 `bot_start()`：

```python
def __init__(self, config: dict) -> None:
    super().__init__(config)
    analysis_bars = validate_analysis_bar_count(config)
    self._analysis_bars = analysis_bars
    self.startup_candle_count = analysis_bars + INDICATOR_WARMUP_BARS + 10
```

原因是 `StrategyResolver` 在策略实例创建后、`bot_start()` 前会读取 `startup_candle_count` 并复制到 config；Backtesting 也会在 `bot_start()` 前计算 `required_startup`。放在 `__init__()` 可以同时覆盖 live 和 backtest。

`bot_start()` 不应再次改变该值，只做一致性断言：

```python
assert self.startup_candle_count == self._analysis_bars + INDICATOR_WARMUP_BARS + 10
```

另外建议限制配置范围。`analysis_bar_count` 当前允许到 5000，但交易所 OHLCV 单次/多次请求能力有限。启动时应调用框架已有的 startup candle 校验；无法取得足够历史数据时应明确报错，而不是永久返回 `frame=None`。

### 6.2 验收矩阵

至少测试以下值：

| analysis_bar_count | 预期 startup |
|---:|---:|
| 30 | 90 |
| 100 | 160 |
| 300 | 360 |

对每个值验证：

- StrategyResolver 完成后，实例属性和 config 一致；
- Backtesting.required_startup 一致；
- 回测请求区间第一根可交易 K 线没有额外丢失；
- live 数据足够后能形成正确长度的 KlineFrame；
- 指标 warmup 仍使用额外 50 根，而不进入 LLM 分析窗口。

## 7. 问题五：wheel 缺少 prompt/参考文件

### 7.1 打包配置

在 `pyproject.toml` 增加 package data：

```toml
[tool.setuptools.package-data]
pa_core = [
  "prompts/*.txt",
  "prompts/_reference/*.md",
  "experience/*/success_cases/.gitkeep",
  "experience/*/failure_cases/.gitkeep",
]
```

同时更新 `MANIFEST.in`，保证 sdist 也包含资产：

```text
recursive-include pa_core/prompts *.txt *.md
recursive-include pa_core/experience .gitkeep
```

项目当前 `zip-safe=false`，安装后的 `pa_core.config.PROMPT_DIR` 仍可作为真实文件系统路径使用。若未来改为 zip import，再考虑用 `importlib.resources` 替代 `Path(__file__)`。

### 7.2 构建验收

CI 中增加 wheel 内容测试：

1. 构建 wheel；
2. 解包后断言正好存在 29 个 `.txt` 和 3 个 `_reference/*.md`；
3. 在干净临时环境安装 wheel；
4. 从非仓库目录导入 `pa_core`；
5. 实例化 `PromptAssembler` 并读取 `二元决策.txt`；
6. 校验 prompt blob hash 与 PA_Agent `1090a5b` 基线清单一致。

Dockerfile 直接 `COPY pa_core/`，当前不受 wheel 漏文件影响，但 Docker 冒烟测试仍应确认 32 个知识资产存在。

## 8. 建议的代码边界调整

为了减少后续问题，建议把职责明确为：

```text
TwoStageOrchestrator
  输入 KlineFrame + previous_record
  输出 AnalysisRecord
  不决定 live/backtest，不直接发送通知

RecordWriter
  只保存分析记录
  live 使用 PG，backtest 默认 Null

SignalLedger
  只在 live/dry-run 的有效 order plan 上写入
  返回 signal_id，提供幂等写入和 executed_trade_id 回填

PriceActionWatch
  判断 RunMode
  维护增量状态
  决定是否写 ledger、发 TG、写 enter_* 列
  trade 模式负责把 signal_id 传入 freqtrade 交易生命周期
```

这比当前把“保存完整分析”和“创建 signal”都放在 `PgRecordStore.save_full()` 内更容易保证副作用边界。

## 9. 自动化测试补充清单

### 9.1 必须新增

- `test_backtest_does_not_write_pg_or_signal`
- `test_backtest_does_not_restore_live_previous_record`
- `test_restart_same_candle_skips_submit`
- `test_signal_unique_per_strategy_pair_timeframe_candle`
- `test_incremental_gap_above_limit_forces_full_analysis`
- `test_analysis_bar_count_sets_startup_before_bot_start`
- `test_wheel_contains_all_prompt_assets`
- `test_trade_mode_rejected_until_execution_enabled`
- `test_trade_aborts_when_signal_persistence_fails`
- `test_long_and_short_plan_execution`
- `test_order_filled_links_signal_and_trade`
- `test_trade_restart_restores_risk_plan`

### 9.2 需要继续保留

- 29 个 prompt + 3 个参考文档的基线 hash 校验；
- DataFrame 最旧→最新到 KlineFrame 最新→最旧的顺序校验；
- amount/pct_chg 默认值校验；
- EMA/ATR 与 PA_Agent 基线一致性；
- previous AnalysisRecord PG 往返完整性；
- `dp.send_msg(always_send=True)` 校验；
- LIVE/DRY_RUN 的 watch 模式绝不写任何 `enter_*` 信号。

### 9.3 当前跳过测试的处理

现有 pa_core 测试有若干以“PA_Agent 基线既有失败”为由跳过。它们可以不阻塞本次迁移修复，但应建立单独 issue 清单，区分：

- 确认属于源项目缺陷、迁移保持兼容；
- 依赖缺失（例如 Hypothesis）；
- 时区依赖；
- 已经不属于迁移范围的 GUI 测试。

不要长期仅用无编号的 skip 理由掩盖失败；至少关联 issue/风险登记。

## 10. 最终验收门槛

满足以下条件后，可以把迁移状态从“核心迁移基本完成”提升为“迁移验收通过”：

- 5 个已发现问题全部有自动化回归测试；
- 默认 watch 模式零订单，且重启不重复分析同一 K 线；
- 默认 backtest 对正式 PG 和 Telegram 零副作用；
- 任意受支持的 `analysis_bar_count` 都能正确预加载；
- wheel、editable install、Docker 三种形态都能读取完整知识资产；
- trade 模式未完成时 fail closed；完成后计划价格、方向、止损、止盈和成交关联均能端到端验证；
- 使用 PostgreSQL 做至少一次集成测试，而不只是在 SQLite 上验证 SQLAlchemy 模型；
- 使用一个可控的 fake OpenAI-compatible server 做成功路径回测，覆盖有效 stage1/stage2、signal 落库和增量续写，而不只验证 LLM 失败降级路径。

## 11. 推荐提交拆分

建议按以下提交拆分，方便审查和回滚：

1. `Fix PA backtest persistence isolation`
2. `Make PA candle processing idempotent`
3. `Initialize PA startup candles before strategy resolution`
4. `Package PA prompt assets in wheels and sdists`
5. `Fail closed when PA trade execution is incomplete`
6. `Add PA signal-to-trade execution lifecycle`
7. `Add PostgreSQL and fake-LLM end-to-end verification`

前五个提交完成后，可重新验收 watch 模式迁移；第六、七个提交完成后，再验收“一键切换 trade 模式”的上位设计目标。
