# PriceActionWatch 运行手册

pa_core（第1-6步迁移产物）已通过 freqtrade 策略层接入（第7步）。本文说明如何在开发环境验证与运行。

## 1. 环境准备（完整 dev 环境）

```bash
pip install -r requirements-dev.txt && pip install -e .   # freqtrade 完整依赖
pip install openai psycopg-binary                           # pa_core LLM 调用 + PG 驱动
# tiktoken 可选（token 估算）
```

> 两个 venv 提示：`.venv311` 只装了 pa_core 测试所需依赖（无 freqtrade/openai）；`.venv` 不完整（缺 cachetools/pytest）。跑 freqtrade 请用完整 dev 环境。

## 2. 数据库

- 默认回退：`sqlite:///user_data/pa_records.sqlite`（无需任何准备，冒烟可用）
- 生产（PG）：`watch_config.json` 的 `pa_db_url` 已指向 `postgresql+psycopg://.../freqtrade_monitor`。建库后表自动创建（`create_all`）。

### 2.1 `watch_pair` 表与标的管理（DatabasePairList）

白名单完全来自 PG，`watch_config.json` 已不含 `exchange.pair_whitelist`：`pairlists` 为
`DatabasePairList`（`refresh_period: 60` 秒），白名单 = `watch_pair` 表中 `enabled = true` 且
`market` 匹配（binance → `crypto`，ashare → `ashare`）的行，按 `id` 排序。

**冷启动自动初始化**：策略 `bot_start` 打开 `WatchPairStore`（建表 `create_all`，幂等），当该
market **没有任何行**时自动写入默认标的 `BTC/USDT`、`ETH/USDT`
（`pa_core/records/watch_pair_store.py::DEFAULT_WATCH_PAIRS`）。freqtrade 引擎本身对表严格只读。
新库/升级部署**无需任何手工步骤**：首轮刷新可能短暂为空（仅日志告警），`bot_start` 播种后下一个
刷新周期（≤60s）白名单恢复。

- 想避免首轮空白的告警噪音，可在启动前预执行 `python tools/watch_pairs.py init`
  （建表 + 冷启动播种，与 bot 行为一致，幂等）。
- psql 等价 DDL（仅在需要手工建表时参考）：

```sql
CREATE TABLE IF NOT EXISTS watch_pair (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    market VARCHAR(16) NOT NULL DEFAULT 'crypto',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    display_name VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_watch_pair_symbol_market UNIQUE (symbol, market)
);
CREATE INDEX IF NOT EXISTS ix_watch_pair_market ON watch_pair (market);
INSERT INTO watch_pair (symbol, market, enabled, display_name) VALUES
    ('BTC/USDT', 'crypto', TRUE, 'BTC/USDT'),
    ('ETH/USDT', 'crypto', TRUE, 'ETH/USDT');
```

> 2026-09-30 已核实：Azure `freqtrade_monitor` 库当时**尚无** `watch_pair` 表。升级到本版本后首次启动会自动
> 建表并播种 BTC/ETH，无需手工 init。

**镜像注意（DatabasePairList overlay）**：watch 镜像基底是上游 `freqtradeorg/freqtrade:stable`，不含 fork 独有的
`DatabasePairList`。`Dockerfile.watch` 构建时会把仓库的
`freqtrade/plugins/pairlist/DatabasePairList.py` 覆盖进镜像内的 freqtrade 包（单一来源仍是仓库文件）。
有意**不**把整个 fork 的 freqtrade 源码放进镜像：fork 的 `SCHEMA_TRADE_REQUIRED` 仍要求
`stoploss`/`minimal_roi` 等字段（`watch_config.json` 故意不带，上游已把它们移到策略层），全量 fork 源码会让
容器启动即校验失败。完整在镜像内运行 fork 是后续独立任务（需先对齐 schema 差异）。
升级重建镜像后，确认容器日志出现 `Using resolved pairlist DatabasePairList ...`。

日常增删改（变更在 `refresh_period` 内生效，无需重启/`/reload_config`）：

```bash
# 宿主机（仓库根目录；db-url 取自 watch_config.json 的 pa_db_url 或 --db-url 覆盖）
python tools/watch_pairs.py list [--all]
python tools/watch_pairs.py add SOL/USDT [--timeframe 4h] [--display-name Solana]
python tools/watch_pairs.py disable SOL/USDT     # 停用（保留行，推荐）
python tools/watch_pairs.py enable SOL/USDT
python tools/watch_pairs.py remove SOL/USDT      # 硬删除

# 容器内（db-url 自动取自 FREQTRADE__PA_DB_URL 环境变量）
docker compose -f docker/docker-compose-watch.yml exec freqtrade-watch \
    python /freqtrade/tools/watch_pairs.py list
```

每标的周期（`watch_pair.timeframe`，NULL = 主周期 1h）：仅支持 ≥ 主周期的间隔
（2h/4h/1d/1w…，freqtrade informative 机制限制）。策略经 `informative_pairs` 拉取
对应周期K线，仅在该周期K线收盘时触发分析；`WatchPairStore.__init__` 会对旧表自动
`ALTER TABLE ... ADD COLUMN timeframe`（无需手工迁移）。修改周期 = `remove` + `add --timeframe`。

Telegram 也可直接增删改（`TG_ENABLED=true` 时，fork 的 telegram RPC 内置命令，同一张表、同一套
`WatchPairStore`，`authorized_only` 鉴权）：`/list`、`/add <SYMBOL> [周期] [显示名]`、`/disable`、`/enable`、
`/remove`、`/signal [n] [SYMBOL]`。命令清单见 usage guide §5。

语义约定：

- **停用标的用 `disable`，不要 `remove`**：`remove` 是硬删除。删光某 market 的**全部**行属于"冷启动"状态，
  下次 bot 启动会重新播种默认 BTC/ETH（`ensure_defaults` 仅在表空时写入）；已有行存在时启动不会增删任何标的。
- `watch_pair` 缺表或 PG 不可达时，引擎每周期重试并打告警（`Failed to load pairs from database` / `Active pair whitelist is empty`），只降级不崩溃；补行/恢复后自动回到正常。PG 在启动瞬间不可用时冷启动播种会失败（仅告警），恢复后重启一次即可。
- A股标的同一张表，`--market ashare`（`freqtrade/exchange/ashare.py` 也读它）。

## 3. 冒烟运行（dry-run 盯盘）

```bash
# 1) 填 pa_llm.api_key（watch_config.json）
# 2) 启动
freqtrade trade --strategy PriceActionWatch --config user_data/watch_config.json
```

预期日志：`PriceActionWatch started (watch mode, 100 bars, N pairs, db=sqlite)`。
每根新K线触发一次两阶段 LLM 分析（增量模式）：
- `analysis_record` 表新增一行（完整记录）
- 有下单计划时 `signal` 表新增一行 + Telegram 文本推送（需在 config 开启 telegram 并填 token/chat_id）
- `watch_only=True`：永不写 `enter_long`（零交易）

## 4. 验证清单

| 检查 | 命令/方法 |
|---|---|
| 策略被 freqtrade 正确加载 | 启动日志无 resolver 报错 |
| PA 分析在跑 | `analysis_record` 表行数随K线增长 |
| 信号落库 | `signal` 表（仅下单计划时） |
| 增量分析生效 | 日志无 "full analysis" 每次全量；重启后 `Restored previous analysis for ...` |
| TG 推送 | 收到 `📊 {pair} {tf} PA信号` 文本 |

## 5. 回测（第8步，LLM 逐K线真实调用）

> 主配置已切 `DatabasePairList`（不支持回测）且无 `pair_whitelist`，**不能直接用于回测**。
> 回测请用保留 `StaticPairList` + `pair_whitelist` 的 `user_data/watch_config_offline_bt.json`
> （在线跑法见下；无交易所网络时走 §6.1 离线驱动）。

```bash
freqtrade download-data --config user_data/watch_config_offline_bt.json \
    --timerange 20250101-20250401
freqtrade backtesting --strategy PriceActionWatch \
    --config user_data/watch_config_offline_bt.json --timerange 20250101-20250401
```

注意：每根K线一次 LLM 调用，3 个月 1h ≈ 2,160 次。请先用 `--timerange 20250301-20250307` 小区间验证。
（offline 配置当前是 5m/testdata 场景；正式回测按需改其 `timeframe`/`pair_whitelist`。）

## 6. 切换实盘（watch → trade）

改三处配置：`dry_run:false`、`max_open_trades:N`、`"pa_llm": {"watch_only": false}`。
（watch_only 是 config 驱动的模式开关，不是 hyperopt 参数。）
代码路径不变；signal 表继续记录意图，`trades` 表记录执行，`executed_trade_id` 供对账。

## 6.1 离线回测（无交易所网络）

受限网络环境可用 `tools/bt_offline_driver.py`（monkeypatch 掉市场/费率联网，
配合 `user_data/watch_config_offline_bt.json` 与本地 feather 数据）：

```bash
.venv_pa/bin/python tools/bt_offline_driver.py --timerange 20251128-20251129
```

已验证（2026-09-13，tests/testdata BTC_USDT-5m，LLM 端点指向本地拒连端口模拟失败）：
289 根K线 → 289 次 PA 分析 → 289 条 strategy_exception partial 记录落库 →
signal 0 行 → 回测 0 trades → exit=0。全链路降级路径（引擎→适配→LLM失败→落库→统计）不崩。

## 7. 已知边界

- `.venv_pa`（会话内创建）：Python 3.11.14 + 清华镜像，freqtrade 完整依赖 + openai +
  psycopg3，**不含 ta-lib C 库**（brew 未装；PA 策略不需要 TA-Lib）。策略加载冒烟、
  7 个集成测试、离线回测均在此环境验证通过。
- `pa_core` 已加入 pyproject 打包清单（`pip install -e .` 后任意 cwd 可 import）。
- `pairlists` 已切 `DatabasePairList`（PG `watch_pair` 表动态标的，冷启动自动播种 BTC/ETH，见 §2.1）；配置不再含 `pair_whitelist`。回测/离线回测仍用 `watch_config_offline_bt.json` 的 `StaticPairList`（`DatabasePairList` 不支持回测）。
- A股标的需 `ashare` 交易所配置（`freqtrade/exchange/ashare.py`），本 config 以 binance/crypto 为例。
