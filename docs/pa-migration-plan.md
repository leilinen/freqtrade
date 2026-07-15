# PA_Agent 核心策略迁移到 freqtrade 的方案

> 源项目：`/Users/lilin/Documents/lilin/code/PA_Agent`（PyQt6 GUI 桌面应用，两阶段 LLM 价格行为决策系统）
> 目标：将 PA 核心策略迁入 freqtrade，复用 freqtrade 的 K线拉取、DatabasePairList、原生 telegram、PG 存储，保留 PA_Agent 的完整 LLM 决策能力。
> 决策：不保留 GUI，核心策略作为 `pa_core/` 包放入 freqtrade 仓库。

---

## 1. PA_Agent 架构回顾

PA_Agent 是一个**两阶段 LLM 决策系统**，price action 逻辑分两层：

```
第一层：客观几何特征计算（确定性代码）
  EMA20 / ATR14 → K线几何特征 → 市场结构特征 → K线类型判定
  ↓ 输出"客观事实"
第二层：LLM 决策
  阶段一：市场诊断（prompt → LLM → JSON）
  策略文件路由（按诊断选 prompt）
  阶段二：交易决策（prompt → LLM → JSON）
  JSON 校验（校验 LLM 输出的几何/盈亏比/一致性）
```

**关键事实**：PA_Agent 的"price action 核心"不是传统代码指标，而是一套 **prompt 工程 + LLM 推理**。客观几何特征只是喂给 LLM 的"事实"，真正做信号判定、方向判断、交易决策的是 LLM。

### PA_Agent 现状：没有自动化回测

PA_Agent 目前**没有回测功能**，只有"实时决策 + 事后人工复盘"：

| 机制 | 性质 |
|---|---|
| `trade_logger.save_trade_record()` | 每次 LLM 决策**实时**存 CSV + 图表（非回测） |
| `analysis_history.py` | 定位"上一次分析记录"用于增量分析（非回测） |
| `demo/replayer.py` | GUI 演示回放，重放已录制的 LLM 响应（非策略回测） |
| `trade_metrics.py` | 决策**时刻**校验盈亏比、交易者方程（实时校验，非事后统计） |
| `experience/` 目录 | 给 LLM 的参考案例库（目前为空） |

"迁移到 freqtrade 并支持 LLM 回测"是**新建能力**，不是迁移现有功能。

---

## 2. LLM 回测的技术约束与解决

### 约束：freqtrade 回测与 live 的调用粒度不同

```
Live / Dry-run（freqtradebot.py → interface.py:1213-1218）:
  每根新K线 → analyze_ticker(整个 df)
  process_only_new_candles=True 时只在新K线触发
  → 策略可以只看 iloc[-1]，LLM 每根新K线调一次 ✅

回测（backtesting.py:1775）:
  advise_all_indicators(data)   ← 一次性传入所有历史K线（几千根）
  ft_advise_signals(pair_data)  ← populate_entry_trend 收到整批 df
  → 策略必须在这一 次 调用里给每一行填好 enter_long
  → LLM 回测 = 逐行循环调 LLM（几千次）
```

### 解决方案：在 populate_* 里区分运行模式

```python
def populate_entry_trend(self, dataframe, metadata):
    if self.dp.runmode in (RunMode.LIVE, RunMode.DRY_RUN):
        # Live：只对最后一根调 LLM
        decision = self._pa_decide(dataframe.iloc[-1], metadata)
        if decision.is_entry:
            dataframe.loc[dataframe.index[-1], 'enter_long'] = 1
    elif self.dp.runmode == RunMode.BACKTEST:
        # 回测：逐根调 LLM，给每根填 enter_long
        for i in range(self.startup_candle_count, len(dataframe)):
            decision = self._pa_decide(dataframe.iloc[:i+1], metadata)
            if decision.is_entry:
                dataframe.loc[dataframe.index[i], 'enter_long'] = 1
    return dataframe
```

两条路径都能跑 LLM，区别是回测逐行遍历、live 只调最后一根。不考虑成本，接受回测时间开销。

---

## 3. 核心策略代码完整清单

核心策略分 **6 层**，共约 **16,800 行代码 + 32 个 prompt 文本文件**。

### 第1层：数据结构（190 行）— 需适配

| 文件 | 行数 | 职责 | 迁移处理 |
|---|---|---|---|
| `data/base.py` | 127 | `KlineBar`/`KlineFrame`/`IndicatorBundle` 数据类 + `DataSource` ABC | 保留数据类，丢弃 `DataSource` ABC（freqtrade 有自己的数据层） |
| `data/datetime_ts.py` | 73 | 时区安全的 epoch/datetime 转换 | 直接移植 |

> ⚠️ 这是核心策略的**数据地基**——所有特征计算都依赖 `KlineBar`。迁移时需写 `df_adapter.py`，把 freqtrade 的 pandas DataFrame 行映射成 `KlineBar`。

### 第2层：指标计算（194 行）— 直接移植

| 文件 | 行数 | 职责 | 迁移处理 |
|---|---|---|---|
| `indicators/ema.py` | 81 | EMA（全量 + 增量） | 直接移植，用原版（保证与 PA 决策一致，不用 TA-Lib） |
| `indicators/atr.py` | 113 | ATR Wilder（全量 + 增量） | 直接移植 |

### 第3层：客观特征计算（2,595 行）— 核心资产

| 文件 | 行数 | 职责 | 迁移处理 |
|---|---|---|---|
| `ai/kline_features.py` | 323 | 单根K线几何特征（body/wick/close_pos、bar_type、inside/ioi、micro-double、gap、breakout、follow-through） | 改 import 路径 |
| `ai/market_features.py` | 822 | 市场结构特征（区间、波段、突破、HL计数、支撑阻力、measured-move） | 改 import 路径 |
| `ai/structure_levels.py` | 296 | 确定性支撑/阻力刷新 | 改 import 路径 |
| `ai/trend_context.py` | 252 | Brooks 趋势上下文 + spike 检测 | 改 import 路径 |
| `ai/decision_nodes.py` | 3,071 | **决策树节点引擎**（PreflightDataGate、§1.1/§2.3/§2.4/§9/§11 判定器、OverrideArbiter） | 改 import 路径 |
| `ai/decision_tree.py` | 666 | 二元决策树加载器（解析 `二元决策.txt`）+ trace 辅助 | 改 import 路径，依赖 `config.paths` 的 PROMPT_DIR |
| `ai/cycle_enums.py` | 119 | cycle_position 枚举唯一真相源 | 直接移植 |

### 第4层：Prompt 组装与路由（3,318 行）— 核心资产

| 文件 | 行数 | 职责 | 迁移处理 |
|---|---|---|---|
| `ai/prompt_assembler.py` | 1,953 | 组装阶段一/二消息列表（人设、术语、K线表、特征、策略文件、经验库） | 改 import 路径 |
| `ai/router.py` | 246 | 阶段一诊断 → 策略 `.txt` 文件列表映射 | 改 import 路径 |
| `ai/pattern_routing.py` | 365 | 形态标签合并 + 阶段一路由简报 | 直接移植 |
| `ai/decision_stance.py` | 155 | 保守/平衡/激进立场配置 | 直接移植 |
| `ai/decision_continuity.py` | 712 | 前决策失效、翻转冷却、阶段二连续性阻断 | 🔴 **改造**：去掉 `trade_records/*.csv` 读取，改读 freqtrade PG |

### 第5层：LLM 调用 + 校验（5,577 行）— 核心资产

| 文件 | 行数 | 职责 | 迁移处理 |
|---|---|---|---|
| `orchestrator/two_stage.py` | 1,217 | **两阶段编排器**——完整流程驱动（核心入口） | 🔴 **改造**：去掉 QClaw/Cursor/WorkBuddy fallback（1058-1164行），去 GUI 回调 |
| `orchestrator/validation_retry.py` | 201 | validate_with_retry 包装 | 改 import 路径 |
| `ai/deepseek_client.py` | 812 | OpenAI 兼容流式 chat client | 改 import（去 `config.settings` GUI 部分） |
| `ai/mimo_compat.py` | 247 | MiMo reasoning_content 兼容 | 直接移植 |
| `ai/json_validator.py` | 1,165 | 阶段一/二 JSON 校验（分类 a–e） | 改 import 路径 |
| `ai/stage1_normalizer.py` | 744 | 阶段一 JSON 规范化 | 改 import 路径 |
| `ai/stage2_normalizer.py` | 1,816 | 阶段二 JSON 规范化 | 改 import 路径 |
| `ai/coherence_checks.py` | 787 | 跨字段一致性校验 | 改 import 路径 |
| `ai/trace_normalize.py` | 912 | gate_trace/decision_trace 规范化 | 改 import 路径 |
| `ai/trace_semantic_checks.py` | 257 | trace 语义校验 | 改 import 路径 |
| `ai/retry_policy.py` | 213 | 重试策略 | 直接移植 |
| `ai/retry_feedback.py` | 211 | 构建重试反馈消息 | 改 import 路径 |
| `ai/prompts/schemas.py` | 611 | JSON schema 定义（trace item、signal_bar、entry_bar、stage1/2） | 直接移植 |
| `ai/response_extract.py` | 33 | 从持久化响应字典提取 content/reasoning | 直接移植 |
| `ai/provider_errors.py` | 27 | 检测不可重试的 provider 配额失败 | 直接移植 |
| `ai/token_counter.py` | 32 | tiktoken token 估算 | 直接移植 |
| `ai/validation_messages.py` | 53 | 校验错误前缀的人类可读标签 | 直接移植 |

### 第6层：知识资产（32 个文本文件）— 原样复制

`prompt_engineering/*.txt`（32 文件），复制到 `pa_core/prompts/`：

- 核心框架：`二元决策.txt`、`提示词大纲_人设与思维方式.txt`、`市场诊断框架.txt`、`逐棒分析检查单.txt`
- 形态规则（文件13~28）：窄通道、宽通道、楔形、二次入场、K线信号识别、止损止盈与仓位管理、突破失败与突破测试、H1H2L1L2计数、AlwaysIn与20GB、铁丝网与无交易环境、信号失败后的磁力位、MeasuredMove与结构目标、最终旗形与趋势末端、主要趋势反转MTR、三角形与收敛形态、双重顶底与微型结构
- 通道/尖峰/区间分析+策略对：上涨/下跌通道分析识别+交易策略、极速上涨/下跌分析识别+交易策略、震荡区间分析识别+交易策略

### 辅助工具（必带）

| 文件 | 行数 | 职责 | 迁移处理 |
|---|---|---|---|
| `util/price_tick.py` | 158 | 价格最小跳动推断、tick 对齐、K-seq 解析 | 直接移植 |
| `util/trade_metrics.py` | 468 | 盈亏比、交易者方程校验（被 json_validator 依赖） | 直接移植 |
| `util/mask_secret.py` | 11 | API key 脱敏 | 直接移植 |
| `util/timefmt.py` | 8 | `now_local_ms()` | 直接移植 |
| `util/threading.py` | 43 | `CancelToken` + `OrchestratorEvent` 枚举 | 直接移植 |
| `config/paths.py` | 29 | 路径常量（PROMPT_DIR、EXPERIENCE_DIR、RECORDS_PENDING_DIR） | 改路径指向 `pa_core/prompts/` |
| `config/settings.py` | 280 | Pydantic 设置 | 🔴 **精简**：只保留 AI/Prompt/Validation/General 部分，去掉 GUI/数据源设置 |

### 记录持久化（改造后带）

| 文件 | 行数 | 职责 | 迁移处理 |
|---|---|---|---|
| `records/schema.py` | 101 | AnalysisRecord 等 Pydantic 模型 | 保留 |
| `records/experience_reader.py` | 208 | 读经验库案例（success/failure） | 保留，改路径 |
| `records/pending_writer.py` | 175 | 存 AnalysisRecord JSON | 🔴 **改造**：改成写 PG signal 表 |
| `records/analysis_history.py` | 138 | 定位历史记录（增量分析） | 🔴 **改造**：改读 PG |

---

## 4. 明确丢弃的（不迁移）

| 类别 | 文件 | 理由 |
|---|---|---|
| **GUI 全部** | `gui/`（30文件）、`main.py`、`app_context.py`、`demo/`、`util/event_bus.py`、`ai/session_ledger.py` | 不保留 GUI |
| **数据源全部** | `data/mt5.py`、`tradingview*.py`、`akshare_source.py`、`eastmoney*.py`（7文件）、`tushare_source.py`、`yfinance_source.py`、`factory.py`、`refresh_loop.py`、`refresh_policy.py`、`market_defaults.py`、`kline_adjust.py` | freqtrade 有自己的数据层 |
| **通知** | `notify/feishu_notifier.py`、`notify/pushplus_notifier.py` | freqtrade 有原生 telegram |
| **LLM 备用通道** | `cursor_connector.py`、`cursor_sdk_client.py`、`qclaw_*.py`（3文件）、`workbuddy_connector.py`、`client_factory.py` | 只保留 deepseek client |
| **trade_logger.py** | `records/trade_logger.py`（CSV+matplotlib 记录器） | freqtrade 有自己的交易 DB |

---

## 5. 迁移后的代码组织

`pa_core/` 放在 freqtrade 仓库内：

```
freqtrade/
├── pa_core/                              ← 迁移自 PA_Agent 的核心策略（纯 Python，无 GUI 依赖）
│   ├── __init__.py
│   ├── data_structures.py                ← data/base.py + data/datetime_ts.py（去 DataSource ABC）
│   ├── indicators.py                      ← indicators/ema.py + atr.py
│   ├── features/                          ← 客观特征计算
│   │   ├── __init__.py
│   │   ├── kline_features.py
│   │   ├── market_features.py
│   │   ├── structure_levels.py
│   │   └── trend_context.py
│   ├── decision/                          ← 决策树与判定
│   │   ├── __init__.py
│   │   ├── decision_nodes.py
│   │   ├── decision_tree.py
│   │   └── cycle_enums.py
│   ├── llm/                               ← LLM 编排
│   │   ├── __init__.py
│   │   ├── two_stage.py                   ← 核心入口（去备用通道 + GUI 回调）
│   │   ├── deepseek_client.py
│   │   ├── mimo_compat.py
│   │   ├── prompt_assembler.py
│   │   ├── router.py
│   │   ├── pattern_routing.py
│   │   ├── decision_stance.py
│   │   ├── decision_continuity.py         ← 改造（去 trade_records CSV，改读 PG）
│   │   └── validation_retry.py
│   ├── validation/                        ← JSON 校验
│   │   ├── __init__.py
│   │   ├── json_validator.py
│   │   ├── stage1_normalizer.py
│   │   ├── stage2_normalizer.py
│   │   ├── coherence_checks.py
│   │   ├── trace_normalize.py
│   │   ├── trace_semantic_checks.py
│   │   ├── retry_policy.py
│   │   ├── retry_feedback.py
│   │   └── schemas.py                     ← ai/prompts/schemas.py
│   ├── records/                           ← 改造成写 PG
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── experience_reader.py
│   │   └── persistence.py                 ← 改造自 pending_writer.py，写 PG signal 表
│   ├── util/
│   │   ├── __init__.py
│   │   ├── price_tick.py
│   │   ├── trade_metrics.py
│   │   ├── mask_secret.py
│   │   ├── timefmt.py
│   │   ├── threading.py
│   │   └── df_adapter.py                  ← 新建：freqtrade DataFrame ↔ KlineBar 适配（关键桥梁）
│   ├── config.py                           ← config/settings.py + paths.py（精简，去 GUI/数据源设置）
│   ├── prompts/                            ← prompt_engineering/ 的 32 个 .txt 文件原样复制
│   │   ├── 二元决策.txt
│   │   ├── 提示词大纲_人设与思维方式.txt
│   │   ├── 市场诊断框架.txt
│   │   ├── ... (共 32 个文件)
│   └── experience/                         ← 经验库目录结构（目前为空，保留结构）
│       ├── tight_channel/{success,failure}_cases/
│       └── ... (按 cycle_position 分类)
│
├── user_data/strategies/
│   └── price_action_watch.py              ← freqtrade 策略，import pa_core
├── docs/
│   ├── watch-system-design.md             ← watch 系统设计（已有）
│   └── pa-migration-plan.md               ← 本文档
└── ...
```

---

## 6. 策略层对接

```python
# user_data/strategies/price_action_watch.py
from freqtrade.strategy import IStrategy, BooleanParameter
from pa_core.llm.two_stage import TwoStageAnalyzer
from pa_core.util.df_adapter import df_to_kline_frame

class PriceActionWatch(IStrategy):
    watch_only = BooleanParameter(default=True, load=True)

    minimal_roi = {}
    stoploss = -0.99
    use_exit_signal = False

    def bot_start(self, **kwargs):
        """初始化 PA 核心 + 建 PG signal 表。"""
        self._pa = TwoStageAnalyzer(self.config["pa_llm"])
        # create_engine + 建 signal 表

    def populate_indicators(self, dataframe, metadata):
        """几何特征 + 结构特征（确定性，向量化，回测/live 都算）。"""
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        """信号判定：调 PA 两阶段 LLM 决策。"""
        if self.dp.runmode in (RunMode.LIVE, RunMode.DRY_RUN):
            # Live：只对最后一根调 LLM
            frame = df_to_kline_frame(dataframe)
            decision = self._pa.decide(frame, metadata)
            if decision.is_entry:
                self._record_signal(decision)  # 写 signal 表 + send_msg
                if not self.watch_only.value:
                    dataframe.loc[dataframe.index[-1], 'enter_long'] = 1
        elif self.dp.runmode == RunMode.BACKTEST:
            # 回测：逐根调 LLM
            for i in range(self.startup_candle_count, len(dataframe)):
                frame = df_to_kline_frame(dataframe.iloc[:i+1])
                decision = self._pa.decide(frame, metadata)
                if decision.is_entry:
                    dataframe.loc[dataframe.index[i], 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        return dataframe
```

---

## 7. 迁移工作量评估

| 部分 | 行数 | 难度 | 说明 |
|---|---|---|---|
| 直接复制（prompt 文本、小工具） | ~3,500 | 🟢 低 | 基本原样，改 import 路径 |
| 特征/决策树代码 | ~6,000 | 🟡 中 | 改 import 路径 + 适配 KlineBar 数据源 |
| LLM 编排/校验 | ~6,000 | 🟡 中 | 改 import + 去 GUI 耦合 + 去 cursor/qclaw 备用通道 |
| **需改造** | | 🔴 高 | |
| - `decision_continuity.py` | 712 | 🔴 | 去 `trade_records/*.csv` 读取，改读 freqtrade PG |
| - `pending_writer.py` → PG | 175 | 🔴 | 改成写 PG signal 表 |
| - `config/settings.py` | 280 | 🔴 | 精简，去 GUI/数据源设置 |
| - `df_adapter.py` | 新建 | 🔴 | freqtrade DataFrame ↔ KlineBar 映射（关键桥梁） |
| - `two_stage.py` | 1,217 | 🔴 | 去 QClaw/Cursor/WorkBuddy fallback（1058-1164行），去 GUI 回调 |

---

## 8. 迁移顺序

按依赖关系自底向上：

| 步骤 | 内容 | 依赖 |
|---|---|---|
| **1** | 数据结构 + 指标 + `df_adapter.py`（第1、2层 + 适配层） | 无（地基） |
| **2** | 客观特征计算（第3层：kline_features、market_features、structure_levels、trend_context、decision_nodes、decision_tree、cycle_enums） | 步骤1 |
| **3** | prompt 文本复制 + config 精简 + 辅助工具（第6层 + config + util） | 步骤1 |
| **4** | Prompt 组装与路由（第4层：prompt_assembler、router、pattern_routing、decision_stance、decision_continuity） | 步骤2、3 |
| **5** | LLM 调用 + 校验（第5层：two_stage、deepseek_client、json_validator、normalizer、coherence_checks 等） | 步骤4 |
| **6** | 记录持久化改造（records → PG signal 表） | 步骤5 |
| **7** | freqtrade 策略层对接（`price_action_watch.py`）+ 验证 live dry-run 能跑 | 步骤6 |
| **8** | 回测验证（freqtrade backtesting 逐根调 LLM） | 步骤7 |

---

## 9. 关键风险

1. **`df_adapter.py` 是迁移成败关键**：PA_Agent 所有特征计算都基于 `KlineBar` 对象（dataclass），freqtrade 用的是 pandas DataFrame。适配层必须正确映射（尤其"最新在前"vs"最旧在前"的顺序约定——PA_Agent 的 bars 是最新在前，freqtrade df 是时间升序最旧在前）。
2. **`two_stage.py` 去 GUI 回调**：编排器有大量 `on_stage1_reasoning`/`on_stage2_content` 等回调（GUI 流式显示用），迁移后这些回调要改成 no-op 或去掉。
3. **prompt 路径硬编码**：`config/paths.py` 的 `PROMPT_DIR = PROJECT_ROOT / "prompt_engineering"`，迁移后要改成指向 `pa_core/prompts/`。
4. **`decision_continuity.py` 的 trade_records 依赖**：这个模块读 CSV 判断"上一笔交易状态"，迁移后要改读 freqtrade PG 的 trades 表，或 signal 表。
5. **LLM 非确定性**：同样历史数据，两次回测结果可能不同。这是 LLM 固有特性，不是 bug——回测结果应多次取平均或看分布。
6. **回测时间**：几千根K线逐根调 LLM，回测耗时会很长（小时~天级）。不考虑成本，但需接受等待时间。
