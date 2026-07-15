# PA 价格行为盯盘系统设计文档

> 本文是整合主文档，统一描述「PA_Agent 核心策略迁移」与「freqtrade 盯盘系统」两层设计。
> 历史文档 `watch-system-design.md`（系统层）与 `pa-migration-plan.md`（策略层）仍保留备查，但权威版本以本文为准。

## 0. 一句话定位

把 PA_Agent 的**两阶段 LLM 价格行为决策系统**（PyQt6 GUI 桌面应用）迁移进 freqtrade，复用 freqtrade 的 K线拉取、DatabasePairList、原生 telegram、PostgreSQL 存储，构建一个「**只盯盘不下单（watch 模式）→ 可一键切到实盘（trade 模式）**」的量化系统。**不保留 GUI**，核心策略作为 `pa_core/` 包放入 freqtrade 仓库。

---

## 1. 目标与定位

- **保留完整 LLM 决策能力**：PA 的 price action 核心**不是传统代码指标**，而是一套 prompt 工程 + LLM 推理。客观几何特征只是喂给 LLM 的「事实」，真正做信号判定、方向判断、交易决策的是 LLM。LLM 不可舍弃。
- **只盯盘，不下单（watch 模式）**：检测信号并推送，不产生任何订单。
- **Telegram 富交互**：能在 TG 里看 K线图、查信号、管理监控标的。
- **PostgreSQL 作为查询/复盘层**：持久化信号记录、监控标的、freqtrade 交易数据。
- **不修改框架核心代码**：保持 fork 与上游 freqtrade 的同步能力。

### 1.1 watch → trade 是改配置，不是重写代码

**盯盘系统不是独立产品，而是量化交易系统的 watch 模式。** 从 watch 到实盘交易应该是**改配置，不是重写代码**：

| 阶段 | 配置 | 代码路径 |
|---|---|---|
| **watch 模式**（当前） | `dry_run:true` + `max_open_trades:0` + `watch_only:true` | 策略算特征 → LLM 决策 → 写 `signal` 表 → 推 TG，**不写 `enter_long`** |
| **trade 模式**（未来） | `dry_run:false` + `max_open_trades:N` + `watch_only:false` | 策略算特征 → LLM 决策 → 写 `signal` 表 → **写 `enter_long`** → freqtrade 自动下单 |

两个阶段共用同一套策略类、同一套 PA 核心决策逻辑、同一张 `signal` 表。差异只在：

1. 策略参数 `watch_only`（控制是否写 `enter_long`）
2. config 的 `dry_run` / `max_open_trades` / telegram 设置

### 1.2 `signal` 表的定位：交易意图账本（trade intent ledger）

在量化系统里，`signal` 表不是「通知记录」，而是**交易意图账本**：

- 记录「策略想做什么」（direction、entry_price、stop_loss、target_price、指标快照、**LLM 决策结果**）
- freqtrade 原生的 `trades`/`orders` 表（通过 `--db-url` 进 PG）记录「实际做了什么」
- 两张表 JOIN，才能算出量化策略迭代的核心指标：**成交率、计划vs实际滑点、目标价vs实际收益**

因此 `signal` 表必须：
- 在**信号检测时刻**由策略写入（`populate_entry_trend` 内），不能延后
- **模式无关**（watch/trade 都写，trade 模式下即使订单未成交也要有意图记录）
- 不依赖外部 bot 是否在线（实盘交易时 bot 可能挂，但意图记录不能丢）

> 复盘的两条价值都依赖这张表：
> - **复盘策略效果**：信号质量分布、胜率、计划vs实际收益
> - **检查决策逻辑**：信号判定时的指标快照 + LLM 输出，验证决策是否如设计工作

### 1.3 PA 迁移的边界

- 迁移 PA_Agent 的**策略核心**（LLM 决策 + 客观特征），**不迁移** GUI、数据源、通知、LLM 备用通道。
- PA_Agent **目前没有回测功能**，只有「实时决策 + 事后人工复盘」。「迁移到 freqtrade 并支持 LLM 回测」是**新建能力**，不是迁移现有功能。

---

## 2. 整体架构

系统分两个正交维度：

1. **纵向（PA 决策核心）**：客观几何特征 → 两阶段 LLM 决策 → 信号
2. **横向（freqtrade 平台）**：单进程引擎 + telegram 交互 + PostgreSQL 存储

```
┌──────────────────────────────────────────────────────────────────┐
│  单进程: freqtrade (dry_run + max_open_trades:0 + watch_only)      │
│                                                                    │
│  ┌─── 主线程 ───────────────────────────────────────────────────┐ │
│  │ user_data/strategies/price_action_watch.py                   │ │
│  │   class PriceActionWatch(IStrategy)                          │ │
│  │     watch_only: BoolParameter(default=True)                  │ │
│  │     bot_start()            ← 初始化 PA 核心 + 建 PG signal 表│ │
│  │     populate_indicators()  ← 调 pa_core 客观特征（确定性）     │ │
│  │     populate_entry_trend() ← PA 两阶段 LLM 决策               │ │
│  │                              → 写 signal 表 + send_msg        │ │
│  │                              → watch_only=False 时写 enter_long│ │
│  │     populate_exit_trend()  ← return df（空）                 │ │
│  │                                                              │ │
│  │ DatabasePairList ← 从 PG watch_pair 读白名单                 │ │
│  │                                                              │ │
│  │ pa_core/  ← PA_Agent 核心策略（纯 Python，无 GUI）           │ │
│  │   ├── TwoStageAnalyzer.decide(frame) → 两阶段 LLM 决策       │ │
│  │   └── features/ → 客观几何特征（EMA/ATR/K线/市场结构/决策树）│ │
│  └──────────────────────────────────────────────────────────────┘ │
│                           │                                        │
│  ┌─── telegram 线程（freqtrade 原生）──────────────────────────────┐ │
│  │ 原生命令: /status /profit /whitelist /health ...              │ │
│  │ dp.send_msg() 文本信号推送（策略侧触发）                       │ │
│  │                                                               │ │
│  │ telegram handler 插件（watch 扩展，本方案新增）:               │ │
│  │   /chart <pair> → self._rpc 拿 analyzed df → mplfinance → 发图│ │
│  │   /signal [n]   → 读 PG signal 表                             │ │
│  │   /watch        → 读 PG watch_pair 表                         │ │
│  │   /add /remove  → 写 PG watch_pair 表                         │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  --db-url postgresql+psycopg://...  (交易数据进PG)                 │
│  telegram.enabled:true + allow_custom_messages:true                │
│    + custom_handlers: ["WatchTelegramHandler"]                     │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  PostgreSQL                                                       │
│  watch_pair | signal | trades | orders | ...                     │
│  （无 ohlcv 表；K线在文件，回测读 feather）                        │
└──────────────────────────────────────────────────────────────────┘
```

### 2.1 三类数据的存储路径

| 数据类型 | 存储 | 写入方 | 机制 |
|---|---|---|---|
| **freqtrade 交易数据**（trades/orders/pairlocks 等） | PG | freqtrade 引擎 | `--db-url postgresql+psycopg://...`，零代码改动 |
| **K线 OHLCV** | 文件（feather） | freqtrade 引擎 | 框架原生，回测读它；插件画图走进程内 RPC。**不写 PG**（见 3.4） |
| **交易对/监控标的** | PG | telegram 插件（`/add` `/remove`）+ freqtrade 读取 | `watch_pair` 表，freqtrade 用 DatabasePairList 插件读 |
| **信号记录（交易意图）** | PG | freqtrade 策略 | 策略在 `populate_entry_trend` 检测到信号时写入 `signal` 表 |

---

## 3. 关键决策与依据

### 3.1 用 dry-run 实现盯盘，不新增 monitor 模式

盯盘通过 `dry_run:true + max_open_trades:0 + watch_only:true` 实现。

freqtrade 引擎从头到尾不知道你在「交易」还是「盯盘」。它只做三件事：拉K线 → 跑策略 `populate_*` → 读 `enter_long` 列决定要不要下单。「要不要下单」是策略层决定的，不是引擎模式决定的。新增 monitor runmode 只是在引擎里加个标签，底下还是同一套循环，却会破坏 fork 的上游同步。

`dry_run:true` 在这里的真实含义是「**需要一个能跑 dataprovider 和 REST API 的引擎实例，但不连真实交易所**」。

### 3.2 策略层职责边界

freqtrade 策略的本职是算指标、写信号列。本方案**唯一**的策略层持久化放宽是：**写 `signal` 表**（交易意图账本，见 1.2）。其余数据全部走框架原生路径。

| 行为 | 允许？ | 说明 |
|---|---|---|
| 算指标、加列 | ✅ | 策略本职 |
| 信号判定（调 LLM） | ✅ | 策略本职（PA 核心） |
| **写 PG（signal 表）** | ✅ | **本方案唯一放宽**：交易意图账本，低频（仅信号事件），模式无关（watch/trade 都写） |
| 写 PG（K线 ohlcv 表） | ❌ | 原生路径已覆盖，见 3.4 |
| `dp.send_msg()` 推 TG 文本 | ✅ | 框架原生的轻量通知通道（注意去重陷阱，见 9.3） |
| HTTP server | ❌ | 旧版 price_action 的反模式，禁止 |
| 自建通知管道（POST 外部服务） | ❌ | 由 telegram 插件负责 |

### 3.3 单进程职责分离（复用 freqtrade 原生 telegram）

**不另起外部 bot 进程**。复用 freqtrade 原生 telegram bot，通过 telegram handler 插件机制（见 7.1）扩展自定义命令。所有逻辑在 freqtrade 一个进程内：

| 组件 | 职责 | 与 PG 的关系 |
|---|---|---|
| **freqtrade 引擎 + 策略 + pa_core** | 拉K线、算特征、LLM 决策、检测信号、**写 signal 表** | 写 signal 表；经 `--db-url` 写 trades |
| **freqtrade 原生 telegram** | 原生命令（/status /profit /whitelist 等）+ `dp.send_msg()` 文本信号推送 | 经 RPC 读 trades |
| **telegram handler 插件**（watch 扩展） | 自定义命令（/chart /signal /watch /add /remove）+ K线图渲染 + 信号卡片推送 | 独立 PG 连接：读 signal/watch_pair，写 watch_pair |
| **PostgreSQL** | 统一存储 信号、交易对、freqtrade 交易数据 | 共享数据层 |

> **为什么不用外部 bot 进程**：外部 bot 要么 REST 轮询（延迟+复杂），要么 WS 订阅（复杂度高），都要处理进程间通信。复用原生 telegram + 插件扩展，命令处理函数在进程内直接拿 RPC 和 PG，最简单。代价是需要给框架加一个小型扩展点（见 7.1，对齐 freqtrade 已有的插件模式）。

### 3.4 不做 K线双写（决策依据）

两个曾列的理由经源码验证均不成立：

| 曾列理由 | 验证结果 | 结论 |
|---|---|---|
| K线写 PG → 画图 | `GET /pair_candles` 返回的是 `get_analyzed_dataframe()`——**策略算完指标的完整 df**，不只是 OHLCV。插件画图调 `self._rpc._rpc_analysed_dataframe()` 即可，且天然带指标列 | PG 不是画图的必要条件 |
| K线写 PG → 回测 | freqtrade 回测只从文件读（`backtesting.py:316` → `history.load_data(datadir=...)`）。`IDataHandler` 三个实现 feather/json/parquet **全部基于文件系统，无 DB 后端** | **PG 里的 K线对回测没用，回测读的是 feather** |

> **结论**：PG ohlcv 表没有真实消费者。freqtrade 文件存储（feather）已经是回测级的历史存储，REST/WS 是实时出口。再写一份 PG 是重复造轮子，还会带来每根K线高频 upsert 的性能负担。

**离线分析脚本需要历史K线怎么办？** 直接读 feather：`freqtrade.data.history.load_data(datadir, ...)` 一行代码拿到 DataFrame，比 SQL 快，且和回测数据源完全一致（消除「分析数据和回测数据不一致」的坑）。

### 3.5 无需 REST/WS 轮询（进程内直调）

因为 telegram handler 插件跑在 freqtrade 进程内（见 3.3），命令处理函数**直接**拿到所需数据，绕开了「外部 bot 怎么和 freqtrade 通信」这个整个问题域：

| 数据需求 | 外部 bot 方案（已弃用） | 进程内插件方案（本方案） |
|---|---|---|
| 画图要 analyzed df | REST `GET /pair_candles` 轮询 | `self._rpc._rpc_analysed_dataframe()` 进程内调用 |
| 查信号/标的 | 轮询 PG | 直读 PG（独立 session） |
| 推送新信号 | bot 轮询 PG signal 表 | `dp.send_msg()` 原生推送（策略侧） |

**REST `/pair_candles` 和 WebSocket `/api/v1/message/ws` 仍然存在**，可供外部工具（如 freqUI 网页）使用，但 watch 系统的 telegram 交互**不依赖它们**。

### 3.6 LLM 回测的技术约束与解决

freqtrade 回测与 live 的调用粒度不同：

```
Live / Dry-run（freqtradebot.py → interface.py:1213-1218）:
  每根新K线 → analyze_ticker(整个 df)
  process_only_new_candles=True 时只在新K线触发
  → 策略可以只看 iloc[-1]，LLM 每根新K线调一次 ✅

回测（backtesting.py:1775）:
  advise_all_indicators(data)   ← 一次性传入所有历史K线（几千根）
  ft_advise_signals(pair_data)  ← populate_entry_trend 收到整批 df
  → 策略必须在这一次调用里给每一行填好 enter_long
  → LLM 回测 = 逐行循环调 LLM（几千次）
```

**解决**：在 `populate_*` 里按 `runmode` 区分（见 6.1 的融合策略代码）。live 只对最后一根调 LLM；回测逐根遍历调 LLM 给每行填 `enter_long`。接受回测时间开销（见 9.6）。

---

## 4. PA 核心策略迁移

PA_Agent 核心策略分 **6 层**，共约 **16,800 行代码 + 32 个 prompt 文本文件**。迁移后放入 `pa_core/` 包。

### 4.1 架构回顾：两阶段 LLM 决策

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

### 4.2 第1层：数据结构（190 行）— 需适配

| 文件 | 行数 | 职责 | 迁移处理 |
|---|---|---|---|
| `data/base.py` | 127 | `KlineBar`/`KlineFrame`/`IndicatorBundle` 数据类 + `DataSource` ABC | 保留数据类，丢弃 `DataSource` ABC（freqtrade 有自己的数据层） |
| `data/datetime_ts.py` | 73 | 时区安全的 epoch/datetime 转换 | 直接移植 |

> ⚠️ 这是核心策略的**数据地基**——所有特征计算都依赖 `KlineBar`。迁移时需写 `df_adapter.py`，把 freqtrade 的 pandas DataFrame 行映射成 `KlineBar`（**这是迁移成败关键**，见 9.1）。

### 4.3 第2层：指标计算（194 行）— 直接移植

| 文件 | 行数 | 职责 | 迁移处理 |
|---|---|---|---|
| `indicators/ema.py` | 81 | EMA（全量 + 增量） | 直接移植，用原版（保证与 PA 决策一致，不用 TA-Lib） |
| `indicators/atr.py` | 113 | ATR Wilder（全量 + 增量） | 直接移植 |

### 4.4 第3层：客观特征计算（2,595 行）— 核心资产

| 文件 | 行数 | 职责 | 迁移处理 |
|---|---|---|---|
| `ai/kline_features.py` | 323 | 单根K线几何特征（body/wick/close_pos、bar_type、inside/ioi、micro-double、gap、breakout、follow-through） | 改 import 路径 |
| `ai/market_features.py` | 822 | 市场结构特征（区间、波段、突破、HL计数、支撑阻力、measured-move） | 改 import 路径 |
| `ai/structure_levels.py` | 296 | 确定性支撑/阻力刷新 | 改 import 路径 |
| `ai/trend_context.py` | 252 | Brooks 趋势上下文 + spike 检测 | 改 import 路径 |
| `ai/decision_nodes.py` | 3,071 | **决策树节点引擎**（PreflightDataGate、§1.1/§2.3/§2.4/§9/§11 判定器、OverrideArbiter） | 改 import 路径 |
| `ai/decision_tree.py` | 666 | 二元决策树加载器（解析 `二元决策.txt`）+ trace 辅助 | 改 import 路径，依赖 `config.paths` 的 PROMPT_DIR |
| `ai/cycle_enums.py` | 119 | cycle_position 枚举唯一真相源 | 直接移植 |

### 4.5 第4层：Prompt 组装与路由（3,318 行）— 核心资产

| 文件 | 行数 | 职责 | 迁移处理 |
|---|---|---|---|
| `ai/prompt_assembler.py` | 1,953 | 组装阶段一/二消息列表（人设、术语、K线表、特征、策略文件、经验库） | 改 import 路径 |
| `ai/router.py` | 246 | 阶段一诊断 → 策略 `.txt` 文件列表映射 | 改 import 路径 |
| `ai/pattern_routing.py` | 365 | 形态标签合并 + 阶段一路由简报 | 直接移植 |
| `ai/decision_stance.py` | 155 | 保守/平衡/激进立场配置 | 直接移植 |
| `ai/decision_continuity.py` | 712 | 前决策失效、翻转冷却、阶段二连续性阻断 | 🔴 **改造**：去掉 `trade_records/*.csv` 读取，改读 freqtrade PG |

### 4.6 第5层：LLM 调用 + 校验（5,577 行）— 核心资产

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

### 4.7 第6层：知识资产（32 个文本文件）— 原样复制

`prompt_engineering/*.txt`（32 文件），复制到 `pa_core/prompts/`：

- 核心框架：`二元决策.txt`、`提示词大纲_人设与思维方式.txt`、`市场诊断框架.txt`、`逐棒分析检查单.txt`
- 形态规则（文件13~28）：窄通道、宽通道、楔形、二次入场、K线信号识别、止损止盈与仓位管理、突破失败与突破测试、H1H2L1L2计数、AlwaysIn与20GB、铁丝网与无交易环境、信号失败后的磁力位、MeasuredMove与结构目标、最终旗形与趋势末端、主要趋势反转MTR、三角形与收敛形态、双重顶底与微型结构
- 通道/尖峰/区间分析+策略对：上涨/下跌通道分析识别+交易策略、极速上涨/下跌分析识别+交易策略、震荡区间分析识别+交易策略

### 4.8 辅助工具（必带）

| 文件 | 行数 | 职责 | 迁移处理 |
|---|---|---|---|
| `util/price_tick.py` | 158 | 价格最小跳动推断、tick 对齐、K-seq 解析 | 直接移植 |
| `util/trade_metrics.py` | 468 | 盈亏比、交易者方程校验（被 json_validator 依赖） | 直接移植 |
| `util/mask_secret.py` | 11 | API key 脱敏 | 直接移植 |
| `util/timefmt.py` | 8 | `now_local_ms()` | 直接移植 |
| `util/threading.py` | 43 | `CancelToken` + `OrchestratorEvent` 枚举 | 直接移植 |
| `config/paths.py` | 29 | 路径常量（PROMPT_DIR、EXPERIENCE_DIR、RECORDS_PENDING_DIR） | 改路径指向 `pa_core/prompts/` |
| `config/settings.py` | 280 | Pydantic 设置 | 🔴 **精简**：只保留 AI/Prompt/Validation/General 部分，去掉 GUI/数据源设置 |

### 4.9 记录持久化（改造后带）

| 文件 | 行数 | 职责 | 迁移处理 |
|---|---|---|---|
| `records/schema.py` | 101 | AnalysisRecord 等 Pydantic 模型 | 保留 |
| `records/experience_reader.py` | 208 | 读经验库案例（success/failure） | 保留，改路径 |
| `records/pending_writer.py` | 175 | 存 AnalysisRecord JSON | 🔴 **改造**：改成写 PG signal 表 |
| `records/analysis_history.py` | 138 | 定位历史记录（增量分析） | 🔴 **改造**：改读 PG |

### 4.10 明确丢弃的（不迁移）

| 类别 | 文件 | 理由 |
|---|---|---|
| **GUI 全部** | `gui/`（30文件）、`main.py`、`app_context.py`、`demo/`、`util/event_bus.py`、`ai/session_ledger.py` | 不保留 GUI |
| **数据源全部** | `data/mt5.py`、`tradingview*.py`、`akshare_source.py`、`eastmoney*.py`（7文件）、`tushare_source.py`、`yfinance_source.py`、`factory.py`、`refresh_loop.py`、`refresh_policy.py`、`market_defaults.py`、`kline_adjust.py` | freqtrade 有自己的数据层 |
| **通知** | `notify/feishu_notifier.py`、`notify/pushplus_notifier.py` | freqtrade 有原生 telegram |
| **LLM 备用通道** | `cursor_connector.py`、`cursor_sdk_client.py`、`qclaw_*.py`（3文件）、`workbuddy_connector.py`、`client_factory.py` | 只保留 deepseek client |
| **trade_logger.py** | `records/trade_logger.py`（CSV+matplotlib 记录器） | freqtrade 有自己的交易 DB |

### 4.11 迁移工作量评估

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

## 5. 数据层设计

### 5.1 freqtrade 原生能力清单

| 能力 | 机制 | 本方案用途 |
|---|---|---|
| 模拟运行 | `dry_run:true` | 跑引擎但不连真实交易所 |
| 禁止下单 | `max_open_trades:0` + `watch_only:true`（不写 `enter_long`） | 双保险零交易 |
| 交易数据进 PG | `--db-url postgresql+psycopg://...` | trades/orders 等直接进 PG |
| **原生 telegram bot** | `telegram.enabled:true` | **复用为 watch 系统的交互入口**（原生命令 + 插件扩展） |
| 策略推 TG 文本 | `self.dp.send_msg(str)` + `telegram.allow_custom_messages:true` | 信号文本推送（策略侧触发） |
| RPC analyzed df | `RPC._rpc_analysed_dataframe()` | 插件画图时进程内调用（无需 REST） |
| 暴露含指标的 df | `GET /pair_candles?pair=&timeframe=&limit=` | 供 freqUI / 外部工具使用（watch 插件不依赖） |
| 暴露白名单 | `GET /whitelist` | 备用；插件直接读 PG watch_pair |
| DatabasePairList 插件 | 从 PG `watch_pair` 表读白名单 | 动态标的管理（develop 分支已有） |
| WebSocket 订阅 | `WS /api/v1/message/ws` | 供 freqUI 等外部消费者（watch 插件不依赖） |

**原生缺失（需自建）**：自定义 TG 命令（通过 7.1 的 handler 插件扩展点）、富文本/图表渲染（插件内 mplfinance）、信号表（PG）。

### 5.2 PG 表结构（初步设计，待细化）

- `watch_pair`（监控标的）：symbol、market、enabled、display_name —— 复用 develop 分支已有的 DatabasePairList 约定
- `signal`（交易意图账本）：symbol、timeframe、candle_time、direction、quality、指标快照、**LLM 决策结果**、entry_price、stop_loss、target_price、**executed_trade_id（可空）** —— 见下方说明
- freqtrade 原生表（trades/orders/pairlocks/...）：由框架自动建表和管理

> **`signal.executed_trade_id` 的用途**：实盘模式下，由对账任务在订单成交后回填 freqtrade `trades` 表的外键。把「策略意图（signal）」和「实际执行（trades）」串起来，才能算成交率、计划vs实际滑点、目标价vs实际收益。watch 模式下此列为 NULL。

> **性能**：`signal` 表只在检测到信号时写入（低频），无需 upsert 优化。原 K线高频 upsert 的性能担忧已随 ohlcv 表取消而消失。

---

## 6. 策略层设计（融合版）

> 这是两份旧文档的**融合点**：watch-system-design 的「待定型骨架⚠️」与 pa-migration-plan 的「具体调 PA 两阶段」合并为一个无矛盾的策略实现。信号判定逻辑 = PA 两阶段 LLM 决策；副作用（写 signal 表 + send_msg）= watch 系统要求；运行模式分支 = LLM 回测约束要求。

### 6.1 策略骨架

```python
# user_data/strategies/price_action_watch.py
from freqtrade.enums import RunMode
from freqtrade.strategy import IStrategy, BooleanParameter
from pa_core.llm.two_stage import TwoStageAnalyzer
from pa_core.util.df_adapter import df_to_kline_frame


class PriceActionWatch(IStrategy):
    # watch/trade 模式开关：watch_only=True 不写 enter_long（盯盘），False 写 enter_long（实盘）
    watch_only = BooleanParameter(default=True, load=True)

    # 零交易配置（watch 模式下与 watch_only=True 双保险）
    minimal_roi = {}
    stoploss = -0.99
    use_exit_signal = False

    def bot_start(self, **kwargs):
        """初始化 PA 核心 + 建 PG signal 表。"""
        self._pa = TwoStageAnalyzer(self.config["pa_llm"])
        # create_engine + 建 signal 表（一次性）

    def populate_indicators(self, dataframe, metadata):
        """客观几何特征 + 结构特征（确定性，向量化，回测/live 都算）。

        调 pa_core.features 计算 K线/市场结构特征，结果挂在 df 列上。
        不写 PG：回测读 feather；插件画图经 self._rpc 进程内拿 analyzed df。
        """
        # pa_core 特征计算（EMA/ATR/K线几何/市场结构/决策树）
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        """信号判定：调 PA 两阶段 LLM 决策 → 写 signal 表 + 推 TG → 按模式写 enter_long。

        融合两份设计：
        - PA 两阶段 LLM 决策（信号判定核心）
        - watch_only 控制是否写 enter_long（watch/trade 切换）
        - 仅 LIVE/DRY_RUN 执行副作用（写 signal 表 + send_msg），回测只填 enter_long
        - 按 runmode 区分调用粒度（live 调最后一根，回测逐根遍历）
        """
        if self.dp.runmode in (RunMode.LIVE, RunMode.DRY_RUN):
            # Live：只对最后一根调 LLM
            frame = df_to_kline_frame(dataframe)
            decision = self._pa.decide(frame, metadata)
            if decision.is_entry:
                # 写 signal 表（交易意图账本）+ send_msg（always_send=True，见 9.3）
                self._record_signal(decision)
                # watch_only=False（trade 模式）时才写 enter_long
                if not self.watch_only.value:
                    dataframe.loc[dataframe.index[-1], 'enter_long'] = 1
        elif self.dp.runmode == RunMode.BACKTEST:
            # 回测：逐根调 LLM，给每根填 enter_long（回测不做副作用）
            for i in range(self.startup_candle_count, len(dataframe)):
                frame = df_to_kline_frame(dataframe.iloc[:i + 1])
                decision = self._pa.decide(frame, metadata)
                if decision.is_entry:
                    dataframe.loc[dataframe.index[i], 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        return dataframe
```

### 6.2 待定型项

- [ ] watch/trade 两套 config 的公共 base 抽取（仅差 `dry_run` / `max_open_trades` / `watch_only` / telegram 设置）
- [ ] 信号推送的 TG 文本格式
- [ ] signal 表中 LLM 决策结果的序列化格式（stage1 诊断 + stage2 决策 + trace）

---

## 7. 交互层设计（telegram 插件）

### 7.1 telegram handler 插件机制（框架扩展点）

为复用 freqtrade 原生 telegram 并扩展自定义命令，给框架加一个**对齐已有插件模式**（pairlist/protection/strategy 都用 `IResolver` 扫描目录发现子类）的扩展点。

**框架侧新增**（约 2 个小文件 + `telegram.py` 约 5 行修改）：

```python
# freqtrade/rpc/telegram_handlers/itelegram_handler.py（新建，约30行）
class ITelegramHandler(ABC):
    """telegram 命令插件基类。子类实现 commands() 返回命令映射。"""

    def __init__(self, config: Config, rpc: RPC) -> None:
        self._config = config
        self._rpc = rpc  # RPCManager：可调 _rpc_analysed_dataframe() 等（画图用）

    @abstractmethod
    def commands(self) -> dict[str, Callable]:
        """返回 {命令名: async 回调}。回调签名: async def cb(update, context)。"""

    def keyboard_buttons(self) -> list[list[str]]:
        """可选：加到 TG 菜单键盘的按钮。默认空。"""
        return []
```

```python
# freqtrade/resolvers/telegram_handler_resolver.py（新建，约40行，复用 IResolver）
class TelegramHandlerResolver(IResolver):
    object_type = ITelegramHandler
    initial_search_path = Path(__file__).parent.parent.joinpath("rpc/telegram_handlers")

    @staticmethod
    def load_handlers(config, rpc) -> list[ITelegramHandler]:
        names = config.get("telegram", {}).get("custom_handlers", [])
        return [TelegramHandlerResolver.load_object(n, config, kwargs={...}) for n in names]
```

```python
# freqtrade/rpc/telegram.py 的 _init() 第337行后（callbacks 循环后）新增约5行：
self._custom_handlers = TelegramHandlerResolver.load_handlers(self._config, self._rpc)
for handler_obj in self._custom_handlers:
    for cmd, callback in handler_obj.commands().items():
        self._app.add_handler(CommandHandler(cmd, callback))
```

config 配置：
```json
"telegram": {
    "enabled": true,
    "allow_custom_messages": true,
    "custom_handlers": ["WatchTelegramHandler"]
}
```

> **对上游同步的影响**：新增文件（`itelegram_handler.py` / `telegram_handler_resolver.py`）零冲突；`telegram.py` 约 5 行修改用 `custom_handlers` config 开关控制——不配置则零行为变化，上游合并时冲突概率极低。

### 7.2 用户侧 watch 插件

`user_data/telegram_handlers/watch_handler.py`（迁移旧 `services/tg_bot.py` 的业务逻辑）：

```python
class WatchTelegramHandler(ITelegramHandler):
    def __init__(self, config, rpc):
        super().__init__(config, rpc)
        # 独立 PG 连接（不和策略共享 session，线程安全）
        self._engine = create_engine(config["pa_db_url"])

    def commands(self):
        return {
            "chart": self._chart,      # /chart <pair> → 画K线图
            "signal": self._signal,    # /signal [n] → 查最近信号
            "watch": self._watch,      # /watch → 查监控标的
            "add": self._add,          # /add <pair> → 加监控标的
            "remove": self._remove,    # /remove <pair> → 删监控标的
        }
        # 命令的业务逻辑从旧 services/tg_bot.py 迁移：
        #   - 3 市场路由（crypto/ashare/usstock）
        #   - A股交易日判断、symbol 归一化
        #   - K线健康检查（_fetch_kline_health_rows）
        #   - /quote 路由逻辑 → /chart
```

### 7.3 线程模型

| 线程 | 跑什么 | PG 访问 |
|---|---|---|
| **主线程** | freqtrade 引擎循环 + 策略 `populate_*` + pa_core LLM 决策 + DatabasePairList | 策略写 signal 表（独立 session）；DatabasePairList 读 watch_pair |
| **telegram 线程** | python-telegram-bot 的 async event loop + 命令回调 | 插件独立 session：读 signal/watch_pair，写 watch_pair |

两个线程各自独立 PG session，无共享状态。`watch_pair` 表的跨线程读写靠 PG 默认 Read Committed 隔离级别保证可见性（handler 写入后，DatabasePairList 下次刷新即可读到）。

---

## 8. 实施计划（合并时间线）

两条工作线并行推进：**A 线 = 平台基础设施**（来自 watch-system-design），**B 线 = PA 核心迁移**（来自 pa-migration-plan），最终在「策略层对接」汇合。

### A 线：平台基础设施

| 步骤 | 内容 | 依赖 |
|---|---|---|
| **A1** | 准备 PostgreSQL；设计并建表 `watch_pair` / `signal`（无 `ohlcv` 表，见 3.4）；freqtrade 交易数据用 `--db-url` 接入 | 无 |
| **A2** | 框架扩展点：新建 `itelegram_handler.py` + `telegram_handler_resolver.py`，改 `telegram.py` 约 5 行，config schema 加 `telegram.custom_handlers` | A1 |
| **A3** | DatabasePairList 接入；`watch_pair` 表插入初始标的，验证白名单动态生效 | A1 |
| **A4** | watch telegram 插件：迁移旧 `tg_bot.py` 业务逻辑，实现 `/chart /signal /watch /add /remove` | A2、B6（signal 表） |

### B 线：PA 核心迁移

| 步骤 | 内容 | 依赖 |
|---|---|---|
| **B1** | 数据结构 + 指标 + `df_adapter.py`（第1、2层 + 适配层） | 无（地基） |
| **B2** | 客观特征计算（第3层：kline_features、market_features、structure_levels、trend_context、decision_nodes、decision_tree、cycle_enums） | B1 |
| **B3** | prompt 文本复制（32 文件）+ config 精简 + 辅助工具（第6层 + config + util） | B1 |
| **B4** | Prompt 组装与路由（第4层：prompt_assembler、router、pattern_routing、decision_stance、decision_continuity） | B2、B3 |
| **B5** | LLM 调用 + 校验（第5层：two_stage、deepseek_client、json_validator、normalizer、coherence_checks 等） | B4 |
| **B6** | 记录持久化改造（records → 写 PG signal 表） | B5、A1 |

### 汇合点

| 步骤 | 内容 | 依赖 |
|---|---|---|
| **C1** | freqtrade 策略层对接（`price_action_watch.py`，见第 6 节）+ 验证 live dry-run 能跑（PA 决策 + 写 signal 表 + send_msg） | B6 |
| **C2** | 回测验证（freqtrade backtesting 逐根调 LLM，见 3.6） | C1 |
| **C3** | Docker 部署：扩展 `docker-compose.yml`（`freqtrade-watch` + `postgres`，**单服务**）；watch 镜像带 psycopg/mplfinance（Dockerfile.custom） | C1 |

---

## 9. 关键约束与风险（去重合并）

### 9.1 `df_adapter.py` 是迁移成败关键
PA_Agent 所有特征计算都基于 `KlineBar` 对象（dataclass），freqtrade 用的是 pandas DataFrame。适配层必须正确映射（尤其「最新在前」vs「最旧在前」的顺序约定——PA_Agent 的 bars 是最新在前，freqtrade df 是时间升序最旧在前）。

### 9.2 `two_stage.py` 去 GUI 耦合
编排器有大量 `on_stage1_reasoning`/`on_stage2_content` 等回调（GUI 流式显示用），迁移后这些回调要改成 no-op 或去掉。同时去掉 QClaw/Cursor/WorkBuddy fallback（1058-1164行），只保留 deepseek client。

### 9.3 `dp.send_msg` 默认去重陷阱
`always_send=False`（默认）时，**同一根K线相同文本只发一次**。信号推送必须用 `always_send=True`，或让文本带唯一元素（如 candle_time），否则同K线多个信号会被吞。`dp.send_msg` 只发字符串（4096字符限制），图表/按钮由 telegram handler 插件渲染（`reply_photo` 等）。

### 9.4 prompt 路径硬编码
`config/paths.py` 的 `PROMPT_DIR = PROJECT_ROOT / "prompt_engineering"`，迁移后要改成指向 `pa_core/prompts/`。

### 9.5 `decision_continuity.py` 的 trade_records 依赖
这个模块读 CSV 判断「上一笔交易状态」，迁移后要改读 freqtrade PG 的 trades 表，或 signal 表。

### 9.6 LLM 非确定性 + 回测时间
- **非确定性**：同样历史数据，两次回测结果可能不同。这是 LLM 固有特性，不是 bug——回测结果应多次取平均或看分布。
- **回测时间**：几千根K线逐根调 LLM，回测耗时会很长（小时~天级）。需接受等待时间。

### 9.7 框架扩展点需要小改 `freqtrade/`
telegram handler 插件需新增 2 个文件 + `telegram.py` 约 5 行修改。这是为复用原生 telegram 付出的代价。用 `custom_handlers` config 开关控制，不配置则零行为变化，上游合并冲突概率低。需长期维护这套 fork 差异。

### 9.8 DatabasePairList 的限制
- **market 映射只有 2 值**：`ashare`/`crypto`（`DatabasePairList.py` 硬编码）。旧 `services/tg_bot.py` 支持 `usstock`。新系统若要美股，插件需扩展为 3 值。
- **refresh_period 缓存延迟**（默认 3600 秒）：`/add` 写完 `watch_pair` 后不会立即生效。handler 可调 freqtrade 原生 `/reload_config` 或 `/pause`+`/start` 触发刷新。

### 9.9 dry-run 与镜像
- `dry_run:true` 仍跑完整交易循环，会有空模拟订单表（进 PG），这是正常的。
- freqtrade 官方镜像不带 psycopg/mplfinance，watch 镜像需自建（Dockerfile.custom）。

### 9.10 psycopg 版本与跨线程 session
- `--db-url postgresql+psycopg://...` 用的是 psycopg3，需确认当前 SQLAlchemy 版本支持（较新版本支持）。
- telegram 线程与主线程各自独立 PG session，无共享状态（见 7.3）。`watch_pair` 跨线程读写靠 PG 默认 Read Committed 隔离级别保证可见性。

### 9.11 写 signal 表是策略层唯一持久化放宽
限定在写信号事件（低频、模式无关），不做 HTTP/通知/K线双写。这是「交易意图账本」的定位决定的（见 1.2），不能延后（信号时刻必须捕获）。

---

## 10. 文件清单（合并）

### 新建（用户侧）

| 文件 | 说明 |
|---|---|
| `docs/pa-watch-system-design.md` | 本文档（整合主文档） |
| `docs/watch-system-design.md` | 系统层历史文档（保留备查） |
| `docs/pa-migration-plan.md` | 策略层历史文档（保留备查） |
| `user_data/strategies/price_action_watch.py` | PriceActionWatch 策略（融合版，见第 6 节） |
| `user_data/telegram_handlers/watch_handler.py` | WatchTelegramHandler 插件，迁移旧 tg_bot.py 业务逻辑 |
| `user_data/watch_config.json` | watch 模式 config（dry_run/max_open_trades:0/db_url/custom_handlers） |
| PG 初始化 SQL | 表结构：`watch_pair` / `signal`，待设计 |

### 新建（框架侧，扩展点）

| 文件 | 说明 |
|---|---|
| `freqtrade/rpc/telegram_handlers/__init__.py` | 包初始化 |
| `freqtrade/rpc/telegram_handlers/itelegram_handler.py` | ITelegramHandler 基类 |
| `freqtrade/resolvers/telegram_handler_resolver.py` | 加载器，复用 IResolver |

### 新建（PA 核心迁移，`pa_core/` 包）

```
freqtrade/
├── pa_core/                              ← 迁移自 PA_Agent 的核心策略（纯 Python，无 GUI 依赖）
│   ├── __init__.py
│   ├── data_structures.py                ← data/base.py + data/datetime_ts.py（去 DataSource ABC）
│   ├── indicators.py                      ← indicators/ema.py + atr.py
│   ├── features/                          ← 客观特征计算（第3层）
│   │   ├── __init__.py
│   │   ├── kline_features.py
│   │   ├── market_features.py
│   │   ├── structure_levels.py
│   │   └── trend_context.py
│   ├── decision/                          ← 决策树与判定（第3层）
│   │   ├── __init__.py
│   │   ├── decision_nodes.py
│   │   ├── decision_tree.py
│   │   └── cycle_enums.py
│   ├── llm/                               ← LLM 编排（第4、5层）
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
│   ├── validation/                        ← JSON 校验（第5层）
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
│   │   └── ... (共 32 个文件)
│   └── experience/                         ← 经验库目录结构（目前为空，保留结构）
│       ├── tight_channel/{success,failure}_cases/
│       └── ... (按 cycle_position 分类)
```

### 复用（已有，不重写）

- `freqtrade/plugins/pairlist/DatabasePairList.py`（develop 分支已有，读 `watch_pair` 表）
- `freqtrade/exchange/ashare.py`（A股数据源）
- freqtrade 原生 telegram bot（复用为 watch 交互入口，通过插件扩展）

### 修改（框架侧，小改）

- `freqtrade/rpc/telegram.py`（`_init()` 约 5 行，加载并注册 custom handler）
- config schema（加 `telegram.custom_handlers` 字段）
- `docker-compose.yml`（加 `freqtrade-watch` + `postgres`，**单服务无 watch-bot**）

### 旧代码去留（`40a809d7c` 删除旧 price_action 系统后遗留的「半死」文件，全部删除）

| 文件 | 处置 |
|---|---|
| `services/tg_bot.py` | 删除。有价值的业务逻辑（3 市场路由、A股交易日判断、K线健康检查、命令逻辑）迁移到 `user_data/telegram_handlers/watch_handler.py` |
| `services/Dockerfile.tg` | 删除（无独立 bot 进程，不再需要） |
| `tools/eval_signal_quality.py` | 删除。如需保留离线分析能力，另写读 feather 的新版本 |
| `docker/docker-compose-pa.yml`、`docker/docker-compose-pa.azure.yml` | 删除，由新 `docker-compose.yml` watch 服务替代 |
| `docker/Dockerfile.pa` | 删除，改用 `docker/Dockerfile.custom` 模式构建带 psycopg/mplfinance 的 watch 镜像 |
| `migrate_add_display_name.sql` | 删除。新系统的 PG 初始化 SQL 直接建带 `display_name` 列的 `watch_pair` 表 |

### 不触碰

- `freqtrade/` 框架核心代码（仅上述 3 个新增文件 + telegram.py 约 5 行修改 + config schema，其余保持上游同步）
