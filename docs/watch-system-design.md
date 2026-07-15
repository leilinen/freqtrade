# 盯盘/模拟系统设计文档

基于 freqtrade dry-run 构建一个"只盯盘不下单、Telegram 富交互、PostgreSQL 统一存储"的价格行为监控系统。

## 1. 目标

- **只盯盘，不下单**：检测信号并推送，不产生任何订单
- **Telegram 富交互**：能在 TG 里看 K线图、查信号、管理监控标的
- **PostgreSQL 作为查询/复盘层**：持久化信号记录、监控标的、freqtrade 交易数据
- **不修改框架代码**：保持 fork 与上游 freqtrade 的同步能力

### 1.1 长期定位：盯盘是量化系统的第一阶段

**盯盘系统不是独立产品，而是量化交易系统的 watch 模式。** 从 watch 到实盘交易应该是**改配置，不是重写代码**：

| 阶段 | 配置 | 代码路径 |
|---|---|---|
| **watch 模式**（当前） | `dry_run:true` + `max_open_trades:0` + `watch_only:true` | 策略算指标 → 检测信号 → 写 `signal` 表 → 推 TG，**不写 `enter_long`** |
| **trade 模式**（未来） | `dry_run:false` + `max_open_trades:N` + `watch_only:false` | 策略算指标 → 检测信号 → 写 `signal` 表 → **写 `enter_long`** → freqtrade 自动下单 |

两个阶段共用同一套策略类、同一套信号检测逻辑、同一张 `signal` 表。差异只在：

1. 策略参数 `watch_only`（控制是否写 `enter_long`）
2. config 的 `dry_run` / `max_open_trades` / telegram 设置

为此，策略加一个 `watch_only: BoolParameter(default=True)`。`watch_only=False` 时写 `enter_long` 列。切换实盘 = 改参数 + 改 config，**代码路径不变**。

### 1.2 `signal` 表的定位：交易意图账本（trade intent ledger）

在量化系统里，`signal` 表不是"通知记录"，而是**交易意图账本**：

- 记录"策略想做什么"（direction、entry_price、stop_loss、target_price、指标快照）
- freqtrade 原生的 `trades`/`orders` 表（通过 `--db-url` 进 PG）记录"实际做了什么"
- 两张表 JOIN，才能算出量化策略迭代的核心指标：**成交率、计划vs实际滑点、目标价vs实际收益**

> 复盘的两条价值都依赖这张表：
> - **复盘策略效果**：信号质量分布、胜率、计划vs实际收益
> - **检查代码是否符合预期**：信号判定时的指标快照、计划入场/止损/目标，验证逻辑是否如设计工作

因此 `signal` 表必须：
- 在**信号检测时刻**由策略写入（`populate_entry_trend` 内），不能延后
- **模式无关**（watch/trade 都写，trade 模式下即使订单未成交也要有意图记录）
- 不依赖外部 bot 是否在线（实盘交易时 bot 可能挂，但意图记录不能丢）

---

## 2. 关键决策与依据

### 2.1 用 dry-run 实现盯盘，不新增 monitor 模式

盯盘通过 `dry_run:true + max_open_trades:0 + watch_only:true（策略不写 enter_long）` 实现。

freqtrade 引擎从头到尾不知道你在"交易"还是"盯盘"。它只做三件事：拉K线 → 跑策略 `populate_*` → 读 `enter_long` 列决定要不要下单。"要不要下单"是策略层决定的，不是引擎模式决定的。新增 monitor runmode 只是在引擎里加个标签，底下还是同一套循环，却会破坏 fork 的上游同步。

`dry_run:true` 在这里的真实含义是"**需要一个能跑 dataprovider 和 REST API 的引擎实例，但不连真实交易所**"。

### 2.2 策略层职责边界

freqtrade 策略的本职是算指标、写信号列。本方案**唯一**的策略层持久化放宽是：**写 `signal` 表**（交易意图账本，见 1.2）。其余数据全部走框架原生路径。

| 行为 | 允许？ | 说明 |
|---|---|---|
| 算指标、加列 | ✅ | 策略本职 |
| 信号判定 | ✅ | 策略本职 |
| **写 PG（signal 表）** | ✅ | **本方案唯一放宽**：交易意图账本，低频（仅信号事件），模式无关（watch/trade 都写） |
| 写 PG（K线 ohlcv 表） | ❌ | 原生路径已覆盖，见 2.4 |
| `dp.send_msg()` 推 TG 文本 | ✅ | 框架原生的轻量通知通道（注意去重陷阱，见 8.3） |
| HTTP server | ❌ | 旧版 price_action 的反模式，禁止 |
| 自建通知管道（POST 外部服务） | ❌ | 由外部 bot 负责 |

### 2.3 单进程职责分离（freqtrade 内置 telegram）

**不另起外部 bot 进程**。复用 freqtrade 原生 telegram bot，通过 telegram handler 插件机制（见 5.5）扩展自定义命令。所有逻辑在 freqtrade 一个进程内：

| 组件 | 职责 | 与 PG 的关系 |
|---|---|---|
| **freqtrade 引擎 + 策略** | 拉K线、算指标、检测信号、**写 signal 表** | 写 signal 表；经 `--db-url` 写 trades |
| **freqtrade 原生 telegram** | 原生命令（/status /profit /whitelist 等）+ `dp.send_msg()` 文本信号推送 | 经 RPC 读 trades |
| **telegram handler 插件**（watch 扩展） | 自定义命令（/chart /signal /watch /add /remove）+ K线图渲染 + 信号卡片推送 | 独立 PG 连接：读 signal/watch_pair，写 watch_pair |
| **PostgreSQL** | 统一存储 信号、交易对、freqtrade 交易数据 | 共享数据层 |

数据流向（全在进程内，无 REST/WS 轮询）：
- **K线**：freqtrade 拉取 → 写文件（feather，框架原生，回测读它）；插件画图时**直接经 `self._rpc` 拿 analyzed df**（带指标的完整 df，进程内调用，无需 REST）
- **信号**：策略检测到信号 → 写 PG `signal` 表 + `dp.send_msg()` 推 TG 文本
- **图表渲染**：`/chart` 命令处理函数 → 经 `self._rpc._rpc_analysed_dataframe()` 拿 df → mplfinance 渲染 → `reply_photo()` 发图
- **信号/标的查询**：插件直读 PG（独立 session，进程内调用，无需 REST）
- **freqtrade 交易数据**：通过 `--db-url postgresql+psycopg://...` 直接进 PG，零代码改动

> **为什么不用外部 bot 进程**：外部 bot 要么 REST 轮询（延迟+复杂），要么 WS 订阅（复杂度高），都要处理进程间通信。复用原生 telegram + 插件扩展，命令处理函数在进程内直接拿 RPC 和 PG，最简单。代价是需要给框架加一个小型扩展点（见 5.5，对齐 freqtrade 已有的插件模式）。

### 2.4 不做 K线双写（决策依据）

两个曾列的理由经源码验证均不成立：

| 曾列理由 | 验证结果 | 结论 |
|---|---|---|
| K线写 PG → 画图 | `GET /pair_candles` 返回的是 `get_analyzed_dataframe()`——**策略算完指标的完整 df**，不只是 OHLCV。bot 画图调这个接口即可，且天然带指标列 | PG 不是画图的必要条件 |
| K线写 PG → 回测 | freqtrade 回测只从文件读（`backtesting.py:316` → `history.load_data(datadir=...)`）。`IDataHandler` 三个实现 feather/json/parquet **全部基于文件系统，无 DB 后端** | **PG 里的 K线对回测没用，回测读的是 feather** |

> **结论**：PG ohlcv 表没有真实消费者。freqtrade 文件存储（feather）已经是回测级的历史存储，REST/WS 是实时出口。再写一份 PG 是重复造轮子，还会带来每根K线高频 upsert 的性能负担（原 8.2 节担忧）。

**离线分析脚本（如 `tools/eval_signal_quality.py`）需要历史K线怎么办？** 直接读 feather：`freqtrade.data.history.load_data(datadir, ...)` 一行代码拿到 DataFrame，比 SQL 快，且和回测数据源完全一致（消除"分析数据和回测数据不一致"的坑）。

### 2.5 无需 REST/WS 轮询（进程内直调）

因为 telegram handler 插件跑在 freqtrade 进程内（见 2.3），命令处理函数**直接**拿到所需数据，绕开了"外部 bot 怎么和 freqtrade 通信"这个整个问题域：

| 数据需求 | 外部 bot 方案（已弃用） | 进程内插件方案（本方案） |
|---|---|---|
| 画图要 analyzed df | REST `GET /pair_candles` 轮询 | `self._rpc._rpc_analysed_dataframe()` 进程内调用 |
| 查信号/标的 | 轮询 PG | 直读 PG（独立 session） |
| 推送新信号 | bot 轮询 PG signal 表 | `dp.send_msg()` 原生推送（策略侧） |

**REST `/pair_candles` 和 WebSocket `/api/v1/message/ws` 仍然存在**，可供外部工具（如 freqUI 网页、`eval_signal_quality.py`）使用，但 watch 系统的 telegram 交互**不依赖它们**。

---

## 3. 数据存储方案（三类数据）

| 数据类型 | 存储 | 写入方 | 机制 |
|---|---|---|---|
| **freqtrade 交易数据**（trades/orders/pairlocks 等） | PG | freqtrade 引擎 | `--db-url postgresql+psycopg://...`，需装 psycopg，零代码改动 |
| **K线 OHLCV** | 文件（feather） | freqtrade 引擎 | 框架原生，回测读它；bot 画图走 REST `/pair_candles`，实时数据走 WS `analyzed_df`。**不写 PG**（见 2.4） |
| **交易对/监控标的** | PG | 外部 bot（`/add` `/remove`）+ freqtrade 读取 | `watch_pair` 表，freqtrade 用 DatabasePairList 插件读 |
| **信号记录（交易意图）** | PG | freqtrade 策略 | 策略在 `populate_entry_trend` 检测到信号时写入 `signal` 表，bot 读 PG 查询 |

### PG 表结构（初步设计，待细化）

- `watch_pair`（监控标的）：symbol、market、enabled、display_name —— 复用 develop 分支已有的 DatabasePairList 约定
- `signal`（交易意图账本）：symbol、timeframe、candle_time、direction、quality、指标快照、entry_price、stop_loss、target_price、**executed_trade_id（可空）** —— 见下方说明
- freqtrade 原生表（trades/orders/pairlocks/...）：由框架自动建表和管理

> **`signal.executed_trade_id` 的用途**：实盘模式下，由对账任务在订单成交后回填 freqtrade `trades` 表的外键。把"策略意图（signal）"和"实际执行（trades）"串起来，才能算成交率、计划vs实际滑点、目标价vs实际收益。watch 模式下此列为 NULL。

> **性能**：`signal` 表只在检测到信号时写入（低频），无需 upsert 优化。原 K线高频 upsert 的性能担忧已随 ohlcv 表取消而消失。

---

## 4. freqtrade 原生能力清单

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

**原生缺失（需自建）**：自定义 TG 命令（通过 5.2 的 handler 插件扩展点）、富文本/图表渲染（插件内 mplfinance）、信号表（PG）。

---

## 5. 架构设计（单进程）

```
┌─────────────────────────────────────────────────────────────┐
│  单进程: freqtrade (dry_run + max_open_trades:0 + watch_only) │
│                                                               │
│  ┌─── 主线程 ──────────────────────────────────────────────┐ │
│  │ user_data/strategies/watch_strategy.py                  │ │
│  │   class PriceActionWatch(IStrategy)                     │ │
│  │     watch_only: BoolParameter(default=True)             │ │
│  │     bot_start()            ← 建 PG 连接、建 signal 表   │ │
│  │     populate_indicators() ← 算指标（纯计算，不写PG）     │ │
│  │     populate_entry_trend() ← 信号判定 + 写signal表+send_msg │
│  │                            （watch_only=True 时不写 enter_long）│ │
│  │     populate_exit_trend()  ← return df（空）            │ │
│  │                                                         │ │
│  │ DatabasePairList ← 从 PG watch_pair 读白名单            │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                                  │
│  ┌─── telegram 线程（freqtrade 原生）────────────────────────┐ │
│  │ 原生命令: /status /profit /whitelist /health ...        │ │
│  │ dp.send_msg() 文本信号推送（策略侧触发）                 │ │
│  │                                                         │ │
│  │ telegram handler 插件（watch 扩展，本方案新增）:         │ │
│  │   user_data/telegram_handlers/watch_handler.py          │ │
│  │   class WatchTelegramHandler(ITelegramHandler)          │ │
│  │     /chart <pair>  → self._rpc 拿 analyzed df           │ │
│  │                    → mplfinance 渲染 → reply_photo      │ │
│  │     /signal [n]    → 读 PG signal 表                    │ │
│  │     /watch         → 读 PG watch_pair 表                │ │
│  │     /add /remove   → 写 PG watch_pair 表                │ │
│  │   (独立 PG session，线程安全)                           │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  --db-url postgresql+psycopg://...  (交易数据进PG)            │
│  --config .../watch_config.json                                │
│  telegram.enabled:true + allow_custom_messages:true           │
│    + custom_handlers: ["WatchTelegramHandler"]  ← 插件加载    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  PostgreSQL                                                 │
│  watch_pair | signal | trades | orders | ...                │
│  （无 ohlcv 表；K线在文件，回测读 feather）                  │
└─────────────────────────────────────────────────────────────┘
```

### 5.1 线程模型

| 线程 | 跑什么 | PG 访问 |
|---|---|---|
| **主线程** | freqtrade 引擎循环 + 策略 `populate_*` + DatabasePairList | 策略写 signal 表（独立 session）；DatabasePairList 读 watch_pair |
| **telegram 线程** | python-telegram-bot 的 async event loop + 命令回调 | 插件独立 session：读 signal/watch_pair，写 watch_pair |

两个线程各自独立 PG session，无共享状态。`watch_pair` 表的跨线程读写靠 PG 默认 Read Committed 隔离级别保证可见性（handler 写入后，DatabasePairList 下次刷新即可读到）。

### 5.2 telegram handler 插件机制（框架扩展点）

为复用 freqtrade 原生 telegram 并扩展自定义命令，给框架加一个**对齐已有插件模式**（pairlist/protection/strategy 都用 `IResolver` 扫描目录发现子类）的扩展点。

**框架侧新增**（约 3 个小文件 + `telegram.py` 约 5 行修改）：

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

**用户侧 watch 插件**（`user_data/telegram_handlers/watch_handler.py`，迁移旧 `tg_bot.py` 的业务逻辑）：

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

---

## 6. 策略实现（待策略逻辑定型后细化）

> ⚠️ 策略的具体指标和信号判定逻辑**还在构思中，尚未定型**。本节描述的是策略的**框架和职责**，具体算法待补充。

### 6.1 策略骨架

```python
class PriceActionWatch(IStrategy):
    # watch/trade 模式开关：watch_only=True 不写 enter_long（盯盘），False 写 enter_long（实盘）
    watch_only = BoolParameter(default=True, load=True)

    # 零交易配置（watch 模式下与 watch_only=True 双保险）
    minimal_roi = {}
    stoploss = -0.99
    use_exit_signal = False
    max_open_trades = 0  # 实盘切换时在 config 里设

    def bot_start(self, **kwargs):
        """初始化 PG 连接、确保 signal 表存在。"""
        # create_engine + 建 signal 表（一次性）

    def populate_indicators(self, dataframe, metadata):
        """算指标（纯计算，不写 PG）。"""
        # 算指标（策略逻辑待定型）
        # K线不写 PG：回测读 feather；telegram 插件画图经 self._rpc 进程内拿 analyzed df
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        """信号判定 + 写 signal 表 + 推 TG 文本。"""
        # 1. 信号判定（策略逻辑待定型）
        # 2. 仅在 LIVE/DRY_RUN 模式执行副作用（排除回测）
        # 3. 有信号 → 写 PG signal 表（symbol, timeframe, candle_time, direction,
        #    quality, 指标快照, entry_price, stop_loss, target_price）
        # 4. 有信号 → self.dp.send_msg("信号文本", always_send=True)
        # 5. watch_only=False 时才写 enter_long（实盘下单）
        if not self.watch_only.value:
            dataframe.loc[signal_cond, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        return dataframe
```

### 6.2 待定型项

- [ ] 具体计算哪些指标
- [ ] 信号判定条件（什么情况算一个有效信号）
- [ ] 信号质量分级（如需要）
- [ ] 推送的 TG 文本格式
- [ ] watch/trade 两套 config 的公共 base 抽取（仅差 `dry_run` / `max_open_trades` / `watch_only` / telegram 设置）

---

## 7. 实施计划

### 第1步：基础设施（PG + 表结构）
- 准备 PostgreSQL 实例
- 设计并建表：`watch_pair` / `signal`（无 `ohlcv` 表，见 2.4）
- freqtrade 交易数据用 `--db-url postgresql+psycopg://...` 接入

### 第2步：框架扩展点（telegram handler 插件机制）
- 新建 `freqtrade/rpc/telegram_handlers/itelegram_handler.py`（基类）
- 新建 `freqtrade/resolvers/telegram_handler_resolver.py`（加载器，复用 IResolver）
- 修改 `freqtrade/rpc/telegram.py` `_init()`（约 5 行，加载并注册自定义 handler）
- config schema 加 `telegram.custom_handlers` 字段
- **验证**：写一个最简 echo handler 插件，TG 发 `/echo` 能收到回显

### 第3步：策略骨架（PG 连接 + signal 表）
- 新建 `watch_strategy.py`：`PriceActionWatch(IStrategy)`，含 `watch_only` 参数
- `bot_start` 建 PG 连接 + 建 signal 表
- `populate_indicators` 用最简指标（纯计算，不写 PG）
- 新建 `watch_config.json`：`dry_run:true`、`max_open_trades:0`、`db_url` 指向 PG、`telegram.custom_handlers`
- **验证**：freqtrade 跑起来，PG signal 表建好，原生 TG 命令（/status）可用

### 第4步：DatabasePairList 接入
- 配置 `pairlists: [{"method": "DatabasePairList", ...}]`
- 在 `watch_pair` 表插入初始标的
- **验证**：freqtrade 白名单随 watch_pair 表变化

### 第5步：信号检测 + 落库 + 推送
- `populate_entry_trend` 实现信号判定（策略逻辑定型后）
- 写 `signal` 表 + `dp.send_msg()` 推 TG 文本
- **验证**：检测到信号时 PG 有记录 + 原生 TG 收到文本

### 第6步：watch telegram 插件（迁移旧 tg_bot.py 能力）
- 新建 `user_data/telegram_handlers/watch_handler.py`：`WatchTelegramHandler(ITelegramHandler)`
- 从旧 `services/tg_bot.py` 迁移业务逻辑：3 市场路由、A股交易日判断、K线健康检查
- 实现 `/chart <pair>`（经 `self._rpc` 拿 df → mplfinance 渲染 → reply_photo）
- 实现 `/signal`、`/watch`、`/add`、`/remove`（读/写 PG）
- **验证**：各命令可用，/chart 能发 K线图

### 第7步：Docker 部署
- 扩展 `docker-compose.yml`：`freqtrade-watch` + `postgres`（**单服务**，无 watch-bot）
- freqtrade 容器需带 psycopg/mplfinance（用 Dockerfile.custom）

---

## 8. 关键约束与风险

1. **写 signal 表是策略层唯一持久化放宽**：限定在写信号事件（低频、模式无关），不做 HTTP/通知/K线双写。这是"交易意图账本"的定位决定的（见 1.2），不能延后（信号时刻必须捕获）
2. **框架扩展点（telegram handler 插件）需要小改 `freqtrade/`**：新增 2 个文件 + `telegram.py` 约 5 行修改。这是为复用原生 telegram 付出的代价（见 5.2）。用 `custom_handlers` config 开关控制，不配置则零行为变化，上游合并冲突概率低。需长期维护这套 fork 差异
3. **dp.send_msg 默认去重陷阱**：`always_send=False`（默认）时，**同一根K线相同文本只发一次**。信号推送必须用 `always_send=True`，或让文本带唯一元素（如 candle_time），否则同K线多个信号会被吞
4. **dp.send_msg 只发字符串**（4096字符限制），图表/按钮由 telegram handler 插件渲染（`reply_photo` 等）
5. **DatabasePairList 的 market 映射只有 2 值**：`ashare`/`crypto`（`DatabasePairList.py` 硬编码）。旧 `services/tg_bot.py` 支持 `usstock`。新系统若要美股，插件需扩展为 3 值
6. **DatabasePairList 的 refresh_period 缓存延迟**（默认 3600 秒）：`/add` 写完 `watch_pair` 后不会立即生效。handler 可调 freqtrade 原生 `/reload_config` 或 `/pause`+`/start` 触发刷新
7. **freqtrade 官方镜像不带 psycopg/mplfinance**，watch 镜像需自建（Dockerfile.custom）
8. **dry_run 仍跑完整交易循环**，会有空模拟订单表（进 PG），这是正常的
9. **策略逻辑尚未定型**（第6节标 ⚠️），实施计划第5步依赖策略设计完成
10. **psycopg 版本确认**：`--db-url postgresql+psycopg://...` 用的是 psycopg3，需确认当前 SQLAlchemy 版本支持（较新版本支持）
11. **telegram 线程与主线程独立 PG session**：插件 handler 跑在 telegram 线程，策略 `populate_*` 跑在主线程，各自独立 SQLAlchemy session，无共享状态。`watch_pair` 跨线程读写靠 PG 默认 Read Committed 隔离级别保证可见性

---

## 9. 文件清单

**新建（用户侧）**：
- `docs/watch-system-design.md`（本文档）
- `user_data/strategies/watch_strategy.py`（PriceActionWatch 策略）
- `user_data/telegram_handlers/watch_handler.py`（WatchTelegramHandler 插件，迁移旧 tg_bot.py 业务逻辑）
- `user_data/watch_config.json`
- PG 初始化 SQL（表结构：`watch_pair` / `signal`，待设计）

**新建（框架侧，扩展点）**：
- `freqtrade/rpc/telegram_handlers/__init__.py`
- `freqtrade/rpc/telegram_handlers/itelegram_handler.py`（ITelegramHandler 基类）
- `freqtrade/resolvers/telegram_handler_resolver.py`（加载器，复用 IResolver）

**复用（已有，不重写）**：
- `freqtrade/plugins/pairlist/DatabasePairList.py`（develop 分支已有，读 `watch_pair` 表）
- `freqtrade/exchange/ashare.py`（A股数据源）
- freqtrade 原生 telegram bot（复用为 watch 交互入口，通过插件扩展）

**修改（框架侧，小改）**：
- `freqtrade/rpc/telegram.py`（`_init()` 约 5 行，加载并注册 custom handler）
- config schema（加 `telegram.custom_handlers` 字段）
- `docker-compose.yml`（加 `freqtrade-watch` + `postgres`，**单服务无 watch-bot**）

**旧代码去留**（`40a809d7c` 删除旧 price_action 系统后遗留的"半死"文件，全部删除）：

| 文件 | 处置 |
|---|---|
| `services/tg_bot.py` | 删除。有价值的业务逻辑（3 市场路由、A股交易日判断、K线健康检查、命令逻辑）迁移到 `user_data/telegram_handlers/watch_handler.py` |
| `services/Dockerfile.tg` | 删除（无独立 bot 进程，不再需要） |
| `tools/eval_signal_quality.py` | 删除。如需保留离线分析能力，另写读 feather 的新版本 |
| `docker/docker-compose-pa.yml`、`docker/docker-compose-pa.azure.yml` | 删除，由新 `docker-compose.yml` watch 服务替代 |
| `docker/Dockerfile.pa` | 删除，改用 `docker/Dockerfile.custom` 模式构建带 psycopg/mplfinance 的 watch 镜像 |
| `migrate_add_display_name.sql` | 删除。新系统的 PG 初始化 SQL 直接建带 `display_name` 列的 `watch_pair` 表 |

**不触碰**：
- `freqtrade/` 框架核心代码（仅上述 3 个新增文件 + telegram.py 约 5 行修改 + config schema，其余保持上游同步）
