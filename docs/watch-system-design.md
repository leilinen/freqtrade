# 盯盘/模拟系统设计文档

基于 freqtrade dry-run 构建一个"只盯盘不下单、Telegram 富交互、PostgreSQL 统一存储"的价格行为监控系统。

## 1. 目标

- **只盯盘，不下单**：检测信号并推送，不产生任何订单
- **Telegram 富交互**：能在 TG 里看 K线图、查信号、管理监控标的
- **PostgreSQL 统一存储**：所有需要持久化的数据（K线、信号、交易对、freqtrade 交易数据）都放 PG
- **不修改框架代码**：保持 fork 与上游 freqtrade 的同步能力

---

## 2. 关键决策与依据

### 2.1 用 dry-run 实现盯盘，不新增 monitor 模式

盯盘通过 `dry_run:true + max_open_trades:0 + 不写 enter_long` 实现。

freqtrade 引擎从头到尾不知道你在"交易"还是"盯盘"。它只做三件事：拉K线 → 跑策略 `populate_*` → 读 `enter_long` 列决定要不要下单。"要不要下单"是策略层决定的，不是引擎模式决定的。新增 monitor runmode 只是在引擎里加个标签，底下还是同一套循环，却会破坏 fork 的上游同步。

`dry_run:true` 在这里的真实含义是"**需要一个能跑 dataprovider 和 REST API 的引擎实例，但不连真实交易所**"。

### 2.2 策略层职责边界

freqtrade 策略的本职是算指标、写信号列，策略层承担**数据持久化**职责（写 PG），但仍然禁止其他基础设施：

| 行为 | 允许？ | 说明 |
|---|---|---|
| 算指标、加列 | ✅ | 策略本职 |
| 信号判定 | ✅ | 策略本职 |
| **写 PG（K线、信号）** | ✅ | **本方案放宽**：PG 作为共享数据层，策略是写入方 |
| `dp.send_msg()` 推 TG 文本 | ✅ | 框架原生的轻量通知通道 |
| HTTP server | ❌ | 旧版 price_action 的反模式，禁止 |
| 自建通知管道（POST 外部服务） | ❌ | 由外部 bot 负责 |

### 2.3 三进程职责分离

| 进程 | 职责 | 与 PG 的关系 |
|---|---|---|
| **freqtrade（策略）** | 拉K线、算指标、检测信号、**写 K线和信号到 PG** | 只写 |
| **外部 TG bot** | 接收命令、调 freqtrade API 渲染图表、**读 PG 查信号/标的** | 只读 |
| **PostgreSQL** | 统一存储 K线、信号、交易对、freqtrade 交易数据 | 共享数据层 |

数据流向：
- **K线/信号**：freqtrade 拉取 → 写文件（feather，框架原生）+ 写 PG（双写）
- **图表渲染**：bot 调 freqtrade REST API（`/pair_candles`，走内存缓存，实时性好）
- **信号/标的查询**：bot 直读 PG（历史查询、聚合统计）
- **freqtrade 交易数据**：通过 `--db-url postgresql+psycopg://...` 直接进 PG，零代码改动

### 2.4 K线双写策略

freqtrade 的 K线存储**架构上是文件系统**（feather/json/parquet，`IDataHandler` 基于 `datadir: Path`，无 DB 后端）。本方案保留文件存储作为框架原生方式，同时把 OHLCV **双写**到 PG：

- **文件层**：freqtrade 正常读写，用于策略计算（pandas 整表加载，高效）
- **PG 层**：策略在 `populate_indicators` 里增量写入，用于外部查询/分析/展示

这样既不破坏框架架构，又满足"PG 里有 K线"的需求。

### 2.5 REST 轮询先行，WebSocket 后续

第一版外部 bot 用定时调 `GET /pair_candles` 拉数据渲染图表（实现简单）。信号推送也可先用 bot 轮询 PG 实现。稳定后再升级到 WebSocket `/api/v1/message/ws` 订阅（实时性好但复杂度高）。

---

## 3. 数据存储方案（四类数据）

| 数据类型 | 存储 | 写入方 | 机制 |
|---|---|---|---|
| **freqtrade 交易数据**（trades/orders/pairlocks 等） | PG | freqtrade 引擎 | `--db-url postgresql+psycopg://...`，需装 psycopg，零代码改动 |
| **K线 OHLCV** | 文件（feather）+ PG（双写） | freqtrade 策略 | 文件：框架原生；PG：策略 `populate_indicators` 增量 upsert |
| **交易对/监控标的** | PG | 外部 bot（`/add` `/remove`）+ freqtrade 读取 | `watch_pair` 表，freqtrade 用 DatabasePairList 插件读 |
| **信号记录** | PG | freqtrade 策略 | 策略检测到信号时写入 `signal` 表，bot 读 PG 查询 |

### PG 表结构（初步设计，待细化）

- `watch_pair`（监控标的）：symbol、market、enabled、display_name —— 复用 develop 分支已有的 DatabasePairList 约定
- `ohlcv`（K线）：symbol、timeframe、candle_time、open、high、low、close、volume，唯一约束 (symbol, timeframe, candle_time)
- `signal`（信号记录）：symbol、timeframe、candle_time、direction、quality、指标快照、entry_price、stop_loss、target_price
- freqtrade 原生表（trades/orders/pairlocks/...）：由框架自动建表和管理

> ⚠️ K线 upsert 性能注意：策略每根新K线触发一次，需用 `INSERT ... ON CONFLICT DO UPDATE`（或批量 flush）避免高频小写入压垮 PG。

---

## 4. freqtrade 原生能力清单

| 能力 | 机制 | 本方案用途 |
|---|---|---|
| 模拟运行 | `dry_run:true` | 跑引擎但不连真实交易所 |
| 禁止下单 | `max_open_trades:0` + 不写 `enter_long` | 双保险零交易 |
| 交易数据进 PG | `--db-url postgresql+psycopg://...` | trades/orders 等直接进 PG |
| 策略推 TG 文本 | `self.dp.send_msg(str)` + `telegram.allow_custom_messages:true` | 基础信号通知通道 |
| 暴露含指标的 df | `GET /pair_candles?pair=&timeframe=&limit=` | bot 拉数据渲染图表 |
| 暴露白名单 | `GET /whitelist` | bot 显示监控标的 |
| DatabasePairList 插件 | 从 PG `watch_pair` 表读白名单 | 动态标的管理（develop 分支已有） |
| WebSocket 订阅 | `WS /api/v1/message/ws` | 第二阶段实时信号 |

**原生缺失（需自建）**：自定义 TG 命令、富文本/图表渲染、K线双写到 PG、信号表。

---

## 5. 架构设计

```
┌──────────────────────────────────────────────────────────┐
│  进程1: freqtrade (dry_run + max_open_trades:0)           │
│  ┌────────────────────────────────────────────────────┐  │
│  │ user_data/strategies/watch_strategy.py             │  │
│  │   class PriceActionWatch(IStrategy)                │  │
│  │     bot_start()            ← 建 PG 连接、建表      │  │
│  │     populate_indicators() ← 算指标 + 双写K线到PG    │  │
│  │     populate_entry_trend() ← 信号判定 + 写PG + send_msg │
│  │     populate_exit_trend()  ← return df（空）       │  │
│  └────────────────────────────────────────────────────┘  │
│  --db-url postgresql+psycopg://...  (交易数据进PG)        │
│  --config .../watch_config.json                          │
│  pairlists: DatabasePairList (从PG读监控标的)             │
│  telegram.enabled:true + allow_custom_messages:true      │
└───────┬───────────────────────────────┬──────────────────┘
        │ 写 (K线/信号)                  │ REST: GET /pair_candles
        ▼                                ▼
┌──────────────────────────────────────────────────────────┐
│  PostgreSQL                                              │
│  watch_pair | ohlcv | signal | trades | orders | ...     │
└───────────────────────▲──────────────────────────────────┘
                        │ 读 (信号/标的查询)
                        │
┌──────────────────────────────────────────────────────────┐
│  进程2: services/watch_bot.py（外部 TG bot）              │
│  ┌────────────────────────────────────────────────────┐  │
│  │ /watch              查监控标的（读PG watch_pair）     │  │
│  │ /chart <pair>       渲染图（调freqtrade /pair_candles）│
│  │ /signal [n]         查最近信号（读PG signal）        │  │
│  │ /add /remove <pair> 管理标的（写PG watch_pair）      │  │
│  │ 定时查PG新信号 → 自动推送富文本卡片                  │  │
│  └────────────────────────────────────────────────────┘  │
│  技术栈: python-telegram-bot + httpx + mplfinance        │
└──────────────────────────────────────────────────────────┘
```

---

## 6. 策略实现（待策略逻辑定型后细化）

> ⚠️ 策略的具体指标和信号判定逻辑**还在构思中，尚未定型**。本节描述的是策略的**框架和职责**，具体算法待补充。

### 6.1 策略骨架

```python
class PriceActionWatch(IStrategy):
    # 零交易配置
    minimal_roi = {}
    stoploss = -0.99
    use_exit_signal = False
    max_open_trades = 0  # 也可在 config 里设

    def bot_start(self, **kwargs):
        """初始化 PG 连接、确保表存在。"""
        # create_engine + 建表（一次性）

    def populate_indicators(self, dataframe, metadata):
        """算指标 + 双写 K线到 PG。"""
        # 1. 算指标（策略逻辑待定型）
        # 2. 仅在 LIVE/DRY_RUN 模式写 PG（排除回测）
        # 3. upsert 最后一根K线到 PG ohlcv 表
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        """信号判定 + 写信号到 PG + 推 TG 文本。"""
        # 1. 信号判定（策略逻辑待定型）
        # 2. 有信号 → 写 PG signal 表
        # 3. 有信号 → self.dp.send_msg("信号文本")
        # 不写 enter_long（零交易）
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        return dataframe
```

### 6.2 待定型项

- [ ] 具体计算哪些指标
- [ ] 信号判定条件（什么情况算一个有效信号）
- [ ] 信号质量分级（如需要）
- [ ] 推送的 TG 文本格式

### 6.3 K线双写的实现要点

- **位置**：`populate_indicators` 末尾
- **触发频率**：受 `process_only_new_candles=True` 控制，只有新K线收盘才触发
- **数据源**：入参 `dataframe`（完整 OHLCV + 指标），取 `iloc[-1]` 为新K线
- **去重**：用 `INSERT ... ON CONFLICT (symbol, timeframe, candle_time) DO UPDATE`
- **模式过滤**：`if self.dp.runmode.value not in ("live", "dry_run"): return`，避免回测时误写
- **pair 过滤**：按 `metadata["pair"]` 区分主 pair 和 informative pair

---

## 7. 实施计划

### 第1步：基础设施（PG + 表结构）
- 准备 PostgreSQL 实例
- 设计并建表：`watch_pair` / `ohlcv` / `signal`
- freqtrade 交易数据用 `--db-url postgresql+psycopg://...` 接入

### 第2步：策略骨架（PG 连接 + 双写验证）
- 新建 `watch_strategy.py`：`PriceActionWatch(IStrategy)`
- `bot_start` 建 PG 连接
- `populate_indicators` 用最简指标 + 双写 K线到 PG
- 新建 `watch_config.json`：`dry_run:true`、`max_open_trades:0`、`api_server.enabled:true`、`db_url` 指向 PG
- **验证**：跑起来后查 PG 的 ohlcv 表有数据

### 第3步：DatabasePairList 接入
- 配置 `pairlists: [{"method": "DatabasePairList", ...}]`
- 在 `watch_pair` 表插入初始标的
- **验证**：freqtrade 白名单随 watch_pair 表变化

### 第4步：信号检测 + 落库 + 推送
- `populate_entry_trend` 实现信号判定（策略逻辑定型后）
- 写 `signal` 表 + `dp.send_msg()` 推 TG 文本
- **验证**：检测到信号时 PG 有记录 + 内置 TG 收到文本

### 第5步：外部 bot 骨架
- 新建 `services/watch_bot.py` + `services/Dockerfile.watch`
- 实现 `/chart <pair>`：调 `/pair_candles` → mplfinance 渲染 → 发图
- **验证**：TG 发 /chart 收到 K线图

### 第6步：bot 读 PG + 完善命令
- `/watch`（读 watch_pair）、`/signal`（读 signal）、`/add` `/remove`（写 watch_pair）
- 定时查 PG 新信号 → 自动推送富文本卡片
- **验证**：各命令可用，自动推送生效

### 第7步：Docker 部署
- 扩展 `docker-compose.yml`：`freqtrade-watch` + `watch-bot` + `postgres`
- 同 network，bot 通过 `http://freqtrade-watch:8080` 访问 API

---

## 8. 关键约束与风险

1. **K线双写是策略层职责**：放宽了"策略不碰 DB"的约束，但严格限定在写 K线/信号，不做 HTTP/通知
2. **K线 upsert 性能**：每根新K线一次写入，需用 `ON CONFLICT DO UPDATE` 或批量 flush
3. **dp.send_msg 只发字符串**（4096字符限制，每根K线去重），图表/按钮必须外部 bot 渲染
4. **freqtrade 官方镜像不带 psycopg/mplfinance**，watch-bot 和策略镜像都需自建（或用 Dockerfile.custom）
5. **dry_run 仍跑完整交易循环**，会有空模拟订单表（进 PG），这是正常的
6. **DatabasePairList 的 refresh_period 有缓存延迟**（默认 3600 秒），`/add` 后不会立即生效
7. **第一版用 REST 轮询**，稳定后升级 WebSocket
8. **策略逻辑尚未定型**（第6节标 ⚠️），实施计划第4步依赖策略设计完成

---

## 9. 文件清单

**新建**：
- `docs/watch-system-design.md`（本文档）
- `user_data/strategies/watch_strategy.py`
- `user_data/watch_config.json`
- `services/watch_bot.py`
- `services/Dockerfile.watch`
- PG 初始化 SQL（表结构，待设计）

**复用（已有，不重写）**：
- `freqtrade/plugins/pairlist/DatabasePairList.py`（develop 分支已有）

**修改**：
- `docker-compose.yml`（加 watch-bot + postgres 服务）

**不触碰**：
- `freqtrade/` 框架代码（保持上游同步）
