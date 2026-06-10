# Price Action Signal Monitor — Docker 部署指南

## 架构概览

```
┌──────────────┐  HTTP POST  ┌──────────────────┐  TG API  ┌──────┐
│ freqtrade 1h │─────────────→│   freqtrade-tg-bot │──────────→│ 用户  │
│ freqtrade 4h │             │   (HTTP :8090)    │          └──────┘
└──────┬───────┘             └────────┬───────────┘
       │ write                        │ read/write
       ▼                              ▼
┌──────────────────────────────────────────┐
│          PostgreSQL (freqtrade_monitor)  │
│   - watch_pair  (监控标的配置)            │
│   - pa_signal   (信号记录)               │
└──────────────────────────────────────────┘
```

三个独立服务通过 Docker 网络通信：
- **price-action-1h** — 1 小时周期信号检测
- **price-action-4h** — 4 小时周期信号检测
- **freqtrade-tg-bot** — Telegram Bot + HTTP API

## 前置条件

- Docker + Docker Compose
- PostgreSQL（需要已有实例或新建）
- Telegram Bot Token（通过 @BotFather 创建）
- OKX 交易所账号（当前使用 OKX，可改为其他）

## 第一步：准备 PostgreSQL

确保 PostgreSQL 可访问，创建数据库：

```sql
CREATE DATABASE freqtrade_monitor;
```

如果使用现有的 Docker PG 实例：

```bash
docker exec postgres psql -U postgres -c "CREATE DATABASE freqtrade_monitor;"
```

## 第二步：构建镜像

### 2.1 构建 freqtrade 镜像

```bash
cd freqtrade
docker build -f docker/Dockerfile.pa -t freqtrade-pa:latest .
```

构建时间较长（需编译 TA-Lib），约 10-20 分钟。

### 2.2 构建 TG Bot 镜像

```bash
cd freqtrade-strategies
docker-compose -f docker-compose-pa.yml build tg-bot
```

## 第三步：配置

### 3.1 创建 Telegram Bot

1. 在 Telegram 中找 **@BotFather**，发送 `/newbot`
2. 按提示设置名称，获取 Bot Token
3. 获取你的 Chat ID（给 bot 发消息后访问 `https://api.telegram.org/bot<TOKEN>/getUpdates`）

### 3.2 修改配置文件

编辑 `freqtrade-strategies/user_data/price_action_config_1h.json`：

```json
{
  "exchange": {
    "name": "okx",
    "key": "your-api-key",
    "secret": "your-api-secret"
  }
}
```

> 注意：dry_run 模式不需要真实 API key，可以留占位符。

### 3.3 修改 docker-compose 环境变量

编辑 `freqtrade-strategies/docker-compose-pa.yml`：

```yaml
tg-bot:
  environment:
    TG_TOKEN: "你的Bot Token"
    TG_CHAT_ID: "你的Chat ID"
    DB_URL: "postgresql://postgres:postgres@postgres:5432/freqtrade_monitor"
```

### 3.4 Docker 网络

确保 Docker 网络存在且 PG 在同一网络：

```bash
docker network create panwatch_default 2>/dev/null || true
```

如果 PG 在其他网络，修改 `docker-compose-pa.yml` 的 `networks` 部分。

## 第四步：启动服务

```bash
cd freqtrade-strategies
docker-compose -f docker-compose-pa.yml up -d
```

验证三个服务都启动：

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

预期输出：

```
NAMES                STATUS
freqtrade-tg-bot     Up ...
price-action-4h      Up ...
price-action-1h      Up ...
```

## 第五步：初始化和验证

### 5.1 初始化监控标的

首次启动时，策略会自动在 `watch_pair` 表插入默认标的：
- BTC/USDT、ETH/USDT、SOL/USDT、BNB/USDT

### 5.2 验证 TG Bot

在 Telegram 中给 Bot 发送：

```
/pa_help
/pa_watch
```

如果收到回复，说明 Bot 正常。

### 5.3 查看日志

```bash
# 1h 实例日志
docker logs price-action-1h --tail 50

# 4h 实例日志
docker logs price-action-4h --tail 50

# TG Bot 日志
docker logs freqtrade-tg-bot --tail 50

# 实时跟踪
docker logs -f price-action-1h
```

正常情况下，每个周期会看到：

```
Scan BTC/USDT 1h: long good
Signal saved to PG: BTC/USDT long good
Signal notified to tg-bot: BTC/USDT long good
```

### 5.4 测试信号推送

手动发送测试信号：

```bash
docker exec price-action-1h python3 -c "
import requests
r = requests.post('http://tg-bot:8090/signal', json={
    'symbol': 'TEST/USDT', 'timeframe': '1h', 'direction': 'long',
    'quality': 'good', 'body_pct': 0.9, 'close_location': 0.95,
    'body_ratio': 2.0, 'bar_types': ['engulfing'], 'ema20_above': True,
    'ema_gap': 0.5, 'bull_strength_5': 0.7, 'entry_price': 69850.0,
    'stop_loss': 69200.0, 'target_price': 71150.0
}, timeout=5)
print('Status:', r.status_code)
"
```

TG 收到消息说明整条链路正常。

## 日常操作

### TG 命令

| 命令 | 说明 |
|------|------|
| `/pa_watch` | 查看监控标的列表 |
| `/pa_add DOGE/USDT` | 添加监控标的 |
| `/pa_remove BTC/USDT` | 禁用监控标的 |
| `/pa_signals` | 查看最近 10 条信号 |
| `/pa_signals 20` | 查看最近 20 条信号 |
| `/pa_help` | 帮助信息 |

添加/移除标的后，策略会在缓存刷新后（默认 1 小时）自动生效。

### 重启服务

```bash
cd freqtrade-strategies
docker-compose -f docker-compose-pa.yml restart
```

### 停止服务

```bash
cd freqtrade-strategies
docker-compose -f docker-compose-pa.yml down
```

### 更新策略代码

修改策略后需要重建镜像并重启：

```bash
# 重建 freqtrade 镜像
cd freqtrade
docker build -f docker/Dockerfile.pa -t freqtrade-pa:latest .

# 重启
cd ../freqtrade-strategies
docker-compose -f docker-compose-pa.yml up -d price-action-1h price-action-4h
```

### 更新 TG Bot 代码

```bash
cd freqtrade-strategies
docker-compose -f docker-compose-pa.yml build tg-bot
docker-compose -f docker-compose-pa.yml up -d tg-bot
```

## 信号消息格式说明

收到信号时，消息格式如下：

```
+ BTC/USDT 1h
做多 [Good] 当前价格: 69850.00
实体占比=0.90 收盘位置=0.95 实体比=2.0
止损 69200.00 | 目标 71150.00 (盈亏比 1.8)
特殊K线: 吞噬
EMA20: 均线上方 (偏离=0.5x ATR)
近5K偏向: 多头 (70%)
```

| 字段 | 说明 |
|------|------|
| +/- | 做多/做空方向 |
| Good/Acceptable/Fair | 信号质量等级 |
| 当前价格 | K线收盘价 |
| 实体占比 | K线实体占总长度的比例 |
| 收盘位置 | 收盘价在K线中的位置 (0=最低, 1=最高) |
| 实体比 | 实体相对近20根K线实体中位数的倍数 |
| 止损/目标 | 止损放在信号K极值，目标为2倍盈亏比 |
| 特殊K线 | 惊喜/吞噬/内包/2K反转/十字星 |
| EMA20 | 价格与均线的位置关系 |
| 近5K偏向 | 近5根K线多空力量对比 |

## 常见问题

### TG Bot 报 Conflict 错误

说明有另一个进程在用同一个 Bot Token polling。确保只有一个 tg-bot 实例运行。

### freqtrade 日志只有 heartbeat

这是正常的——策略只在检测到信号时才输出 INFO 日志。如需查看无信号时的扫描活动，可调整日志级别为 DEBUG。

### DatabasePairList 读取不到标的

检查 PG 中 `watch_pair` 表的 `enabled` 字段是否为 `true`。缓存默认 1 小时刷新一次。

### 交易所连接失败

确认网络可访问交易所 API。如在国内使用 OKX 需要确保域名 `www.okx.com` 可达。
