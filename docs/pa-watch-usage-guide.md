# PA 盯盘系统使用说明书

> 适用版本：`dev_pa_mig` 分支（PA_Agent 基线 `1090a5b` 迁移完成）
> 配套文档：架构设计见 `docs/pa-watch-system-design.md`，迁移过程见 `docs/pa-migration-plan.md`，运维细节见 `docs/pa-watch-runbook.md`。本文面向**使用者**：怎么配、怎么跑、怎么看信号、怎么排障。

---

## 1. 系统简介

一个以 LLM 两阶段价格行为分析为核心的**盯盘系统**（watch 模式，默认不下单）：

- 每根收盘K线自动触发一次完整分析：客观几何特征 → 阶段一市场诊断（LLM）→ 策略知识路由 → 阶段二交易决策（LLM）→ 多层 JSON 校验 → 连续性守卫
- 检测到下单计划时：写入 `signal` 表（交易意图账本）+ Telegram 推送决策详情
- 支持增量分析（每根新K线只发增量，省 ~14.5K token/次），重启后自动从数据库恢复上一轮分析状态
- 与 freqtrade 引擎同进程：`watch_only=false` + 实盘配置即可切换为自动交易，代码路径不变

## 2. 快速开始

### 2.1 服务器部署（推荐，Docker）

```bash
git clone git@github.com:leilinen/freqtrade.git && cd freqtrade
git checkout dev_pa_mig

cp docker/.env.example docker/.env
vim docker/.env          # 必填 PA_LLM_API_KEY；要 TG 推送则填 TG_TOKEN/TG_CHAT_ID 并置 TG_ENABLED=true

docker compose -f docker/docker-compose-watch.yml --env-file docker/.env up -d --build
docker logs -f freqtrade-watch
```

启动成功的标志（日志）：

```
PriceActionWatch started (watch mode, 100 bars, 0 pairs, db=postgresql)
```

> 服务器需能访问：Docker Hub（拉基础镜像，国内可配镜像加速）、binance API（拉K线）、`api.deepseek.com`（LLM）。LLM 端点可经 `PA_LLM_BASE_URL` 换成任何 OpenAI 兼容网关。

### 2.2 本机运行（venv）

```bash
pip install -r requirements.txt && pip install -e . && pip install openai "psycopg[binary]"

export FREQTRADE__PA_LLM__API_KEY="sk-xxx"     # 密钥走环境变量，别写进 config
freqtrade trade --strategy PriceActionWatch --config user_data/watch_config.json
```

没有 PG 时系统自动回退到本地 sqlite（`user_data/pa_records.sqlite`），日志会有一次 warning，功能不受影响。

### 2.3 三分钟看懂它干什么

启动后什么都不用做。每根K线收盘（默认 timeframe `1h`，标的 BTC/ETH）自动分析一轮；轮次记录进 `analysis_record` 表；**只有当 LLM 给出实际下单计划**（限价单/突破单/市价单）时才会写 `signal` 表并向 Telegram 推送。观望结论（"不下单"）不会推送。

## 3. 配置说明

### 3.1 密钥管理（重要）

`user_data/watch_config.json` 已被 git 跟踪，**任何密钥都不要写进去**。统一用 freqtrade 的环境变量覆盖机制（`FREQTRADE__节__键`，Docker 部署时由 `docker/.env` 注入）：

| 环境变量 | 说明 |
|---|---|
| `FREQTRADE__PA_LLM__API_KEY` | LLM API Key（必填） |
| `FREQTRADE__PA_LLM__BASE_URL` | OpenAI 兼容端点（默认 `https://api.deepseek.com`） |
| `FREQTRADE__TELEGRAM__ENABLED` / `__TOKEN` / `__CHAT_ID` | 开启 TG 推送 |
| `FREQTRADE__PA_DB_URL` | 覆盖存储库连接串 |

### 3.2 `watch_config.json` 关键项

| 配置项 | 默认 | 说明 |
|---|---|---|
| `dry_run` | `true` | 模拟运行（不连真实交易所下单） |
| `max_open_trades` | `0` | 零交易双保险之一 |
| `timeframe` | `"1h"` | 分析周期 |
| `exchange.pair_whitelist` | BTC/ETH | 监控标的（StaticPairList） |
| `pa_db_url` | PG 连接串 | signal/analysis_record 存储；不可用时自动回退 sqlite |
| `pa_llm.watch_only` | `true` | **盯盘模式开关**；`false` 时写入场信号列 |
| `pa_llm.model` / `base_url` / `api_key` | deepseek | LLM 提供方 |
| `pa_llm.analysis_bar_count` | `100` | 每轮分析送给 LLM 的K线根数 |
| `pa_llm.decision_stance` | `"balanced"` | 交易倾向：`conservative` / `balanced` / `aggressive` / `extreme_aggressive` |
| `pa_llm.reasoning_effort` | `"high"` | LLM 推理强度（low/medium/high/max） |
| `pa_llm.enable_next_bar_prediction` | `false` | 开启后让 LLM 附带预测下根K线（多耗 token） |
| `pa_llm.structure_flip_cooldown_bars` | `3` | 同结构位反向方案的冷却K线数 |

## 4. 信号解读

### 4.1 Telegram 推送格式

```
📊 BTC/USDT 1h PA信号
周期: normal_channel | 方向: bullish
决策: 做多 限价单 @ 100000.0
止损: 98000.0 | TP1: 104000.0 | TP2: 108000.0
置信度: 65 | 预估胜率: 55%
```

- **周期**：阶段一判定的市场状态（spike / tight_channel / normal_channel / broad_channel / trading_range 等 Brooks 周期位置）
- **方向**：阶段一的趋势方向（bullish / bearish / neutral）
- **决策**：阶段二的下单计划；`order_type` 为"不下单"时不推送
- **置信度/胜率**：LLM 自评，仅供决策参考

### 4.2 `signal` 表（交易意图账本）

| 字段 | 含义 |
|---|---|
| `symbol` / `timeframe` / `candle_time` | 标的、周期、决策依据K线（K1）的开盘时间 |
| `order_type` / `order_direction` | 限价单/突破单/市价单；做多/做空 |
| `entry_price` / `stop_loss_price` / `take_profit_price` / `take_profit_price_2` | 计划入场/止损/止盈1/止盈2 |
| `risk_reward` | 系统按计划价计算的盈亏比（reward:risk） |
| `trade_confidence` / `estimated_win_rate` | LLM 置信度 / 预估胜率 |
| `cycle_position` / `direction` | 阶段一市场状态摘要 |
| `snapshot` | JSONB：阶段一摘要（形态/支撑/阻力）+ 完整决策 + 终止节点 |
| `record_id` | 关联的完整分析轮次（`analysis_record.id`） |
| `executed_trade_id` | 实盘模式对账后回填的 freqtrade `trades.id`；watch 模式恒为 NULL |

### 4.3 常用查询

```sql
-- 最近 10 条信号
SELECT created_at, symbol, order_direction, order_type, entry_price,
       stop_loss_price, risk_reward, cycle_position
FROM signal ORDER BY created_at DESC LIMIT 10;

-- 各周期状态下的信号分布与平均盈亏比
SELECT cycle_position, COUNT(*) AS n, ROUND(AVG(risk_reward), 2) AS avg_rr
FROM signal GROUP BY cycle_position ORDER BY n DESC;

-- 分析轮次的失败原因统计（LLM 稳定性监控）
SELECT is_partial, partial_reason, COUNT(*) FROM analysis_record
GROUP BY 1, 2 ORDER BY 3 DESC;

-- 复盘某条信号的完整上下文（两阶段 prompts + 原始响应）
SELECT record_json FROM analysis_record
WHERE id = (SELECT record_id FROM signal ORDER BY created_at DESC LIMIT 1);
```

## 5. Telegram 交互

复用 freqtrade **原生** telegram bot（同一 token/chat）。除上面的自动推送外，可用全部原生命令：

| 常用命令 | 作用 |
|---|---|
| `/status` | 当前持仓（watch 模式恒空） |
| `/profit` `/daily` `/count` | 收益统计（watch 模式为 0） |
| `/whitelist` | 当前监控标的 |
| `/health` / `/version` / `/show_config` | 运行状态/版本/配置 |
| `/pause` `/stop` / `/start` | 暂停/停止/恢复分析循环 |
| `/reload_config` | 重载配置（改 config 文件后免重启） |

> 自定义命令（`/chart` 画图、`/signal` 查库、`/add` `/remove` 动态标的）属于 telegram 插件阶段（设计文档 §5.2），尚未实施。当前改监控标的 = 改 `watch_config.json` 的 `pair_whitelist` + `/reload_config`。

## 6. 回测

> LLM 是策略核心：回测时**每根K线都会真实调用一次 LLM**（增量模式）。先小区间试跑再放大。

```bash
# 下载数据（需交易所网络）
freqtrade download-data --config user_data/watch_config.json --timerange 20250901-20251201

# 回测（一周区间先验证）
freqtrade backtesting --strategy PriceActionWatch \
  --config user_data/watch_config.json --timerange 20251201-20251208
```

Token 消耗参考：1h 周期一天 24 次调用；增量模式下每次约几 K token（首轮全量 ~30K+）。

**离线/无密钥回测**（验证管道，LLM 全部失败降级为 partial 记录，信号为空）：

```bash
python tools/bt_offline_driver.py --timerange 20251128-20251129
```

## 7. 切换实盘（watch → trade）

改三处配置（代码不变）：

```jsonc
// watch_config.json 或环境变量
"dry_run": false,          // 连真实交易所
"max_open_trades": 3,      // 允许持仓
"pa_llm": {"watch_only": false}
```

同时补齐 `exchange.key/secret`（环境变量 `FREQTRADE__EXCHANGE__KEY/__SECRET`）。切换后：LLM 下单计划照常写入 `signal` 表（意图），freqtrade 自动下单并写入 `trades` 表（执行），事后可按 `executed_trade_id` 对账复盘。**建议先小额 + 保持 TG 推送盯一段再放量。**

## 8. 故障排查

| 症状（日志/现象） | 原因 | 处理 |
|---|---|---|
| `PgRecordStore(...) unavailable; falling back to sqlite` | PG 未启动/密码错/网络不通 | 检查 PG 容器与健康检查；不修也能跑（数据进 sqlite） |
| `pa_llm.api_key is empty — every PA analysis will fail` | 未配置密钥 | 配 `FREQTRADE__PA_LLM__API_KEY` |
| `analysis_record` 每行 `partial_reason='strategy_exception'` | LLM 调用本身失败（无凭据/端点不通） | 查 `record_json->exception`；确认 key 与 `base_url` 可达 |
| 长期 `signal` 表 0 行 | LLM 判定均为"不下单"，或全部分析失败 | 先查上一条的 partial 原因分布；确认K线数据在增长 |
| `Missing credentials` | openai 客户端构造时无 key | 同上，配环境变量 |
| 回测 `No trades made` | 无密钥降级（预期）或确实无信号 | 看回测日志中 PA 分析次数与 partial 统计 |
| 启动后白名单为空 | `pair_whitelist` 为空或数据不足 `startup_candle_count`(160) | 补白名单；等K线预热（新标的约 160 根后开始出分析） |
| 重启后无 `Restored previous analysis` | 上一轮无成功记录（首轮正常）或库被清 | 无需处理；第二轮起自动恢复 |

## 9. 目录与文档索引

```
user_data/strategies/price_action_watch.py   策略入口（组装 pa_core + freqtrade 回调）
user_data/watch_config.json                  主配置
pa_core/                                     PA 策略核心（20,699 行，结构详见 docs/pa-migration-plan.md §5）
pa_core/prompts/                             32 个 PA 知识文件（决策树/形态规则/通道策略）
tools/bt_offline_driver.py                   离线回测驱动
docker/                                      部署资产（Dockerfile.watch + compose + .env 模板）
```

| 文档 | 内容 |
|---|---|
| 本文 | 使用说明 |
| `docs/pa-watch-system-design.md` | 系统架构与关键决策 |
| `docs/pa-migration-plan.md` | PA_Agent 迁移方案与代码结构 |
| `docs/pa-watch-runbook.md` | 环境搭建与运维细节 |
