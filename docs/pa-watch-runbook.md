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

```bash
freqtrade download-data --config user_data/watch_config.json --timerange 20250101-20250401
freqtrade backtesting --strategy PriceActionWatch --config user_data/watch_config.json \
    --timerange 20250101-20250401
```

注意：每根K线一次 LLM 调用，3 个月 1h ≈ 2,160 次。请先用 `--timerange 20250301-20250307` 小区间验证。

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
- `pairlists` 默认 `StaticPairList`（BTC/ETH）。接 PG 动态标的管理时换 `DatabasePairList`（设计文档第3步）。
- A股标的需 `ashare` 交易所配置（`freqtrade/exchange/ashare.py`），本 config 以 binance/crypto 为例。
