# PA Watch：静态标的 + watch-only 部署验证计划

> 目标分支：`dev_pa_mig`  
> 目标主机：`azureuser@azure`（Azure）  
> 范围：仅启动一个 `freqtrade-watch` 与一个专用 PostgreSQL；不下单、不启用动态标的或 Telegram 自定义命令。

## 1. 目标与成功标准

本次验证的是完整的低频盯盘链路：

```text
Binance K线 → PriceActionWatch → DeepSeek → PostgreSQL → 容器日志
```

初始固定标的为 `BTC/USDT`、`ETH/USDT`，周期 `1h`。配置必须保持：

- `dry_run: true`
- `max_open_trades: 0`
- `pa_llm.watch_only: true`
- `StaticPairList`

验收时，应确认两个容器持续健康；策略成功加载；PG 自动创建 freqtrade 原生表和 PA 的 `analysis_record`/`signal` 表；每个收盘周期仅出现一轮分析；无 `trades` 记录和真实订单。只有模型给出下单计划时才应新增 `signal`。

## 2. 已确认的部署前提

Azure 主机已具备 Docker 29.5.3 和 Docker Compose 5.1.4，外网可访问 GitHub、Docker Hub、Binance 与 DeepSeek API 域名。Docker 构建缓存已清理，根分区约有 48 GB 可用空间。

主机内存约 892 MB，已有 4 GB swap。该资源只适用于本次两标的、1 小时周期、远端 LLM 的低频 watch 服务：不设置 Docker `mem_limit`，避免因硬限制误杀；服务可能在 swap 压力下变慢。部署后必须观察 `free -h`、`docker stats`、容器重启次数和日志。

旧服务已由操作者停止。本次不得复用或修改以下旧工作目录、容器或数据卷：

- `/home/azureuser/freqtrade`
- `/home/azureuser/freqtrade-pa-agent/freqtrade`
- 旧 `freqtrade-pa:*` 镜像、旧 PostgreSQL 卷

## 3. 隔离与配置原则

在 Azure 新建独立目录 `/home/azureuser/freqtrade-pa-watch`，从 GitHub 的 `dev_pa_mig` 分支克隆。Compose 项目名固定为 `pa-watch`，使新容器、网络和卷与旧服务隔离；不得使用仓库中已有的 `container_name` 与旧服务重名。

部署前需要在新目录的 `docker/.env` 创建仅本服务使用的密钥文件（权限 `600`），内容来自 `docker/.env.example`：

- `PA_LLM_API_KEY`：由操作者提供，不写入 Git 或命令历史；
- `POSTGRES_PASSWORD`：新生成的高强度随机值；
- `PA_LLM_BASE_URL`：保留 DeepSeek 默认值，除非操作者明确切换；
- `TG_ENABLED=false`，`TG_TOKEN`/`TG_CHAT_ID` 留空；
- `TZ=Asia/Shanghai`。

`user_data/watch_config.json` 保持仓库默认的静态 BTC/ETH、`StaticPairList` 与 watch-only 三重保护。Compose 通过环境变量覆盖连接串和 API key，因此配置文件中的占位数据库密码不会被实际使用。

## 4. 执行步骤

### 4.1 部署前只读检查

1. 确认远端 `dev_pa_mig` 指向预期提交，并检查新目录无未跟踪配置文件。
2. 确认旧服务保持停止，80/443 无关紧要；新 Compose 不发布宿主机端口。
3. 确认 Docker 磁盘可用空间不少于 15 GB，内存/swap 余量可接受。
4. 核对 `watch_config.json` 的四项安全值：`dry_run=true`、`max_open_trades=0`、`watch_only=true`、`StaticPairList`。

若任一安全值被修改，停止部署并恢复默认值；本轮不允许以“先跑起来”为理由切换到 trade 模式。

### 4.2 首次构建与启动

1. 在独立目录拉取指定分支，并创建 `docker/.env`。
2. 使用以下形式启动：

   ```bash
   docker compose --project-name pa-watch \
     -f docker/docker-compose-watch.yml --env-file docker/.env up -d --build
   ```

3. 等待 PostgreSQL healthcheck 通过，再确认 `freqtrade-watch` 正常启动。
4. 不运行 `docker system prune`，不清理旧镜像、旧容器或旧卷。

构建与启动阶段只允许新增本项目的镜像、容器、网络和命名卷。失败时先读取日志定位，不反复无差别重建。

### 4.3 启动后验证

按以下顺序验证：

1. `docker compose ... ps`：`postgres` healthy，`freqtrade-watch` running。
2. 查看最近策略日志，确认加载 `PriceActionWatch`，模式日志包含 `watch mode`，白名单为 BTC/ETH，且没有 API key、数据库密码输出。
3. 在 `pa-postgres` 内只读检查表：应至少存在 `analysis_record`、`signal` 及 freqtrade 原生表（例如 `trades`、`orders`）。
4. 检查 `trades` 行数为 `0`；容器日志中不应出现创建/提交订单行为。
5. 等待一个 1h 收盘周期，确认每个静态标的各有一次 PA 分析尝试；若模型给出有效计划，`signal` 增加一行并关联 `analysis_record`；若观望，仅 `analysis_record` 增加。
6. 观察至少 15 分钟：容器没有循环重启、OOM、持续数据库连接失败或 API 重试风暴；检查内存、swap 和 `docker stats`。

LLM 调用失败（无效 key、额度不足或上游异常）不等于容器部署失败，但必须明确记录为功能验证未通过，且不得据此放宽 watch-only 安全配置。

## 5. 本轮明确不做的事项

- 不启用 `watch_only=false`、`dry_run=false` 或真实交易所密钥。
- 不启用做空、计划限价执行、止损止盈映射或 `executed_trade_id` 回填。
- 不使用 `DatabasePairList`，不创建/依赖 `watch_pair` 表，也不实施 `/add`、`/remove` 等 Telegram 插件功能。
- 不导入、覆盖或删除旧 PG 数据与 Docker 卷。
- 不将数据库暴露到宿主机或公网。

## 6. 观察阈值与处置

| 现象 | 处理 |
|---|---|
| `freqtrade-watch` 连续重启或被 OOM 杀死 | 立即停止本项目容器，保留日志和卷；先检查内存、swap、日志量与 LLM 并发。 |
| PG 未 healthy | 查看 PG 日志与卷权限；不删除卷重试，以免误伤新数据。 |
| API key/模型错误 | 仅修正 `docker/.env` 后重启 `freqtrade-watch`；不提交 `.env`。 |
| 没有 `signal` | 若分析记录正常且模型结论为“不下单”，这是预期；不能以此为由关闭 watch-only。 |
| 出现 `trades` 或订单日志 | 立即 `docker compose ... stop`，检查三项安全配置与环境变量后再决定是否恢复。 |

## 7. 回滚

本次验证的可逆回滚为停止隔离项目：

```bash
docker compose --project-name pa-watch \
  -f docker/docker-compose-watch.yml --env-file docker/.env stop
```

该操作保留 PostgreSQL 卷和日志，便于复盘。删除新项目卷属于数据删除行为，只有在确认不需要保留任何分析记录、且获得明确授权后才执行；绝不触及旧服务卷。

## 8. 验收后的下一步

只有本计划通过后，才处理数据库完善工作：版本化 migration、`watch_pair` 表、`signal` 唯一约束和重启幂等。完成这些修复并通过独立验证前，不进入 trade 模式。
