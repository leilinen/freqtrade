-- 一次性迁移：为 watch_pair 表新增 display_name 列
--
-- 用途：存储 A 股标的中文名称（如"贵州茅台"），加密货币/美股留空。
-- 背景：create_all() 不会自动 ALTER 已存在的表，需手动执行。
--
-- 执行方式（任选其一）：
--   psql -h <host> -p <port> -U postgres -d freqtrade_monitor -f migrate_add_display_name.sql
--   或在 psql 交互终端直接粘贴执行。
--
-- 幂等：IF NOT EXISTS 保证重复执行不报错。

ALTER TABLE watch_pair ADD COLUMN IF NOT EXISTS display_name VARCHAR;
