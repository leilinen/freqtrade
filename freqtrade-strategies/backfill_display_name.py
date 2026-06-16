#!/usr/bin/env python3
"""一次性脚本：回填 watch_pair 表中存量 A 股标的的中文名称。

前置条件：
  1. 已执行 migrate_add_display_name.sql（列已存在）。
  2. 能 import freqtrade.exchange.ashare（在 freqtrade 主仓库的 venv 下运行）。

用法：
  # 在 freqtrade 主仓库根目录运行（需已 pip install -e .）
  DB_URL="postgresql://postgres:postgres@localhost:15432/freqtrade_monitor" \
      .venv/bin/python freqtrade-strategies/backfill_display_name.py

  # 只回填、不实际写库（预演）
  DRY_RUN=1 DB_URL="..." .venv/bin/python freqtrade-strategies/backfill_display_name.py

说明：
  - 仅处理 market='ashare' 且 display_name IS NULL 的记录。
  - 每个标的调用 fetch_ashare_name（新浪主 + 腾讯兜底），失败则跳过（留空）。
  - 标的间间隔 1 秒，与 ashare.py 反爬节奏一致。
"""
import logging
import os
import time

from sqlalchemy import create_engine, text

from freqtrade.exchange.ashare import fetch_ashare_name


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s")
logger = logging.getLogger(__name__)

DB_URL = os.environ.get("DB_URL", "postgresql://postgres:postgres@localhost:5432/freqtrade_monitor")
DRY_RUN = os.environ.get("DRY_RUN", "") == "1"


def main() -> None:
    engine = create_engine(DB_URL)
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT id, symbol FROM watch_pair "
                "WHERE market = 'ashare' AND display_name IS NULL "
                "ORDER BY id"
            )
        ).fetchall()

    if not rows:
        logger.info("No ashare pairs to backfill.")
        return

    logger.info("Found %d ashare pair(s) to backfill (dry_run=%s).", len(rows), DRY_RUN)
    updated = 0
    for rid, symbol in rows:
        name = fetch_ashare_name(symbol)
        if name:
            logger.info("%s -> %s", symbol, name)
            if not DRY_RUN:
                with engine.begin() as conn:
                    conn.execute(
                        text("UPDATE watch_pair SET display_name = :n WHERE id = :i"),
                        {"n": name, "i": rid},
                    )
            updated += 1
        else:
            logger.warning("%s -> (name not found, skipped)", symbol)
        time.sleep(1)

    logger.info("Done. Updated %d/%d pair(s).", updated, len(rows))


if __name__ == "__main__":
    main()
