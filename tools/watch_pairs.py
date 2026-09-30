"""Ops CLI for the watch_pair table (monitored symbols of the PA watch system).

Writes go through pa_core's WatchPairStore; freqtrade itself reads the table
read-only via DatabasePairList (changes become visible within the pairlist's
refresh_period, 60s by default). The bot seeds the default pairs (BTC/ETH)
itself on cold start when the market has no rows — `init` is only a
convenience to pre-create/seed without waiting for a bot start.

Usage (host, from repo root):
    python tools/watch_pairs.py list
    python tools/watch_pairs.py add SOL/USDT
    python tools/watch_pairs.py disable SOL/USDT && python tools/watch_pairs.py enable SOL/USDT

In the watch container (db-url picked up from FREQTRADE__PA_DB_URL):
    docker compose -f docker/docker-compose-watch.yml exec freqtrade-watch \
        python /freqtrade/tools/watch_pairs.py list

db-url resolution order: --db-url > $PA_DB_URL > $FREQTRADE__PA_DB_URL >
pa_db_url in --config (default user_data/watch_config.json).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pa_core.records.watch_pair_store import (  # noqa: E402
    MARKET_ASHARE,
    MARKET_CRYPTO,
    WatchPairDuplicateError,
    WatchPairNotFoundError,
    WatchPairStore,
)

DEFAULT_CONFIG = REPO_ROOT / "user_data" / "watch_config.json"


def _resolve_db_url(args: argparse.Namespace) -> str:
    if args.db_url:
        return args.db_url
    env_url = os.environ.get("PA_DB_URL") or os.environ.get("FREQTRADE__PA_DB_URL")
    if env_url:
        return env_url
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"error: cannot read config {args.config}: {exc}")
    url = config.get("pa_db_url")
    if not url:
        sys.exit(
            f"error: no pa_db_url in {args.config}; pass --db-url or set PA_DB_URL"
        )
    return url


def _check_symbol(symbol: str, market: str) -> str:
    symbol = symbol.strip()
    if not symbol:
        sys.exit("error: symbol must not be empty")
    if market == MARKET_CRYPTO and "/" not in symbol:
        print(
            f"warning: {symbol!r} has no '/' separator — crypto pairs usually "
            "look like BTC/USDT",
            file=sys.stderr,
        )
    return symbol


def _print_rows(rows: list[dict]) -> None:
    if not rows:
        print("(no rows)")
        return
    header = {"id": "ID", "symbol": "SYMBOL", "market": "MARKET", "enabled": "ENABLED",
              "display_name": "DISPLAY_NAME", "updated_at": "UPDATED_AT"}
    widths = {
        key: max(len(header[key]), *(len(str(row.get(key, ""))) for row in rows))
        for key in header
    }
    fmt = "  ".join(f"{{:{widths[key]}}}" for key in header)
    print(fmt.format(*header.values()))
    for row in rows:
        enabled = "yes" if row["enabled"] else "NO"
        print(fmt.format(row["id"], row["symbol"], row["market"], enabled,
                         row.get("display_name") or "", row.get("updated_at") or ""))


def symbol_ref(args: argparse.Namespace) -> str:
    return f"{args.symbol.strip()} ({args.market})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="watch_pairs.py",
        description="Manage the watch_pair table (PA watch monitored symbols).",
    )
    parser.add_argument(
        "--db-url",
        help="SQLAlchemy URL of the PA database (default: PA_DB_URL / "
        "FREQTRADE__PA_DB_URL env, else pa_db_url from --config)",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="freqtrade config to read pa_db_url / pair_whitelist from "
        f"(default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--market",
        choices=[MARKET_CRYPTO, MARKET_ASHARE],
        default=MARKET_CRYPTO,
        help="which market the symbols belong to (default: crypto)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "init",
        help="create watch_pair (idempotent) and seed the default pairs "
        "when the market has no rows — same as the bot's cold-start behavior",
    )

    p_list = sub.add_parser("list", help="list monitored symbols")
    p_list.add_argument(
        "--all", action="store_true", help="include disabled rows (default: enabled only)"
    )

    p_add = sub.add_parser("add", help="add a symbol (enabled)")
    p_add.add_argument("symbol")
    p_add.add_argument("--display-name", help="human-facing label (default: symbol)")

    p_remove = sub.add_parser(
        "remove", help="hard-delete a symbol (prefer 'disable' to pause watching)"
    )
    p_remove.add_argument("symbol")

    p_enable = sub.add_parser("enable", help="re-enable a disabled symbol")
    p_enable.add_argument("symbol")

    p_disable = sub.add_parser("disable", help="stop watching a symbol (keeps the row)")
    p_disable.add_argument("symbol")

    args = parser.parse_args(argv)
    db_url = _resolve_db_url(args)
    store = WatchPairStore(db_url)
    market = args.market

    try:
        if args.command == "init":
            added = store.ensure_defaults(market)
            if added:
                print(f"Seeded default rows: {', '.join(added)}")
            else:
                n = len(store.list_pairs(market=market))
                print(f"watch_pair already has {n} row(s) in market {market}; nothing seeded")
        elif args.command == "list":
            _print_rows(store.list_pairs(market=market, enabled_only=not args.all))
        elif args.command == "add":
            symbol = _check_symbol(args.symbol, market)
            store.add(symbol, market=market, display_name=args.display_name)
            print(f"Added {symbol} ({market}, enabled)")
        elif args.command == "remove":
            store.remove(_check_symbol(args.symbol, market), market=market)
            print(f"Removed {symbol_ref(args)}")
        elif args.command == "enable":
            store.set_enabled(_check_symbol(args.symbol, market), market=market, enabled=True)
            print(f"Enabled {symbol_ref(args)}")
        elif args.command == "disable":
            store.set_enabled(_check_symbol(args.symbol, market), market=market, enabled=False)
            print(f"Disabled {symbol_ref(args)}")
    except WatchPairDuplicateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except WatchPairNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI boundary: surface, don't traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.command in ("add", "remove", "enable", "disable"):
        print(
            "freqtrade picks this up within its DatabasePairList refresh_period "
            "(60s by default) — no restart needed."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
