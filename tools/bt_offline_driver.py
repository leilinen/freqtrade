"""Offline backtest driver for the PA watch system.

Patches exchange market/fee loading (no network needed) and runs the
standard freqtrade backtesting CLI. Useful in restricted-network
environments with local feather data (e.g. tests/testdata).

Usage:
    python tools/bt_offline_driver.py [extra freqtrade backtesting args...]

Requires a config with: dry_run enabled, "fee" set (short-circuits the
network fee lookup), and pa_llm.base_url pointing at a refused local port
to simulate LLM failures (or a reachable endpoint for real runs).
"""
from __future__ import annotations

import sys
from unittest.mock import patch

from freqtrade.exchange.exchange import Exchange


def _fake_market(symbol: str, base: str, price_prec: int, amount_prec: int) -> dict:
    return {
        "symbol": symbol,
        "base": base,
        "quote": "USDT",
        "spot": True,
        "swap": False,
        "future": False,
        "margin": False,
        "active": True,
        "precision": {"amount": amount_prec, "price": price_prec},
        "limits": {
            "amount": {"min": 10 ** -amount_prec, "max": None},
            "cost": {"min": 0.0, "max": None},
            "leverage": {"min": None, "max": 1},
            "price": {"min": None, "max": None},
        },
        "contractSize": None,
        "linear": None,
        "inverse": None,
        "info": {},
    }


FAKE_MARKETS = {
    "BTC/USDT": _fake_market("BTC/USDT", "BTC", 2, 6),
    "ETH/USDT": _fake_market("ETH/USDT", "ETH", 2, 5),
    "XRP/USDT": _fake_market("XRP/USDT", "XRP", 4, 1),
}


def main() -> None:
    extra = sys.argv[1:]
    sys.argv = [
        "freqtrade", "backtesting",
        "--strategy", "PriceActionWatch",
        "--config", "user_data/watch_config_offline_bt.json",
        "--datadir", "tests/testdata",
        *extra,
    ]
    from freqtrade.main import main as ft_main

    with (
        patch.object(Exchange, "reload_markets", lambda self, *a, **k: None),
        patch.object(Exchange, "markets", new=property(lambda self: FAKE_MARKETS)),
        patch.object(Exchange, "validate_config", lambda self, config: None),
    ):
        ft_main()


if __name__ == "__main__":
    main()
