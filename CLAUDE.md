# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup

```bash
# Full dev install (preferred, uses uv if available)
pip install -r requirements-dev.txt
pip install -e ft_client/
pip install -e .

# Or use the interactive setup script
./setup.sh --install
```

Optional dependency groups: `pip install -e ".[plot]"`, `.[hyperopt]`, `.[freqai]`, `.[freqai_rl]`, `.[all]`, `.[dev]`

### Testing

```bash
# All tests (parallel, randomized)
pytest --random-order -n auto

# Single test file
pytest tests/exchange/test_binance.py

# Single test case
pytest tests/exchange/test_binance.py::test_function_name

# With coverage
pytest --random-order --cov=freqtrade --cov-config=.coveragerc

# Long-running/live tests
pytest --longrun
```

Test configuration uses `asyncio_mode = "auto"` and `--dist loadscope` (groups tests in same class/module together). Tests live in `tests/` organized by module: `exchange/`, `freqtradebot/`, `optimize/`, `persistence/`, `plugins/`, `rpc/`, `strategy/`, `data/`, `freqai/`.

### Linting and Formatting

```bash
ruff check              # Lint
ruff check --fix        # Auto-fix lint issues
ruff format --check     # Check formatting
ruff format             # Auto-format
mypy freqtrade scripts tests  # Type check
pre-commit run --all-files    # Run all hooks
```

## Architecture

Freqtrade is a crypto trading bot. Entry point: `freqtrade.main:main` → CLI dispatch via `argparse` subcommands.

### Core Packages

| Package | Purpose |
|---------|---------|
| `freqtrade/strategy/` | `IStrategy` ABC — users subclass to define trading logic (populate_indicators, populate_entry/exit_trend, plus optional callbacks) |
| `freqtrade/exchange/` | Exchange abstraction wrapping ccxt/ccxt.pro. Base `Exchange` class with per-exchange subclasses (Binance, Bybit, OKX, etc.) |
| `freqtrade/resolvers/` | `IResolver` — dynamic class loading. Scans `user_data/` dirs for Python files, imports and instantiates strategy/exchange/pairlist/protection classes |
| `freqtrade/plugins/` | Pairlist (`IPairList`) and protection (`IProtection`) plugin systems |
| `freqtrade/freqtradebot.py` | `FreqtradeBot` — central orchestrator. `process()` is the main loop iteration |
| `freqtrade/worker.py` | `Worker` — event loop with state machine and throttling |
| `freqtrade/data/` | `DataProvider` facade for strategies, pluggable data handlers (JSON/Feather/Parquet) |
| `freqtrade/persistence/` | SQLAlchemy ORM (Trade, Order, PairLock models), SQLite by default |
| `freqtrade/optimize/` | Backtesting engine and hyperopt (Hyperparameter optimization) |
| `freqtrade/rpc/` | Telegram bot, REST API (FastAPI), Discord webhooks, WebSocket |
| `freqtrade/configuration/` | Config loading: JSON files + CLI args + env vars → merged dict → JSON-Schema validation |
| `freqtrade/freqai/` | Optional ML subsystem (`IFreqaiModel` for LightGBM, XGBoost, RL models) |

### Live Trading Flow

```
main() → start_trading() → Worker.run() [infinite loop]
  → FreqtradeBot.process() [each iteration]:
    1. reload_markets()
    2. refresh_active_whitelist()  (pairlist plugins)
    3. dataprovider.refresh()      (fetch candles)
    4. strategy.analyze()          (populate_indicators + entry/exit signals)
    5. manage_open_orders()        (timeouts/cancellations)
    6. exit_positions()            (check exit conditions)
    7. enter_positions()           (execute new entries)
```

### Plugin Pattern

Custom strategies, pairlists, and protections are loaded at runtime by resolvers scanning `user_data/` directories. To add a new pairlist: subclass `IPairList`, implement `gen_pairlist()`/`filter_pairlist()`, register in `constants.py` `AVAILABLE_PAIRLISTS`.

## Code Style

- **Ruff** for lint + format: line length 100, max McCabe complexity 12
- Enabled rule sets: pyflakes, pycodestyle, pyupgrade, isort, bugbear, flake8-builtins, flake8-tidy-imports, flake8-bandit, flake8-async, numpy, mccabe
- **Docstrings**: double-quoted, reST format (`:param xxx:`, `:return:`) on all public methods
- **Tests**: assertions allowed (ruff ignores S101 in tests); use `log_has`/`log_has_re` from `tests/conftest.py` for log assertions
- **Type checking**: mypy with `ignore_missing_imports = true`, tests excluded from type checking

## Conventions

- **Branching**: All PRs target `develop` (never `stable`). Feature branches use `feat/*`.
- **Pre-commit**: Install with `pre-commit install`. Hooks run ruff, mypy, codespell, and custom schema validation.
- **Tests required**: Every new feature must include unit tests.
- **Docs required**: New features must include documentation updates in the same PR.
- **Config**: Plain `dict` (typed as `Config = dict[str, Any]`) — no config objects, just dicts passed throughout.
- **Custom exceptions**: Hierarchy rooted at `FreqtradeException` → `OperationalException`, `DependencyException`, `StrategyError`.
- **Language**: English for all commit messages, PR descriptions, code comments, and variable names.
