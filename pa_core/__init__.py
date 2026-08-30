"""pa_core — price action strategy core, ported from PA_Agent.

Migrated from the PA_Agent project (https://github.com/.../PA_Agent, AGPL-3.0)
at baseline commit ``1090a5b`` (2026-08-28). See docs/pa-migration-plan.md for
the full migration plan and the layer-by-layer porting order.

This package is pure Python with no GUI/data-source/notification dependencies.
Runtime deps: pandas (freqtrade already ships it). The LLM layer (later porting
steps) additionally needs ``openai``/``tiktoken``.

Layer 1 (this step): data structures + indicators + snapshot helpers +
freqtrade DataFrame adapter.
"""
