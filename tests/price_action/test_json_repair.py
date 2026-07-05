"""Tests for JSON repair helpers in validation.py.

Covers the LLM-output repair pipeline ported from PA_Agent:
smart-quote normalization, stray-string-separator removal, unescaped-quote
repair, and semicolon-separator repair. The golden case is the real id=251
``}," "entry_setup_type"`` lesion recovered from production.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parents[2] / "user_data" / "strategies"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from price_action.validation import (  # noqa: E402
    _balance_json_brackets,
    _drop_stray_string_separators,
    _normalize_smart_quotes,
    _repair_semicolon_separator,
    _repair_unescaped_quotes,
    parse_json_object,
)


# ── Smart-quote normalization ──


class TestNormalizeSmartQuotes:
    def test_curly_double_quotes_replaced(self):
        assert _normalize_smart_quotes('“hello”') == '"hello"'

    def test_curly_single_quotes_replaced(self):
        assert _normalize_smart_quotes('‘it’s’') == "'it's'"

    def test_en_em_dash_replaced(self):
        assert _normalize_smart_quotes('a–b—c') == 'a-b-c'

    def test_ascii_unchanged(self):
        assert _normalize_smart_quotes('plain "ascii"') == 'plain "ascii"'


# ── Stray string separator (the id=251 lesion) ──


class TestDropStrayStringSeparators:
    """The headline case: LLM inserts an isolated ``" "`` between object fields."""

    def test_real_id251_lesion_variant_b(self):
        """id=251 pattern: ``}," "key":`` where key has no quote of its own.

        In production (id=251) the lesion sits **inside** bar_analysis, after
        a nested object closes. We reproduce that structure here.
        """
        raw = '{"bar_analysis": {"pattern": {"name": "四重确认"}, " "entry_setup_type": "breakout_pullback"}, "direction": "bullish"}'
        out = _drop_stray_string_separators(raw)
        # The stray " " is dropped; the closing quote of the stray string
        # doubles as the opening quote of the real key.
        assert '" "entry_setup_type"' not in out
        assert json.loads(out)["bar_analysis"]["entry_setup_type"] == "breakout_pullback"

    def test_variant_a_key_has_own_quote(self):
        """Variant: ``}," " "key":`` where the real key already has a quote."""
        raw = '{"a": {"x":"确认"}, " " "key": "v"}'
        out = _drop_stray_string_separators(raw)
        # Stray " " removed, real key's own quote preserved.
        assert json.loads(out)["key"] == "v"

    def test_does_not_touch_legitimate_string_array(self):
        """A whitespace-only string inside a JSON array is a legal element."""
        raw = '[" ", "real"]'
        out = _drop_stray_string_separators(raw)
        # Must remain unchanged: array elements are not object-field positions.
        assert json.loads(out) == [" ", "real"]

    def test_does_not_touch_string_value_with_space(self):
        """A regular string value containing spaces is left alone."""
        raw = '{"a": "hello world", "b": "x"}'
        out = _drop_stray_string_separators(raw)
        assert json.loads(out)["a"] == "hello world"

    def test_preserves_non_whitespace_stray_content(self):
        """A stray string with real content (not just whitespace) is NOT dropped.

        We only remove whitespace-only stray separators; content-bearing ones
        might be legitimate and are left for other repair passes.
        """
        raw = '{"a":"x"}, "real content" "key":"v"}'
        out = _drop_stray_string_separators(raw)
        # Unchanged — "real content" is not whitespace-only.
        assert raw == out


# ── Unescaped quotes inside string values ──


class TestRepairUnescapedQuotes:
    def test_value_with_embedded_unescaped_quote(self):
        """LLM writes ``"a": "he said "hi""`` — the inner quotes break parsing."""
        raw = '{"a": "he said "hi" then"}'
        out = _repair_unescaped_quotes(raw)
        assert json.loads(out)["a"] == 'he said "hi" then'

    def test_properly_escaped_quotes_unchanged(self):
        raw = '{"a": "he said \\"hi\\""}'
        out = _repair_unescaped_quotes(raw)
        assert json.loads(out)["a"] == 'he said "hi"'

    def test_normal_object_unchanged(self):
        raw = '{"a": "1", "b": "2"}'
        out = _repair_unescaped_quotes(raw)
        assert json.loads(out) == {"a": "1", "b": "2"}


# ── Semicolon separator ──


class TestRepairSemicolonSeparator:
    def test_semicolon_after_string_value_replaced(self):
        raw = '{"a": "1"; "b": "2"}'
        out = _repair_semicolon_separator(raw)
        assert json.loads(out) == {"a": "1", "b": "2"}

    def test_semicolon_inside_string_preserved(self):
        raw = '{"a": "x;y"}'
        out = _repair_semicolon_separator(raw)
        assert json.loads(out)["a"] == "x;y"


# ── End-to-end: parse_json_object on the real id=251 payload ──


class TestParseJsonObjectEndToEnd:
    """The full repair pipeline must recover the id=251 production failure."""

    def test_real_id251_payload_recovers_entry_setup_type(self, tmp_path):
        """Reconstruct the id=251 lesion and verify parse_json_object recovers it.

        This is the production case: glm-5.x emitted ``}," "entry_setup_type"``
        causing ``JSONDecodeError: Expecting ':' delimiter at char 5929``.
        Before the fix, the whole diagnosis failed. After the fix, the field
        is recovered intact.

        In production the lesion sits **inside** ``bar_analysis``, right after
        a nested confirmation object closes (the bar_analysis itself is still
        open) — we reproduce that structure.
        """
        # Build the clean nested structure first. The key ordering mirrors
        # production: confirmation sub-object first, then entry_setup_type.
        raw = json.dumps(
            {
                "bar_analysis": {
                    "confirmation": {"text": "因K1同时完成突破前高+实体最大+放量+gate_break四重确认"},
                    "entry_setup_type": "breakout_pullback",
                },
                "direction": "bullish",
            },
            ensure_ascii=False,
        )
        # Inject the id=251 lesion: after the nested confirmation object closes
        # (still INSIDE bar_analysis), LLM inserted a stray " " before the real
        # entry_setup_type key (which loses its own opening quote to the stray).
        lesioned = raw.replace(
            '"confirmation": {"text": "因K1同时完成突破前高+实体最大+放量+gate_break四重确认"}, "entry_setup_type"',
            '"confirmation": {"text": "因K1同时完成突破前高+实体最大+放量+gate_break四重确认"}, " "entry_setup_type"',
        )
        assert '" "entry_setup_type"' in lesioned  # sanity: lesion present

        obj = parse_json_object(lesioned)
        assert obj["bar_analysis"]["entry_setup_type"] == "breakout_pullback"
        assert obj["bar_analysis"]["confirmation"]["text"].startswith("因K1")
        assert obj["direction"] == "bullish"

    def test_clean_json_passes_through(self):
        raw = '{"a": 1, "b": "two"}'
        obj = parse_json_object(raw)
        assert obj == {"a": 1, "b": "two"}

    def test_markdown_fenced_json_parsed(self):
        raw = '```json\n{"a": 1}\n```'
        assert parse_json_object(raw) == {"a": 1}

    def test_smart_quote_json_parsed(self):
        raw = '{“a”: “value”}'
        assert parse_json_object(raw) == {"a": "value"}

    def test_truncated_json_balanced(self):
        """Truncated JSON missing its closing braces is closed by the repair."""
        raw = '{"a": {"b": 1'
        obj = parse_json_object(raw)
        assert obj["a"]["b"] == 1
