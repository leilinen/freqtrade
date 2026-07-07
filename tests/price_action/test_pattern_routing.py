"""Tests for pattern_routing coherence + validator (ported from upstream)."""
from __future__ import annotations

import sys
from pathlib import Path

_STRATEGY_DIR = Path(__file__).resolve().parents[2] / "user_data" / "strategies"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

import pytest  # noqa: E402

from price_action.pattern_routing import (  # noqa: E402
    ensure_detected_patterns_coherent,
    sync_detected_patterns_field,
    validate_detected_patterns_vs_key_signals,
)


class TestSyncDetectedPatternsField:
    def test_preserves_model_tags(self):
        s1 = {"detected_patterns": ["wedge"]}
        out = sync_detected_patterns_field(s1)
        assert "wedge" in out
        assert s1["detected_patterns"] == out

    def test_overlays_from_entry_setup_type(self):
        s1 = {
            "detected_patterns": [],
            "bar_analysis": {"entry_setup_type": "wedge"},
        }
        out = sync_detected_patterns_field(s1)
        assert "wedge" in out

    def test_overlays_from_cycle_position(self):
        s1 = {"detected_patterns": [], "cycle_position": "trading_range"}
        out = sync_detected_patterns_field(s1)
        assert "middle_range" in out
        assert "overlap" in out

    def test_drops_barbwire_when_blocked(self):
        s1 = {
            "detected_patterns": ["barbwire"],
            "program_features": {"barbwire_candidate": False},
        }
        out = sync_detected_patterns_field(s1)
        # model-emitted barbwire gets dropped because program_features denies it
        assert "barbwire" not in out


class TestEnsureDetectedPatternsCoherent:
    def test_adds_wedge_from_keyword(self):
        s1 = {
            "detected_patterns": [],
            "key_signals": ["出现楔形结构"],
        }
        assert ensure_detected_patterns_coherent(s1) is True
        assert "wedge" in s1["detected_patterns"]

    def test_adds_wedge_from_entry_setup_type(self):
        s1 = {
            "detected_patterns": [],
            "bar_analysis": {"entry_setup_type": "wedge"},
        }
        ensure_detected_patterns_coherent(s1)
        assert "wedge" in s1["detected_patterns"]

    def test_adds_breakout_pullback_from_entry_setup_type(self):
        s1 = {
            "detected_patterns": [],
            "bar_analysis": {"entry_setup_type": "breakout_pullback"},
        }
        ensure_detected_patterns_coherent(s1)
        assert "breakout_pullback" in s1["detected_patterns"]

    def test_adds_h1_from_keyword(self):
        s1 = {
            "detected_patterns": [],
            "key_signals": ["H1 计数入场"],
        }
        ensure_detected_patterns_coherent(s1)
        assert "h1" in s1["detected_patterns"]

    def test_skips_barbwire_when_keyword_matches_but_no_program_features(self):
        s1 = {
            "detected_patterns": [],
            "key_signals": ["铁丝网结构"],
        }
        ensure_detected_patterns_coherent(s1)
        # No program_features.barbwire_candidate, so not auto-added
        assert "barbwire" not in s1["detected_patterns"]

    def test_adds_barbwire_when_program_features_confirms(self):
        s1 = {
            "detected_patterns": [],
            "key_signals": ["铁丝网结构"],
            "program_features": {"barbwire_candidate": True},
        }
        ensure_detected_patterns_coherent(s1)
        assert "barbwire" in s1["detected_patterns"]

    def test_tr_boundary_adds_middle_range(self):
        s1 = {
            "detected_patterns": [],
            "bar_analysis": {"entry_setup_type": "tr_boundary"},
        }
        ensure_detected_patterns_coherent(s1)
        assert "middle_range" in s1["detected_patterns"]

    def test_tr_boundary_skips_barbwire_without_program_features(self):
        s1 = {
            "detected_patterns": [],
            "bar_analysis": {"entry_setup_type": "tr_boundary"},
        }
        ensure_detected_patterns_coherent(s1)
        assert "barbwire" not in s1["detected_patterns"]

    def test_no_change_when_already_complete(self):
        s1 = {
            "detected_patterns": ["wedge"],
            "bar_analysis": {"entry_setup_type": "wedge"},
        }
        assert ensure_detected_patterns_coherent(s1) is False

    def test_returns_bool(self):
        s1 = {"detected_patterns": []}
        result = ensure_detected_patterns_coherent(s1)
        assert isinstance(result, bool)


class TestValidateDetectedPatternsVsKeySignals:
    def test_no_errors_when_coherent(self):
        s1 = {
            "detected_patterns": ["wedge"],
            "key_signals": ["楔形结构"],
        }
        assert validate_detected_patterns_vs_key_signals(s1) == []

    def test_error_when_keyword_missing_from_patterns(self):
        s1 = {
            "detected_patterns": [],
            "key_signals": ["出现楔形结构"],
        }
        errors = validate_detected_patterns_vs_key_signals(s1)
        assert len(errors) == 1
        assert "wedge" in errors[0]

    def test_h1_keyword_handled_via_hl_path_not_pattern_table(self):
        # PATTERN_KEYWORD_TAGS has no h1 entry; h1 sync/validate goes via
        # the dedicated HL_COUNT_RE path in sync_detected_patterns_field.
        # So validate does not emit an h1-specific error here.
        s1 = {
            "detected_patterns": [],
            "key_signals": ["H1 计数入场"],
        }
        errors = validate_detected_patterns_vs_key_signals(s1)
        assert not any("h1" in e.lower() for e in errors)

    def test_error_when_wedge_setup_missing(self):
        s1 = {
            "detected_patterns": [],
            "bar_analysis": {"entry_setup_type": "wedge"},
        }
        errors = validate_detected_patterns_vs_key_signals(s1)
        assert any("wedge" in e for e in errors)

    def test_error_when_breakout_pullback_setup_missing(self):
        s1 = {
            "detected_patterns": [],
            "bar_analysis": {"entry_setup_type": "breakout_pullback"},
        }
        errors = validate_detected_patterns_vs_key_signals(s1)
        assert any("breakout_pullback" in e for e in errors)

    def test_error_when_tr_boundary_missing_middle_range(self):
        s1 = {
            "detected_patterns": [],
            "bar_analysis": {"entry_setup_type": "tr_boundary"},
        }
        errors = validate_detected_patterns_vs_key_signals(s1)
        assert any("middle_range" in e for e in errors)

    def test_no_barbwire_error_without_program_features(self):
        s1 = {
            "detected_patterns": [],
            "key_signals": ["铁丝网"],
        }
        errors = validate_detected_patterns_vs_key_signals(s1)
        # program_features.barbwire_candidate absent → barbwire sync skipped → no error
        assert not any("barbwire" in e for e in errors)

    def test_does_not_mutate_input(self):
        s1 = {
            "detected_patterns": [],
            "key_signals": ["楔形"],
        }
        before_patterns = list(s1["detected_patterns"])
        validate_detected_patterns_vs_key_signals(s1)
        assert s1["detected_patterns"] == before_patterns

    def test_uses_risk_warning_text(self):
        s1 = {
            "detected_patterns": [],
            "key_signals": [],
            "risk_warning": "楔形反转风险",
        }
        errors = validate_detected_patterns_vs_key_signals(s1)
        assert any("wedge" in e for e in errors)
