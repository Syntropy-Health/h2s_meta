"""Sealed challenge suite — layer 0 + ``post`` (``daypart.membrane.post``).

Category: integration/membrane-post
Spec: docs/daypart/contracts/CONTRACT-slice-B-membrane.md section B3 (with B2
for the DSL verdicts the drafts below produce).

Integration with the REAL pinned LHM guard (no mocks): drafts are chosen so
that layer 0 (``assert_narration_safe``) and the DSL disagree, which pins
both the ordering and the fact that layer 0 runs on every draft.

Ambiguities resolved here (questions for the principal):

* When ``regenerate()`` raises, only ``decision == "escalate"``,
  ``text is None``, ``len(attempt_log) == attempts`` and JSON-serializability
  are asserted; whether the failed regeneration counts as an attempt or sets
  ``regenerated`` is left open.
* ``regenerate`` is passed as a zero-argument callable and must be invoked
  with no arguments.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from syntropy.lhm.metrics.claims import (
    CLAIMS_GUARD_VERSION,
    VALENCE_GUARD_VERSION,
    ClaimScopeError,
    assert_narration_safe,
)

from daypart.membrane import dsl
from daypart.membrane.post import PostResult, post

# Clean everywhere; deliberately odd bytes (zero-width, double spaces,
# trailing newline) to pin byte-identical pass-through.
CLEAN = "Nice​ work —  three days in a row!  Want to keep the streak going?\n"
CLEAN_2 = "Would a reminder at 8pm fit your evening better?"
# Fails layer 0 (advice lexicon "recommend") but passes the DSL (D10/HQ2).
LAYER0_ONLY = "We recommend moving your evening reminder to 8pm."
# Passes layer 0 (past tense, no take/start/stop) but fails the DSL (G5).
DSL_ONLY = "You logged 500 mg yesterday, nice consistency."
DSL_ONLY_2 = "Your log shows two capsules and 400MG at breakfast."
# Passes layer 0; fires exactly two DSL rules, one of them twice.
MULTI = "You logged 500 mg and 200 mg alongside your metformin."
MULTI_IDS = ["dosing.unit_quantity", "substance.rx_named"]
CRISIS_SILENT = "I hear you, and I am here with you right now."
CRISIS_HANDOFF = "I am here with you. Please call or text 988 right now."

ALL_DRAFTS = [CLEAN, CLEAN_2, LAYER0_ONLY, DSL_ONLY, DSL_ONLY_2, MULTI, CRISIS_SILENT, CRISIS_HANDOFF]


def _layer0_ok(text: str) -> bool:
    try:
        assert_narration_safe(text)
    except ClaimScopeError:
        return False
    return True


class Regen:
    """Counting zero-arg regenerate stub."""

    def __init__(self, result: str | None = None, exc: BaseException | None = None):
        self.result = result
        self.exc = exc
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        assert self.result is not None
        return self.result


def _assert_trace_shape(trace: dict, *drafts: str) -> None:
    assert isinstance(trace, dict)
    encoded = json.dumps(trace)  # must be JSON-serializable
    assert set(trace) >= {"layer0", "dsl", "attempts", "regenerated", "decision", "attempt_log"}
    assert trace["layer0"]["guard_versions"] == {
        "claims": CLAIMS_GUARD_VERSION,
        "valence": VALENCE_GUARD_VERSION,
    }
    assert trace["layer0"]["outcome"] in ("pass", "fail")
    assert trace["dsl"]["rule_version"] == "boundary-v1"
    assert trace["dsl"]["passed"] in (True, False, None)
    viol = trace["dsl"]["violations"]
    assert isinstance(viol, list)
    assert viol == sorted(set(viol))
    assert trace["attempts"] in (1, 2)
    assert isinstance(trace["regenerated"], bool)
    assert trace["decision"] in ("pass", "escalate")
    log = trace["attempt_log"]
    assert isinstance(log, list)
    assert len(log) == trace["attempts"]
    for entry in log:
        assert set(entry) >= {"layer0", "dsl_passed", "violations"}
        assert entry["layer0"] in ("pass", "fail")
        assert entry["dsl_passed"] in (True, False, None)
        assert entry["violations"] == sorted(set(entry["violations"]))
        if entry["layer0"] == "fail":
            assert entry["dsl_passed"] is None
            assert entry["violations"] == []
    for d in drafts:
        if d.strip():
            assert d not in encoded
            assert json.dumps(d)[1:-1] not in encoded


# --------------------------------------------------------------------------- #
# Fixture sanity — the drafts mean what this suite says they mean (real LHM).
# --------------------------------------------------------------------------- #


class TestDraftPremises:
    """integration/membrane-post :: draft premises against real LHM + DSL."""

    def test_layer0_only_draft(self):
        assert not _layer0_ok(LAYER0_ONLY)
        assert dsl.check(LAYER0_ONLY).passed is True

    @pytest.mark.parametrize("draft", [DSL_ONLY, DSL_ONLY_2, MULTI])
    def test_dsl_only_drafts(self, draft):
        assert _layer0_ok(draft)
        assert dsl.check(draft).passed is False

    @pytest.mark.parametrize("draft", [CLEAN, CLEAN_2, CRISIS_HANDOFF])
    def test_clean_drafts(self, draft):
        assert _layer0_ok(draft)
        assert dsl.check(draft).passed is True
        assert dsl.check(CRISIS_HANDOFF, crisis_context=True).passed is True


# --------------------------------------------------------------------------- #
# Pass path
# --------------------------------------------------------------------------- #


class TestPassPath:
    """integration/membrane-post :: first draft passes both layers."""

    def test_returns_byte_identical_text(self):
        r = post(CLEAN)
        assert isinstance(r, PostResult)
        assert r.decision == "pass"
        assert r.text == CLEAN
        assert r.text.encode("utf-8") == CLEAN.encode("utf-8")

    def test_regenerate_not_called_when_first_passes(self):
        regen = Regen(result=CLEAN_2)
        r = post(CLEAN, regenerate=regen)
        assert regen.calls == 0
        assert r.decision == "pass"
        assert r.text == CLEAN

    def test_trace(self):
        r = post(CLEAN)
        t = r.trace
        _assert_trace_shape(t, CLEAN)
        assert t["layer0"]["outcome"] == "pass"
        assert t["dsl"]["passed"] is True
        assert t["dsl"]["violations"] == []
        assert t["attempts"] == 1
        assert t["regenerated"] is False
        assert t["decision"] == "pass"
        assert t["attempt_log"] == [{"layer0": "pass", "dsl_passed": True, "violations": []}]

    def test_result_frozen(self):
        r = post(CLEAN)
        assert dataclasses.is_dataclass(r)
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.decision = "escalate"  # type: ignore[misc]

    def test_keyword_only(self):
        with pytest.raises(TypeError):
            post(CLEAN, False)  # type: ignore[misc]

    def test_deterministic(self):
        a, b = post(CLEAN), post(CLEAN)
        assert a.decision == b.decision and a.text == b.text and a.trace == b.trace


# --------------------------------------------------------------------------- #
# Layer 0 failure
# --------------------------------------------------------------------------- #


class TestLayer0:
    """integration/membrane-post :: layer 0 (real LHM narration guard)."""

    def test_layer0_fail_no_regenerate_escalates(self):
        r = post(LAYER0_ONLY)
        assert r.decision == "escalate"
        assert r.text is None
        t = r.trace
        _assert_trace_shape(t, LAYER0_ONLY)
        assert t["layer0"]["outcome"] == "fail"
        assert t["dsl"]["passed"] is None
        assert t["dsl"]["violations"] == []
        assert t["attempts"] == 1
        assert t["regenerated"] is False
        assert t["decision"] == "escalate"
        assert t["attempt_log"] == [{"layer0": "fail", "dsl_passed": None, "violations": []}]

    def test_layer0_fail_then_clean_regen_passes(self):
        regen = Regen(result=CLEAN_2)
        r = post(LAYER0_ONLY, regenerate=regen)
        assert regen.calls == 1
        assert r.decision == "pass"
        assert r.text == CLEAN_2
        t = r.trace
        _assert_trace_shape(t, LAYER0_ONLY, CLEAN_2)
        assert t["attempts"] == 2
        assert t["regenerated"] is True
        assert t["decision"] == "pass"
        assert t["layer0"]["outcome"] == "pass"
        assert t["dsl"]["passed"] is True
        assert t["dsl"]["violations"] == []
        assert t["attempt_log"] == [
            {"layer0": "fail", "dsl_passed": None, "violations": []},
            {"layer0": "pass", "dsl_passed": True, "violations": []},
        ]

    def test_layer0_runs_on_regenerated_draft(self):
        regen = Regen(result=LAYER0_ONLY)
        r = post(DSL_ONLY, regenerate=regen)
        assert regen.calls == 1
        assert r.decision == "escalate"
        assert r.text is None
        t = r.trace
        _assert_trace_shape(t, DSL_ONLY, LAYER0_ONLY)
        assert t["attempts"] == 2
        assert t["regenerated"] is True
        assert t["layer0"]["outcome"] == "fail"
        assert t["dsl"]["passed"] is None
        assert t["dsl"]["violations"] == []
        assert t["attempt_log"][0] == {
            "layer0": "pass",
            "dsl_passed": False,
            "violations": ["dosing.unit_quantity"],
        }
        assert t["attempt_log"][1] == {"layer0": "fail", "dsl_passed": None, "violations": []}

    def test_layer0_fails_closed_on_every_draft(self):
        regen = Regen(result=LAYER0_ONLY)
        r = post(LAYER0_ONLY, regenerate=regen)
        assert regen.calls == 1
        assert r.decision == "escalate"
        assert r.text is None
        assert [e["layer0"] for e in r.trace["attempt_log"]] == ["fail", "fail"]


# --------------------------------------------------------------------------- #
# DSL failure
# --------------------------------------------------------------------------- #


class TestDslFailure:
    """integration/membrane-post :: DSL failure and single regeneration."""

    def test_dsl_fail_no_regenerate_escalates(self):
        r = post(DSL_ONLY)
        assert r.decision == "escalate"
        assert r.text is None
        t = r.trace
        _assert_trace_shape(t, DSL_ONLY)
        assert t["layer0"]["outcome"] == "pass"
        assert t["dsl"]["passed"] is False
        assert t["dsl"]["violations"] == ["dosing.unit_quantity"]
        assert t["attempts"] == 1
        assert t["regenerated"] is False

    def test_dsl_fail_regenerated_into_fail_escalates_after_one_call(self):
        regen = Regen(result=DSL_ONLY_2)
        r = post(DSL_ONLY, regenerate=regen)
        assert regen.calls == 1
        assert r.decision == "escalate"
        assert r.text is None
        t = r.trace
        _assert_trace_shape(t, DSL_ONLY, DSL_ONLY_2)
        assert t["attempts"] == 2
        assert t["regenerated"] is True
        assert t["decision"] == "escalate"
        assert t["layer0"]["outcome"] == "pass"
        assert t["dsl"]["passed"] is False
        assert t["dsl"]["violations"] == ["dosing.unit_quantity"]
        assert [e["dsl_passed"] for e in t["attempt_log"]] == [False, False]

    def test_dsl_fail_then_clean_regen_passes(self):
        regen = Regen(result=CLEAN)
        r = post(DSL_ONLY, regenerate=regen)
        assert regen.calls == 1
        assert r.decision == "pass"
        assert r.text == CLEAN
        assert r.text.encode("utf-8") == CLEAN.encode("utf-8")
        assert r.trace["attempts"] == 2
        assert r.trace["regenerated"] is True

    def test_violations_sorted_unique(self):
        r = post(MULTI)
        t = r.trace
        _assert_trace_shape(t, MULTI)
        assert t["dsl"]["violations"] == MULTI_IDS
        assert t["attempt_log"][0]["violations"] == MULTI_IDS
        expected = sorted({v.rule_id for v in dsl.check(MULTI).violations})
        assert t["dsl"]["violations"] == expected


# --------------------------------------------------------------------------- #
# regenerate raising
# --------------------------------------------------------------------------- #


class TestRegenerateRaises:
    """integration/membrane-post :: exceptions from regenerate escalate."""

    @pytest.mark.parametrize("first", [DSL_ONLY, LAYER0_ONLY])
    @pytest.mark.parametrize("exc", [RuntimeError("boom"), ValueError("bad"), TimeoutError()])
    def test_escalates_without_propagating(self, first, exc):
        regen = Regen(exc=exc)
        r = post(first, regenerate=regen)
        assert regen.calls == 1
        assert r.decision == "escalate"
        assert r.text is None
        t = r.trace
        _assert_trace_shape(t, first)
        assert t["decision"] == "escalate"


# --------------------------------------------------------------------------- #
# Crisis context
# --------------------------------------------------------------------------- #


class TestCrisisContext:
    """integration/membrane-post :: crisis_context threading."""

    def test_crisis_without_handoff_escalates(self):
        r = post(CRISIS_SILENT, crisis_context=True)
        assert r.decision == "escalate"
        assert r.text is None
        t = r.trace
        _assert_trace_shape(t, CRISIS_SILENT)
        assert t["layer0"]["outcome"] == "pass"
        assert t["dsl"]["passed"] is False
        assert t["dsl"]["violations"] == ["crisis.missing_handoff"]

    def test_same_draft_passes_without_crisis_context(self):
        r = post(CRISIS_SILENT)
        assert r.decision == "pass"
        assert r.text == CRISIS_SILENT

    def test_crisis_with_handoff_passes(self):
        r = post(CRISIS_HANDOFF, crisis_context=True)
        assert r.decision == "pass"
        assert r.text == CRISIS_HANDOFF

    def test_crisis_regenerated_with_handoff_passes(self):
        regen = Regen(result=CRISIS_HANDOFF)
        r = post(CRISIS_SILENT, crisis_context=True, regenerate=regen)
        assert regen.calls == 1
        assert r.decision == "pass"
        assert r.text == CRISIS_HANDOFF
        t = r.trace
        _assert_trace_shape(t, CRISIS_SILENT, CRISIS_HANDOFF)
        assert t["attempts"] == 2
        assert t["attempt_log"][0]["violations"] == ["crisis.missing_handoff"]
        assert t["attempt_log"][1] == {"layer0": "pass", "dsl_passed": True, "violations": []}

    def test_crisis_context_applies_to_regenerated_draft(self):
        regen = Regen(result=CLEAN_2)  # clean copy, but no handoff marker
        r = post(CRISIS_SILENT, crisis_context=True, regenerate=regen)
        assert regen.calls == 1
        assert r.decision == "escalate"
        assert r.trace["dsl"]["violations"] == ["crisis.missing_handoff"]


# --------------------------------------------------------------------------- #
# Trace never leaks drafts
# --------------------------------------------------------------------------- #


class TestTracePrivacy:
    """integration/membrane-post :: trace JSON-serializable, no draft text."""

    @pytest.mark.parametrize("draft", ALL_DRAFTS)
    @pytest.mark.parametrize("crisis", [False, True])
    def test_trace_never_contains_draft(self, draft, crisis):
        regen = Regen(result=CLEAN_2)
        r = post(draft, crisis_context=crisis, regenerate=regen)
        assert regen.calls <= 1
        _assert_trace_shape(r.trace, draft, CLEAN_2)
        assert r.trace["decision"] == r.decision
        assert (r.text is None) == (r.decision == "escalate")
        if r.decision == "pass":
            assert r.text in (draft, CLEAN_2)
