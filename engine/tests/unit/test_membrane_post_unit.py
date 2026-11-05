"""Implementer unit tests for daypart.membrane.post (CONTRACT slice B, B3)."""

from __future__ import annotations

import json

import pytest
from syntropy.lhm.metrics import claims

from daypart.membrane import post as post_mod
from daypart.membrane.post import PostResult, post

GOOD = "Would a reminder at 8pm fit your evening better?"
GOOD2 = "Nice work logging three days in a row — want to keep the streak going?"
DSL_BAD = "Magnesium 400 mg before bed."  # passes layer 0, fails DSL
L0_BAD = "We recommend a walk after dinner."  # fails layer 0 (advice lexicon)


class Regen:
    def __init__(self, out):
        self.out = out
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if isinstance(self.out, BaseException):
            raise self.out
        return self.out


def _assert_trace_shape(trace, draft_texts=()):
    assert set(trace) == {"layer0", "dsl", "attempts", "regenerated", "decision", "attempt_log"}
    assert trace["layer0"]["guard_versions"] == {"claims": "claims-v3", "valence": "valence-v2"}
    assert trace["layer0"]["guard_versions"]["claims"] == claims.CLAIMS_GUARD_VERSION
    assert trace["dsl"]["rule_version"] == "boundary-v1"
    assert trace["attempts"] == len(trace["attempt_log"])
    blob = json.dumps(trace)
    for t in draft_texts:
        if t:
            assert t not in blob


def test_pass_first_try_byte_identical_no_regen():
    draft = "Would a reminder at 8pm fit your evening better?  — ok"
    r = Regen(GOOD2)
    res = post(draft, regenerate=r)
    assert isinstance(res, PostResult)
    assert res.decision == "pass" and res.text == draft
    assert r.calls == 0
    t = res.trace
    _assert_trace_shape(t, [draft])
    assert t["attempts"] == 1 and t["regenerated"] is False and t["decision"] == "pass"
    assert t["layer0"]["outcome"] == "pass"
    assert t["dsl"] == {"rule_version": "boundary-v1", "passed": True, "violations": []}
    assert t["attempt_log"] == [{"layer0": "pass", "dsl_passed": True, "violations": []}]


def test_dsl_fail_then_regen_pass():
    r = Regen(GOOD)
    res = post(DSL_BAD, regenerate=r)
    assert r.calls == 1
    assert res.decision == "pass" and res.text == GOOD
    t = res.trace
    _assert_trace_shape(t, [DSL_BAD, GOOD])
    assert t["attempts"] == 2 and t["regenerated"] is True
    assert t["attempt_log"][0] == {
        "layer0": "pass",
        "dsl_passed": False,
        "violations": ["dosing.unit_quantity"],
    }
    assert t["attempt_log"][1] == {"layer0": "pass", "dsl_passed": True, "violations": []}


def test_layer0_fail_dsl_not_run():
    res = post(L0_BAD)
    assert res.decision == "escalate" and res.text is None
    t = res.trace
    _assert_trace_shape(t, [L0_BAD])
    assert t["layer0"]["outcome"] == "fail"
    assert t["dsl"]["passed"] is None and t["dsl"]["violations"] == []
    assert t["attempts"] == 1 and t["regenerated"] is False


def test_regen_called_at_most_once_and_escalates():
    r = Regen("Take 500 mg of metformin tonight.")
    res = post(DSL_BAD, regenerate=r)
    assert r.calls == 1
    assert res.decision == "escalate" and res.text is None
    t = res.trace
    assert t["attempts"] == 2
    assert t["layer0"]["outcome"] == "fail"  # LHM imperative-dose check
    assert t["dsl"]["passed"] is None


def test_regen_dsl_violations_sorted_unique():
    r = Regen("Magnesium 400mg and 200 mg, plus metformin.")
    res = post(L0_BAD, regenerate=r)
    assert res.decision == "escalate"
    assert res.trace["dsl"]["violations"] == ["dosing.unit_quantity", "substance.rx_named"]
    assert res.trace["dsl"]["passed"] is False
    assert res.trace["attempt_log"][0]["layer0"] == "fail"


@pytest.mark.parametrize("bad_regen", [RuntimeError("boom"), ValueError("x")])
def test_regen_exception_escalates(bad_regen):
    r = Regen(bad_regen)
    res = post(DSL_BAD, regenerate=r)
    assert r.calls == 1
    assert res.decision == "escalate" and res.text is None
    json.dumps(res.trace)


def test_regen_non_string_escalates():
    res = post(DSL_BAD, regenerate=Regen(None))
    assert res.decision == "escalate" and res.text is None


def test_crisis_context_threaded_through():
    msg = "I'm so sorry you're dealing with this. I'm here."
    assert post(msg).decision == "pass"
    res = post(msg, crisis_context=True)
    assert res.decision == "escalate"
    assert res.trace["dsl"]["violations"] == ["crisis.missing_handoff"]
    ok = "I'm so sorry. Please call or text 988 now — they can help right away."
    res = post(msg, crisis_context=True, regenerate=Regen(ok))
    assert res.decision == "pass" and res.text == ok


def test_layer0_is_the_real_lhm_function_called_every_draft(monkeypatch):
    seen = []
    real = post_mod.assert_narration_safe
    assert real is claims.assert_narration_safe

    def spy(text):
        seen.append(text)
        return real(text)

    monkeypatch.setattr(post_mod, "assert_narration_safe", spy)
    post(DSL_BAD, regenerate=Regen(GOOD))
    assert seen == [DSL_BAD, GOOD]


def test_layer0_unexpected_exception_fails_closed(monkeypatch):
    def boom(text):
        raise RuntimeError("guard crashed")

    monkeypatch.setattr(post_mod, "assert_narration_safe", boom)
    res = post(GOOD)
    assert res.decision == "escalate" and res.trace["layer0"]["outcome"] == "fail"


def test_no_regenerate_given_fail_escalates():
    res = post(DSL_BAD)
    assert res.decision == "escalate" and res.text is None
    assert res.trace["attempts"] == 1 and res.trace["regenerated"] is False
    assert res.trace["dsl"]["violations"] == ["dosing.unit_quantity"]
