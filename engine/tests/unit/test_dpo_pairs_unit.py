"""Implementer unit tests for daypart.train.dpo_pairs (CONTRACT slice F, F5)."""

from __future__ import annotations

import pytest

from daypart.membrane import dsl
from daypart.simmember.cohort import generate
from daypart.simmember.manifest import CuratedLeakError
from daypart.simmember.narrative import NarrativeCache, NarrativeRow, TemplateClient, realize
from daypart.train.dpo_pairs import (
    PRINCIPLES,
    RUBRIC_VERSION,
    SHAME_TERMS,
    DpoContext,
    build_pairs,
    make_contexts,
    rubric_check,
)
from daypart.train.sft_dataset import COACH_SYSTEM_PROMPT

GOOD = "Missed days happen. Want to pick one small step for tomorrow?"


class Script:
    model_id = "script"

    def __init__(self, principled, unconstrained):
        self.principled = list(principled)
        self.unconstrained = unconstrained
        self.calls = []

    def complete(self, messages, *, seed, temperature=0.7, max_tokens=256):
        sys = messages[0]["content"]
        kind = "p" if PRINCIPLES in sys else "u"
        self.calls.append((kind, seed))
        if kind == "p":
            out = self.principled.pop(0)
            if isinstance(out, BaseException):
                raise out
            return out
        return self.unconstrained


def ctx(state="lapsed", mid="syn-001"):
    return DpoContext(member_id=mid, day=3, member_state=state, user_turn="Day 3. I missed it.")


def test_constants():
    for t in ("failed", "failure", "lazy", "disappointed", "disappointing", "no excuse",
              "should have", "shame", "guilt"):
        assert t in SHAME_TERMS
    assert dsl.check(PRINCIPLES).passed and RUBRIC_VERSION == "lapse-recovery-v1"


def test_rubric():
    assert rubric_check("engaged", "Nice work.").passed
    assert rubric_check("lapsed", "Nice work.").failed == ("invitation",)
    assert "shame" in rubric_check("engaged", "You FAILED again.").failed
    assert "shame" in rubric_check("engaged", "You should  have done it.").failed
    assert rubric_check("engaged", "Shameless plug").passed  # word boundary
    assert "length" in rubric_check("engaged", "   ").failed
    assert "length" in rubric_check("engaged", "a" * 481).failed
    assert rubric_check("engaged", "a" * 480).passed
    with pytest.raises(ValueError):
        rubric_check("happy", "x")


def test_pair_row_and_call_sequence():
    s = Script(["Take 2 pills.", "You failed? Try again?", GOOD, "unused"], "You were lazy.")
    rows = build_pairs([ctx()], s, n_candidates=4)
    assert [k for k, _ in s.calls] == ["p", "p", "p", "u"]
    assert len({seed for _, seed in s.calls}) == 4
    (row,) = rows
    assert row["prompt"] == [{"role": "system", "content": COACH_SYSTEM_PROMPT},
                             {"role": "user", "content": "Day 3. I missed it."}]
    assert row["chosen"] == [{"role": "assistant", "content": GOOD}]
    assert row["rejected"] == [{"role": "assistant", "content": "You were lazy."}]
    assert row["meta"] == {"member_id": "syn-001", "day": 3, "member_state": "lapsed",
                           "rubric_version": RUBRIC_VERSION, "rejected_dsl_passed": True,
                           "rejected_rubric_passed": False}


def test_no_qualifying_candidate_no_unconstrained_call():
    s = Script(["Take 2 pills.", "No question here."], "x")
    assert build_pairs([ctx()], s, n_candidates=2) == []
    assert [k for k, _ in s.calls] == ["p", "p"]


def test_identical_rejected_dropped_and_determinism():
    assert build_pairs([ctx()], Script([GOOD], GOOD), n_candidates=1) == []
    a, b = Script([GOOD], "Ok."), Script([GOOD], "Ok.")
    assert build_pairs([ctx()], a, seed=5) == build_pairs([ctx()], b, seed=5)
    assert a.calls == b.calls
    c = Script([GOOD], "Ok.")
    build_pairs([ctx()], c, seed=6)
    assert c.calls != a.calls


def test_errors():
    with pytest.raises(ValueError):
        build_pairs([ctx()], Script([], ""), n_candidates=0)
    s = Script([GOOD], "x")
    with pytest.raises(CuratedLeakError):
        build_pairs([ctx(), ctx(mid="cur-002")], s)
    assert s.calls == []
    with pytest.raises(RuntimeError):
        build_pairs([ctx()], Script([RuntimeError("boom")], "x"))


def test_make_contexts_prefers_non_engaged(tmp_path):
    cohort = generate(0, n_synthetic=10, days=21)
    cache = NarrativeCache(tmp_path / "n.jsonl")
    for p in cohort.training_profiles():
        realize(p, cohort.trajectories[p.member_id], TemplateClient(), cache)
    cache.put(NarrativeRow(member_id="cur-001", day=0, kind="journal", prompt_hash="f" * 64,
                           model_id="m", decision="pass", text="x", trace={}))
    out = make_contexts(cohort, cache, per_member=2)
    assert out == make_contexts(cohort, cache, per_member=2)
    assert all(not c.member_id.startswith("cur-") for c in out)
    for p in cohort.training_profiles():
        mine = [c for c in out if c.member_id == p.member_id]
        assert len(mine) <= 2
        traj = cohort.trajectories[p.member_id]
        avail = [r for r in traj.days if r.checkin]
        if mine and any(r.state != "engaged" for r in avail):
            assert mine[0].member_state != "engaged"
            first = min(r.day for r in avail if r.state != "engaged")
            assert mine[0].day == first
        for c in mine:
            assert c.member_id not in c.user_turn
    assert len({c.member_id for c in make_contexts(cohort, cache, max_members=3)}) == 3
    pairs = build_pairs(out, TemplateClient())
    for row in pairs:
        assert dsl.check(row["chosen"][0]["content"]).passed
        assert rubric_check(row["meta"]["member_state"], row["chosen"][0]["content"]).passed
