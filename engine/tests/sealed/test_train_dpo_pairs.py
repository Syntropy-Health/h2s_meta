"""Sealed slice F — F5 lapse-recovery rubric, DPO contexts and pairs.

Contract: docs/daypart/contracts/CONTRACT-slice-F-train.md (F5).
Categories: functional/dpo-rubric, functional/dpo-contexts, functional/dpo-pairs,
integration/dpo-i1.

Ambiguities (questions for the principal, NOT asserted here):
- When no principled candidate qualifies, whether the unconstrained call is still
  made is unspecified; we assert only the candidate count and "no pair".
- Output order of ``make_contexts`` across members/days is unspecified beyond
  "deterministic"; we assert the selected set and per-member bounds.
"""

from __future__ import annotations

import dataclasses
import json

import pydantic
import pytest

from daypart.membrane import dsl
from daypart.simmember.cohort import CURATED_PROFILES
from daypart.simmember.manifest import CuratedLeakError
from daypart.simmember.narrative import NarrativeCache, NarrativeRow, realize
from daypart.train.dpo_pairs import (
    PRINCIPLES,
    RUBRIC_VERSION,
    SHAME_TERMS,
    DpoContext,
    RubricVerdict,
    build_pairs,
    make_contexts,
    rubric_check,
)
from daypart.train.sft_dataset import COACH_SYSTEM_PROMPT

from .conftest_train import (
    FAIL_TEXT,
    FAIL_TEXT_2,
    GOOD_CHOSEN,
    GOOD_CHOSEN_2,
    NO_INVITATION_TEXT,
    SHAME_TEXT,
    RecordingClient,
    cohort0,
    fail_client,
    pass_client,
    training,
)

STATES = ("engaged", "strained", "lapsed", "recovering")


def _principled_system() -> str:
    return COACH_SYSTEM_PROMPT + "\n\n" + PRINCIPLES


# =========================================================================== #
# functional/dpo-rubric
# =========================================================================== #


class TestConstants:
    def test_rubric_version(self):
        assert RUBRIC_VERSION == "lapse-recovery-v1"

    def test_shame_terms_superset(self):
        required = {"failed", "failure", "lazy", "disappointed", "disappointing", "no excuse", "should have", "shame", "guilt"}
        assert isinstance(SHAME_TERMS, tuple)
        assert required <= set(SHAME_TERMS)

    def test_principles_pass_dsl(self):
        assert isinstance(PRINCIPLES, str) and PRINCIPLES.strip()
        assert dsl.check(PRINCIPLES).passed


class TestRubricCheck:
    def test_verdict_type(self):
        v = rubric_check("engaged", "Nice work today.")
        assert isinstance(v, RubricVerdict)
        assert dataclasses.is_dataclass(v)
        assert v.passed is True
        assert v.failed == ()
        with pytest.raises(dataclasses.FrozenInstanceError):
            v.passed = False  # type: ignore[misc]

    @pytest.mark.parametrize("state", STATES)
    def test_good_text_passes_all_states(self, state):
        v = rubric_check(state, GOOD_CHOSEN)
        assert v.passed and v.failed == ()

    @pytest.mark.parametrize("text", ["", "   ", "\n\t "])
    def test_length_empty(self, text):
        v = rubric_check("engaged", text)
        assert not v.passed
        assert "length" in v.failed

    def test_length_bounds(self):
        assert rubric_check("engaged", "a").passed
        assert rubric_check("engaged", "x" * 480).passed
        assert rubric_check("engaged", "  " + "x" * 480 + "  ").passed  # stripped
        v = rubric_check("engaged", "x" * 481)
        assert not v.passed and v.failed == ("length",)

    @pytest.mark.parametrize(
        "term", ["failed", "failure", "lazy", "disappointed", "disappointing", "no excuse", "should have", "shame", "guilt"]
    )
    def test_shame_terms(self, term):
        v = rubric_check("engaged", f"Well, {term} happens sometimes.")
        assert not v.passed
        assert "shame" in v.failed

    def test_all_declared_shame_terms_fire(self):
        for term in SHAME_TERMS:
            assert "shame" in rubric_check("engaged", f"So: {term}.").failed, term

    def test_shame_casefolded(self):
        assert "shame" in rubric_check("engaged", "You FAILED yesterday.").failed
        assert "shame" in rubric_check("engaged", "No Excuse this time.").failed

    def test_shame_word_boundary(self):
        # "lazy" embedded inside a longer token is not a word match
        assert rubric_check("engaged", "We could try the glazyware mug today.").passed

    @pytest.mark.parametrize("state", ["strained", "lapsed", "recovering"])
    def test_invitation_required_when_struggling(self, state):
        v = rubric_check(state, NO_INVITATION_TEXT)
        assert not v.passed
        assert v.failed == ("invitation",)
        assert rubric_check(state, NO_INVITATION_TEXT[:-1] + "?").passed

    def test_engaged_invitation_always_passes(self):
        assert rubric_check("engaged", NO_INVITATION_TEXT).passed

    def test_multiple_failures(self):
        v = rubric_check("lapsed", "")
        assert not v.passed
        assert set(v.failed) == {"length", "invitation"}
        v = rubric_check("lapsed", "You failed, that is lazy.")
        assert set(v.failed) == {"shame", "invitation"}
        assert set(v.failed) <= {"length", "shame", "invitation"}

    @pytest.mark.parametrize("state", ["", "Engaged", "relapsed", "unknown"])
    def test_unknown_state(self, state):
        with pytest.raises(ValueError):
            rubric_check(state, GOOD_CHOSEN)


# =========================================================================== #
# functional/dpo-contexts
# =========================================================================== #


class TestDpoContext:
    def test_fields_and_validation(self):
        c = DpoContext(member_id="syn-001", day=3, member_state="lapsed", user_turn="hi")
        assert c.member_state == "lapsed"
        with pytest.raises(pydantic.ValidationError):
            DpoContext(member_id="syn-001", day=3, member_state="sad", user_turn="hi")
        with pytest.raises(pydantic.ValidationError):
            DpoContext(member_id="syn-001", day=3, member_state="lapsed", user_turn="hi", extra=1)
        with pytest.raises((pydantic.ValidationError, TypeError, AttributeError)):
            c.day = 4  # type: ignore[misc]


def _populated(tmp_path, members, client=None):
    cache = NarrativeCache(tmp_path / "c.jsonl")
    for p in members:
        realize(p, cohort0().trajectories[p.member_id], client or pass_client(), cache)
    return cache


def _expected_days(cache, member_id, per_member):
    tr = cohort0().trajectories[member_id]
    state = {r.day: r.state for r in tr.days}
    days = sorted(
        {r.day for r in cache.rows() if r.member_id == member_id and r.kind == "journal" and r.decision == "pass"}
    )
    days.sort(key=lambda d: (state[d] == "engaged", d))
    return days[:per_member]


class TestMakeContexts:
    def test_selection_rule(self, tmp_path):
        members = training(5)
        cache = _populated(tmp_path, members)
        ctxs = make_contexts(cohort0(), cache, per_member=2)
        assert all(isinstance(c, DpoContext) for c in ctxs)
        got: dict[str, set[int]] = {}
        for c in ctxs:
            got.setdefault(c.member_id, set()).add(c.day)
        for p in members:
            assert got.get(p.member_id, set()) == set(_expected_days(cache, p.member_id, 2)), p.member_id
        assert set(got) <= {p.member_id for p in members}
        assert len(ctxs) == len({(c.member_id, c.day) for c in ctxs})

    @pytest.mark.parametrize("per_member", [1, 3])
    def test_per_member(self, tmp_path, per_member):
        members = training(3)
        cache = _populated(tmp_path, members)
        ctxs = make_contexts(cohort0(), cache, per_member=per_member)
        for p in members:
            days = {c.day for c in ctxs if c.member_id == p.member_id}
            assert days == set(_expected_days(cache, p.member_id, per_member))

    def test_state_and_user_turn(self, tmp_path):
        members = training(3)
        cache = _populated(tmp_path, members)
        journals = {(r.member_id, r.day): r.text for r in cache.rows() if r.kind == "journal"}
        for c in make_contexts(cohort0(), cache):
            rec = cohort0().trajectories[c.member_id].days[c.day]
            assert c.member_state == rec.state
            assert journals[(c.member_id, c.day)] in c.user_turn
            assert c.member_id not in c.user_turn
            assert "cur-" not in c.user_turn

    def test_max_members(self, tmp_path):
        members = training(5)
        cache = _populated(tmp_path, members)
        ctxs = make_contexts(cohort0(), cache, max_members=2)
        assert len({c.member_id for c in ctxs}) <= 2
        assert ctxs

    def test_escalated_journals_skipped(self, tmp_path):
        a, b = training(2)
        cache = NarrativeCache(tmp_path / "c.jsonl")
        realize(a, cohort0().trajectories[a.member_id], pass_client(), cache)
        realize(b, cohort0().trajectories[b.member_id], fail_client(), cache)
        ids = {c.member_id for c in make_contexts(cohort0(), cache)}
        assert ids == {a.member_id}

    def test_deterministic(self, tmp_path):
        members = training(4)
        cache = _populated(tmp_path, members)
        assert make_contexts(cohort0(), cache) == make_contexts(cohort0(), cache)

    def test_curated_rows_ignored(self, tmp_path):
        cache = _populated(tmp_path, training(2))
        for cur in CURATED_PROFILES[:2]:
            day = next(r.day for r in cohort0().trajectories[cur.member_id].days if r.checkin)
            cache.put(
                NarrativeRow(
                    member_id=cur.member_id, day=day, kind="journal", prompt_hash="e" * 64,
                    model_id="fake-v1", decision="pass", text="Curated journal, logged.", trace={},
                )
            )
        ctxs = make_contexts(cohort0(), cache)
        assert ctxs
        assert all(not c.member_id.startswith("cur-") for c in ctxs)
        assert "Curated journal" not in json.dumps([c.model_dump() for c in ctxs])


# =========================================================================== #
# functional/dpo-pairs
# =========================================================================== #


def _ctx(member="syn-001", day=3, state="lapsed", turn="Yesterday felt hard and I skipped my routine."):
    return DpoContext(member_id=member, day=day, member_state=state, user_turn=turn)


def scripted_sampler(principled, unconstrained):
    """Principled calls (PRINCIPLES in system) consume ``principled`` in order;
    unconstrained calls consume ``unconstrained``. Exhaustion fails the test."""
    queues = {"p": list(principled), "u": list(unconstrained)}

    def respond(messages, seed, n):
        system = messages[0]["content"]
        key = "p" if PRINCIPLES in system else "u"
        if not queues[key]:
            raise AssertionError(f"unexpected extra {key} call")
        return queues[key].pop(0)

    return RecordingClient(respond, model_id="sampler-v1")


def _kinds(client):
    return ["p" if PRINCIPLES in c.messages[0]["content"] else "u" for c in client.calls]


class TestBuildPairs:
    def test_first_candidate_qualifies(self):
        s = scripted_sampler([GOOD_CHOSEN], [FAIL_TEXT])
        ctx = _ctx()
        rows = build_pairs([ctx], s, n_candidates=4)
        assert _kinds(s) == ["p", "u"]
        assert s.calls[0].messages == [
            {"role": "system", "content": _principled_system()},
            {"role": "user", "content": ctx.user_turn},
        ]
        assert s.calls[1].messages == [
            {"role": "system", "content": COACH_SYSTEM_PROMPT},
            {"role": "user", "content": ctx.user_turn},
        ]
        assert len(rows) == 1
        assert rows[0]["chosen"] == [{"role": "assistant", "content": GOOD_CHOSEN}]
        assert rows[0]["rejected"] == [{"role": "assistant", "content": FAIL_TEXT}]

    def test_first_qualifying_candidate_chosen(self):
        # dsl-fail, rubric-fail (no invitation in lapsed), rubric-fail (shame), good, good2
        s = scripted_sampler([FAIL_TEXT, NO_INVITATION_TEXT, SHAME_TEXT, GOOD_CHOSEN, GOOD_CHOSEN_2], [FAIL_TEXT_2])
        rows = build_pairs([_ctx()], s, n_candidates=5)
        assert _kinds(s) == ["p", "p", "p", "p", "u"]
        assert rows[0]["chosen"][0]["content"] == GOOD_CHOSEN

    def test_exactly_one_unconstrained_call(self):
        s = scripted_sampler([FAIL_TEXT, GOOD_CHOSEN], [NO_INVITATION_TEXT])
        build_pairs([_ctx()], s, n_candidates=4)
        assert _kinds(s).count("u") == 1
        assert _kinds(s)[-1] == "u"

    def test_no_qualifying_candidate_no_pair(self):
        s = scripted_sampler([FAIL_TEXT, NO_INVITATION_TEXT, SHAME_TEXT], [FAIL_TEXT_2, FAIL_TEXT_2])
        rows = build_pairs([_ctx()], s, n_candidates=3)
        assert rows == []
        assert _kinds(s).count("p") == 3

    def test_identical_rejected_no_pair(self):
        s = scripted_sampler([GOOD_CHOSEN], [GOOD_CHOSEN])
        assert build_pairs([_ctx()], s) == []

    def test_seeds_distinct_and_deterministic(self):
        def run(seed):
            s = scripted_sampler([FAIL_TEXT, FAIL_TEXT, GOOD_CHOSEN], [FAIL_TEXT_2])
            build_pairs([_ctx()], s, n_candidates=4, seed=seed)
            return [c.seed for c in s.calls]

        a = run(0)
        assert len(set(a)) == len(a)  # candidate seeds distinct, rejected seed distinct
        assert run(0) == a
        assert run(1) != a

    def test_seeds_depend_on_context(self):
        seeds = []
        for ctx in (_ctx(day=3), _ctx(day=4)):
            s = scripted_sampler([GOOD_CHOSEN], [FAIL_TEXT])
            build_pairs([ctx], s)
            seeds.append(s.calls[0].seed)
        assert seeds[0] != seeds[1]

    def test_row_shape(self):
        ctx = _ctx(state="engaged")
        s = scripted_sampler([GOOD_CHOSEN], [FAIL_TEXT])  # FAIL_TEXT: DSL fails, rubric passes when engaged
        (row,) = build_pairs([ctx], s)
        assert set(row) == {"prompt", "chosen", "rejected", "meta"}
        assert row["prompt"] == [
            {"role": "system", "content": COACH_SYSTEM_PROMPT},
            {"role": "user", "content": ctx.user_turn},
        ]
        assert row["meta"] == {
            "member_id": ctx.member_id,
            "day": ctx.day,
            "member_state": "engaged",
            "rubric_version": RUBRIC_VERSION,
            "rejected_dsl_passed": False,
            "rejected_rubric_passed": True,
        }
        json.dumps(row)

    def test_meta_rejected_flags_dsl_pass_rubric_fail(self):
        s = scripted_sampler([GOOD_CHOSEN], [SHAME_TEXT])
        (row,) = build_pairs([_ctx(state="lapsed")], s)
        assert row["meta"]["rejected_dsl_passed"] is True
        assert row["meta"]["rejected_rubric_passed"] is False

    def test_meta_rejected_flags_both_pass(self):
        s = scripted_sampler([GOOD_CHOSEN], [GOOD_CHOSEN_2])
        (row,) = build_pairs([_ctx(state="strained")], s)
        assert row["meta"]["rejected_dsl_passed"] is True
        assert row["meta"]["rejected_rubric_passed"] is True
        assert row["rejected"][0]["content"] == GOOD_CHOSEN_2

    def test_engaged_state_no_invitation_needed(self):
        s = scripted_sampler([NO_INVITATION_TEXT], [FAIL_TEXT])
        (row,) = build_pairs([_ctx(state="engaged")], s)
        assert row["chosen"][0]["content"] == NO_INVITATION_TEXT

    def test_multiple_contexts_in_order(self):
        ctxs = [_ctx(member="syn-001", day=2), _ctx(member="syn-002", day=5), _ctx(member="syn-003", day=1)]
        # middle context gets no qualifying candidate (n_candidates=1)
        s = scripted_sampler([GOOD_CHOSEN, FAIL_TEXT, GOOD_CHOSEN_2], [FAIL_TEXT_2, FAIL_TEXT_2, FAIL_TEXT_2])
        rows = build_pairs(ctxs, s, n_candidates=1)
        assert [(r["meta"]["member_id"], r["meta"]["day"]) for r in rows] == [("syn-001", 2), ("syn-003", 1)]

    def test_empty_contexts(self):
        s = scripted_sampler([], [])
        assert build_pairs([], s) == []
        assert s.n_calls == 0

    @pytest.mark.parametrize("n", [0, -1])
    def test_n_candidates_validation(self, n):
        s = scripted_sampler([], [])
        with pytest.raises(ValueError):
            build_pairs([_ctx()], s, n_candidates=n)
        assert s.n_calls == 0

    def test_curated_context_refused_before_any_call(self):
        s = scripted_sampler([GOOD_CHOSEN] * 5, [FAIL_TEXT] * 5)
        ctxs = [_ctx(member="syn-001"), _ctx(member="cur-004")]
        with pytest.raises(CuratedLeakError):
            build_pairs(ctxs, s)
        assert s.n_calls == 0

    def test_sampler_exception_propagates(self):
        class Boom(Exception):
            pass

        def respond(messages, seed, n):
            raise Boom("sampler down")

        with pytest.raises(Boom):
            build_pairs([_ctx()], RecordingClient(respond))

    def test_deterministic_output(self):
        def run():
            s = scripted_sampler([FAIL_TEXT, GOOD_CHOSEN], [SHAME_TEXT])
            return build_pairs([_ctx()], s, seed=5)

        assert run() == run()


# =========================================================================== #
# integration/dpo-i1
# =========================================================================== #


class TestDpoIntegration:
    def test_end_to_end_no_curated(self, tmp_path):
        from daypart.simmember.narrative import TemplateClient

        cache = _populated(tmp_path, training(3), client=TemplateClient())
        for cur in CURATED_PROFILES[:2]:
            day = next(r.day for r in cohort0().trajectories[cur.member_id].days if r.checkin)
            cache.put(
                NarrativeRow(
                    member_id=cur.member_id, day=day, kind="journal", prompt_hash="f" * 64,
                    model_id="t", decision="pass", text="Curated journal, logged.", trace={},
                )
            )
        ctxs = make_contexts(cohort0(), cache)
        rows = build_pairs(ctxs, TemplateClient())
        blob = json.dumps(rows)
        assert "cur-" not in blob
        for r in rows:
            assert dsl.check(r["chosen"][0]["content"]).passed
            assert rubric_check(r["meta"]["member_state"], r["chosen"][0]["content"]).passed
            assert r["chosen"] != r["rejected"]
