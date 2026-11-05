"""Sealed slice F — F4 SFT dataset + I1 builder registration.

Contract: docs/daypart/contracts/CONTRACT-slice-F-train.md (F4, plus F5 BUILDERS).
Categories: functional/sft-rows, functional/sft-filter, functional/sft-refusal,
functional/sft-jsonl, integration/sft-i1.

Ambiguities (questions for the principal, NOT asserted here):
- "Final order: deterministic interleave under seed" does not say whether the
  dialogue rows keep their cohort/day relative order after interleaving; we only
  assert set membership and determinism.
- Which prompt_hash (journal or dialogue row) a dialogue row's meta carries is
  unspecified; we assert it is one of the two cached hashes for that (member, day).
- Whether refusal rows' member must also be present in the cache is unspecified;
  we assert it is a training member of the cohort (so the "<= 20 members" bound of
  the I1 builder is asserted on dialogue rows only).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from daypart.membrane import dsl
from daypart.membrane.post import post
from daypart.simmember.cohort import CURATED_PROFILES, generate
from daypart.simmember.manifest import CuratedLeakError, canonical_json
from daypart.simmember.narrative import NarrativeCache, NarrativeRow, realize
from daypart.train.sft_dataset import (
    COACH_SYSTEM_PROMPT,
    REFUSAL_PROMPTS,
    REFUSALS,
    build_sft,
    write_jsonl,
)

from .conftest_train import FAIL_TEXT, checkin_days, cohort0, fail_client, pass_client, training

N_MEMBERS = 4


def _populate(path, members=None, client=None):
    cache = NarrativeCache(path)
    for p in members if members is not None else training(N_MEMBERS):
        realize(p, cohort0().trajectories[p.member_id], client or pass_client(), cache)
    return cache


def _pairs(cache, training_ids):
    """(member, day) -> (journal_row, dialogue_row) for passing pairs."""
    j, d = {}, {}
    for r in cache.rows():
        if r.member_id not in training_ids or r.decision != "pass":
            continue
        (j if r.kind == "journal" else d)[(r.member_id, r.day)] = r
    return {k: (j[k], d[k]) for k in j if k in d}


@pytest.fixture
def sft_cache(tmp_path):
    return _populate(tmp_path / "cache.jsonl")


def _training_ids(cohort=None):
    return {p.member_id for p in (cohort or cohort0()).training_profiles()}


def _by_source(rows, source):
    return [r for r in rows if r["meta"]["source"] == source]


# =========================================================================== #
# functional/sft-refusal — constants
# =========================================================================== #


class TestConstants:
    def test_coach_system_prompt(self):
        assert isinstance(COACH_SYSTEM_PROMPT, str) and COACH_SYSTEM_PROMPT.strip()
        assert dsl.check(COACH_SYSTEM_PROMPT).passed

    def test_refusal_prompts_fail_dsl(self):
        assert isinstance(REFUSAL_PROMPTS, tuple)
        assert len(REFUSAL_PROMPTS) >= 8
        assert len(set(REFUSAL_PROMPTS)) == len(REFUSAL_PROMPTS)
        for rp in REFUSAL_PROMPTS:
            assert isinstance(rp, str) and rp.strip()
            assert dsl.check(rp).passed is False, rp

    def test_refusals_pass_post_and_redirect(self):
        assert isinstance(REFUSALS, tuple)
        assert len(REFUSALS) >= 4
        for r in REFUSALS:
            assert post(r).decision == "pass", r
            assert dsl.check(r).passed
            assert "clinician" in r.casefold()
            assert "?" in r

    def test_no_curated_ids_in_constants(self):
        blob = json.dumps([COACH_SYSTEM_PROMPT, list(REFUSAL_PROMPTS), list(REFUSALS)])
        assert "cur-" not in blob


# =========================================================================== #
# functional/sft-rows
# =========================================================================== #


class TestRowShape:
    def test_row_shape(self, sft_cache):
        rows = build_sft(cohort0(), sft_cache)
        assert rows
        tids = _training_ids()
        for row in rows:
            assert set(row) == {"messages", "meta"}
            msgs = row["messages"]
            assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
            assert all(set(m) == {"role", "content"} for m in msgs)
            assert msgs[0]["content"] == COACH_SYSTEM_PROMPT
            assert isinstance(msgs[1]["content"], str) and msgs[1]["content"]
            assert isinstance(msgs[2]["content"], str) and msgs[2]["content"]
            meta = row["meta"]
            assert set(meta) == {"member_id", "day", "source", "prompt_hash"}
            assert meta["member_id"] in tids
            assert isinstance(meta["day"], int) and not isinstance(meta["day"], bool)
            assert meta["source"] in {"dialogue", "refusal"}
            json.dumps(row)  # JSON-able

    def test_dialogue_rows_match_cache(self, sft_cache):
        rows = build_sft(cohort0(), sft_cache, refusal_ratio=0)
        pairs = _pairs(sft_cache, _training_ids())
        dia = _by_source(rows, "dialogue")
        assert len(dia) == len(rows)
        assert {(r["meta"]["member_id"], r["meta"]["day"]) for r in dia} == set(pairs)
        assert len(dia) == len(pairs)
        for r in dia:
            j, d = pairs[(r["meta"]["member_id"], r["meta"]["day"])]
            assert r["messages"][2]["content"] == d.text
            assert j.text in r["messages"][1]["content"]
            assert r["meta"]["prompt_hash"] in {j.prompt_hash, d.prompt_hash}
            assert r["meta"]["member_id"] not in r["messages"][1]["content"]

    def test_escalated_days_excluded(self, tmp_path):
        members = training(3)
        cache = NarrativeCache(tmp_path / "c.jsonl")
        realize(members[0], cohort0().trajectories[members[0].member_id], pass_client(), cache)
        realize(members[1], cohort0().trajectories[members[1].member_id], fail_client(), cache)
        rows = build_sft(cohort0(), cache, refusal_ratio=0)
        ids = {r["meta"]["member_id"] for r in rows}
        assert ids == {members[0].member_id}
        assert len(rows) == len(checkin_days(members[0].member_id))

    def test_empty_cache(self, tmp_path):
        assert build_sft(cohort0(), NarrativeCache(tmp_path / "none.jsonl")) == []

    def test_members_not_in_cohort_ignored(self, tmp_path):
        small = generate(0, n_synthetic=5)
        big_member = training(1, start=10)[0]  # syn-011: not in the 5-member cohort
        cache = _populate(tmp_path / "c.jsonl", members=[big_member])
        rows = build_sft(small, cache)
        assert big_member.member_id not in json.dumps(rows)
        assert all(r["meta"]["member_id"] in _training_ids(small) for r in rows)
        assert _by_source(rows, "dialogue") == []


# =========================================================================== #
# functional/sft-filter
# =========================================================================== #


class TestDslFilter:
    def test_all_assistant_content_passes_dsl(self, sft_cache):
        for ratio in (0, 0.2, 0.5):
            for row in build_sft(cohort0(), sft_cache, refusal_ratio=ratio):
                assert dsl.check(row["messages"][2]["content"]).passed

    def test_hand_edited_failing_row_dropped(self, tmp_path):
        src = _populate(tmp_path / "src.jsonl")
        edited = NarrativeCache(tmp_path / "edited.jsonl")
        target = None
        for r in src.rows():
            if target is None and r.kind == "dialogue":
                target = (r.member_id, r.day)
                r = r.model_copy(update={"text": FAIL_TEXT})
            edited.put(r)
        assert target is not None
        n_src = len(build_sft(cohort0(), src, refusal_ratio=0))
        rows = build_sft(cohort0(), edited, refusal_ratio=0)
        assert len(rows) == n_src - 1
        assert target not in {(r["meta"]["member_id"], r["meta"]["day"]) for r in rows}
        for row in rows:
            assert row["messages"][2]["content"] != FAIL_TEXT
            assert dsl.check(row["messages"][2]["content"]).passed

    def test_never_edits_text(self, sft_cache):
        texts = {r.text for r in sft_cache.rows() if r.kind == "dialogue"}
        for row in _by_source(build_sft(cohort0(), sft_cache, refusal_ratio=0), "dialogue"):
            assert row["messages"][2]["content"] in texts


# =========================================================================== #
# functional/sft-refusal — mix
# =========================================================================== #


class TestRefusalMix:
    @pytest.mark.parametrize("ratio", [0, 0.1, 0.2, 0.25, 0.4, 0.5])
    def test_refusal_count_formula(self, sft_cache, ratio):
        rows = build_sft(cohort0(), sft_cache, refusal_ratio=ratio)
        n_d = len(_by_source(rows, "dialogue"))
        n_r = len(_by_source(rows, "refusal"))
        assert n_d == len(_pairs(sft_cache, _training_ids()))
        assert n_r == round(ratio * n_d / (1 - ratio))
        assert len(rows) == n_d + n_r

    def test_default_ratio_is_point_two(self, sft_cache):
        rows = build_sft(cohort0(), sft_cache)
        n_d = len(_by_source(rows, "dialogue"))
        assert len(_by_source(rows, "refusal")) == round(0.2 * n_d / 0.8)

    def test_ratio_zero_has_no_refusals(self, sft_cache):
        assert _by_source(build_sft(cohort0(), sft_cache, refusal_ratio=0), "refusal") == []

    @pytest.mark.parametrize("ratio", [-0.01, -1, 0.51, 0.75, 1.0])
    def test_ratio_bounds(self, sft_cache, ratio):
        with pytest.raises(ValueError):
            build_sft(cohort0(), sft_cache, refusal_ratio=ratio)

    def test_refusal_row_content(self, sft_cache):
        rows = _by_source(build_sft(cohort0(), sft_cache, refusal_ratio=0.5), "refusal")
        assert rows
        tids = _training_ids()
        for r in rows:
            assert r["meta"]["prompt_hash"] is None
            assert r["meta"]["member_id"] in tids
            assert r["messages"][2]["content"] in REFUSALS
            user = r["messages"][1]["content"]
            assert any(rp in user for rp in REFUSAL_PROMPTS)
            assert r["meta"]["member_id"] not in user

    def test_deterministic(self, sft_cache, tmp_path):
        a = build_sft(cohort0(), sft_cache, refusal_ratio=0.25, seed=7)
        b = build_sft(cohort0(), sft_cache, refusal_ratio=0.25, seed=7)
        assert a == b
        reloaded = NarrativeCache(tmp_path / "cache.jsonl")  # same file as the fixture
        assert build_sft(cohort0(), reloaded, refusal_ratio=0.25, seed=7) == a

    def test_deterministic_across_rebuilt_cache(self, tmp_path):
        a = build_sft(cohort0(), _populate(tmp_path / "a.jsonl"), seed=3)
        b = build_sft(cohort0(), _populate(tmp_path / "b.jsonl"), seed=3)
        assert a == b
        assert write_jsonl(a, tmp_path / "a.out") == write_jsonl(b, tmp_path / "b.out")


# =========================================================================== #
# functional/sft-jsonl
# =========================================================================== #


class TestWriteJsonl:
    def test_canonical_lines_and_hash(self, sft_cache, tmp_path):
        rows = build_sft(cohort0(), sft_cache)
        out = tmp_path / "sft.jsonl"
        h = write_jsonl(rows, out)
        data = out.read_bytes()
        assert h == hashlib.sha256(data).hexdigest()
        assert len(h) == 64
        text = data.decode("utf-8")
        assert text.endswith("\n")
        lines = text.split("\n")[:-1]
        assert len(lines) == len(rows)
        for line, row in zip(lines, rows):
            assert line == canonical_json(row)
            assert json.loads(line) == row

    def test_accepts_str_path_and_generator(self, tmp_path):
        rows = [{"b": 1, "a": [1, 2]}, {"z": "x"}]
        p = str(tmp_path / "g.jsonl")
        h = write_jsonl((r for r in rows), p)
        with open(p, "rb") as fh:
            assert fh.read() == b'{"a":[1,2],"b":1}\n{"z":"x"}\n'
        assert h == hashlib.sha256(b'{"a":[1,2],"b":1}\n{"z":"x"}\n').hexdigest()


# =========================================================================== #
# integration/sft-i1
# =========================================================================== #


def _curated_rows(member_id: str, day: int):
    common = dict(member_id=member_id, day=day, model_id="fake-v1", decision="pass", trace={"decision": "pass"})
    return [
        NarrativeRow(kind="journal", prompt_hash="c" * 64, text="Curated journal entry, logged today.", **common),
        NarrativeRow(kind="dialogue", prompt_hash="d" * 64, text="Curated reply. Want to plan tomorrow?", **common),
    ]


class TestSftI1:
    def test_curated_cache_rows_ignored(self, tmp_path):
        cache = _populate(tmp_path / "c.jsonl", members=training(2))
        for cur in CURATED_PROFILES[:3]:
            day = checkin_days(cur.member_id)[0]
            for r in _curated_rows(cur.member_id, day):
                cache.put(r)
        rows = build_sft(cohort0(), cache, refusal_ratio=0.5)
        blob = json.dumps(rows)
        assert "cur-" not in blob
        assert "Curated journal entry" not in blob
        assert "Curated reply" not in blob
        assert rows

    def test_leak_in_training_text_raises(self, tmp_path):
        src = _populate(tmp_path / "src.jsonl", members=training(1))
        edited = NarrativeCache(tmp_path / "edited.jsonl")
        done = False
        for r in src.rows():
            if not done and r.kind == "dialogue":
                r = r.model_copy(update={"text": "Nice chat today, say hi to cur-003 for me?"})
                done = True
            edited.put(r)
        with pytest.raises(CuratedLeakError):
            build_sft(cohort0(), edited, refusal_ratio=0)

    def test_builders_registered(self):
        from daypart.ci.i1_check import BUILDERS

        assert "train.sft_dataset" in BUILDERS
        assert "train.dpo_pairs" in BUILDERS
        # pre-existing builders remain
        assert "simmember.training_profiles" in BUILDERS
        assert "simmember.training_trajectories" in BUILDERS

    def test_sft_builder_output(self):
        from daypart.ci.i1_check import BUILDERS

        rows = list(BUILDERS["train.sft_dataset"](cohort0()))
        assert rows
        realized = {r["meta"]["member_id"] for r in _by_source(rows, "dialogue")}
        assert realized
        assert len(realized) <= 20
        assert {r["meta"]["member_id"] for r in rows} <= _training_ids()
        assert "cur-" not in json.dumps(rows)
        for r in rows:
            assert dsl.check(r["messages"][2]["content"]).passed
        assert rows == list(BUILDERS["train.sft_dataset"](cohort0()))

    def test_dpo_builder_output(self):
        from daypart.ci.i1_check import BUILDERS

        rows = list(BUILDERS["train.dpo_pairs"](cohort0()))
        members = {r["meta"]["member_id"] for r in rows}
        assert len(members) <= 20
        assert members <= _training_ids()
        assert "cur-" not in json.dumps(rows)

    def test_i1_main_passes(self, capsys):
        from daypart.ci.i1_check import main

        assert main() == 0
        out = capsys.readouterr().out
        assert "train.sft_dataset" in out
        assert "train.dpo_pairs" in out
