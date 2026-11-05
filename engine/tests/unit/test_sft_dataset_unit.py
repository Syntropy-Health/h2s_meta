"""Implementer unit tests for daypart.train.sft_dataset (CONTRACT slice F, F4)."""

from __future__ import annotations

import hashlib
import json

import pytest

from daypart.membrane import dsl
from daypart.membrane.post import post
from daypart.simmember.cohort import generate
from daypart.simmember.manifest import CuratedLeakError
from daypart.simmember.narrative import NarrativeCache, NarrativeRow, TemplateClient, realize
from daypart.train.sft_dataset import (
    COACH_SYSTEM_PROMPT,
    REFUSAL_PROMPTS,
    REFUSALS,
    build_sft,
    write_jsonl,
)


@pytest.fixture(scope="module")
def cohort():
    return generate(0, n_synthetic=20, days=14)


@pytest.fixture()
def cache(cohort, tmp_path):
    c = NarrativeCache(tmp_path / "n.jsonl")
    for p in cohort.training_profiles()[:5]:
        realize(p, cohort.trajectories[p.member_id], TemplateClient(), c)
    return c


def test_constants():
    assert COACH_SYSTEM_PROMPT and dsl.check(COACH_SYSTEM_PROMPT).passed
    assert len(REFUSAL_PROMPTS) >= 8 and all(not dsl.check(p).passed for p in REFUSAL_PROMPTS)
    assert len(REFUSALS) >= 4
    for r in REFUSALS:
        assert post(r).decision == "pass" and "clinician" in r and "?" in r


def test_dialogue_rows_shape_and_order(cohort, cache):
    rows = build_sft(cohort, cache, refusal_ratio=0)
    assert rows and all(r["meta"]["source"] == "dialogue" for r in rows)
    order = [p.member_id for p in cohort.training_profiles()]
    keys = [(order.index(r["meta"]["member_id"]), r["meta"]["day"]) for r in rows]
    assert keys == sorted(keys)
    for r in rows:
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert r["messages"][0]["content"] == COACH_SYSTEM_PROMPT
        assert r["meta"]["member_id"] not in r["messages"][1]["content"]
        assert len(r["meta"]["prompt_hash"]) == 64
    n_pairs = sum(1 for x in cache.rows() if x.kind == "dialogue" and x.decision == "pass")
    assert len(rows) == n_pairs


def test_refusal_mix_count_and_determinism(cohort, cache):
    base = len(build_sft(cohort, cache, refusal_ratio=0))
    rows = build_sft(cohort, cache, refusal_ratio=0.2, seed=7)
    refusals = [r for r in rows if r["meta"]["source"] == "refusal"]
    assert len(refusals) == round(0.2 * base / 0.8)
    train_ids = {p.member_id for p in cohort.training_profiles()}
    for r in refusals:
        assert r["meta"]["prompt_hash"] is None and r["meta"]["member_id"] in train_ids
        assert r["messages"][1]["content"] in REFUSAL_PROMPTS
        assert r["messages"][2]["content"] in REFUSALS
    assert rows == build_sft(cohort, cache, refusal_ratio=0.2, seed=7)
    assert all(dsl.check(r["messages"][2]["content"]).passed for r in rows)


@pytest.mark.parametrize("ratio", [-0.1, 0.51, 1.0, float("nan")])
def test_bad_ratio(cohort, cache, ratio):
    with pytest.raises(ValueError):
        build_sft(cohort, cache, refusal_ratio=ratio)


def test_hand_edited_failing_dialogue_dropped(cohort, tmp_path, cache):
    path = tmp_path / "edited.jsonl"
    lines = []
    edited = False
    for row in cache.rows():
        d = row.model_dump(mode="json")
        if not edited and row.kind == "dialogue":
            d["text"] = "Take 400 mg of magnesium tonight."
            edited = True
        lines.append(json.dumps(d))
    path.write_text("\n".join(lines) + "\n")
    rows = build_sft(cohort, NarrativeCache(path), refusal_ratio=0)
    assert len(rows) == len(build_sft(cohort, cache, refusal_ratio=0)) - 1
    assert all("magnesium" not in r["messages"][2]["content"] for r in rows)


def test_curated_cache_rows_ignored(cohort, cache):
    cache.put(NarrativeRow(member_id="cur-001", day=1, kind="journal", prompt_hash="b" * 64,
                           model_id="m", decision="pass", text="x", trace={}))
    cache.put(NarrativeRow(member_id="cur-001", day=1, kind="dialogue", prompt_hash="c" * 64,
                           model_id="m", decision="pass", text="y", trace={}))
    rows = build_sft(cohort, cache)
    assert "cur-" not in json.dumps(rows)


def test_leak_in_text_raises(cohort, cache):
    p = cohort.training_profiles()[10]  # not realized by the fixture
    traj = cohort.trajectories[p.member_id]
    day = next(r.day for r in traj.days if r.checkin)
    cache.put(NarrativeRow(member_id=p.member_id, day=day, kind="journal", prompt_hash="d" * 64,
                           model_id="m", decision="pass", text="hi", trace={}))
    cache.put(NarrativeRow(member_id=p.member_id, day=day, kind="dialogue", prompt_hash="e" * 64,
                           model_id="m", decision="pass", text="Say hi to cur-004?", trace={}))
    with pytest.raises(CuratedLeakError):
        build_sft(cohort, cache)


def test_write_jsonl(tmp_path):
    rows = [{"b": 1, "a": [1, 2]}, {"z": None}]
    digest = write_jsonl(rows, tmp_path / "o.jsonl")
    data = (tmp_path / "o.jsonl").read_bytes()
    assert data == b'{"a":[1,2],"b":1}\n{"z":null}\n'
    assert digest == hashlib.sha256(data).hexdigest()
