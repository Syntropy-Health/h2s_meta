"""Implementer unit tests for daypart.simmember.narrative (CONTRACT slice F, F1-F3)."""

from __future__ import annotations

import io
import json

import pydantic
import pytest

from daypart.membrane.post import post
from daypart.simmember import narrative as nar
from daypart.simmember.cohort import CURATED_PROFILES, Profile, generate
from daypart.simmember.manifest import CuratedLeakError
from daypart.simmember.narrative import (
    MissingCredentialsError,
    MuseClient,
    NarrativeCache,
    NarrativeRow,
    TemplateClient,
    build_prompt,
    realize,
)

ENV = {"MUSE_API_KEY": "sk-secret-123", "MUSE_BASE_URL": "https://muse.example/v1", "MUSE_SPARK_MODEL": "spark-1"}


@pytest.fixture(scope="module")
def cohort():
    return generate(0, n_synthetic=30, days=14)


def _syn(cohort, i=0):
    p = cohort.training_profiles()[i]
    return p, cohort.trajectories[p.member_id]


def _checkin(traj):
    return next(r for r in traj.days if r.checkin)


class Counting:
    def __init__(self, inner=None, outputs=None):
        self.inner = inner or TemplateClient()
        self.outputs = list(outputs) if outputs is not None else None
        self.model_id = "counting"
        self.calls = []

    def complete(self, messages, *, seed, temperature=0.7, max_tokens=256):
        self.calls.append((messages, seed))
        if self.outputs is not None:
            return self.outputs.pop(0) if self.outputs else "Thanks for checking in. What helped today?"
        return self.inner.complete(messages, seed=seed)


# ---- F1 ---------------------------------------------------------------------


def test_muse_missing_vars_named_not_values():
    with pytest.raises(MissingCredentialsError) as ei:
        MuseClient(env={"MUSE_API_KEY": "sk-secret-123", "MUSE_BASE_URL": ""})
    msg = str(ei.value)
    assert "MUSE_BASE_URL" in msg and "MUSE_SPARK_MODEL" in msg and "sk-secret-123" not in msg


def test_muse_model_override_skips_model_var_and_repr_hides_key():
    c = MuseClient(model_id="m", env={k: v for k, v in ENV.items() if k != "MUSE_SPARK_MODEL"})
    assert c.model_id == "m"
    assert "sk-secret-123" not in repr(c)
    assert MuseClient(env=ENV).model_id == "spark-1"


def test_muse_complete_posts_chat_completions(monkeypatch):
    seen = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data)
        return Resp(json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode())

    monkeypatch.setattr(nar.urllib.request, "urlopen", fake_urlopen)
    out = MuseClient(env=ENV).complete([{"role": "user", "content": "x"}], seed=3)
    assert out == "hi"
    assert seen["url"] == "https://muse.example/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-secret-123"
    assert seen["body"]["model"] == "spark-1" and seen["body"]["seed"] == 3


def test_template_client_deterministic_and_pools_pass_post():
    c = TemplateClient()
    msgs = [{"role": "user", "content": "hello"}]
    assert c.complete(msgs, seed=1) == c.complete(msgs, seed=1)
    assert c.model_id == "template-v1"
    for pool in (nar.TEMPLATE_JOURNALS, nar.TEMPLATE_DIALOGUES, nar.TEMPLATE_REFUSALS):
        for text in pool:
            assert post(text).decision == "pass", text
    assert c.complete([{"role": "user", "content": "Should I take 500 mg of magnesium?"}], seed=0) in nar.TEMPLATE_REFUSALS


# ---- F2 ---------------------------------------------------------------------


def test_prompt_is_latent_conditioned_and_deterministic(cohort):
    p, traj = _syn(cohort)
    rec = _checkin(traj)
    a = build_prompt(p, rec, "journal")
    assert a == build_prompt(p, rec, "journal")
    blob = " ".join(m["content"] for m in a.messages)
    lat = p.latents
    for lvl in (lat.motivation, lat.barrier_type, lat.literacy, lat.engagement_pattern):
        assert lvl.replace("_", " ") in blob
    assert rec.state in blob
    assert p.member_id not in blob and "cur-" not in blob
    assert a.member_id == p.member_id and a.kind == "journal" and len(a.prompt_hash) == 64


@pytest.mark.parametrize("axis", ["motivation", "barrier_type", "literacy", "engagement_pattern"])
def test_each_latent_changes_hash(cohort, axis):
    from daypart.simmember.latents import AXES

    p, traj = _syn(cohort)
    rec = _checkin(traj)
    other = next(lvl for lvl in AXES[axis] if lvl != getattr(p.latents, axis))
    q = Profile(member_id=p.member_id, cohort="synthetic", latents=p.latents.model_copy(update={axis: other}))
    assert build_prompt(p, rec, "journal").prompt_hash != build_prompt(q, rec, "journal").prompt_hash


def test_prompt_kind_rules(cohort):
    p, traj = _syn(cohort)
    rec = _checkin(traj)
    d = build_prompt(p, rec, "dialogue", journal="I walked today.")
    assert "I walked today." in " ".join(m["content"] for m in d.messages)
    with pytest.raises(ValueError):
        build_prompt(p, rec, "dialogue")
    with pytest.raises(ValueError):
        build_prompt(p, rec, "dialogue", journal="  ")
    with pytest.raises(ValueError):
        build_prompt(p, rec, "journal", journal="x")
    with pytest.raises(ValueError):
        build_prompt(p, rec, "summary")


def test_curated_profiles_rejected(cohort):
    p, traj = _syn(cohort)
    rec = _checkin(traj)
    with pytest.raises(CuratedLeakError):
        build_prompt(CURATED_PROFILES[0], rec, "journal")
    relabeled = Profile(member_id=p.member_id, cohort="synthetic", latents=CURATED_PROFILES[3].latents)
    with pytest.raises(CuratedLeakError):
        build_prompt(relabeled, rec, "journal")


# ---- F3 ---------------------------------------------------------------------


def _row(**kw):
    base = dict(member_id="syn-001", day=1, kind="journal", prompt_hash="a" * 64, model_id="m",
                decision="pass", text="ok", trace={})
    base.update(kw)
    return NarrativeRow(**base)


def test_row_text_iff_pass():
    with pytest.raises(pydantic.ValidationError):
        _row(text=None)
    with pytest.raises(pydantic.ValidationError):
        _row(decision="escalate", text="x")
    assert _row(decision="escalate", text=None).text is None


def test_cache_put_get_first_write_wins_and_reload(tmp_path):
    path = tmp_path / "c.jsonl"
    cache = NarrativeCache(path)
    assert len(cache) == 0 and not path.exists()
    cache.put(_row())
    cache.put(_row(text="second"))
    assert len(cache) == 1 and cache.get("syn-001", 1, "a" * 64).text == "ok"
    assert cache.get("syn-001", 2, "a" * 64) is None
    again = NarrativeCache(path)
    assert again.rows() == cache.rows()
    assert path.read_text().endswith("\n") and path.read_text().count("\n") == 1


def test_cache_malformed_line(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text("{not json\n")
    with pytest.raises(ValueError):
        NarrativeCache(path)
    path.write_text(json.dumps({"member_id": "syn-001"}) + "\n")
    with pytest.raises(ValueError):
        NarrativeCache(path)


def test_realize_rows_and_cache_hits(cohort, tmp_path):
    p, traj = _syn(cohort)
    cache = NarrativeCache(tmp_path / "c.jsonl")
    client = Counting()
    rows = realize(p, traj, client, cache)
    checkins = [r.day for r in traj.days if r.checkin]
    assert [r.day for r in rows if r.kind == "journal"] == checkins
    assert all(r.decision == "pass" and post(r.text).decision == "pass" for r in rows)
    assert [r.kind for r in rows] == ["journal", "dialogue"] * len(checkins)
    n = len(client.calls)
    assert n == 2 * len(checkins)
    again = realize(p, traj, client, NarrativeCache(tmp_path / "c.jsonl"))
    assert len(client.calls) == n and again == rows


def test_realize_byte_identical_and_days_filter(cohort, tmp_path):
    p, traj = _syn(cohort, 1)
    realize(p, traj, TemplateClient(), NarrativeCache(tmp_path / "a.jsonl"))
    realize(p, traj, TemplateClient(), NarrativeCache(tmp_path / "b.jsonl"))
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    day = _checkin(traj).day
    rows = realize(p, traj, TemplateClient(), NarrativeCache(tmp_path / "c.jsonl"), days=[day])
    assert {r.day for r in rows} == {day}


def test_realize_escalation_cached_without_dialogue(cohort, tmp_path):
    p, traj = _syn(cohort)
    day = _checkin(traj).day
    bad = "Take 2 pills tonight."
    client = Counting(outputs=[bad, bad])
    cache = NarrativeCache(tmp_path / "c.jsonl")
    rows = realize(p, traj, client, cache, days=[day])
    assert len(rows) == 1 and rows[0].decision == "escalate" and rows[0].text is None
    assert rows[0].trace["regenerated"] is True and bad not in json.dumps(rows[0].trace)
    s0, s1 = client.calls[0][1], client.calls[1][1]
    assert s1 == s0 + 1 and s0 == nar.narrative_seed(p.member_id, day, "journal")
    realize(p, traj, client, cache, days=[day])
    assert len(client.calls) == 2


def test_realize_regenerate_pass(cohort, tmp_path):
    p, traj = _syn(cohort)
    day = _checkin(traj).day
    good = "Quiet day, I kept my routine."
    client = Counting(outputs=["Take 2 pills tonight.", good])
    rows = realize(p, traj, client, NarrativeCache(tmp_path / "c.jsonl"), days=[day])
    assert rows[0].text == good and rows[1].kind == "dialogue"
    assert good in client.calls[2][0][1]["content"]


def test_realize_guards(cohort, tmp_path):
    p, traj = _syn(cohort)
    other = cohort.trajectories[cohort.training_profiles()[1].member_id]
    client = Counting()
    cache = NarrativeCache(tmp_path / "c.jsonl")
    with pytest.raises(ValueError):
        realize(p, other, client, cache)
    cur = CURATED_PROFILES[0]
    with pytest.raises(CuratedLeakError):
        realize(cur, cohort.trajectories[cur.member_id], client, cache)
    assert client.calls == [] and len(cache) == 0 and not (tmp_path / "c.jsonl").exists()


# ---------------------------------------------------------------- HAC-28: reasoning models


class _JsonResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _serve(monkeypatch, payload, seen=None):
    def fake_urlopen(req, timeout):
        if seen is not None:
            seen["body"] = json.loads(req.data)
        return _JsonResp(json.dumps(payload).encode())

    monkeypatch.setattr(nar.urllib.request, "urlopen", fake_urlopen)


def test_muse_sends_minimal_reasoning_effort_by_default(monkeypatch):
    seen = {}
    _serve(monkeypatch, {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}, seen)
    MuseClient(env=ENV).complete([{"role": "user", "content": "x"}], seed=1)
    assert seen["body"]["reasoning_effort"] == "minimal"


def test_muse_reasoning_effort_configurable_and_omittable(monkeypatch):
    seen = {}
    _serve(monkeypatch, {"choices": [{"message": {"content": "hi"}}]}, seen)
    MuseClient(env=ENV, reasoning_effort="low").complete([{"role": "user", "content": "x"}], seed=1)
    assert seen["body"]["reasoning_effort"] == "low"
    MuseClient(env={**ENV, "MUSE_REASONING_EFFORT": "medium"}).complete([{"role": "user", "content": "x"}], seed=1)
    assert seen["body"]["reasoning_effort"] == "medium"
    MuseClient(env=ENV, reasoning_effort=None).complete([{"role": "user", "content": "x"}], seed=1)
    assert "reasoning_effort" not in seen["body"]


@pytest.mark.parametrize("content", [None, ""])
def test_muse_truncated_reasoning_is_typed(monkeypatch, content):
    usage = {"prompt_tokens": 40, "completion_tokens": 256, "completion_tokens_details": {"reasoning_tokens": 254}}
    _serve(monkeypatch, {"choices": [{"message": {"content": content}, "finish_reason": "length"}], "usage": usage})
    c = MuseClient(env=ENV)
    with pytest.raises(nar.TruncatedResponseError) as ei:
        c.complete([{"role": "user", "content": "x"}], seed=1)
    assert ei.value.max_tokens == 256
    assert c.last_usage == {"prompt_tokens": 40, "completion_tokens": 256}


def test_muse_records_usage(monkeypatch):
    _serve(monkeypatch, {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 30}})
    c = MuseClient(env=ENV)
    assert c.complete([{"role": "user", "content": "x"}], seed=1) == "hi"
    assert c.last_usage == {"prompt_tokens": 7, "completion_tokens": 30}
    _serve(monkeypatch, {"choices": [{"message": {"content": "hi"}}]})
    c.complete([{"role": "user", "content": "x"}], seed=1)
    assert c.last_usage is None
