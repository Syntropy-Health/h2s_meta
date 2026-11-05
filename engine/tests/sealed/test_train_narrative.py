"""Sealed slice F — F1 text clients, F2 prompts, F3 cache + realization.

Contract: docs/daypart/contracts/CONTRACT-slice-F-train.md (F1-F3).
Categories: functional/narrative-client, functional/narrative-prompt,
functional/narrative-cache, functional/narrative-realize, integration/narrative-i1.

Ambiguities (questions for the principal, NOT asserted here):
- F3 regenerate: contract fixes only the seed (s + 1) of the second call, not
  whether the messages are identical; only the seed is asserted.
- F1 TemplateClient "refusal pool": the selection rule is implementation-defined;
  we assert every output passes ``post`` for journal, dialogue and dosing-request
  messages alike.
"""

from __future__ import annotations

import hashlib
import io
import json
import socket
import urllib.request

import pydantic
import pytest

from daypart.membrane.post import post
from daypart.simmember.cohort import CURATED_PROFILES, Profile
from daypart.simmember.latents import AXES
from daypart.simmember.manifest import CuratedLeakError, canonical_json
from daypart.simmember.narrative import (
    KINDS,
    NARRATIVE_VERSION,
    MissingCredentialsError,
    MuseClient,
    NarrativeCache,
    NarrativePrompt,
    NarrativeRow,
    TemplateClient,
    build_prompt,
    realize,
)

from .conftest_train import (
    FAIL_TEXT,
    PASS_DIALOGUE,
    PASS_JOURNAL,
    ExplodingClient,
    RecordingClient,
    checkin_days,
    cohort0,
    fail_client,
    fail_then_pass_client,
    joined,
    pass_client,
    training,
)

FULL_ENV = {
    "MUSE_API_KEY": "sk-secret-value-123",
    "MUSE_BASE_URL": "https://muse.example/v1",
    "MUSE_SPARK_MODEL": "spark-test-model",
}


@pytest.fixture
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", boom)


def _profile_record(i: int = 0):
    p = training(1, start=i)[0]
    tr = cohort0().trajectories[p.member_id]
    rec = next(r for r in tr.days if r.checkin)
    return p, tr, rec


# =========================================================================== #
# functional/narrative-client — F1
# =========================================================================== #


class TestMuseClientCredentials:
    def test_missing_error_is_runtime_error(self):
        assert issubclass(MissingCredentialsError, RuntimeError)

    def test_empty_env_names_all_missing(self, no_network):
        with pytest.raises(MissingCredentialsError) as ei:
            MuseClient(env={})
        msg = str(ei.value)
        for name in ("MUSE_API_KEY", "MUSE_BASE_URL", "MUSE_SPARK_MODEL"):
            assert name in msg

    @pytest.mark.parametrize("missing", ["MUSE_API_KEY", "MUSE_BASE_URL", "MUSE_SPARK_MODEL"])
    def test_single_missing_named_without_values(self, missing, no_network):
        env = {k: v for k, v in FULL_ENV.items() if k != missing}
        with pytest.raises(MissingCredentialsError) as ei:
            MuseClient(env=env)
        msg = str(ei.value)
        assert missing in msg
        for v in env.values():
            assert v not in msg

    @pytest.mark.parametrize("empty", ["MUSE_API_KEY", "MUSE_BASE_URL", "MUSE_SPARK_MODEL"])
    def test_empty_value_counts_as_missing(self, empty, no_network):
        env = dict(FULL_ENV, **{empty: ""})
        with pytest.raises(MissingCredentialsError) as ei:
            MuseClient(env=env)
        assert empty in str(ei.value)
        assert FULL_ENV["MUSE_API_KEY"] not in str(ei.value)

    def test_explicit_model_id_does_not_need_spark_model(self, no_network):
        env = {k: v for k, v in FULL_ENV.items() if k != "MUSE_SPARK_MODEL"}
        c = MuseClient(model_id="explicit-model", env=env)
        assert c.model_id == "explicit-model"

    def test_model_id_from_env(self, no_network):
        c = MuseClient(env=dict(FULL_ENV))
        assert c.model_id == FULL_ENV["MUSE_SPARK_MODEL"]

    def test_construction_no_network_and_repr_hides_key(self, no_network):
        c = MuseClient(env=dict(FULL_ENV))
        assert FULL_ENV["MUSE_API_KEY"] not in repr(c)
        assert FULL_ENV["MUSE_API_KEY"] not in str(c)

    def test_reads_os_environ_by_default(self, monkeypatch, no_network):
        for k in FULL_ENV:
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(MissingCredentialsError):
            MuseClient()
        for k, v in FULL_ENV.items():
            monkeypatch.setenv(k, v)
        assert MuseClient().model_id == FULL_ENV["MUSE_SPARK_MODEL"]


class _FakeResp(io.BytesIO):
    status = 200

    def getcode(self):
        return 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


class TestMuseClientComplete:
    def test_posts_openai_chat_request(self, monkeypatch):
        seen = []

        def fake_open(self, req, data=None, timeout=None, *a, **k):
            seen.append((req, data))
            body = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "hello there?"}}]}
            return _FakeResp(json.dumps(body).encode())

        monkeypatch.setattr(urllib.request.OpenerDirector, "open", fake_open)
        c = MuseClient(env=dict(FULL_ENV))
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        out = c.complete(msgs, seed=3)
        assert out == "hello there?"
        assert len(seen) == 1
        req, data = seen[0]
        assert isinstance(req, urllib.request.Request)
        assert req.full_url == "https://muse.example/v1/chat/completions"
        assert req.get_method() == "POST"
        assert req.get_header("Authorization") == f"Bearer {FULL_ENV['MUSE_API_KEY']}"
        payload = json.loads((data if data is not None else req.data).decode())
        assert payload["messages"] == msgs
        assert payload["model"] == FULL_ENV["MUSE_SPARK_MODEL"]


class TestTemplateClient:
    def test_default_model_id(self):
        assert TemplateClient().model_id == "template-v1"
        assert TemplateClient("tpl-x").model_id == "tpl-x"

    def test_deterministic_in_messages_and_seed(self, no_network):
        p, _, rec = _profile_record()
        msgs = build_prompt(p, rec, "journal").messages
        a = [TemplateClient().complete(msgs, seed=s) for s in range(20)]
        b = [TemplateClient().complete(msgs, seed=s) for s in range(20)]
        assert a == b
        assert all(isinstance(t, str) and t for t in a)

    def test_pool_has_variety(self, no_network):
        p, _, rec = _profile_record()
        msgs = build_prompt(p, rec, "journal").messages
        outs = {TemplateClient().complete(msgs, seed=s) for s in range(64)}
        assert len(outs) >= 2

    def test_all_outputs_pass_post(self, no_network):
        from daypart.train.sft_dataset import COACH_SYSTEM_PROMPT, REFUSAL_PROMPTS

        tc = TemplateClient()
        msg_sets = []
        for i in range(3):
            p, _, rec = _profile_record(i)
            msg_sets.append(build_prompt(p, rec, "journal").messages)
            msg_sets.append(build_prompt(p, rec, "dialogue", journal=PASS_JOURNAL).messages)
        for rp in REFUSAL_PROMPTS:
            msg_sets.append(
                ({"role": "system", "content": COACH_SYSTEM_PROMPT}, {"role": "user", "content": rp})
            )
        for msgs in msg_sets:
            for s in range(40):
                t = tc.complete(msgs, seed=s)
                assert post(t).decision == "pass", t


# =========================================================================== #
# functional/narrative-prompt — F2
# =========================================================================== #


def _text_has(text: str, level: str) -> bool:
    t = text.casefold()
    return level.casefold() in t or level.replace("_", " ").casefold() in t


def _alt_profile(p: Profile, axis: str) -> Profile:
    curated = {c.latents for c in CURATED_PROFILES}
    cur = getattr(p.latents, axis)
    for lvl in AXES[axis]:
        if lvl == cur:
            continue
        lat = p.latents.model_copy(update={axis: lvl})
        if lat not in curated:
            return Profile(member_id=p.member_id, cohort=p.cohort, latents=lat)
    raise AssertionError("no alternative level")


class TestBuildPrompt:
    def test_constants(self):
        assert NARRATIVE_VERSION == "narrative-v1"
        assert KINDS == ("journal", "dialogue")

    def test_journal_prompt_fields(self):
        p, _, rec = _profile_record()
        pr = build_prompt(p, rec, "journal")
        assert isinstance(pr, NarrativePrompt)
        assert pr.member_id == p.member_id
        assert pr.day == rec.day
        assert pr.kind == "journal"
        assert len(pr.messages) >= 1
        for m in pr.messages:
            assert m["role"] in {"system", "user", "assistant"}
            assert isinstance(m["content"], str)

    def test_latent_conditioned(self):
        for i in range(4):
            p, _, rec = _profile_record(i)
            for kind, kw in (("journal", {}), ("dialogue", {"journal": PASS_JOURNAL})):
                text = joined(build_prompt(p, rec, kind, **kw).messages)
                for axis in ("motivation", "barrier_type", "literacy", "engagement_pattern"):
                    assert _text_has(text, getattr(p.latents, axis)), (axis, kind)
                assert _text_has(text, rec.state)

    @pytest.mark.parametrize("axis", ["motivation", "barrier_type", "literacy", "engagement_pattern"])
    @pytest.mark.parametrize("kind", ["journal", "dialogue"])
    def test_latent_change_changes_hash(self, axis, kind):
        p, _, rec = _profile_record()
        kw = {"journal": PASS_JOURNAL} if kind == "dialogue" else {}
        q = _alt_profile(p, axis)
        assert build_prompt(p, rec, kind, **kw).prompt_hash != build_prompt(q, rec, kind, **kw).prompt_hash

    def test_deterministic(self):
        p, _, rec = _profile_record()
        assert build_prompt(p, rec, "journal") == build_prompt(p, rec, "journal")
        a = build_prompt(p, rec, "dialogue", journal=PASS_JOURNAL)
        assert a == build_prompt(p, rec, "dialogue", journal=PASS_JOURNAL)

    @pytest.mark.parametrize("kind", ["journal", "dialogue"])
    def test_prompt_hash_formula(self, kind):
        p, _, rec = _profile_record()
        kw = {"journal": PASS_JOURNAL} if kind == "dialogue" else {}
        pr = build_prompt(p, rec, kind, **kw)
        expect = hashlib.sha256(
            canonical_json(
                {"version": NARRATIVE_VERSION, "kind": kind, "messages": [dict(m) for m in pr.messages]}
            ).encode("utf-8")
        ).hexdigest()
        assert pr.prompt_hash == expect

    def test_journal_vs_dialogue_hash_differ(self):
        p, _, rec = _profile_record()
        assert (
            build_prompt(p, rec, "journal").prompt_hash
            != build_prompt(p, rec, "dialogue", journal=PASS_JOURNAL).prompt_hash
        )

    def test_dialogue_contains_journal_verbatim(self):
        p, _, rec = _profile_record()
        j = "Rainy morning; I still logged my routine at 7:15 (barely)."
        pr = build_prompt(p, rec, "dialogue", journal=j)
        assert pr.kind == "dialogue"
        assert j in joined(pr.messages)
        other = build_prompt(p, rec, "dialogue", journal=PASS_JOURNAL)
        assert pr.prompt_hash != other.prompt_hash

    def test_journal_kind_rejects_journal_arg(self):
        p, _, rec = _profile_record()
        with pytest.raises(ValueError):
            build_prompt(p, rec, "journal", journal="something")

    @pytest.mark.parametrize("j", [None, ""])
    def test_dialogue_requires_journal(self, j):
        p, _, rec = _profile_record()
        with pytest.raises(ValueError):
            build_prompt(p, rec, "dialogue", journal=j)

    @pytest.mark.parametrize("kind", ["summary", "", "Journal", "refusal"])
    def test_unknown_kind(self, kind):
        p, _, rec = _profile_record()
        with pytest.raises(ValueError):
            build_prompt(p, rec, kind)

    def test_no_member_id_in_messages(self):
        for i in range(5):
            p, tr, _ = _profile_record(i)
            for rec in tr.days[:10]:
                for kind, kw in (("journal", {}), ("dialogue", {"journal": PASS_JOURNAL})):
                    text = joined(build_prompt(p, rec, kind, **kw).messages)
                    assert p.member_id not in text
                    assert "cur-" not in text
                    assert "syn-" not in text

    def test_prompt_model_frozen_and_forbids_extra(self):
        p, _, rec = _profile_record()
        pr = build_prompt(p, rec, "journal")
        with pytest.raises((pydantic.ValidationError, TypeError, AttributeError)):
            pr.day = 99  # type: ignore[misc]
        data = pr.model_dump()
        data["extra_field"] = 1
        with pytest.raises(pydantic.ValidationError):
            NarrativePrompt.model_validate(data)


class TestBuildPromptI1:
    @pytest.mark.parametrize("kind", ["journal", "dialogue"])
    def test_curated_profile_refused(self, kind):
        cur = CURATED_PROFILES[0]
        rec = next(r for r in cohort0().trajectories[cur.member_id].days if r.checkin)
        kw = {"journal": PASS_JOURNAL} if kind == "dialogue" else {}
        with pytest.raises(CuratedLeakError):
            build_prompt(cur, rec, kind, **kw)

    def test_relabeled_curated_latents_refused(self):
        cur = CURATED_PROFILES[3]
        disguised = Profile(member_id="syn-999", cohort="synthetic", latents=cur.latents)
        _, _, rec = _profile_record()
        with pytest.raises(CuratedLeakError):
            build_prompt(disguised, rec, "journal")


# =========================================================================== #
# functional/narrative-cache — F3
# =========================================================================== #


def _row(member="syn-001", day=1, kind="journal", h="a" * 64, decision="pass", text=PASS_JOURNAL, **kw):
    return NarrativeRow(
        member_id=member,
        day=day,
        kind=kind,
        prompt_hash=h,
        model_id=kw.get("model_id", "fake-v1"),
        decision=decision,
        text=text,
        trace=kw.get("trace", {"decision": decision}),
    )


class TestNarrativeRow:
    def test_pass_requires_text(self):
        with pytest.raises(pydantic.ValidationError):
            _row(decision="pass", text=None)

    def test_escalate_forbids_text(self):
        with pytest.raises(pydantic.ValidationError):
            _row(decision="escalate", text="hello")

    def test_escalate_with_none_ok(self):
        r = _row(decision="escalate", text=None)
        assert r.text is None

    def test_extra_forbidden_and_kind_literal(self):
        data = _row().model_dump(mode="json")
        with pytest.raises(pydantic.ValidationError):
            NarrativeRow.model_validate({**data, "extra": 1})
        with pytest.raises(pydantic.ValidationError):
            NarrativeRow.model_validate({**data, "kind": "summary"})
        with pytest.raises(pydantic.ValidationError):
            NarrativeRow.model_validate({**data, "decision": "maybe"})


class TestNarrativeCache:
    def test_missing_file_is_empty(self, tmp_path):
        c = NarrativeCache(tmp_path / "nope.jsonl")
        assert len(c) == 0
        assert c.rows() == []
        assert c.get("syn-001", 1, "a" * 64) is None

    def test_accepts_str_path(self, tmp_path):
        c = NarrativeCache(str(tmp_path / "c.jsonl"))
        c.put(_row())
        assert len(c) == 1

    def test_put_get_and_key(self, tmp_path):
        c = NarrativeCache(tmp_path / "c.jsonl")
        r = _row()
        c.put(r)
        assert c.get("syn-001", 1, "a" * 64) == r
        assert c.get("syn-001", 2, "a" * 64) is None
        assert c.get("syn-002", 1, "a" * 64) is None
        assert c.get("syn-001", 1, "b" * 64) is None
        assert len(c) == 1

    def test_first_write_wins(self, tmp_path):
        path = tmp_path / "c.jsonl"
        c = NarrativeCache(path)
        first = _row(text=PASS_JOURNAL)
        c.put(first)
        before = path.read_bytes()
        c.put(_row(text=PASS_DIALOGUE, model_id="other"))
        assert len(c) == 1
        assert c.get("syn-001", 1, "a" * 64) == first
        assert path.read_bytes() == before
        assert NarrativeCache(path).rows() == [first]

    def test_appends_canonical_lines_in_order(self, tmp_path):
        path = tmp_path / "c.jsonl"
        c = NarrativeCache(path)
        rows = [_row(day=3), _row(day=1, kind="dialogue", text=PASS_DIALOGUE), _row(day=2, decision="escalate", text=None)]
        for r in rows:
            c.put(r)
        assert c.rows() == rows
        raw = path.read_text(encoding="utf-8")
        assert raw.endswith("\n")
        lines = raw.split("\n")[:-1]
        assert len(lines) == 3
        for line, r in zip(lines, rows):
            obj = json.loads(line)
            assert line == canonical_json(obj)
            assert NarrativeRow.model_validate(obj) == r

    def test_reload(self, tmp_path):
        path = tmp_path / "c.jsonl"
        c = NarrativeCache(path)
        rows = [_row(day=d) for d in (5, 2, 9)]
        for r in rows:
            c.put(r)
        c2 = NarrativeCache(path)
        assert c2.rows() == rows
        assert len(c2) == 3
        assert c2.get("syn-001", 2, "a" * 64) == rows[1]
        c2.put(_row(day=2))  # existing key loaded from disk -> no-op
        assert len(NarrativeCache(path)) == 3

    @pytest.mark.parametrize(
        "bad",
        [
            "not json at all\n",
            '{"member_id": "syn-001"}\n',
            "[1, 2, 3]\n",
        ],
    )
    def test_malformed_line_raises(self, tmp_path, bad):
        path = tmp_path / "c.jsonl"
        good = canonical_json(_row().model_dump(mode="json")) + "\n"
        path.write_text(good + bad, encoding="utf-8")
        with pytest.raises(ValueError):
            NarrativeCache(path)

    def test_invalid_pass_row_on_disk_raises(self, tmp_path):
        path = tmp_path / "c.jsonl"
        data = _row().model_dump(mode="json")
        data["text"] = None
        path.write_text(canonical_json(data) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            NarrativeCache(path)


# =========================================================================== #
# functional/narrative-realize — F3
# =========================================================================== #


def _member(i: int = 0):
    p = training(1, start=i)[0]
    return p, cohort0().trajectories[p.member_id]


class TestRealize:
    def test_rows_cover_checkin_days_in_order(self, tmp_path):
        p, tr = _member()
        cache = NarrativeCache(tmp_path / "c.jsonl")
        client = pass_client()
        rows = realize(p, tr, client, cache)
        days = checkin_days(p.member_id)
        assert days, "fixture member must have check-ins"
        expect = [(d, k) for d in days for k in ("journal", "dialogue")]
        assert [(r.day, r.kind) for r in rows] == expect
        assert all(r.member_id == p.member_id for r in rows)
        assert all(r.decision == "pass" for r in rows)
        assert all(r.model_id == client.model_id for r in rows)
        assert client.n_calls == 2 * len(days)
        assert len(cache) == 2 * len(days)
        assert cache.rows() == rows

    def test_prompt_hashes_match_build_prompt(self, tmp_path):
        p, tr = _member()
        rows = realize(p, tr, pass_client(), NarrativeCache(tmp_path / "c.jsonl"))
        by_day = {r.day: r for r in tr.days}
        for j, d in zip(rows[0::2], rows[1::2]):
            rec = by_day[j.day]
            assert j.prompt_hash == build_prompt(p, rec, "journal").prompt_hash
            assert d.prompt_hash == build_prompt(p, rec, "dialogue", journal=j.text).prompt_hash

    def test_client_sees_prompt_messages(self, tmp_path):
        p, tr = _member()
        client = pass_client()
        rows = realize(p, tr, client, NarrativeCache(tmp_path / "c.jsonl"), days=[checkin_days(p.member_id)[0]])
        rec = {r.day: r for r in tr.days}[rows[0].day]
        assert client.calls[0].messages == [dict(m) for m in build_prompt(p, rec, "journal").messages]
        assert client.calls[1].messages == [
            dict(m) for m in build_prompt(p, rec, "dialogue", journal=rows[0].text).messages
        ]
        for call in client.calls:
            assert p.member_id not in joined(call.messages)

    def test_text_and_trace_from_post(self, tmp_path):
        p, tr = _member()
        client = pass_client()
        rows = realize(p, tr, client, NarrativeCache(tmp_path / "c.jsonl"))
        for r in rows:
            assert post(r.text).decision == "pass"
            assert r.trace["decision"] == "pass"
            assert r.trace["attempts"] == 1
            assert r.text not in json.dumps(r.trace)

    def test_days_restriction(self, tmp_path):
        p, tr = _member()
        cdays = checkin_days(p.member_id)
        non_checkin = next(r.day for r in tr.days if not r.checkin)
        pick = [cdays[-1], non_checkin, cdays[0]]
        rows = realize(p, tr, pass_client(), NarrativeCache(tmp_path / "c.jsonl"), days=pick)
        assert [(r.day, r.kind) for r in rows] == [
            (cdays[0], "journal"),
            (cdays[0], "dialogue"),
            (cdays[-1], "journal"),
            (cdays[-1], "dialogue"),
        ]

    def test_cache_hit_zero_calls(self, tmp_path):
        p, tr = _member()
        path = tmp_path / "c.jsonl"
        cache = NarrativeCache(path)
        first = realize(p, tr, pass_client(), cache)
        before = path.read_bytes()
        again = RecordingClient(lambda m, s, n: (_ for _ in ()).throw(AssertionError("called")))
        second = realize(p, tr, again, cache)
        assert again.n_calls == 0
        assert second == first
        third_client = ExplodingClient()
        third = realize(p, tr, third_client, NarrativeCache(path))
        assert third_client.calls == 0
        assert third == first
        assert path.read_bytes() == before

    def test_partial_cache_only_missing_days_called(self, tmp_path):
        p, tr = _member()
        cdays = checkin_days(p.member_id)
        cache = NarrativeCache(tmp_path / "c.jsonl")
        realize(p, tr, pass_client(), cache, days=cdays[:2])
        client = pass_client()
        rows = realize(p, tr, client, cache)
        assert client.n_calls == 2 * (len(cdays) - 2)
        assert len(rows) == 2 * len(cdays)

    def test_byte_identical_file_fresh_cache(self, tmp_path):
        out = []
        for name in ("a.jsonl", "b.jsonl"):
            cache = NarrativeCache(tmp_path / name)
            for p in training(3):
                realize(p, cohort0().trajectories[p.member_id], TemplateClient(), cache)
            out.append((tmp_path / name).read_bytes())
        assert out[0] == out[1]
        assert out[0]

    def test_byte_identical_with_recording_client(self, tmp_path):
        out = []
        for name in ("a.jsonl", "b.jsonl"):
            cache = NarrativeCache(tmp_path / name)
            p, tr = _member(1)
            realize(p, tr, pass_client(), cache)
            out.append((tmp_path / name).read_bytes())
        assert out[0] == out[1]

    def test_seed_depends_only_on_member_day_kind(self, tmp_path):
        p, tr = _member()
        cdays = checkin_days(p.member_id)
        a = pass_client("model-a")
        realize(p, tr, a, NarrativeCache(tmp_path / "a.jsonl"))
        b = pass_client("model-b")
        realize(p, tr, b, NarrativeCache(tmp_path / "b.jsonl"), days=[cdays[2]])
        seeds_a = {(i // 2, i % 2): c.seed for i, c in enumerate(a.calls)}
        idx = cdays.index(cdays[2])
        assert b.calls[0].seed == seeds_a[(idx, 0)]
        assert b.calls[1].seed == seeds_a[(idx, 1)]

    def test_escalation_cached_text_none_no_dialogue(self, tmp_path):
        p, tr = _member()
        cdays = checkin_days(p.member_id)
        path = tmp_path / "c.jsonl"
        cache = NarrativeCache(path)
        client = fail_client()
        rows = realize(p, tr, client, cache)
        assert [(r.day, r.kind) for r in rows] == [(d, "journal") for d in cdays]
        for r in rows:
            assert r.decision == "escalate"
            assert r.text is None
            assert r.trace["decision"] == "escalate"
            assert FAIL_TEXT not in json.dumps(r.trace)
        # draft + one regeneration per day, nothing for dialogue
        assert client.n_calls == 2 * len(cdays)
        assert all(r.kind == "journal" for r in cache.rows())
        assert FAIL_TEXT not in path.read_text(encoding="utf-8")
        # rerun neither re-spends nor changes output
        again = ExplodingClient()
        assert realize(p, tr, again, NarrativeCache(path)) == rows
        assert again.calls == 0

    def test_regenerate_called_with_seed_plus_one(self, tmp_path):
        p, tr = _member()
        day = checkin_days(p.member_id)[0]
        client = fail_then_pass_client()
        rows = realize(p, tr, client, NarrativeCache(tmp_path / "c.jsonl"), days=[day])
        j = rows[0]
        assert j.kind == "journal" and j.decision == "pass"
        assert j.text == PASS_JOURNAL
        assert j.trace["regenerated"] is True
        assert j.trace["attempts"] == 2
        assert client.calls[1].seed == client.calls[0].seed + 1
        # dialogue proceeds after a regenerated-pass journal
        assert [r.kind for r in rows] == ["journal", "dialogue"]
        assert client.n_calls == 3

    def test_dialogue_escalation_keeps_journal(self, tmp_path):
        p, tr = _member()
        day = checkin_days(p.member_id)[0]
        client = RecordingClient(lambda m, s, n: PASS_JOURNAL if n == 0 else FAIL_TEXT)
        rows = realize(p, tr, client, NarrativeCache(tmp_path / "c.jsonl"), days=[day])
        assert [(r.kind, r.decision) for r in rows] == [("journal", "pass"), ("dialogue", "escalate")]
        assert rows[1].text is None
        assert client.n_calls == 3


# =========================================================================== #
# integration/narrative-i1 — F3 I1 + validation
# =========================================================================== #


class TestRealizeI1:
    def test_curated_refused_before_any_call(self, tmp_path):
        cur = CURATED_PROFILES[0]
        path = tmp_path / "c.jsonl"
        cache = NarrativeCache(path)
        client = ExplodingClient()
        with pytest.raises(CuratedLeakError):
            realize(cur, cohort0().trajectories[cur.member_id], client, cache)
        assert client.calls == 0
        assert len(cache) == 0
        assert not path.exists() or path.read_bytes() == b""

    def test_relabeled_curated_refused(self, tmp_path):
        cur = CURATED_PROFILES[1]
        p, tr = _member()
        disguised = Profile(member_id=p.member_id, cohort="synthetic", latents=cur.latents)
        client = ExplodingClient()
        cache = NarrativeCache(tmp_path / "c.jsonl")
        with pytest.raises(CuratedLeakError):
            realize(disguised, tr, client, cache)
        assert client.calls == 0
        assert len(cache) == 0

    def test_member_mismatch(self, tmp_path):
        p, _ = _member(0)
        _, other_tr = _member(1)
        client = ExplodingClient()
        cache = NarrativeCache(tmp_path / "c.jsonl")
        with pytest.raises(ValueError):
            realize(p, other_tr, client, cache)
        assert client.calls == 0
        assert len(cache) == 0

    def test_no_curated_ids_in_cache_file(self, tmp_path):
        path = tmp_path / "c.jsonl"
        cache = NarrativeCache(path)
        for p in training(2):
            realize(p, cohort0().trajectories[p.member_id], TemplateClient(), cache)
        assert "cur-" not in path.read_text(encoding="utf-8")
