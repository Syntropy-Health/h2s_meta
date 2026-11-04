"""Sealed slice F — F6 Modal job specs, JobRunner orchestration, plan_pipeline.

Contract: docs/daypart/contracts/CONTRACT-slice-F-train.md (F6, plus the
"no modal import" and "no new dependency" rules in the header).
Categories: functional/jobspec, functional/jobspec-builders, functional/job-runner,
integration/no-modal-import.

Ambiguities (questions for the principal, NOT asserted here):
- Whether ``call_ids`` includes anything for an attempt whose submit raised is
  unspecified beyond "a tuple of call ids"; we assert it holds exactly the ids
  returned by successful submits.
- Exact sleep count on a poll-limit attempt is unspecified; only the interval value
  is asserted there.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pydantic
import pytest

from daypart.simmember.manifest import canonical_json
from daypart.train.modal_jobs import (
    JobResult,
    JobRunner,
    JobSpec,
    LoraConfig,
    PollResult,
    dpo_spec,
    plan_pipeline,
    sft_spec,
)

from .conftest_train import Attempt, FakeBackend, RecordingSleep

ENGINE = Path(__file__).resolve().parents[2]
SHA = "0123456789abcdef" * 4


def _sft(**kw):
    base = dict(kind="sft", base_model="org/glimmer-1b", dataset_path="datasets/sft.jsonl", dataset_sha256=SHA)
    return JobSpec(**{**base, **kw})


def _dpo(**kw):
    base = dict(
        kind="dpo",
        base_model="org/glimmer-1b",
        dataset_path="datasets/dpo.jsonl",
        dataset_sha256=SHA,
        beta=0.1,
        init_adapter="adapters/sft-1",
    )
    return JobSpec(**{**base, **kw})


@pytest.fixture
def dataset(tmp_path):
    p = tmp_path / "sft-data.jsonl"
    p.write_bytes(b'{"messages":[]}\n{"messages":[1]}\n')
    return p


@pytest.fixture
def no_real_sleep(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real time.sleep called")

    monkeypatch.setattr(time, "sleep", boom)


# =========================================================================== #
# functional/jobspec
# =========================================================================== #


class TestLoraConfig:
    def test_defaults(self):
        c = LoraConfig()
        assert c.rank == 8
        assert c.alpha == 16
        assert c.dropout == 0.05
        assert c.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")

    def test_extra_forbidden(self):
        with pytest.raises(pydantic.ValidationError):
            LoraConfig(rank=8, bogus=1)


class TestJobSpec:
    def test_sft_defaults(self):
        s = _sft()
        assert s.lora == LoraConfig()
        assert (s.epochs, s.learning_rate, s.batch_size, s.max_seq_len, s.seed) == (2, 2e-4, 8, 2048, 0)
        assert s.beta is None and s.init_adapter is None
        assert s.gpu == "H100"
        assert s.timeout_s == 14400

    def test_dpo_valid(self):
        s = _dpo()
        assert s.kind == "dpo" and s.beta == 0.1 and s.init_adapter == "adapters/sft-1"

    @pytest.mark.parametrize(
        "kw",
        [
            {"beta": 0.1},
            {"init_adapter": "adapters/x"},
            {"kind": "ppo"},
            {"base_model": ""},
            {"dataset_sha256": SHA.upper()},
            {"dataset_sha256": SHA[:-1]},
            {"dataset_sha256": SHA + "0"},
            {"dataset_sha256": "g" * 64},
            {"epochs": 0},
            {"learning_rate": 0.0},
            {"learning_rate": -1e-4},
            {"batch_size": 0},
            {"max_seq_len": 0},
            {"timeout_s": 59},
            {"unknown_key": 1},
        ],
    )
    def test_sft_violations(self, kw):
        with pytest.raises(pydantic.ValidationError):
            _sft(**kw)

    @pytest.mark.parametrize(
        "kw",
        [
            {"beta": None},
            {"beta": 0.0},
            {"beta": -0.1},
            {"init_adapter": None},
            {"init_adapter": ""},
        ],
    )
    def test_dpo_violations(self, kw):
        with pytest.raises(pydantic.ValidationError):
            _dpo(**kw)

    def test_bounds_inclusive(self):
        s = _sft(epochs=1, batch_size=1, max_seq_len=1, timeout_s=60)
        assert s.timeout_s == 60

    def test_frozen(self):
        s = _sft()
        with pytest.raises((pydantic.ValidationError, TypeError, AttributeError)):
            s.epochs = 5  # type: ignore[misc]

    @pytest.mark.parametrize("make", [_sft, _dpo])
    def test_round_trip(self, make):
        s = make(lora=LoraConfig(rank=16, target_modules=("q_proj",)), seed=3)
        assert JobSpec.model_validate(json.loads(s.model_dump_json())) == s

    @pytest.mark.parametrize("make", [_sft, _dpo])
    def test_hash_name_checkpoint(self, make):
        s = make()
        h = hashlib.sha256(canonical_json(s.model_dump(mode="json")).encode("utf-8")).hexdigest()
        assert s.spec_hash == h
        assert s.name == f"daypart-{s.kind}-{h[:12]}"
        assert s.checkpoint_dir == f"checkpoints/{s.name}"

    def test_hash_sensitive_and_stable(self):
        assert _sft().spec_hash == _sft().spec_hash
        assert _sft().spec_hash != _sft(seed=1).spec_hash
        assert _sft().name != _sft(epochs=3).name


# =========================================================================== #
# functional/jobspec-builders
# =========================================================================== #


class TestSpecBuilders:
    def test_sft_spec(self, dataset):
        s = sft_spec("org/glimmer-1b", dataset)
        assert s.kind == "sft"
        assert s.base_model == "org/glimmer-1b"
        assert s.dataset_sha256 == hashlib.sha256(dataset.read_bytes()).hexdigest()
        assert s.dataset_path == f"datasets/{dataset.name}"
        assert s.lora.rank == 8
        assert s.beta is None and s.init_adapter is None

    def test_sft_spec_str_path_volume_and_overrides(self, dataset):
        s = sft_spec("org/glimmer-1b", str(dataset), volume_path="vol/custom.jsonl", epochs=3, seed=9)
        assert s.dataset_path == "vol/custom.jsonl"
        assert s.epochs == 3 and s.seed == 9
        assert s.lora.rank == 8

    def test_sft_spec_override_violation(self, dataset):
        with pytest.raises(pydantic.ValidationError):
            sft_spec("org/glimmer-1b", dataset, epochs=0)

    def test_sft_spec_hash_tracks_bytes(self, tmp_path):
        a = tmp_path / "d.jsonl"
        a.write_bytes(b"one\n")
        h1 = sft_spec("m", a).dataset_sha256
        a.write_bytes(b"two\n")
        assert sft_spec("m", a).dataset_sha256 != h1

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            sft_spec("m", tmp_path / "missing.jsonl")
        with pytest.raises(FileNotFoundError):
            dpo_spec("m", tmp_path / "missing.jsonl", "adapters/a")

    def test_empty_file(self, tmp_path):
        e = tmp_path / "empty.jsonl"
        e.write_bytes(b"")
        with pytest.raises(ValueError):
            sft_spec("m", e)
        with pytest.raises(ValueError):
            dpo_spec("m", e, "adapters/a")

    def test_dpo_spec(self, dataset):
        s = dpo_spec("org/glimmer-1b", dataset, "adapters/sft-abc")
        assert s.kind == "dpo"
        assert s.init_adapter == "adapters/sft-abc"
        assert s.beta == 0.1
        assert s.dataset_sha256 == hashlib.sha256(dataset.read_bytes()).hexdigest()
        assert s.dataset_path == f"datasets/{dataset.name}"

    def test_dpo_spec_beta_and_volume(self, dataset):
        s = dpo_spec("m", dataset, "adapters/a", beta=0.25, volume_path="v/p.jsonl", learning_rate=1e-5)
        assert s.beta == 0.25 and s.dataset_path == "v/p.jsonl" and s.learning_rate == 1e-5

    @pytest.mark.parametrize("beta", [0.0, -0.5])
    def test_dpo_spec_bad_beta(self, dataset, beta):
        with pytest.raises(ValueError):
            dpo_spec("m", dataset, "adapters/a", beta=beta)

    def test_dpo_spec_requires_adapter(self, dataset):
        with pytest.raises(ValueError):
            dpo_spec("m", dataset, "")


class TestPlanPipeline:
    def test_plan(self, tmp_path):
        sft_d = tmp_path / "sft.jsonl"
        dpo_d = tmp_path / "dpo.jsonl"
        sft_d.write_bytes(b"sft-rows\n")
        dpo_d.write_bytes(b"dpo-rows\n")
        sft, make_dpo = plan_pipeline("org/glimmer-1b", sft_d, dpo_d)
        assert isinstance(sft, JobSpec) and sft.kind == "sft"
        assert sft.base_model == "org/glimmer-1b"
        assert sft.dataset_sha256 == hashlib.sha256(b"sft-rows\n").hexdigest()
        assert sft.lora.rank == 8
        assert callable(make_dpo)
        d = make_dpo("adapters/sft-out")
        assert isinstance(d, JobSpec) and d.kind == "dpo"
        assert d.init_adapter == "adapters/sft-out"
        assert d.base_model == "org/glimmer-1b"
        assert d.dataset_sha256 == hashlib.sha256(b"dpo-rows\n").hexdigest()
        assert d.beta is not None and d.beta > 0
        assert make_dpo("adapters/other").spec_hash != d.spec_hash
        assert make_dpo("adapters/sft-out") == d

    def test_plan_requires_adapter(self, tmp_path):
        a = tmp_path / "a.jsonl"
        a.write_bytes(b"x\n")
        _, make_dpo = plan_pipeline("m", a, a)
        with pytest.raises(ValueError):
            make_dpo("")

    def test_plan_missing_dataset(self, tmp_path):
        a = tmp_path / "a.jsonl"
        a.write_bytes(b"x\n")
        with pytest.raises(FileNotFoundError):
            plan_pipeline("m", tmp_path / "nope.jsonl", a)


# =========================================================================== #
# functional/job-runner
# =========================================================================== #


def R(status, checkpoint=None, adapter=None, error=None):
    return PollResult(status=status, checkpoint=checkpoint, adapter=adapter, error=error)


class TestDataclasses:
    def test_poll_result_defaults_frozen(self):
        p = PollResult("running")
        assert (p.checkpoint, p.adapter, p.error) == (None, None, None)
        assert dataclasses.is_dataclass(p)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.status = "failed"  # type: ignore[misc]

    def test_job_result_frozen(self):
        r = JobResult("failed", None, 1, ("c",), ("e",))
        assert r.status == "failed" and r.attempts == 1
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.attempts = 2  # type: ignore[misc]


@pytest.mark.usefixtures("no_real_sleep")
class TestJobRunner:
    def test_success_first_try(self):
        backend = FakeBackend([Attempt([R("pending"), R("running", checkpoint="ck/1"), R("succeeded", adapter="adapters/a")])])
        sleep = RecordingSleep()
        spec = _sft()
        res = JobRunner(backend, poll_interval_s=5.0, sleep=sleep).run(spec)
        assert isinstance(res, JobResult)
        assert res.status == "succeeded"
        assert res.adapter == "adapters/a"
        assert res.attempts == 1
        assert res.call_ids == ("call-1",)
        assert res.errors == ()
        assert backend.submits == [(spec, None)]
        assert backend.polls == ["call-1"] * 3
        assert sleep.calls == [5.0, 5.0]

    def test_immediate_success_no_sleep(self):
        backend = FakeBackend([Attempt([R("succeeded", adapter="a")])])
        sleep = RecordingSleep()
        res = JobRunner(backend, sleep=sleep).run(_sft())
        assert res.status == "succeeded"
        assert sleep.calls == []

    def test_retry_resumes_from_latest_checkpoint(self):
        backend = FakeBackend(
            [
                Attempt([R("running", checkpoint="ck/1"), R("running", checkpoint="ck/2"), R("failed", error="OOM at step 300")]),
                Attempt([R("running"), R("succeeded", adapter="adapters/final")]),
            ]
        )
        spec = _dpo()
        res = JobRunner(backend, max_retries=2, poll_interval_s=1.0, sleep=RecordingSleep()).run(spec)
        assert res.status == "succeeded"
        assert res.adapter == "adapters/final"
        assert res.attempts == 2
        assert res.call_ids == ("call-1", "call-2")
        assert len(res.errors) == 1
        assert "OOM at step 300" in res.errors[0]
        assert backend.submits == [(spec, None), (spec, "ck/2")]

    def test_checkpoint_carried_across_attempts(self):
        backend = FakeBackend(
            [
                Attempt([R("running", checkpoint="ck/a"), R("failed", error="e1")]),
                Attempt([R("running"), R("failed", error="e2")]),
                Attempt([R("succeeded", adapter="ad")]),
            ]
        )
        res = JobRunner(backend, max_retries=2, sleep=RecordingSleep()).run(_sft())
        assert res.status == "succeeded" and res.attempts == 3
        assert [r for _, r in backend.submits] == [None, "ck/a", "ck/a"]
        assert len(res.errors) == 2

    def test_most_recent_checkpoint_wins(self):
        backend = FakeBackend(
            [
                Attempt([R("running", checkpoint="ck/a"), R("failed", error="e1")]),
                Attempt([R("running", checkpoint="ck/b"), R("failed", checkpoint=None, error="e2")]),
                Attempt([R("succeeded", adapter="ad")]),
            ]
        )
        JobRunner(backend, max_retries=2, sleep=RecordingSleep()).run(_sft())
        assert [r for _, r in backend.submits] == [None, "ck/a", "ck/b"]

    def test_checkpoint_on_failed_poll_counts(self):
        backend = FakeBackend(
            [
                Attempt([R("failed", checkpoint="ck/at-fail", error="preempted")]),
                Attempt([R("succeeded", adapter="ad")]),
            ]
        )
        JobRunner(backend, max_retries=1, sleep=RecordingSleep()).run(_sft())
        assert [r for _, r in backend.submits] == [None, "ck/at-fail"]

    def test_no_checkpoint_resumes_none(self):
        backend = FakeBackend([Attempt([R("failed", error="x")]), Attempt([R("succeeded", adapter="ad")])])
        res = JobRunner(backend, max_retries=1, sleep=RecordingSleep()).run(_sft())
        assert res.status == "succeeded"
        assert [r for _, r in backend.submits] == [None, None]

    def test_max_retries_exhaustion(self):
        backend = FakeBackend([Attempt([R("failed", error=f"err{i}")]) for i in range(3)])
        res = JobRunner(backend, max_retries=2, sleep=RecordingSleep()).run(_sft())
        assert res.status == "failed"
        assert res.adapter is None
        assert res.attempts == 3
        assert len(backend.submits) == 3
        assert res.call_ids == ("call-1", "call-2", "call-3")
        assert len(res.errors) == 3
        for i, e in enumerate(res.errors):
            assert f"err{i}" in e

    def test_default_max_retries_is_two(self):
        backend = FakeBackend([Attempt([R("failed", error="e")]) for _ in range(3)])
        res = JobRunner(backend, sleep=RecordingSleep()).run(_sft())
        assert res.status == "failed" and res.attempts == 3 and len(backend.submits) == 3

    def test_zero_retries(self):
        backend = FakeBackend([Attempt([R("failed", error="e")])])
        res = JobRunner(backend, max_retries=0, sleep=RecordingSleep()).run(_sft())
        assert res.status == "failed" and res.attempts == 1 and len(backend.submits) == 1
        assert len(res.errors) == 1

    def test_submit_exception_is_failed_attempt(self):
        backend = FakeBackend([Attempt(submit_exc=RuntimeError("quota exceeded")), Attempt([R("succeeded", adapter="ad")])])
        res = JobRunner(backend, max_retries=1, sleep=RecordingSleep()).run(_sft())
        assert res.status == "succeeded"
        assert res.attempts == 2
        assert res.errors == ("RuntimeError: quota exceeded",)
        assert res.call_ids == ("call-2",)

    def test_poll_exception_is_failed_attempt(self):
        backend = FakeBackend(
            [
                Attempt([R("running", checkpoint="ck/1"), ConnectionError("lost connection")]),
                Attempt([R("succeeded", adapter="ad")]),
            ]
        )
        res = JobRunner(backend, max_retries=1, sleep=RecordingSleep()).run(_sft())
        assert res.status == "succeeded"
        assert res.errors == ("ConnectionError: lost connection",)
        assert [r for _, r in backend.submits] == [None, "ck/1"]

    def test_all_exceptions_never_raise(self):
        backend = FakeBackend([Attempt(submit_exc=ValueError("bad")), Attempt([KeyError("k")]), Attempt(submit_exc=OSError("io"))])
        res = JobRunner(backend, max_retries=2, sleep=RecordingSleep()).run(_sft())
        assert res.status == "failed"
        assert res.adapter is None
        assert res.attempts == 3
        assert len(res.errors) == 3
        assert res.errors[0] == "ValueError: bad"
        assert res.errors[1].startswith("KeyError: ")
        assert res.errors[2] == "OSError: io"

    def test_poll_limit(self):
        backend = FakeBackend([Attempt([R("running")])])
        sleep = RecordingSleep()
        res = JobRunner(backend, max_retries=0, max_polls=3, poll_interval_s=2.5, sleep=sleep).run(_sft())
        assert res.status == "failed"
        assert len(backend.polls) == 3
        assert len(res.errors) == 1
        assert "poll" in res.errors[0].lower()
        assert sleep.calls and all(s == 2.5 for s in sleep.calls)

    def test_poll_limit_then_retry_with_checkpoint(self):
        backend = FakeBackend([Attempt([R("running", checkpoint="ck/p")]), Attempt([R("succeeded", adapter="ad")])])
        res = JobRunner(backend, max_retries=1, max_polls=2, sleep=RecordingSleep()).run(_sft())
        assert res.status == "succeeded" and res.attempts == 2
        assert [r for _, r in backend.submits] == [None, "ck/p"]
        assert len(backend.polls) == 3

    def test_success_without_adapter_is_failure(self):
        backend = FakeBackend([Attempt([R("succeeded", adapter=None)]), Attempt([R("succeeded", adapter="ad")])])
        res = JobRunner(backend, max_retries=1, sleep=RecordingSleep()).run(_sft())
        assert res.status == "succeeded" and res.adapter == "ad"
        assert res.attempts == 2 and len(res.errors) == 1

    def test_success_without_adapter_exhausts(self):
        backend = FakeBackend([Attempt([R("succeeded")])])
        res = JobRunner(backend, max_retries=0, sleep=RecordingSleep()).run(_sft())
        assert res.status == "failed" and res.adapter is None and len(res.errors) == 1

    def test_sleep_default_interval(self):
        backend = FakeBackend([Attempt([R("running"), R("succeeded", adapter="a")])])
        sleep = RecordingSleep()
        JobRunner(backend, sleep=sleep).run(_sft())
        assert sleep.calls == [30.0]

    @pytest.mark.parametrize(
        "kw", [{"max_retries": -1}, {"max_polls": 0}, {"max_polls": -3}, {"poll_interval_s": -0.1}]
    )
    def test_argument_validation(self, kw):
        with pytest.raises(ValueError):
            JobRunner(FakeBackend([]), sleep=RecordingSleep(), **kw)

    def test_boundary_arguments_ok(self):
        backend = FakeBackend([Attempt([R("running"), R("succeeded", adapter="a")])])
        sleep = RecordingSleep()
        res = JobRunner(backend, max_retries=0, max_polls=2, poll_interval_s=0.0, sleep=sleep).run(_sft())
        assert res.status == "succeeded"
        assert sleep.calls == [0.0]


# =========================================================================== #
# integration/no-modal-import
# =========================================================================== #


_NO_MODAL_SCRIPT = r"""
import sys
sys.modules["modal"] = None  # any `import modal` now raises ImportError
import daypart.train
import daypart.train.modal_jobs as mj
import daypart.train.sft_dataset
import daypart.train.dpo_pairs
import daypart.simmember.narrative
assert hasattr(mj, "JobRunner") and hasattr(mj, "JobSpec")
assert "daypart.train.modal_app" not in sys.modules, "modal_app must not be imported by the package"
print("OK")
"""

_FRESH_SCRIPT = r"""
import sys
import daypart.train
import daypart.train.modal_jobs
import daypart.train.sft_dataset
import daypart.train.dpo_pairs
import daypart.simmember.narrative
import daypart.ci.i1_check
assert sys.modules.get("modal") is None, "modal imported at module import time"
assert "daypart.train.modal_app" not in sys.modules
print("OK")
"""


class TestNoModalImport:
    @pytest.mark.parametrize("script", [_NO_MODAL_SCRIPT, _FRESH_SCRIPT], ids=["modal-blocked", "fresh"])
    def test_import_without_modal(self, script):
        env = {k: v for k, v in os.environ.items() if not k.startswith("MUSE_")}
        proc = subprocess.run(
            [sys.executable, "-c", script], cwd=str(ENGINE), env=env, capture_output=True, text=True, timeout=120
        )
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout

    def test_inprocess_reload_with_modal_blocked(self, monkeypatch):
        import importlib

        import daypart.train.modal_jobs as mj

        monkeypatch.setitem(sys.modules, "modal", None)
        importlib.reload(mj)
        assert mj.JobSpec is not None

    def test_no_modal_dependency_added(self):
        data = tomllib.loads((ENGINE / "pyproject.toml").read_text(encoding="utf-8"))
        deps = list(data.get("project", {}).get("dependencies", []))
        for group in data.get("dependency-groups", {}).values():
            deps.extend(d for d in group if isinstance(d, str))
        for extra in data.get("project", {}).get("optional-dependencies", {}).values():
            deps.extend(extra)
        names = {d.split(";")[0].split("[")[0].split("@")[0].split(">")[0].split("<")[0].split("=")[0].strip().lower() for d in deps}
        assert "modal" not in names
