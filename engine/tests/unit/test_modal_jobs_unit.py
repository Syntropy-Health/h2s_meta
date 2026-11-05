"""Implementer unit tests for daypart.train.modal_jobs (CONTRACT slice F, F6)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pydantic
import pytest

import daypart.train.modal_jobs  # noqa: F401  (names re-bound per test below: other suites may reload this module)


@pytest.fixture(autouse=True)
def _fresh_names():
    import importlib
    mod = importlib.import_module("daypart.train.modal_jobs")
    g = globals()
    for n in ("JobRunner", "JobSpec", "LoraConfig", "PollResult", "dpo_spec", "plan_pipeline", "sft_spec"):
        g[n] = getattr(mod, n)
    yield


@pytest.fixture()
def data(tmp_path):
    p = tmp_path / "sft.jsonl"
    p.write_bytes(b'{"a":1}\n')
    return p


def test_no_modal_import():
    code = "import sys, daypart.train, daypart.train.modal_jobs, daypart.train.sft_dataset, daypart.train.dpo_pairs; print('modal' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_sft_spec_defaults(data):
    s = sft_spec("org/glimmer", data)
    assert s.kind == "sft" and s.lora.rank == 8 and s.lora == LoraConfig()
    assert s.dataset_sha256 == hashlib.sha256(data.read_bytes()).hexdigest()
    assert s.dataset_path == "datasets/sft.jsonl"
    assert sft_spec("org/glimmer", data, volume_path="x/y.jsonl").dataset_path == "x/y.jsonl"
    assert s.name == f"daypart-sft-{s.spec_hash[:12]}" and s.checkpoint_dir == f"checkpoints/{s.name}"
    assert JobSpec.model_validate(json.loads(s.model_dump_json())) == s
    assert sft_spec("org/glimmer", data, epochs=3).spec_hash != s.spec_hash


def test_dpo_spec(data):
    d = dpo_spec("org/glimmer", data, "adapters/x")
    assert d.beta == 0.1 and d.init_adapter == "adapters/x"
    assert JobSpec.model_validate(json.loads(d.model_dump_json())) == d


def test_dataset_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        sft_spec("m", tmp_path / "nope.jsonl")
    (tmp_path / "e.jsonl").write_bytes(b"")
    with pytest.raises(ValueError):
        sft_spec("m", tmp_path / "e.jsonl")


@pytest.mark.parametrize(
    "kw",
    [
        {"kind": "sft", "beta": 0.1},
        {"kind": "sft", "init_adapter": "a"},
        {"kind": "dpo"},
        {"kind": "dpo", "beta": 0.1},
        {"kind": "dpo", "beta": 0, "init_adapter": "a"},
        {"kind": "dpo", "beta": 0.1, "init_adapter": ""},
        {"kind": "sft", "dataset_sha256": "A" * 64},
        {"kind": "sft", "dataset_sha256": "a" * 63},
        {"kind": "sft", "epochs": 0},
        {"kind": "sft", "learning_rate": 0},
        {"kind": "sft", "batch_size": 0},
        {"kind": "sft", "max_seq_len": 0},
        {"kind": "sft", "timeout_s": 59},
        {"kind": "sft", "base_model": ""},
        {"kind": "sft", "bogus": 1},
        {"kind": "rl"},
    ],
)
def test_validation(kw):
    base = {"base_model": "m", "dataset_path": "datasets/x", "dataset_sha256": "a" * 64}
    with pytest.raises(pydantic.ValidationError):
        JobSpec(**{**base, **kw})


class Backend:
    def __init__(self, script):
        self.script = script  # list of lists of PollResult|Exception per attempt; or Exception for submit
        self.submits = []
        self.n = 0

    def submit(self, spec, *, resume_from):
        self.submits.append(resume_from)
        item = self.script[len(self.submits) - 1]
        if isinstance(item, BaseException):
            raise item
        cid = f"call-{len(self.submits)}"
        self._polls = list(item)
        return cid

    def poll(self, call_id):
        r = self._polls.pop(0) if self._polls else PollResult("running")
        if isinstance(r, BaseException):
            raise r
        return r


def _spec():
    return JobSpec(kind="sft", base_model="m", dataset_path="d", dataset_sha256="a" * 64)


def test_runner_success_sleeps_between_polls():
    sleeps = []
    b = Backend([[PollResult("pending"), PollResult("running"), PollResult("succeeded", adapter="ad")]])
    r = JobRunner(b, poll_interval_s=5, sleep=sleeps.append).run(_spec())
    assert (r.status, r.adapter, r.attempts, r.call_ids, r.errors) == ("succeeded", "ad", 1, ("call-1",), ())
    assert sleeps == [5, 5] and b.submits == [None]


def test_runner_retry_from_checkpoint():
    b = Backend([
        [PollResult("running", checkpoint="ck/1"), PollResult("failed", error="oom")],
        RuntimeError("boom"),
        [PollResult("succeeded", adapter="ad")],
    ])
    r = JobRunner(b, max_retries=2, sleep=lambda s: None).run(_spec())
    assert r.status == "succeeded" and r.attempts == 3 and r.call_ids == ("call-1", "call-3")
    assert b.submits == [None, "ck/1", "ck/1"]
    assert r.errors == ("oom", "RuntimeError: boom")


def test_runner_all_fail_and_poll_limit():
    b = Backend([[], [PollResult("succeeded")], [ValueError("x")]])
    r = JobRunner(b, max_retries=2, max_polls=3, sleep=lambda s: None).run(_spec())
    assert r.status == "failed" and r.adapter is None and r.attempts == 3 and len(r.errors) == 3
    assert "poll limit" in r.errors[0] and r.errors[2] == "ValueError: x"
    assert len(b.submits) == 3


def test_runner_bad_args():
    for kw in ({"max_retries": -1}, {"max_polls": 0}, {"poll_interval_s": -1}):
        with pytest.raises(ValueError):
            JobRunner(Backend([]), **kw)


def test_plan_pipeline(tmp_path, data):
    dpo = tmp_path / "dpo.jsonl"
    dpo.write_bytes(b'{"p":1}\n')
    sft, make = plan_pipeline("m", data, dpo)
    assert sft.kind == "sft"
    d = make("adapters/daypart-sft-x")
    assert d.kind == "dpo" and d.init_adapter == "adapters/daypart-sft-x" and d.beta == 0.1
    assert d.dataset_sha256 == hashlib.sha256(dpo.read_bytes()).hexdigest()


def test_modal_app_compiles():
    import py_compile
    from pathlib import Path

    import daypart.train

    py_compile.compile(str(Path(daypart.train.__file__).parent / "modal_app.py"), doraise=True)


# --- HAC-17: LoRA targets hit the Muse Glimmer language tower only ---------------------------

_LANG = "model.language_model.layers.{i}.{blk}.{proj}"
_VISION = "model.vision_tower.layers.{i}.attn.{proj}"


def test_language_lora_targets_hits_language_tower_only():
    import re

    from daypart.train.modal_jobs import language_lora_targets

    rx = re.compile(language_lora_targets(("q_proj", "k_proj", "v_proj", "o_proj")))
    for i in (0, 7, 51):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert rx.fullmatch(_LANG.format(i=i, blk="self_attn", proj=proj))
    for proj in ("q_proj", "k_proj", "v_proj", "proj"):
        assert not rx.fullmatch(_VISION.format(i=3, proj=proj))
    assert not rx.fullmatch(_LANG.format(i=0, blk="self_attn", proj="gate_proj"))  # attention gate untouched
    assert not rx.fullmatch(_LANG.format(i=0, blk="mlp", proj="up_proj"))
    assert not rx.fullmatch("model.language_model.layers.0.self_attn.q_proj.lora_A")
    assert not rx.fullmatch("lm_head")


def test_language_lora_targets_mlp_and_escaping():
    import re

    from daypart.train.modal_jobs import language_lora_targets

    rx = re.compile(language_lora_targets(("up_proj", "down_proj"), prefix="a.b"))
    assert rx.fullmatch("a.b.3.mlp.up_proj")
    assert not rx.fullmatch("aXb.3.mlp.up_proj")  # prefix dots are literal
    assert not rx.fullmatch("model.language_model.layers.3.mlp.up_proj")


def test_language_lora_targets_rejects_bad_names():
    from daypart.train.modal_jobs import language_lora_targets

    for bad in ((), ("",), ("q_proj", "a.b"), ("q_proj|.*",)):
        with pytest.raises(ValueError):
            language_lora_targets(bad)
