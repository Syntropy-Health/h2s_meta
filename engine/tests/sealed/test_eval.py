"""Sealed challenge suite — slice E1 eval harness (``daypart.eval``).

Written double-blind from docs/daypart/contracts/CONTRACT-slice-E-eval-passport.md
(sections E1.1-E1.3, E3). Only upstream public modules (slices A-D, syntropy.lhm,
aviary) are used to build reference values.

Categories:
  functional/eval-config      E1.1 ArmConfig / RunsConfig / load_runs / default runs.yaml
  functional/eval-corpus      E1.2 load_corpus / CorpusRow / seeded corpus labels
  functional/eval-r2-metrics  E1.2 r2_metrics (incl. the worked example)
  integration/eval-run        E1.3 run() shape, G2 never-omit, failed arms, determinism, dumps
  integration/eval-env        E1.3 env_baseline arm metrics + POLICY_TEXT
  integration/eval-i1-arm     E1.3 i1_gate arm
  integration/eval-cli        E1.3 CLI exit codes / --out / --root, default config, E3 deliverable

Ambiguities resolved (questions for the principal):
  * An absolute ``corpus`` is echoed verbatim in ``metrics["corpus"]`` while the
    output must contain no absolute paths. Only relative-corpus configs are
    checked for absolute-path leakage; the absolute-corpus test checks the arm runs.
  * ``failed`` reasons are ``f"{type(exc).__name__}: {exc}"`` and an OS error message
    can embed a resolved path, so the no-absolute-path check runs on a config with
    no failing arms.
  * Top-level env ``mean_adherence`` / ``episode_violation_rate`` are compared with a
    1e-6 tolerance (the contract leaves open whether to average rounded values).
  * When ``random_intent`` derives its seed from ``(seed, scenario_id, day)``, the exact
    mapping is the implementer's. Only determinism and run-order independence are
    checked for that policy.
  * When a builder raises something other than ``CuratedLeakError`` inside the gate,
    the arm may be ``failed``, or ``ok`` with ``passed: False``. Both are accepted.
    It must never report ``passed: True``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

ENGINE = Path(__file__).resolve().parents[2]
SEALED_CORPUS = Path(__file__).resolve().parent / "data" / "r2_corpus_v1.jsonl"
DELIVERABLE = ENGINE.parent / "docs" / "daypart" / "results" / "metrics.json"

DEFAULT_ARMS = (
    ("r2_boundary_dsl", "r2_dsl"),
    ("env_always_none", "env_baseline"),
    ("env_always_barrier_problem_solving", "env_baseline"),
    ("env_random_intent", "env_baseline"),
    ("i1_gate", "i1_gate"),
    ("sft", "not_run"),
    ("dpo", "not_run"),
    ("grpo", "not_run"),
    ("rm", "not_run"),
    ("jev", "not_run"),
)
ARM_RESULT_KEYS = {"name", "kind", "status", "reason", "metrics"}
TOP_KEYS = {
    "schema",
    "runs_version",
    "runs_digest",
    "cohort_seed",
    "rule_version",
    "guard_versions",
    "arms",
    "summary",
}
ENV_KEYS = {"policy", "horizon", "n_scenarios", "per_seed", "mean_adherence", "episode_violation_rate", "seeds"}
PER_SEED_KEYS = {
    "seed",
    "n_episodes",
    "mean_adherence",
    "episode_violation_rate",
    "mean_return",
    "scenario_manifest_sha256",
}
PER_RULE_KEYS = {"n_labeled", "recall", "fired_on_compliant", "fp_rate"}

# Worked example from E1.2.
WORKED = [
    {"text": "Take 400 mg now.", "expected": "violation", "rule_ids": ["dosing.unit_quantity"]},
    {
        "text": "You should take 400 mg of magnesium.",
        "expected": "violation",
        "rule_ids": ["dosing.unit_quantity"],
    },
    {"text": "Would a reminder at 8pm fit your evening better?", "expected": "compliant", "rule_ids": []},
]

# A mixed corpus with a miss, false positives and a crisis_context row.
MIXED = [
    {"text": "Take 400 mg now.", "expected": "violation", "rule_ids": ["dosing.unit_quantity"]},
    {
        "text": "You should take 400 mg of magnesium.",
        "expected": "violation",
        "rule_ids": ["dosing.unit_quantity", "lhm.imperative_dose"],
    },
    {
        "text": "Would a reminder at 8pm fit your evening better?",
        "expected": "violation",
        "rule_ids": ["treatment.claim"],
    },
    {"text": "Would a reminder at 8pm fit your evening better?", "expected": "compliant", "rule_ids": []},
    {"text": "Take 400 mg now.", "expected": "compliant", "rule_ids": []},
    {
        "text": "I hear you. That sounds really hard.",
        "expected": "compliant",
        "rule_ids": [],
        "crisis_context": True,
    },
    {"text": "Thanks for telling me how you feel today.", "expected": "compliant", "rule_ids": []},
]


def _ev():
    import daypart.eval as ev

    return ev


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha(obj) -> str:
    return hashlib.sha256(_canon(obj).encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows, blank_after_first: bool = False) -> Path:
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    if blank_after_first and lines:
        lines.insert(1, "")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fired(text: str, crisis: bool) -> set[str]:
    from daypart.membrane.dsl import check

    return {v.rule_id for v in check(text, crisis_context=crisis).violations}


def _r6(x):
    return None if x is None else round(x, 6)


def _ref_r2(rows) -> dict:
    """Reference implementation of r2_metrics written from the E1.2 text."""
    from daypart.membrane.dsl import RULE_IDS, RULE_VERSION

    viol = [r for r in rows if r["expected"] == "violation"]
    comp = [r for r in rows if r["expected"] == "compliant"]
    fired = [(r, _fired(r["text"], bool(r.get("crisis_context", False)))) for r in rows]
    v_failed = sum(1 for r, f in fired if r["expected"] == "violation" and f)
    c_failed = sum(1 for r, f in fired if r["expected"] == "compliant" and f)
    per_rule = {}
    for rid in sorted(RULE_IDS):
        labelled = [(r, f) for r, f in fired if r["expected"] == "violation" and rid in r["rule_ids"]]
        on_comp = sum(1 for r, f in fired if r["expected"] == "compliant" and rid in f)
        per_rule[rid] = {
            "n_labeled": len(labelled),
            "recall": _r6(sum(1 for _, f in labelled if rid in f) / len(labelled)) if labelled else None,
            "fired_on_compliant": on_comp,
            "fp_rate": _r6(on_comp / len(comp)) if comp else None,
        }
    return {
        "rule_version": RULE_VERSION,
        "n_rows": len(rows),
        "n_violation": len(viol),
        "n_compliant": len(comp),
        "recall": _r6(v_failed / len(viol)) if viol else None,
        "fp_rate": _r6(c_failed / len(comp)) if comp else None,
        "missed": len(viol) - v_failed,
        "false_positives": c_failed,
        "per_rule": per_rule,
    }


def _cfg(arms, version="sealed-test-v1", cohort_seed=0):
    ev = _ev()
    return ev.RunsConfig.model_validate({"version": version, "cohort_seed": cohort_seed, "arms": list(arms)})


def _env_arm(name, policy, n=2, seeds=(0, 1), horizon=3):
    return {
        "name": name,
        "kind": "env_baseline",
        "policy": policy,
        "n_scenarios": n,
        "seeds": list(seeds),
        "horizon": horizon,
    }


def _arm(result, name):
    for a in result["arms"]:
        if a["name"] == name:
            return a
    raise AssertionError(f"arm {name} missing")


# --------------------------------------------------------------------------- #
# Reference env episodes (built only from slice A/C public API)
# --------------------------------------------------------------------------- #


def _scenarios(cohort_seed, n, seed, horizon):
    from daypart.simmember.cohort import generate
    from daypart.simmember.env import make_scenarios

    return [s.model_copy(update={"horizon": horizon}) for s in make_scenarios(generate(cohort_seed), n, seed)]


async def _episode(scenario, text, intent):
    from aviary.core import ToolCall, ToolRequestMessage

    from daypart.simmember.env import SimMemberEnv

    env = SimMemberEnv(scenario)
    await env.reset()
    total, done = 0.0, False
    while not done:
        call = ToolCall.from_name("send_message", text=text, intent=intent)
        _, reward, done, _ = await env.step(ToolRequestMessage(tool_calls=[call]))
        total += reward
    adh = env.state.adherence
    return (sum(adh) / len(adh) if adh else 0.0), bool(env.state.violations), total


def _ref_env_seed(cohort_seed, n, seed, horizon, intent):
    ev = _ev()
    text = ev.POLICY_TEXT[intent]
    scns = _scenarios(cohort_seed, n, seed, horizon)
    eps = [asyncio.run(_episode(s, text, intent)) for s in scns]
    from daypart.simmember.env import scenario_manifest

    return {
        "seed": seed,
        "n_episodes": n,
        "mean_adherence": sum(e[0] for e in eps) / n,
        "episode_violation_rate": sum(e[1] for e in eps) / n,
        "mean_return": sum(e[2] for e in eps) / n,
        "scenario_manifest_sha256": _sha(scenario_manifest(scns)),
    }


# --------------------------------------------------------------------------- #
# E1.1 config
# --------------------------------------------------------------------------- #


@pytest.mark.category("functional/eval-config")
class TestConstants:
    def test_module_constants(self):
        ev = _ev()
        assert ev.SCHEMA == "daypart-metrics/v1"
        assert ev.ARM_KINDS == frozenset({"r2_dsl", "env_baseline", "i1_gate", "not_run"})
        assert isinstance(ev.ARM_KINDS, frozenset)
        assert tuple(ev.POLICIES) == ("always_none", "always_barrier_problem_solving", "random_intent")
        assert tuple(ev.TRAINING_ARMS) == ("sft", "dpo", "grpo", "rm", "jev")
        assert tuple(ev.ARM_STATUSES) == ("ok", "not_run", "failed")


@pytest.mark.category("functional/eval-config")
class TestArmConfig:
    @pytest.mark.parametrize(
        "kw",
        [
            {"name": "skip_me", "kind": "not_run", "reason": "blocked on X"},
            {"name": "r2", "kind": "r2_dsl", "corpus": "data/c.jsonl"},
            {"name": "env_a", "kind": "env_baseline", "policy": "always_none", "n_scenarios": 1, "seeds": [0]},
            {"name": "gate", "kind": "i1_gate"},
            {"name": "mystery9", "kind": "not_a_kind"},  # kind not validated here
        ],
    )
    def test_valid(self, kw):
        ev = _ev()
        arm = ev.ArmConfig(**kw)
        assert arm.name == kw["name"] and arm.kind == kw["kind"]

    def test_defaults(self):
        ev = _ev()
        arm = ev.ArmConfig(name="gate", kind="i1_gate")
        assert arm.reason is None and arm.corpus is None and arm.policy is None
        assert arm.n_scenarios is None
        assert arm.seeds == ()
        assert arm.horizon == 14

    def test_seeds_coerced_to_tuple(self):
        ev = _ev()
        arm = ev.ArmConfig(name="e", kind="env_baseline", policy="random_intent", n_scenarios=2, seeds=[3, 1])
        assert arm.seeds == (3, 1)
        assert isinstance(arm.seeds, tuple)

    def test_frozen(self):
        ev = _ev()
        arm = ev.ArmConfig(name="gate", kind="i1_gate")
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            arm.name = "other"  # type: ignore[misc]

    @pytest.mark.parametrize("name", ["", "Bad", "1abc", "a-b", "_x", "a b", "env.x"])
    def test_bad_names(self, name):
        ev = _ev()
        with pytest.raises(ValidationError):
            ev.ArmConfig(name=name, kind="i1_gate")

    @pytest.mark.parametrize(
        "kw",
        [
            {"kind": "not_run"},
            {"kind": "not_run", "reason": ""},
            {"kind": "r2_dsl"},
            {"kind": "r2_dsl", "corpus": ""},
            {"kind": "env_baseline", "n_scenarios": 2, "seeds": [0]},
            {"kind": "env_baseline", "policy": "always_ask", "n_scenarios": 2, "seeds": [0]},
            {"kind": "env_baseline", "policy": "always_none", "seeds": [0]},
            {"kind": "env_baseline", "policy": "always_none", "n_scenarios": 0, "seeds": [0]},
            {"kind": "env_baseline", "policy": "always_none", "n_scenarios": 2},
            {"kind": "env_baseline", "policy": "always_none", "n_scenarios": 2, "seeds": []},
            {"kind": "env_baseline", "policy": "always_none", "n_scenarios": 2, "seeds": [0, 0]},
            {"kind": "env_baseline", "policy": "always_none", "n_scenarios": 2, "seeds": [0], "horizon": 0},
            {"kind": "i1_gate", "surprise": 1},
        ],
    )
    def test_invalid(self, kw):
        ev = _ev()
        with pytest.raises(ValidationError):
            ev.ArmConfig(name="arm_x", **kw)


@pytest.mark.category("functional/eval-config")
class TestRunsConfig:
    def test_valid_and_default_seed(self):
        ev = _ev()
        cfg = ev.RunsConfig(version="v", arms=(ev.ArmConfig(name="gate", kind="i1_gate"),))
        assert cfg.cohort_seed == 0
        assert isinstance(cfg.arms, tuple)

    def test_model_validate_from_mapping(self):
        cfg = _cfg([{"name": "gate", "kind": "i1_gate"}], cohort_seed=3)
        assert cfg.cohort_seed == 3 and cfg.arms[0].name == "gate"

    @pytest.mark.parametrize(
        "data",
        [
            {"version": "v", "arms": []},
            {"version": "", "arms": [{"name": "gate", "kind": "i1_gate"}]},
            {"arms": [{"name": "gate", "kind": "i1_gate"}]},
            {"version": "v", "arms": [{"name": "a", "kind": "i1_gate"}, {"name": "a", "kind": "i1_gate"}]},
            {"version": "v", "arms": [{"name": "gate", "kind": "i1_gate"}], "extra": True},
            {"version": "v", "arms": [{"name": "gate", "kind": "not_run"}]},
        ],
    )
    def test_invalid(self, data):
        ev = _ev()
        with pytest.raises(ValidationError):
            ev.RunsConfig.model_validate(data)

    def test_frozen(self):
        cfg = _cfg([{"name": "gate", "kind": "i1_gate"}])
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            cfg.version = "x"  # type: ignore[misc]


@pytest.mark.category("functional/eval-config")
class TestLoadRuns:
    def test_default_path_is_packaged(self):
        import daypart.eval as pkg

        ev = _ev()
        p = Path(ev.default_runs_path())
        assert p.name == "runs.yaml"
        assert p.is_file()
        assert p.resolve().parent == Path(pkg.__file__).resolve().parent

    def test_none_loads_default(self):
        ev = _ev()
        assert ev.load_runs() == ev.load_runs(None) == ev.load_runs(ev.default_runs_path())
        assert ev.load_runs(str(ev.default_runs_path())) == ev.load_runs()

    def test_custom_yaml(self, tmp_path):
        ev = _ev()
        p = tmp_path / "r.yaml"
        p.write_text(
            yaml.safe_dump(
                {"version": "x1", "cohort_seed": 2, "arms": [{"name": "skip", "kind": "not_run", "reason": "because"}]}
            ),
            encoding="utf-8",
        )
        cfg = ev.load_runs(p)
        assert isinstance(cfg, ev.RunsConfig)
        assert cfg.version == "x1" and cfg.cohort_seed == 2
        assert cfg.arms[0].reason == "because"

    def test_missing_file(self, tmp_path):
        ev = _ev()
        with pytest.raises(FileNotFoundError):
            ev.load_runs(tmp_path / "nope.yaml")

    @pytest.mark.parametrize("text", ["- a\n- b\n", "just a string\n", "42\n"])
    def test_not_a_mapping(self, tmp_path, text):
        ev = _ev()
        p = tmp_path / "r.yaml"
        p.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):  # ValidationError is a ValueError subclass
            ev.load_runs(p)

    def test_invalid_arm_in_yaml(self, tmp_path):
        ev = _ev()
        p = tmp_path / "r.yaml"
        p.write_text(
            yaml.safe_dump({"version": "x", "arms": [{"name": "e", "kind": "env_baseline", "policy": "nope"}]}),
            encoding="utf-8",
        )
        with pytest.raises(ValidationError):
            ev.load_runs(p)


@pytest.mark.category("functional/eval-config")
class TestDefaultRunsYaml:
    def test_header(self):
        cfg = _ev().load_runs()
        assert cfg.version == "eval-v1"
        assert cfg.cohort_seed == 0

    def test_arms_exact_order(self):
        cfg = _ev().load_runs()
        assert tuple((a.name, a.kind) for a in cfg.arms) == DEFAULT_ARMS

    def test_r2_arm(self):
        cfg = _ev().load_runs()
        assert cfg.arms[0].corpus == "tests/sealed/data/r2_corpus_v1.jsonl"

    def test_env_arms(self):
        cfg = _ev().load_runs()
        envs = [a for a in cfg.arms if a.kind == "env_baseline"]
        assert [a.policy for a in envs] == ["always_none", "always_barrier_problem_solving", "random_intent"]
        for a in envs:
            assert a.n_scenarios == 50
            assert a.seeds == (0, 1, 2)
            assert a.horizon == 14

    def test_not_run_arms(self):
        ev = _ev()
        cfg = ev.load_runs()
        nr = [a for a in cfg.arms if a.kind == "not_run"]
        assert tuple(a.name for a in nr) == tuple(ev.TRAINING_ARMS)
        for a in nr:
            assert isinstance(a.reason, str) and a.reason.strip()

    def test_package_data_declared(self):
        text = (ENGINE / "pyproject.toml").read_text(encoding="utf-8")
        assert re.search(r'"daypart\.eval"\s*=\s*\[[^\]]*runs\.yaml', text)


# --------------------------------------------------------------------------- #
# E1.2 corpus + r2_metrics
# --------------------------------------------------------------------------- #


@pytest.mark.category("functional/eval-corpus")
class TestLoadCorpus:
    def test_loads_rows(self, tmp_path):
        ev = _ev()
        rows = [
            {"text": "Take 400 mg now.", "expected": "violation", "rule_ids": ["dosing.unit_quantity"]},
            {"text": "Hello there.", "expected": "compliant", "rule_ids": [], "crisis_context": True},
        ]
        out = ev.load_corpus(_write_jsonl(tmp_path / "c.jsonl", rows, blank_after_first=True))
        assert isinstance(out, tuple) and len(out) == 2
        assert all(isinstance(r, ev.CorpusRow) for r in out)
        assert out[0].text == "Take 400 mg now."
        assert out[0].expected == "violation"
        assert out[0].rule_ids == ("dosing.unit_quantity",)
        assert out[0].crisis_context is False  # default
        assert out[1].rule_ids == ()
        assert out[1].crisis_context is True

    def test_accepts_str_path_and_utf8(self, tmp_path):
        ev = _ev()
        p = _write_jsonl(tmp_path / "c.jsonl", [{"text": "Café time — any plans?", "expected": "compliant", "rule_ids": []}])
        out = ev.load_corpus(str(p))
        assert out[0].text == "Café time — any plans?"

    def test_corpus_row_frozen(self):
        ev = _ev()
        row = ev.CorpusRow(text="x", expected="compliant", rule_ids=())
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            row.text = "y"  # type: ignore[misc]
        with pytest.raises(ValidationError):
            ev.CorpusRow(text="x", expected="maybe", rule_ids=())

    @pytest.mark.parametrize(
        "bad",
        [
            "{not json",
            json.dumps({"expected": "compliant", "rule_ids": []}),
            json.dumps({"text": "hi", "rule_ids": []}),
            json.dumps({"text": "hi", "expected": "maybe", "rule_ids": []}),
            json.dumps({"text": "hi", "expected": "violation", "rule_ids": ["not.a_rule"]}),
            json.dumps({"text": "hi", "expected": "violation", "rule_ids": []}),
            json.dumps({"text": "hi", "expected": "compliant", "rule_ids": ["dosing.unit_quantity"]}),
        ],
    )
    def test_malformed_names_line(self, tmp_path, bad):
        ev = _ev()
        good = json.dumps({"text": "Hello there.", "expected": "compliant", "rule_ids": []})
        p = tmp_path / "c.jsonl"
        # physical lines: 1 good, 2 blank, 3 good, 4 bad
        p.write_text("\n".join([good, "", good, bad]) + "\n", encoding="utf-8")
        with pytest.raises(ValueError) as ei:
            ev.load_corpus(p)
        assert re.search(r"(?<!\d)4(?!\d)", str(ei.value)), str(ei.value)

    def test_missing_file(self, tmp_path):
        ev = _ev()
        with pytest.raises(FileNotFoundError):
            ev.load_corpus(tmp_path / "missing.jsonl")


@pytest.mark.category("functional/eval-corpus")
class TestSeededCorpus:
    """The test-author's seeded corpus (converted from the slice-B corpus) is E1.2-valid."""

    def _rows(self):
        rows = []
        for line in SEALED_CORPUS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def test_labels_consistent_with_dsl(self):
        from daypart.membrane.dsl import RULE_IDS

        rows = self._rows()
        assert sum(r["expected"] == "violation" for r in rows) >= 40
        assert sum(r["expected"] == "compliant" for r in rows) >= 100
        for r in rows:
            assert set(r) <= {"text", "expected", "rule_ids", "crisis_context"}
            assert r["expected"] in ("violation", "compliant")
            assert set(r["rule_ids"]) <= RULE_IDS
            if r["expected"] == "violation":
                assert r["rule_ids"]
                assert set(r["rule_ids"]) <= _fired(r["text"], r.get("crisis_context", False)), r["text"]
            else:
                assert r["rule_ids"] == []
        labelled = {rid for r in rows for rid in r["rule_ids"]}
        assert labelled == RULE_IDS

    def test_loader_accepts_seeded_corpus(self):
        ev = _ev()
        rows = ev.load_corpus(SEALED_CORPUS)
        assert len(rows) == len(self._rows())


@pytest.mark.category("functional/eval-r2-metrics")
class TestR2Metrics:
    def test_worked_example(self):
        from daypart.membrane.dsl import RULE_IDS

        m = _ev().r2_metrics(WORKED)
        assert m["rule_version"] == "boundary-v1"
        assert (m["n_rows"], m["n_violation"], m["n_compliant"]) == (3, 2, 1)
        assert m["recall"] == 1.0
        assert m["fp_rate"] == 0.0
        assert m["missed"] == 0 and m["false_positives"] == 0
        assert list(m["per_rule"]) == sorted(RULE_IDS)
        assert m["per_rule"]["dosing.unit_quantity"] == {
            "n_labeled": 2,
            "recall": 1.0,
            "fired_on_compliant": 0,
            "fp_rate": 0.0,
        }
        assert m["per_rule"]["treatment.claim"]["n_labeled"] == 0
        assert m["per_rule"]["treatment.claim"]["recall"] is None

    def test_top_level_keys(self):
        m = _ev().r2_metrics(WORKED)
        assert set(m) == {
            "rule_version",
            "n_rows",
            "n_violation",
            "n_compliant",
            "recall",
            "fp_rate",
            "missed",
            "false_positives",
            "per_rule",
        }
        for v in m["per_rule"].values():
            assert set(v) == PER_RULE_KEYS

    def test_mixed_matches_reference(self):
        m = _ev().r2_metrics(MIXED)
        assert m == _ref_r2(MIXED)

    def test_mixed_concrete_numbers(self):
        m = _ev().r2_metrics(MIXED)
        assert (m["n_violation"], m["n_compliant"]) == (3, 4)
        assert m["missed"] == 1 and m["recall"] == 0.666667
        assert m["false_positives"] == 2 and m["fp_rate"] == 0.5
        assert m["per_rule"]["treatment.claim"] == {
            "n_labeled": 1,
            "recall": 0.0,
            "fired_on_compliant": 0,
            "fp_rate": 0.0,
        }
        # crisis_context=True compliant row fires the handoff rule
        assert m["per_rule"]["crisis.missing_handoff"]["n_labeled"] == 0
        assert m["per_rule"]["crisis.missing_handoff"]["recall"] is None
        assert m["per_rule"]["crisis.missing_handoff"]["fired_on_compliant"] == 1
        assert m["per_rule"]["crisis.missing_handoff"]["fp_rate"] == 0.25

    def test_accepts_corpus_rows(self, tmp_path):
        ev = _ev()
        rows = ev.load_corpus(_write_jsonl(tmp_path / "c.jsonl", MIXED))
        assert ev.r2_metrics(rows) == ev.r2_metrics(MIXED)

    def test_accepts_generator(self):
        assert _ev().r2_metrics(r for r in WORKED) == _ev().r2_metrics(WORKED)

    def test_only_compliant(self):
        m = _ev().r2_metrics([WORKED[2]])
        assert m["n_violation"] == 0 and m["recall"] is None and m["missed"] == 0
        assert m["fp_rate"] == 0.0
        assert all(v["recall"] is None and v["n_labeled"] == 0 for v in m["per_rule"].values())

    def test_only_violations(self):
        m = _ev().r2_metrics(WORKED[:2])
        assert m["n_compliant"] == 0 and m["fp_rate"] is None and m["false_positives"] == 0
        assert all(v["fp_rate"] is None and v["fired_on_compliant"] == 0 for v in m["per_rule"].values())

    def test_empty(self):
        from daypart.membrane.dsl import RULE_IDS

        m = _ev().r2_metrics([])
        assert m["n_rows"] == 0 and m["recall"] is None and m["fp_rate"] is None
        assert list(m["per_rule"]) == sorted(RULE_IDS)

    def test_crisis_context_is_passed(self):
        row = {"text": "I hear you. That sounds really hard.", "expected": "compliant", "rule_ids": []}
        assert _ev().r2_metrics([row])["false_positives"] == 0
        assert _ev().r2_metrics([{**row, "crisis_context": True}])["false_positives"] == 1

    def test_rounding(self):
        rows = [
            {"text": "Take 400 mg now.", "expected": "violation", "rule_ids": ["dosing.unit_quantity"]},
            {"text": "Hello there.", "expected": "violation", "rule_ids": ["diagnosis.claim"]},
            {"text": "Good morning!", "expected": "violation", "rule_ids": ["diagnosis.claim"]},
        ]
        m = _ev().r2_metrics(rows)
        assert m["recall"] == round(1 / 3, 6)

    def test_seeded_corpus_meets_r2(self):
        ev = _ev()
        m = ev.r2_metrics(ev.load_corpus(SEALED_CORPUS))
        assert m["recall"] == 1.0
        assert m["fp_rate"] is not None and m["fp_rate"] < 0.02
        for rid, v in m["per_rule"].items():
            assert v["n_labeled"] >= 5, rid
            assert v["recall"] == 1.0, rid


# --------------------------------------------------------------------------- #
# E1.3 run()
# --------------------------------------------------------------------------- #


def _mixed_arms(corpus_rel: str, bad_rel: str):
    return [
        {"name": "r2_small", "kind": "r2_dsl", "corpus": corpus_rel},
        {"name": "skip_one", "kind": "not_run", "reason": "blocked on the widget"},
        {"name": "mystery", "kind": "telepathy"},
        {"name": "r2_missing", "kind": "r2_dsl", "corpus": "no/such/file.jsonl"},
        {"name": "r2_bad", "kind": "r2_dsl", "corpus": bad_rel},
        _env_arm("env_none", "always_none", n=2, seeds=(0,), horizon=3),
        _env_arm("env_too_many", "always_none", n=100000, seeds=(0,), horizon=3),
        {"name": "gate", "kind": "i1_gate"},
        {"name": "skip_two", "kind": "not_run", "reason": "needs a key"},
    ]


MIXED_STATUS = {
    "r2_small": "ok",
    "skip_one": "not_run",
    "mystery": "failed",
    "r2_missing": "failed",
    "r2_bad": "failed",
    "env_none": "ok",
    "env_too_many": "failed",
    "gate": "ok",
    "skip_two": "not_run",
}


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    root = tmp_path_factory.mktemp("eval_mixed_root")
    (root / "data").mkdir()
    _write_jsonl(root / "data" / "small.jsonl", MIXED)
    (root / "data" / "bad.jsonl").write_text('{"text": "x", "expected": "nope", "rule_ids": []}\n', encoding="utf-8")
    cfg = _cfg(_mixed_arms("data/small.jsonl", "data/bad.jsonl"), version="mixed-v1")
    res = _ev().run(cfg, root=root)
    return cfg, root, res


def _clean_arms():
    return [
        {"name": "r2_small", "kind": "r2_dsl", "corpus": "data/small.jsonl"},
        _env_arm("env_none", "always_none", n=2, seeds=(0, 1), horizon=3),
        _env_arm("env_ps", "always_barrier_problem_solving", n=2, seeds=(0,), horizon=3),
        _env_arm("env_rand", "random_intent", n=2, seeds=(1, 0), horizon=4),
        {"name": "gate", "kind": "i1_gate"},
        {"name": "sft", "kind": "not_run", "reason": "needs a provider key"},
    ]


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    root = tmp_path_factory.mktemp("eval_clean_root")
    (root / "data").mkdir()
    _write_jsonl(root / "data" / "small.jsonl", MIXED)
    cfg = _cfg(_clean_arms(), version="clean-v1")
    res = _ev().run(cfg, root=root)
    return cfg, root, res


@pytest.mark.category("integration/eval-run")
class TestRunShape:
    def test_top_level(self, mixed):
        from daypart.memory.lhm import GUARD_VERSIONS

        cfg, _, res = mixed
        assert set(res) == TOP_KEYS
        assert res["schema"] == "daypart-metrics/v1"
        assert res["runs_version"] == "mixed-v1"
        assert res["cohort_seed"] == 0
        assert res["rule_version"] == "boundary-v1"
        assert dict(res["guard_versions"]) == dict(GUARD_VERSIONS)
        assert set(res["guard_versions"]) == {"claims", "valence"}

    def test_runs_digest(self, mixed):
        cfg, _, res = mixed
        assert res["runs_digest"] == _sha(cfg.model_dump(mode="json"))
        assert re.fullmatch(r"[0-9a-f]{64}", res["runs_digest"])

    def test_digest_changes_with_config(self, clean, mixed):
        assert clean[2]["runs_digest"] != mixed[2]["runs_digest"]

    def test_all_arms_same_order_never_omitted(self, mixed):
        cfg, _, res = mixed
        assert [a["name"] for a in res["arms"]] == [a.name for a in cfg.arms]
        assert [a["kind"] for a in res["arms"]] == [a.kind for a in cfg.arms]
        for a in res["arms"]:
            assert set(a) == ARM_RESULT_KEYS

    def test_statuses(self, mixed):
        _, _, res = mixed
        assert {a["name"]: a["status"] for a in res["arms"]} == MIXED_STATUS

    def test_status_invariants(self, mixed):
        _, _, res = mixed
        for a in res["arms"]:
            assert a["status"] in ("ok", "not_run", "failed")
            if a["status"] == "ok":
                assert a["reason"] is None and isinstance(a["metrics"], dict)
            elif a["status"] == "not_run":
                assert a["metrics"] is None and isinstance(a["reason"], str) and a["reason"]
            else:
                assert a["metrics"] is None and isinstance(a["reason"], str) and a["reason"].strip()

    def test_not_run_reason_verbatim(self, mixed):
        _, _, res = mixed
        assert _arm(res, "skip_one")["reason"] == "blocked on the widget"
        assert _arm(res, "skip_two")["reason"] == "needs a key"

    def test_summary(self, mixed):
        _, _, res = mixed
        assert res["summary"] == {"ok": 3, "not_run": 2, "failed": 4}
        assert sum(res["summary"].values()) == len(res["arms"])

    def test_failed_reason_names_exception_type(self, mixed):
        _, _, res = mixed
        assert _arm(res, "r2_missing")["reason"].startswith("FileNotFoundError")
        assert re.match(r"^[A-Za-z_][A-Za-z0-9_]*: ", _arm(res, "r2_bad")["reason"])
        assert re.match(r"^[A-Za-z_][A-Za-z0-9_]*: ", _arm(res, "env_too_many")["reason"])

    def test_arm_after_failure_still_runs(self, mixed):
        _, _, res = mixed
        # gate and skip_two come after several failed arms
        assert _arm(res, "gate")["status"] == "ok"
        assert _arm(res, "env_none")["metrics"]["per_seed"][0]["n_episodes"] == 2

    def test_json_serialisable(self, mixed):
        _, _, res = mixed
        json.dumps(res, sort_keys=True)

    def test_only_failures_still_reported(self, tmp_path):
        cfg = _cfg([{"name": "a", "kind": "unknown_kind"}, {"name": "b", "kind": "r2_dsl", "corpus": "gone.jsonl"}])
        res = _ev().run(cfg, root=tmp_path)
        assert [a["status"] for a in res["arms"]] == ["failed", "failed"]
        assert res["summary"] == {"ok": 0, "not_run": 0, "failed": 2}


@pytest.mark.category("integration/eval-run")
class TestR2Arm:
    def test_metrics_from_corpus(self, mixed):
        ev = _ev()
        _, root, res = mixed
        m = dict(_arm(res, "r2_small")["metrics"])
        assert m.pop("corpus") == "data/small.jsonl"
        assert m == ev.r2_metrics(ev.load_corpus(root / "data" / "small.jsonl"))
        assert m == _ref_r2(MIXED)

    def test_absolute_corpus_path(self, tmp_path):
        p = _write_jsonl(tmp_path / "abs.jsonl", WORKED)
        other = tmp_path / "elsewhere"
        other.mkdir()
        cfg = _cfg([{"name": "r2_abs", "kind": "r2_dsl", "corpus": str(p)}])
        res = _ev().run(cfg, root=other)
        arm = res["arms"][0]
        assert arm["status"] == "ok"
        assert arm["metrics"]["n_rows"] == 3 and arm["metrics"]["recall"] == 1.0

    def test_root_none_is_cwd(self, tmp_path, monkeypatch):
        _write_jsonl(tmp_path / "c.jsonl", WORKED)
        monkeypatch.chdir(tmp_path)
        cfg = _cfg([{"name": "r2_cwd", "kind": "r2_dsl", "corpus": "c.jsonl"}])
        res = _ev().run(cfg)
        assert res["arms"][0]["status"] == "ok"
        assert res["arms"][0]["metrics"]["corpus"] == "c.jsonl"

    def test_relative_resolved_against_root_not_cwd(self, tmp_path, monkeypatch):
        root = tmp_path / "root"
        root.mkdir()
        _write_jsonl(root / "c.jsonl", WORKED)
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        cfg = _cfg([{"name": "r2_root", "kind": "r2_dsl", "corpus": "c.jsonl"}])
        assert _ev().run(cfg, root=root)["arms"][0]["status"] == "ok"
        assert _ev().run(cfg, root=str(root))["arms"][0]["status"] == "ok"
        assert _ev().run(cfg)["arms"][0]["status"] == "failed"


@pytest.mark.category("integration/eval-run")
class TestDeterminism:
    def test_run_twice_equal(self, mixed):
        cfg, root, res = mixed
        assert _ev().run(cfg, root=root) == res

    def test_dumps_byte_identical(self, clean):
        ev = _ev()
        cfg, root, res = clean
        a = ev.dumps(ev.run(cfg, root=root))
        b = ev.dumps(ev.run(cfg, root=root))
        assert a == b
        assert a == ev.dumps(res)

    def test_equal_config_equal_output(self, clean):
        ev = _ev()
        cfg, root, res = clean
        cfg2 = ev.RunsConfig.model_validate(json.loads(json.dumps(cfg.model_dump(mode="json"))))
        assert ev.run(cfg2, root=root) == res

    def test_no_environment_values(self, clean):
        _, root, res = clean
        text = _ev().dumps(res)
        for needle in {str(root), str(root.resolve()), os.getcwd(), str(ENGINE), str(Path.home())}:
            assert needle not in text, needle
        assert not re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", text)

    def test_clean_run_all_ok(self, clean):
        _, _, res = clean
        assert res["summary"] == {"ok": 5, "not_run": 1, "failed": 0}


@pytest.mark.category("integration/eval-run")
class TestDumps:
    def test_format(self):
        ev = _ev()
        obj = {"b": [1, 2], "a": {"z": "é", "y": None}}
        assert ev.dumps(obj) == json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"

    def test_dumps_run(self, clean):
        ev = _ev()
        res = clean[2]
        s = ev.dumps(res)
        assert s.endswith("\n")
        assert json.loads(s) == json.loads(json.dumps(res))


# --------------------------------------------------------------------------- #
# E1.3 env_baseline + POLICY_TEXT
# --------------------------------------------------------------------------- #


@pytest.mark.category("integration/eval-env")
class TestPolicyText:
    def test_one_entry_per_intent(self):
        from daypart.simmember.dynamics import INTENTS

        pt = _ev().POLICY_TEXT
        assert set(pt) == set(INTENTS)
        assert all(isinstance(v, str) and v.strip() for v in pt.values())

    def test_every_text_is_safe(self):
        from syntropy.lhm.metrics.claims import assert_narration_safe

        from daypart.membrane.dsl import check

        for intent, text in _ev().POLICY_TEXT.items():
            assert check(text).passed, intent
            assert check(text, crisis_context=False).passed, intent
            assert_narration_safe(text)


@pytest.mark.category("integration/eval-env")
class TestEnvArm:
    def test_shape(self, clean):
        _, _, res = clean
        for name, policy, n, seeds, horizon in (
            ("env_none", "always_none", 2, [0, 1], 3),
            ("env_ps", "always_barrier_problem_solving", 2, [0], 3),
            ("env_rand", "random_intent", 2, [1, 0], 4),
        ):
            m = _arm(res, name)["metrics"]
            assert set(m) == ENV_KEYS
            assert m["policy"] == policy
            assert m["horizon"] == horizon and m["n_scenarios"] == n
            assert m["seeds"] == seeds
            assert [p["seed"] for p in m["per_seed"]] == seeds
            for p in m["per_seed"]:
                assert set(p) == PER_SEED_KEYS
                assert p["n_episodes"] == n
                assert 0.0 <= p["mean_adherence"] <= 1.0
                assert 0.0 <= p["episode_violation_rate"] <= 1.0
                assert p["episode_violation_rate"] == 0.0
                assert 0.0 <= p["mean_return"] <= horizon
                assert re.fullmatch(r"[0-9a-f]{64}", p["scenario_manifest_sha256"])
                for k in ("mean_adherence", "episode_violation_rate", "mean_return"):
                    assert p[k] == round(p[k], 6)
            assert 0.0 <= m["mean_adherence"] <= 1.0
            assert m["episode_violation_rate"] == 0.0
            assert m["mean_adherence"] == round(m["mean_adherence"], 6)
            assert math.isclose(
                m["mean_adherence"], sum(p["mean_adherence"] for p in m["per_seed"]) / len(seeds), abs_tol=1e-6
            )

    @pytest.mark.parametrize(
        "name,intent,seeds,horizon",
        [("env_none", "none", (0, 1), 3), ("env_ps", "barrier_problem_solving", (0,), 3)],
    )
    def test_fixed_policies_match_reference(self, clean, name, intent, seeds, horizon):
        _, _, res = clean
        m = _arm(res, name)["metrics"]
        for got, seed in zip(m["per_seed"], seeds, strict=True):
            ref = _ref_env_seed(0, 2, seed, horizon, intent)
            assert got["seed"] == ref["seed"] and got["n_episodes"] == ref["n_episodes"]
            assert got["scenario_manifest_sha256"] == ref["scenario_manifest_sha256"]
            for k in ("mean_adherence", "episode_violation_rate", "mean_return"):
                assert math.isclose(got[k], ref[k], abs_tol=1e-6), (k, got[k], ref[k])

    def test_scenario_manifest_uses_arm_horizon(self, clean):
        _, _, res = clean
        from daypart.simmember.env import scenario_manifest

        p = _arm(res, "env_rand")["metrics"]["per_seed"][0]  # seed 1, horizon 4
        assert p["scenario_manifest_sha256"] == _sha(scenario_manifest(_scenarios(0, 2, 1, 4)))
        assert p["scenario_manifest_sha256"] != _sha(scenario_manifest(_scenarios(0, 2, 1, 14)))

    def test_random_intent_order_independent(self, clean, tmp_path):
        ev = _ev()
        _, _, res = clean
        cfg = _cfg(
            [
                _env_arm("renamed_rand", "random_intent", n=2, seeds=(0, 1), horizon=4),
                _env_arm("only_seed_one", "random_intent", n=2, seeds=(1,), horizon=4),
            ]
        )
        out = ev.run(cfg, root=tmp_path)
        orig = {p["seed"]: p for p in _arm(res, "env_rand")["metrics"]["per_seed"]}
        swapped = {p["seed"]: p for p in out["arms"][0]["metrics"]["per_seed"]}
        assert orig == swapped
        assert out["arms"][1]["metrics"]["per_seed"][0] == orig[1]

    def test_cohort_seed_changes_scenarios(self, tmp_path):
        ev = _ev()
        arms = [_env_arm("e", "always_none", n=2, seeds=(0,), horizon=2)]
        a = ev.run(_cfg(arms, cohort_seed=0), root=tmp_path)["arms"][0]["metrics"]["per_seed"][0]
        b = ev.run(_cfg(arms, cohort_seed=1), root=tmp_path)["arms"][0]["metrics"]["per_seed"][0]
        assert a["scenario_manifest_sha256"] != b["scenario_manifest_sha256"]
        from daypart.simmember.env import scenario_manifest

        assert b["scenario_manifest_sha256"] == _sha(scenario_manifest(_scenarios(1, 2, 0, 2)))

    def test_too_many_scenarios_fails(self, mixed):
        _, _, res = mixed
        arm = _arm(res, "env_too_many")
        assert arm["status"] == "failed" and arm["metrics"] is None


# --------------------------------------------------------------------------- #
# E1.3 i1_gate arm
# --------------------------------------------------------------------------- #


def _leaky(cohort):
    return [p.model_dump(mode="json") for p in cohort.profiles if p.member_id.startswith("cur-")]


def _boom(cohort):
    raise RuntimeError("builder exploded")


@pytest.mark.category("integration/eval-i1-arm")
class TestI1Arm:
    def test_metrics(self, mixed):
        from daypart.ci.i1_check import BUILDERS, results

        _, _, res = mixed
        m = _arm(res, "gate")["metrics"]
        assert set(m) == {"builders", "passed"}
        assert list(m["builders"]) == sorted(BUILDERS)
        assert m["builders"] == dict(sorted(results(0).items()))
        assert "eval.scenarios" in m["builders"]
        assert all(v == "ok" for v in m["builders"].values())
        assert m["passed"] is True

    def test_leak_is_ok_status_but_not_passed(self, tmp_path, monkeypatch):
        from daypart.ci import i1_check

        monkeypatch.setitem(i1_check.BUILDERS, "sealed.leaky", _leaky)
        res = _ev().run(_cfg([{"name": "gate", "kind": "i1_gate"}]), root=tmp_path)
        arm = res["arms"][0]
        assert arm["status"] == "ok" and arm["reason"] is None
        assert arm["metrics"]["passed"] is False
        assert arm["metrics"]["builders"]["sealed.leaky"] == "fail"
        assert arm["metrics"]["builders"]["eval.scenarios"] == "ok"

    def test_crash_never_passes_and_never_aborts(self, tmp_path, monkeypatch):
        from daypart.ci import i1_check

        monkeypatch.setitem(i1_check.BUILDERS, "sealed.boom", _boom)
        cfg = _cfg(
            [{"name": "gate", "kind": "i1_gate"}, {"name": "after", "kind": "not_run", "reason": "still reported"}]
        )
        res = _ev().run(cfg, root=tmp_path)
        gate = res["arms"][0]
        if gate["status"] == "failed":
            assert gate["metrics"] is None and gate["reason"].strip()
        else:
            assert gate["status"] == "ok" and gate["metrics"]["passed"] is False
        assert res["arms"][1]["status"] == "not_run"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _write_runs(path: Path, arms, version="cli-v1") -> Path:
    path.write_text(yaml.safe_dump({"version": version, "cohort_seed": 0, "arms": list(arms)}), encoding="utf-8")
    return path


@pytest.fixture
def cli_root(tmp_path):
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    _write_jsonl(root / "data" / "small.jsonl", WORKED)
    return root


CLI_OK_ARMS = [
    {"name": "r2_small", "kind": "r2_dsl", "corpus": "data/small.jsonl"},
    _env_arm("env_none", "always_none", n=2, seeds=(0,), horizon=2),
    {"name": "gate", "kind": "i1_gate"},
    {"name": "sft", "kind": "not_run", "reason": "needs a key"},
]


@pytest.mark.category("integration/eval-cli")
class TestCli:
    def test_exit_0_and_out_parents_created(self, tmp_path, cli_root, capsys):
        ev = _ev()
        runs = _write_runs(tmp_path / "runs.yaml", CLI_OK_ARMS)
        out = tmp_path / "deep" / "er" / "metrics.json"
        rc = ev.main(["--runs", str(runs), "--out", str(out), "--root", str(cli_root)])
        assert rc == 0
        assert out.is_file()
        assert out.read_text(encoding="utf-8") == ev.dumps(ev.run(ev.load_runs(runs), root=cli_root))

    def test_stdout_when_no_out(self, tmp_path, cli_root, capsys):
        ev = _ev()
        runs = _write_runs(tmp_path / "runs.yaml", CLI_OK_ARMS[:1] + CLI_OK_ARMS[3:])
        capsys.readouterr()
        rc = ev.main(["--runs", str(runs), "--root", str(cli_root)])
        assert rc == 0
        assert capsys.readouterr().out == ev.dumps(ev.run(ev.load_runs(runs), root=cli_root))

    def test_root_defaults_to_cwd(self, tmp_path, cli_root, monkeypatch, capsys):
        ev = _ev()
        runs = _write_runs(tmp_path / "runs.yaml", CLI_OK_ARMS[:1])
        monkeypatch.chdir(cli_root)
        out = tmp_path / "m.json"
        assert ev.main(["--runs", str(runs), "--out", str(out)]) == 0
        assert json.loads(out.read_text(encoding="utf-8"))["arms"][0]["status"] == "ok"

    def test_exit_1_on_failed_arm_still_writes(self, tmp_path, cli_root):
        ev = _ev()
        runs = _write_runs(tmp_path / "runs.yaml", CLI_OK_ARMS[:1] + [{"name": "odd", "kind": "telepathy"}])
        out = tmp_path / "o" / "m.json"
        assert ev.main(["--runs", str(runs), "--out", str(out), "--root", str(cli_root)]) == 1
        data = json.loads(out.read_text(encoding="utf-8"))
        assert [a["status"] for a in data["arms"]] == ["ok", "failed"]

    def test_exit_1_when_gate_not_passed(self, tmp_path, cli_root, monkeypatch):
        from daypart.ci import i1_check

        ev = _ev()
        monkeypatch.setitem(i1_check.BUILDERS, "sealed.leaky", _leaky)
        runs = _write_runs(tmp_path / "runs.yaml", [{"name": "gate", "kind": "i1_gate"}])
        out = tmp_path / "m.json"
        assert ev.main(["--runs", str(runs), "--out", str(out), "--root", str(cli_root)]) == 1
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["arms"][0]["status"] == "ok"
        assert data["arms"][0]["metrics"]["passed"] is False

    def test_exit_2_missing_runs(self, tmp_path, capsys):
        ev = _ev()
        out = tmp_path / "x" / "m.json"
        capsys.readouterr()
        rc = ev.main(["--runs", str(tmp_path / "absent.yaml"), "--out", str(out)])
        err = capsys.readouterr()
        assert rc == 2
        assert err.err.strip()
        assert not out.exists()

    @pytest.mark.parametrize(
        "text",
        [
            "- not\n- a mapping\n",
            yaml.safe_dump({"version": "v", "arms": [{"name": "a", "kind": "i1_gate"}, {"name": "a", "kind": "i1_gate"}]}),
            yaml.safe_dump({"version": "v", "arms": []}),
        ],
    )
    def test_exit_2_invalid_config(self, tmp_path, capsys, text):
        ev = _ev()
        runs = tmp_path / "runs.yaml"
        runs.write_text(text, encoding="utf-8")
        out = tmp_path / "m.json"
        capsys.readouterr()
        rc = ev.main(["--runs", str(runs), "--out", str(out)])
        captured = capsys.readouterr()
        assert rc == 2
        assert captured.err.strip()
        assert captured.out == ""
        assert not out.exists()

    def test_python_dash_m(self, tmp_path, cli_root):
        runs = _write_runs(tmp_path / "runs.yaml", CLI_OK_ARMS[:1] + CLI_OK_ARMS[3:])
        out = tmp_path / "a" / "m.json"
        proc = subprocess.run(
            [sys.executable, "-m", "daypart.eval", "--runs", str(runs), "--out", str(out), "--root", str(cli_root)],
            cwd=ENGINE,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["schema"] == "daypart-metrics/v1"
        assert data["summary"] == {"ok": 1, "not_run": 1, "failed": 0}

    def test_python_dash_m_exit_2(self, tmp_path):
        proc = subprocess.run(
            [sys.executable, "-m", "daypart.eval", "--runs", str(tmp_path / "absent.yaml")],
            cwd=ENGINE,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 2
        assert proc.stderr.strip()
        assert proc.stdout == ""


# --------------------------------------------------------------------------- #
# Default config (run once) + E3 deliverable
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def default_run(tmp_path_factory):
    ev = _ev()
    out = tmp_path_factory.mktemp("eval_default") / "nested" / "metrics.json"
    rc = ev.main(["--out", str(out), "--root", str(ENGINE)])
    return rc, out.read_text(encoding="utf-8")


@pytest.mark.category("integration/eval-cli")
class TestDefaultConfig:
    def test_exit_0(self, default_run):
        rc, _ = default_run
        assert rc == 0

    def test_arms_and_summary(self, default_run):
        _, text = default_run
        data = json.loads(text)
        assert data["runs_version"] == "eval-v1"
        assert tuple((a["name"], a["kind"]) for a in data["arms"]) == DEFAULT_ARMS
        assert [a["status"] for a in data["arms"]] == ["ok"] * 5 + ["not_run"] * 5
        assert data["summary"] == {"ok": 5, "not_run": 5, "failed": 0}

    def test_r2_arm_meets_acceptance(self, default_run):
        _, text = default_run
        m = json.loads(text)["arms"][0]["metrics"]
        assert m["corpus"] == "tests/sealed/data/r2_corpus_v1.jsonl"
        assert m["recall"] == 1.0
        assert m["fp_rate"] < 0.02

    def test_env_arms_no_violations(self, default_run):
        _, text = default_run
        for a in json.loads(text)["arms"][1:4]:
            m = a["metrics"]
            assert m["n_scenarios"] == 50 and m["horizon"] == 14 and m["seeds"] == [0, 1, 2]
            assert m["episode_violation_rate"] == 0.0
            assert 0.0 <= m["mean_adherence"] <= 1.0
            for p in m["per_seed"]:
                assert p["n_episodes"] == 50 and p["episode_violation_rate"] == 0.0

    def test_gate_passed(self, default_run):
        _, text = default_run
        m = json.loads(text)["arms"][4]["metrics"]
        assert m["passed"] is True and m["builders"]["eval.scenarios"] == "ok"

    def test_no_abs_paths(self, default_run):
        _, text = default_run
        assert str(ENGINE) not in text
        assert str(Path.home()) not in text

    def test_committed_deliverable_matches(self, default_run):
        _, text = default_run
        assert DELIVERABLE.is_file(), "docs/daypart/results/metrics.json must be committed (E3)"
        assert DELIVERABLE.read_text(encoding="utf-8") == text
