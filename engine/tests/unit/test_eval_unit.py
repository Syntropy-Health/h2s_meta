import json

import pydantic
import pytest

from daypart.ci import i1_check
from daypart.eval import (
    POLICIES,
    POLICY_TEXT,
    TRAINING_ARMS,
    ArmConfig,
    CorpusRow,
    RunsConfig,
    default_runs_path,
    dumps,
    load_corpus,
    load_runs,
    main,
    r2_metrics,
    run,
)
from daypart.eval.policies import choose
from daypart.membrane import dsl
from daypart.memory.lhm import GUARD_VERSIONS
from daypart.simmember.dynamics import INTENTS

V1 = {"text": "Take 400 mg now.", "expected": "violation", "rule_ids": ["dosing.unit_quantity"]}
V2 = {"text": "You should take 400 mg of magnesium.", "expected": "violation", "rule_ids": ["dosing.unit_quantity"]}
C1 = {"text": "Would a reminder at 8pm fit your evening better?", "expected": "compliant", "rule_ids": []}


def _write_corpus(path, rows, blank=False):
    lines = [json.dumps(r) for r in rows]
    if blank:
        lines.insert(1, "")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _env_arm(name="env_x", policy="always_none", n=3, seeds=(0, 1), horizon=4):
    return ArmConfig(name=name, kind="env_baseline", policy=policy, n_scenarios=n, seeds=seeds, horizon=horizon)


# --------------------------------------------------------------------------- config


def test_default_runs():
    cfg = load_runs()
    assert default_runs_path().name == "runs.yaml" and default_runs_path().is_file()
    assert cfg == load_runs(default_runs_path())
    assert cfg.version == "eval-v1" and cfg.cohort_seed == 0
    assert [a.name for a in cfg.arms] == [
        "r2_boundary_dsl", "env_always_none", "env_always_barrier_problem_solving",
        "env_random_intent", "i1_gate", "base", "sft", "dpo", "grpo", "rm", "jev",
    ]
    assert cfg.arms[0].corpus == "tests/sealed/data/r2_corpus_v1.jsonl"
    for arm, pol in zip(cfg.arms[1:4], POLICIES):
        assert (arm.kind, arm.policy, arm.n_scenarios, arm.seeds, arm.horizon) == ("env_baseline", pol, 50, (0, 1, 2), 14)
    assert all((a.kind, a.model) == ("boundary_eval", a.name) for a in cfg.arms[5:8])
    assert all(a.kind == "not_run" and a.reason.strip() for a in cfg.arms[8:])


@pytest.mark.parametrize(
    "kw",
    [
        {"name": "", "kind": "i1_gate"},
        {"name": "Bad", "kind": "i1_gate"},
        {"name": "x", "kind": "not_run"},
        {"name": "x", "kind": "not_run", "reason": "  "},
        {"name": "x", "kind": "r2_dsl"},
        {"name": "x", "kind": "env_baseline", "n_scenarios": 1, "seeds": (0,)},
        {"name": "x", "kind": "env_baseline", "policy": "greedy", "n_scenarios": 1, "seeds": (0,)},
        {"name": "x", "kind": "env_baseline", "policy": "always_none", "n_scenarios": 0, "seeds": (0,)},
        {"name": "x", "kind": "env_baseline", "policy": "always_none", "n_scenarios": 1},
        {"name": "x", "kind": "env_baseline", "policy": "always_none", "n_scenarios": 1, "seeds": (0, 0)},
        {"name": "x", "kind": "env_baseline", "policy": "always_none", "n_scenarios": 1, "seeds": (0,), "horizon": 0},
        {"name": "x", "kind": "i1_gate", "bogus": 1},
    ],
)
def test_arm_config_invalid(kw):
    with pytest.raises(pydantic.ValidationError):
        ArmConfig(**kw)


def test_arm_config_unknown_kind_allowed():
    assert ArmConfig(name="x", kind="mystery").kind == "mystery"


def test_runs_config_invalid():
    arm = ArmConfig(name="a", kind="i1_gate")
    with pytest.raises(pydantic.ValidationError):
        RunsConfig(version="v", arms=())
    with pytest.raises(pydantic.ValidationError):
        RunsConfig(version="v", arms=(arm, arm))
    with pytest.raises(pydantic.ValidationError):
        RunsConfig(version="", arms=(arm,))
    with pytest.raises(pydantic.ValidationError):
        RunsConfig.model_validate({"version": "v", "arms": [{"name": "a", "kind": "i1_gate"}], "x": 1})


def test_load_runs_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_runs(tmp_path / "missing.yaml")
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError):
        load_runs(p)


# --------------------------------------------------------------------------- corpus


def test_load_corpus_and_example(tmp_path):
    rows = load_corpus(_write_corpus(tmp_path / "c.jsonl", [V1, V2, C1], blank=True))
    assert len(rows) == 3 and all(isinstance(r, CorpusRow) for r in rows)
    assert rows[0].rule_ids == ("dosing.unit_quantity",) and rows[2].crisis_context is False
    m = r2_metrics(rows)
    assert m == r2_metrics([V1, V2, C1])
    assert m["rule_version"] == "boundary-v1"
    assert (m["n_rows"], m["n_violation"], m["n_compliant"]) == (3, 2, 1)
    assert (m["recall"], m["fp_rate"], m["missed"], m["false_positives"]) == (1.0, 0.0, 0, 0)
    assert list(m["per_rule"]) == sorted(dsl.RULE_IDS)
    assert m["per_rule"]["dosing.unit_quantity"] == {"n_labeled": 2, "recall": 1.0, "fired_on_compliant": 0, "fp_rate": 0.0}
    assert m["per_rule"]["treatment.claim"]["n_labeled"] == 0 and m["per_rule"]["treatment.claim"]["recall"] is None


def test_r2_metrics_empty_sides():
    m = r2_metrics([C1])
    assert m["recall"] is None and m["fp_rate"] == 0.0
    m = r2_metrics([V1])
    assert m["fp_rate"] is None and all(v["fp_rate"] is None for v in m["per_rule"].values())


def test_r2_metrics_missed_and_fp():
    miss = {"text": "Hello there.", "expected": "violation", "rule_ids": ["treatment.claim"]}
    fp = {"text": "Take 400 mg now.", "expected": "compliant", "rule_ids": []}
    m = r2_metrics([miss, fp, C1])
    assert m["missed"] == 1 and m["recall"] == 0.0
    assert m["false_positives"] == 1 and m["fp_rate"] == 0.5
    assert m["per_rule"]["treatment.claim"]["recall"] == 0.0
    assert m["per_rule"]["dosing.unit_quantity"]["fired_on_compliant"] == 1
    assert m["per_rule"]["dosing.unit_quantity"]["fp_rate"] == 0.5


@pytest.mark.parametrize(
    "bad",
    [
        "{not json",
        json.dumps({"expected": "compliant", "rule_ids": []}),
        json.dumps({"text": "x", "rule_ids": []}),
        json.dumps({"text": "x", "expected": "maybe", "rule_ids": []}),
        json.dumps({"text": "x", "expected": "violation", "rule_ids": ["nope.rule"]}),
        json.dumps({"text": "x", "expected": "violation", "rule_ids": []}),
        json.dumps({"text": "x", "expected": "compliant", "rule_ids": ["dosing.unit_quantity"]}),
    ],
)
def test_load_corpus_malformed_names_line(tmp_path, bad):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps(C1) + "\n\n" + bad + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="3"):
        load_corpus(p)


# --------------------------------------------------------------------------- policies


def test_policy_text_safe():
    from syntropy.lhm.metrics.claims import assert_narration_safe

    assert set(POLICY_TEXT) == set(INTENTS)
    for text in POLICY_TEXT.values():
        assert dsl.check(text).passed
        assert_narration_safe(text)


def test_choose():
    assert choose("always_none", 0, "scn-0-syn-001", 0) == (POLICY_TEXT["none"], "none")
    assert choose("always_barrier_problem_solving", 0, "s", 3)[1] == "barrier_problem_solving"
    draws = [choose("random_intent", 1, "scn-1-syn-002", d)[1] for d in range(40)]
    assert draws == [choose("random_intent", 1, "scn-1-syn-002", d)[1] for d in range(40)]
    assert len(set(draws)) > 1 and set(draws) <= set(INTENTS)


# --------------------------------------------------------------------------- run


def _cfg(*arms, version="t1"):
    return RunsConfig(version=version, cohort_seed=0, arms=arms)


def test_run_all_arms_reported(tmp_path):
    _write_corpus(tmp_path / "c.jsonl", [V1, V2, C1])
    cfg = _cfg(
        ArmConfig(name="r2", kind="r2_dsl", corpus="c.jsonl"),
        ArmConfig(name="r2_missing", kind="r2_dsl", corpus="nope.jsonl"),
        _env_arm(),
        _env_arm(name="too_many", n=10_000, seeds=(0,)),
        ArmConfig(name="mystery", kind="mystery"),
        ArmConfig(name="sft", kind="not_run", reason="needs a key"),
    )
    out = run(cfg, root=tmp_path)
    assert out["schema"] == "daypart-metrics/v1" and out["runs_version"] == "t1"
    assert out["guard_versions"] == GUARD_VERSIONS and out["rule_version"] == "boundary-v1"
    assert [a["name"] for a in out["arms"]] == [a.name for a in cfg.arms]
    st = {a["name"]: a for a in out["arms"]}
    assert all(set(a) == {"name", "kind", "status", "reason", "metrics"} for a in out["arms"])
    assert st["r2"]["status"] == "ok" and st["r2"]["metrics"]["corpus"] == "c.jsonl"
    assert st["r2"]["metrics"]["recall"] == 1.0
    for name in ("r2_missing", "too_many", "mystery"):
        assert st[name]["status"] == "failed" and st[name]["reason"] and st[name]["metrics"] is None
    assert str(tmp_path) not in dumps(out)
    assert st["sft"] == {"name": "sft", "kind": "not_run", "status": "not_run", "reason": "needs a key", "metrics": None}
    assert out["summary"] == {"ok": 2, "not_run": 1, "failed": 3}


def test_run_digest_and_determinism(tmp_path):
    import hashlib

    cfg = _cfg(_env_arm(policy="random_intent"), _env_arm(name="b", policy="always_barrier_problem_solving"))
    a, b = run(cfg, root=tmp_path), run(cfg, root=tmp_path)
    assert a == b and dumps(a) == dumps(b) and dumps(a).endswith("\n")
    canon = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    assert a["runs_digest"] == hashlib.sha256(canon.encode()).hexdigest()


def test_env_baseline_metrics(tmp_path):
    out = run(_cfg(_env_arm(n=4, seeds=(2, 0), horizon=5)), root=tmp_path)
    m = out["arms"][0]["metrics"]
    assert (m["policy"], m["horizon"], m["n_scenarios"], m["seeds"]) == ("always_none", 5, 4, [2, 0])
    assert [p["seed"] for p in m["per_seed"]] == [2, 0]
    for p in m["per_seed"]:
        assert p["n_episodes"] == 4 and p["episode_violation_rate"] == 0.0
        assert 0.0 <= p["mean_adherence"] <= 1.0 and 0.0 <= p["mean_return"] <= 5.0
        assert len(p["scenario_manifest_sha256"]) == 64
    assert m["episode_violation_rate"] == 0.0


async def test_run_inside_event_loop(tmp_path):
    out = run(_cfg(_env_arm(n=2, seeds=(0,), horizon=2)), root=tmp_path)
    assert out["arms"][0]["status"] == "ok"


def test_i1_gate_arm_and_results():
    res = i1_check.results(0)
    assert set(res) == set(i1_check.BUILDERS) and "eval.scenarios" in res
    assert all(v == "ok" for v in res.values())
    out = run(_cfg(ArmConfig(name="i1_gate", kind="i1_gate")))
    arm = out["arms"][0]
    assert arm["status"] == "ok" and arm["metrics"] == {"builders": dict(sorted(res.items())), "passed": True}


def test_i1_gate_failing_builder(monkeypatch):
    monkeypatch.setitem(i1_check.BUILDERS, "leaky", lambda c: [{"member_id": "cur-001"}])
    res = i1_check.results(0)
    assert res["leaky"] == "fail"
    out = run(_cfg(ArmConfig(name="i1_gate", kind="i1_gate")))
    assert out["arms"][0]["status"] == "ok" and out["arms"][0]["metrics"]["passed"] is False
    assert i1_check.main() == 1


def test_i1_main_output(capsys):
    assert i1_check.main() == 0
    assert "I1 ok   eval.scenarios" in capsys.readouterr().out.splitlines()


# --------------------------------------------------------------------------- CLI


def _yaml(tmp_path, body):
    p = tmp_path / "runs.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_cli_ok_and_out(tmp_path):
    _write_corpus(tmp_path / "c.jsonl", [V1, C1])
    runs = _yaml(tmp_path, "version: v\narms:\n  - {name: r2, kind: r2_dsl, corpus: c.jsonl}\n"
                 "  - {name: sft, kind: not_run, reason: blocked}\n")
    out = tmp_path / "deep" / "dir" / "m.json"
    assert main(["--runs", str(runs), "--root", str(tmp_path), "--out", str(out)]) == 0
    first = out.read_text(encoding="utf-8")
    assert json.loads(first)["summary"] == {"ok": 1, "not_run": 1, "failed": 0}
    assert main(["--runs", str(runs), "--root", str(tmp_path), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == first


def test_cli_failed_arm_exit_1_stdout(tmp_path, capsys):
    runs = _yaml(tmp_path, "version: v\narms:\n  - {name: r2, kind: r2_dsl, corpus: missing.jsonl}\n")
    assert main(["--runs", str(runs), "--root", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["arms"][0]["status"] == "failed"


def test_cli_i1_not_passed_exit_1(tmp_path, monkeypatch):
    monkeypatch.setitem(i1_check.BUILDERS, "leaky", lambda c: [{"member_id": "cur-001"}])
    runs = _yaml(tmp_path, "version: v\narms:\n  - {name: i1_gate, kind: i1_gate}\n")
    out = tmp_path / "m.json"
    assert main(["--runs", str(runs), "--out", str(out)]) == 1
    assert json.loads(out.read_text())["arms"][0]["metrics"]["passed"] is False


@pytest.mark.parametrize("body", [None, "- a\n", "version: v\narms: []\n", "version: v\narms:\n  - {name: x, kind: not_run}\n", "a: [\n"])
def test_cli_config_error_exit_2(tmp_path, capsys, body):
    runs = tmp_path / "missing.yaml" if body is None else _yaml(tmp_path, body)
    out = tmp_path / "m.json"
    assert main(["--runs", str(runs), "--out", str(out)]) == 2
    cap = capsys.readouterr()
    assert cap.err and not cap.out and not out.exists()
