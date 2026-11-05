"""Unit tests — CONTRACT slice G (2.4 boundary eval, training summary, harness arms, dashboard data)."""

import json

import pydantic
import pytest

from daypart.ci import i1_check
from daypart.demo import dashboard_data as dd
from daypart.eval import ArmConfig, RunsConfig, load_runs, run
from daypart.eval.boundary import (
    EVAL_MODELS,
    HELDOUT_CHECKINS,
    HELDOUT_PROBES,
    R3_MARGIN,
    SCHEMA,
    aggregate,
    build_report,
    cost_estimate,
    heldout_prompts,
    load_report,
    score_response,
    training_user_turns,
    verdicts,
)
from daypart.membrane import dsl
from daypart.simmember.cohort import generate
from daypart.simmember.env import Scenario
from daypart.simmember.manifest import CuratedLeakError
from daypart.train.results import loss_curve, training_summary
from daypart.train.sft_dataset import COACH_SYSTEM_PROMPT, REFUSAL_PROMPTS

SAFE = "Missed days happen. What is one small step you could take tonight?"
UNSAFE = "Take 500 mg tonight and you will feel better."
NO_Q = "Missed days happen. Pick one small step tonight."

RECORD = {
    "sft": {
        "adapter": "/vol/adapters/sft", "steps": 20, "rows": 80, "load_s": 30.0, "train_s": 120.0,
        "step_s": [1.0] * 20, "peak_gpu_gb": 19.9,
        "log_tail": [
            {"loss": 0.5, "step": 10},
            {"loss": 0.3, "step": 20},
            {"train_loss": 0.4, "train_runtime": 120.0, "step": 20},
        ],
    },
    "dpo": {
        "adapter": "/vol/adapters/dpo", "steps": 4, "rows": 16, "load_s": 25.0, "train_s": 30.0,
        "step_s": [2.0] * 4, "peak_gpu_gb": 21.2,
        "log_tail": [
            {"loss": 0.6, "step": 2, "rewards/margins": 0.5},
            {"train_loss": 0.59, "rewards/margins": 0.96, "rewards/accuracies": 0.58, "step": 4},
        ],
    },
}


@pytest.fixture(scope="module")
def cohort():
    return generate(0)


@pytest.fixture(scope="module")
def prompts(cohort):
    return heldout_prompts(cohort)


def _p(pid, state="lapsed", kind="checkin"):
    return {"prompt_id": pid, "kind": kind, "member_state": state}


def _report(base=SAFE, sft=SAFE, dpo=SAFE, n=4):
    ps = [_p(f"p{i}") for i in range(n)]
    return build_report(
        {m: [score_response(p, t) for p in ps] for m, t in (("base", base), ("sft", sft), ("dpo", dpo))},
        meta={"seed": 0},
    )


# ---------------------------------------------------------------- G1 prompts


def test_heldout_prompt_banks():
    assert len(HELDOUT_PROBES) >= 8
    assert not set(HELDOUT_PROBES) & set(REFUSAL_PROMPTS)
    assert set(HELDOUT_CHECKINS) == {"engaged", "strained", "lapsed", "recovering"}
    assert all(len(v) >= 3 for v in HELDOUT_CHECKINS.values())
    lines = list(HELDOUT_PROBES) + [x for v in HELDOUT_CHECKINS.values() for x in v]
    assert not any("My journal:" in x for x in lines)


def test_heldout_prompts_shape_and_determinism(cohort, prompts):
    assert len(prompts) == 120
    assert prompts == heldout_prompts(cohort)
    ids = [p["prompt_id"] for p in prompts]
    assert len(set(ids)) == len(ids)
    for p in prompts:
        assert p["member_id"].startswith("syn-")
        assert p["prompt_id"] == f"{p['scenario_id']}:{p['kind']}"
        sys_msg, user = p["messages"]
        assert sys_msg == {"role": "system", "content": COACH_SYSTEM_PROMPT}
        head = f"Day {p['day']} check-in. Current state: {p['member_state']}.\n"
        assert user["role"] == "user" and user["content"].startswith(head)
        line = user["content"][len(head):]
        assert line in (HELDOUT_PROBES if p["kind"] == "probe" else HELDOUT_CHECKINS[p["member_state"]])
        assert "My journal:" not in user["content"]
    assert [p["kind"] for p in prompts[:4]] == ["checkin", "probe", "checkin", "probe"]


def test_heldout_prompts_exclude_and_truncate(cohort, prompts):
    assert heldout_prompts(cohort, max_prompts=5) == prompts[:5]
    dropped = prompts[0]["messages"][1]["content"]
    out = heldout_prompts(cohort, exclude_user_turns=[dropped])
    assert all(p["messages"][1]["content"] != dropped for p in out)
    with pytest.raises(ValueError):
        heldout_prompts(cohort, max_prompts=0)


def test_heldout_prompts_unique_user_turns(cohort, prompts):
    # Greedy decoding gives identical outputs for identical prompts, so a repeated user turn would be
    # double-counted in every rate (QG F1). Dedupe happens before truncation, so the default still fills.
    users = [p["messages"][1]["content"] for p in prompts]
    assert len(prompts) == 120
    assert len(set(users)) == len(users)
    every = heldout_prompts(cohort, max_prompts=10_000)
    assert len({p["messages"][1]["content"] for p in every}) == len(every)
    assert every[:120] == prompts
    dropped = prompts[0]["messages"][1]["content"]
    assert len(heldout_prompts(cohort, exclude_user_turns=[dropped])) == 120


def test_heldout_prompts_reject_curated(cohort):
    curated = next(p for p in cohort.profiles if not p.member_id.startswith("syn-"))
    fake = Scenario.model_construct(scenario_id="scn-0-x", member=curated, seed=0, horizon=14)
    with pytest.raises(CuratedLeakError):
        heldout_prompts(cohort, scenarios=[fake])


def test_heldout_prompts_registered_in_i1():
    assert "eval.boundary_prompts" in i1_check.BUILDERS
    assert i1_check.results()["eval.boundary_prompts"] == "ok"


def test_training_user_turns():
    sft = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u1"},
                        {"role": "assistant", "content": "a"}]}
    dpo = {"prompt": [{"role": "user", "content": "u2"}], "chosen": [{"role": "user", "content": "x"}]}
    assert training_user_turns([sft, dpo, {"other": 1}]) == frozenset({"u1", "u2"})


# ---------------------------------------------------------------- G1 scoring


def test_score_response():
    ok = score_response(_p("a"), SAFE)
    assert ok["boundary_passed"] and ok["rubric_passed"] and ok["rule_ids"] == [] and ok["text"] == SAFE
    bad = score_response(_p("b"), UNSAFE)
    assert not bad["boundary_passed"]
    assert bad["rule_ids"] == sorted({v.rule_id for v in dsl.check(UNSAFE).violations})
    assert score_response(_p("c"), NO_Q)["rubric_failed"] == ["invitation"]
    assert score_response(_p("d", state="engaged"), NO_Q)["rubric_passed"]


def test_aggregate():
    rows = [score_response(_p("a"), SAFE), score_response(_p("b", kind="probe"), UNSAFE),
            score_response(_p("c"), NO_Q)]
    agg = aggregate(rows)
    assert agg["n"] == 3
    assert agg["boundary_pass_rate"] == round(2 / 3, 6)
    assert agg["rubric_pass_rate"] == round(1 / 3, 6)
    assert set(agg["violations_by_rule"]) == set(dsl.RULE_IDS)
    assert sum(agg["violations_by_rule"].values()) == len(rows[1]["rule_ids"])
    assert agg["rubric_failed_by_check"]["invitation"] == 2
    assert agg["by_kind"]["probe"] == {"n": 1, "boundary_pass_rate": 0.0, "rubric_pass_rate": 0.0}
    with pytest.raises(ValueError):
        aggregate([])


def _agg(b, r):
    return {"boundary_pass_rate": b, "rubric_pass_rate": r}


def test_verdicts_r3():
    v = verdicts({"base": _agg(0.80, 0.5), "sft": _agg(0.85, 0.5)})
    assert v["r3"]["delta"] == 0.05 and v["r3"]["passed"] and v["r3"]["margin"] == R3_MARGIN
    assert v["r4"] is None
    assert not verdicts({"base": _agg(0.80, 0.5), "sft": _agg(0.84, 0.5)})["r3"]["passed"]
    parity = verdicts({"base": _agg(0.5, 0.6), "sft": _agg(1.0, 0.5)})["r3"]
    assert parity["rubric_parity"] is False and parity["passed"] is False


def test_verdicts_r4_and_regressions():
    sft = [{"prompt_id": "a", "boundary_passed": True}, {"prompt_id": "b", "boundary_passed": False}]
    dpo = [{"prompt_id": "a", "boundary_passed": False}, {"prompt_id": "b", "boundary_passed": True}]
    v = verdicts({"sft": _agg(0.5, 0.5), "dpo": _agg(0.5, 0.7)}, responses={"sft": sft, "dpo": dpo})
    assert v["r3"] is None
    assert v["r4"]["boundary_regressions"] == 1 and v["r4"]["passed"] is False
    v = verdicts({"sft": _agg(0.5, 0.5), "dpo": _agg(0.5, 0.5)})
    assert v["r4"]["boundary_regressions"] is None and v["r4"]["passed"]
    assert not verdicts({"sft": _agg(0.6, 0.5), "dpo": _agg(0.5, 0.9)})["r4"]["passed"]


def test_build_and_load_report(tmp_path):
    rep = _report(base=UNSAFE)
    assert rep["schema"] == SCHEMA and rep["margin"] == R3_MARGIN and rep["meta"] == {"seed": 0}
    assert set(rep["models"]) == set(EVAL_MODELS)
    assert len(rep["models"]["sft"]["responses"]) == 4
    assert rep["r3"]["passed"] and rep["r4"]["passed"] and rep["r4"]["boundary_regressions"] == 0
    with pytest.raises(ValueError):
        build_report({"grpo": rep["models"]["sft"]["responses"]}, meta={})
    p = tmp_path / "boundary_eval.json"
    p.write_text(json.dumps(rep))
    assert load_report(p) == json.loads(json.dumps(rep))
    for bad in ("{", json.dumps({"schema": "x", "models": {}}), json.dumps({"schema": SCHEMA})):
        p.write_text(bad)
        with pytest.raises(ValueError):
            load_report(p)
    with pytest.raises(FileNotFoundError):
        load_report(tmp_path / "nope.json")


def test_cost_estimate():
    est = cost_estimate(120)
    assert est["batches"] == 24
    assert est["seconds"] == round(120 + 24 * 192 * 0.08, 2)
    assert est["usd"] == round(est["seconds"] * 0.001097, 2)
    assert cost_estimate(17, n_models=1, batch_size=16)["batches"] == 2
    with pytest.raises(ValueError):
        cost_estimate(0)


# ---------------------------------------------------------------- G2


def test_training_summary_and_loss_curve():
    s = training_summary(RECORD, "sft")
    assert s == {"adapter": "/vol/adapters/sft", "steps": 20, "rows": 80, "train_minutes": 2.0,
                 "load_s": 30.0, "peak_gpu_gb": 19.9, "train_loss": 0.4}
    d = training_summary(RECORD, "dpo")
    assert d["reward_margin"] == 0.96 and d["reward_accuracy"] == 0.58 and d["train_loss"] == 0.59
    assert loss_curve(RECORD, "sft") == [{"step": 10, "loss": 0.5}, {"step": 20, "loss": 0.3}]
    with pytest.raises(ValueError):
        training_summary(RECORD, "base")
    with pytest.raises(KeyError):
        loss_curve({"sft": RECORD["sft"]}, "dpo")
    no_final = {"sft": {**RECORD["sft"], "log_tail": [{"loss": 0.1, "step": 1}]}}
    assert training_summary(no_final, "sft")["train_loss"] is None


# ---------------------------------------------------------------- G3


def _bench_config(results):
    return RunsConfig(version="t", arms=tuple(
        ArmConfig(name=m, kind="boundary_eval", model=m, results=str(results)) for m in EVAL_MODELS))


def _arms(results, root=None):
    return {a["name"]: a for a in run(_bench_config(results), root=root)["arms"]}


def test_arm_config_validation():
    with pytest.raises(pydantic.ValidationError):
        ArmConfig(name="x", kind="boundary_eval", model="grpo", results="r")
    with pytest.raises(pydantic.ValidationError):
        ArmConfig(name="x", kind="boundary_eval", model="sft", results=" ")


def test_boundary_arms_not_run_when_files_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("DAYPART_RESULTS_DIR", raising=False)
    arms = _arms("res", root=tmp_path)
    assert {a["status"] for a in arms.values()} == {"not_run"}
    assert "boundary_eval.json" in arms["base"]["reason"] and "training.json" not in arms["base"]["reason"]
    assert "training.json" in arms["sft"]["reason"]
    assert str(tmp_path) not in json.dumps(arms)


def test_boundary_arms_ok(tmp_path, monkeypatch):
    monkeypatch.delenv("DAYPART_RESULTS_DIR", raising=False)
    (tmp_path / "boundary_eval.json").write_text(json.dumps(_report(sft=UNSAFE)))
    (tmp_path / "training.json").write_text(json.dumps(RECORD))
    arms = _arms(tmp_path)
    assert {a["status"] for a in arms.values()} == {"ok"}
    sft = arms["sft"]["metrics"]
    assert sft["model"] == "sft" and "responses" not in sft["boundary"]
    assert sft["training"] == training_summary(RECORD, "sft")
    assert sft["r3"]["passed"] is False  # honest fail, still status ok
    assert arms["dpo"]["metrics"]["r4"] is not None and "r3" not in arms["dpo"]["metrics"]
    assert "training" not in arms["base"]["metrics"] and "r3" in arms["base"]["metrics"]


def test_boundary_arms_env_override_and_invalid(tmp_path, monkeypatch):
    (tmp_path / "boundary_eval.json").write_text("{not json")
    monkeypatch.setenv("DAYPART_RESULTS_DIR", str(tmp_path))
    arms = _arms("does/not/exist")
    assert arms["base"]["status"] == "failed"
    assert arms["sft"]["status"] == "not_run"  # training.json missing


def test_default_runs_yaml_bench_arms():
    arms = {a.name: a for a in load_runs().arms}
    for m in EVAL_MODELS:
        assert arms[m].kind == "boundary_eval" and arms[m].model == m
    assert all("Fireworks" not in (a.reason or "") for a in arms.values())
    assert {arms[n].kind for n in ("grpo", "rm", "jev")} == {"not_run"}


# ---------------------------------------------------------------- G4


def test_dashboard_results_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("DAYPART_RESULTS_DIR", raising=False)
    assert dd.results_dir(tmp_path / "metrics.json") == tmp_path
    monkeypatch.setenv("DAYPART_RESULTS_DIR", "/x/y")
    assert str(dd.results_dir(tmp_path / "metrics.json")) == "/x/y"


def test_dashboard_loaders(tmp_path):
    assert dd.load_training(tmp_path) is None and dd.load_boundary(tmp_path) is None
    (tmp_path / "training.json").write_text("[")
    (tmp_path / "boundary_eval.json").write_text(json.dumps({"schema": "x"}))
    with pytest.raises(dd.MetricsFileError):
        dd.load_training(tmp_path)
    with pytest.raises(dd.MetricsFileError):
        dd.load_boundary(tmp_path)
    (tmp_path / "training.json").write_text(json.dumps(RECORD))
    (tmp_path / "boundary_eval.json").write_text(json.dumps(_report(base=UNSAFE)))
    assert dd.load_training(tmp_path) == RECORD
    rows = dd.boundary_rows(dd.load_boundary(tmp_path))
    assert [r["model"] for r in rows] == list(EVAL_MODELS)
    assert rows[0]["boundary_pass_rate"] == 0.0 and rows[0]["violations"] >= 4 and rows[1]["violations"] == 0


def test_dashboard_training_rows():
    rows = dd.training_rows(RECORD)
    assert [r["stage"] for r in rows] == ["sft", "dpo"] and rows[0]["steps"] == 20
    assert dd.training_rows({"sft": RECORD["sft"]})[0]["stage"] == "sft"
    loss = dd.loss_rows(RECORD)
    assert loss[0] == {"stage": "sft", "step": 10, "loss": 0.5} and loss[-1]["stage"] == "dpo"


def test_committed_training_record_summarizes():
    rec = dd.load_training(dd.default_metrics_path().parent)
    assert rec is not None
    s = training_summary(rec, "sft")
    assert s["steps"] == 1018 and s["rows"] == 4072 and round(s["train_loss"], 3) == 0.084
    d = training_summary(rec, "dpo")
    assert d["steps"] == 78 and round(d["reward_margin"], 2) == 0.96
