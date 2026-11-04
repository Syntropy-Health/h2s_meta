"""HAC-20 — dashboard data loader over ``daypart-metrics/v1`` (all arms reported, G2)."""

import json

import pytest

from daypart.demo.dashboard_data import (
    MetricsFileError,
    default_metrics_path,
    env_rows,
    load_metrics,
)


def _arm(name, kind, status, reason=None, metrics=None):
    return {"name": name, "kind": kind, "status": status, "reason": reason, "metrics": metrics}


def _env_metrics(policy, adherence):
    return {"policy": policy, "horizon": 14, "n_scenarios": 2, "seeds": [0, 1],
            "mean_adherence": adherence, "episode_violation_rate": 0.0,
            "per_seed": [{"seed": s, "n_episodes": 2, "mean_adherence": adherence, "episode_violation_rate": 0.0,
                          "mean_return": 7.0, "scenario_manifest_sha256": "ab" * 32} for s in (0, 1)]}


def _doc(arms):
    summary = {"ok": 0, "not_run": 0, "failed": 0}
    for a in arms:
        summary[a["status"]] += 1
    return {"schema": "daypart-metrics/v1", "runs_version": "eval-v1", "runs_digest": "f" * 64, "cohort_seed": 0,
            "rule_version": "boundary-v1", "guard_versions": {"claims": "c", "valence": "v"},
            "arms": arms, "summary": summary}


ARMS = [
    _arm("r2_boundary_dsl", "r2_dsl", "failed", reason="FileNotFoundError: corpus not found: x.jsonl"),
    _arm("env_always_none", "env_baseline", "ok", metrics=_env_metrics("always_none", 0.5)),
    _arm("env_random_intent", "env_baseline", "ok", metrics=_env_metrics("random_intent", 0.6)),
    _arm("i1_gate", "i1_gate", "ok", metrics={"builders": {"a": "ok"}, "passed": True}),
    _arm("sft", "not_run", "not_run", reason="needs a Fireworks key (G6)"),
]


def _write(tmp_path, doc):
    p = tmp_path / "metrics.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_loads_every_arm_in_order_including_not_run(tmp_path):
    data = load_metrics(_write(tmp_path, _doc(ARMS)))
    assert [a.name for a in data.arms] == [a["name"] for a in ARMS]
    sft = data.arms[-1]
    assert sft.status == "not_run" and sft.reason == "needs a Fireworks key (G6)" and sft.metrics is None
    assert data.summary == {"ok": 3, "not_run": 1, "failed": 1}
    assert data.runs_version == "eval-v1" and data.rule_version == "boundary-v1"


def test_arm_table_rows_have_reason_for_non_ok(tmp_path):
    rows = load_metrics(_write(tmp_path, _doc(ARMS))).arm_table()
    assert len(rows) == len(ARMS)
    assert all(set(r) == {"arm", "kind", "status", "reason"} for r in rows)
    assert all(r["reason"] for r in rows if r["status"] != "ok")
    assert all(r["reason"] == "" for r in rows if r["status"] == "ok")


def test_env_rows_only_ok_env_baselines(tmp_path):
    rows = env_rows(load_metrics(_write(tmp_path, _doc(ARMS))))
    assert [r["policy"] for r in rows] == ["always_none", "random_intent"]
    assert rows[0]["mean_adherence"] == 0.5 and rows[0]["episode_violation_rate"] == 0.0


def test_missing_file_raises_with_hint(tmp_path):
    with pytest.raises(MetricsFileError, match="python -m daypart.eval"):
        load_metrics(tmp_path / "absent.json")


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(schema="other/v9"),
    lambda d: d.pop("arms"),
    lambda d: d["arms"][0].update(status="maybe"),
    lambda d: d["arms"][0].update(reason=None),           # failed arm without reason
    lambda d: d["arms"][-1].update(reason=""),            # not_run arm without reason
    lambda d: d["arms"][1].update(metrics=None),          # ok arm without metrics
    lambda d: d["summary"].update(ok=99),                 # summary disagrees with arms
])
def test_invalid_documents_rejected(tmp_path, mutate):
    doc = json.loads(json.dumps(_doc(ARMS)))
    mutate(doc)
    with pytest.raises(MetricsFileError):
        load_metrics(_write(tmp_path, doc))


def test_bad_json_rejected(tmp_path):
    p = tmp_path / "metrics.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(MetricsFileError):
        load_metrics(p)


def test_default_path_points_at_docs_results():
    p = default_metrics_path()
    assert p.parts[-4:] == ("docs", "daypart", "results", "metrics.json")


def test_reads_real_harness_output(tmp_path):
    from daypart.eval import ArmConfig, RunsConfig, dumps, run

    cfg = RunsConfig(version="t", arms=(
        ArmConfig(name="env_always_none", kind="env_baseline", policy="always_none", n_scenarios=2, seeds=(0,),
                  horizon=3),
        ArmConfig(name="sft", kind="not_run", reason="blocked"),
    ))
    p = tmp_path / "m.json"
    p.write_text(dumps(run(cfg, root=tmp_path)), encoding="utf-8")
    data = load_metrics(p)
    assert [a.status for a in data.arms] == ["ok", "not_run"]
    assert env_rows(data)[0]["policy"] == "always_none"
