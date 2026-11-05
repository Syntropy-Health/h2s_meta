"""Streamlit results dashboard (HAC-20): every arm of ``docs/daypart/results/metrics.json``.

Run: ``uv run streamlit run daypart/demo/dashboard.py`` (from ``engine/``), or ``uv run daypart-demo dashboard``.
Optional: ``-- --metrics PATH``. Offline, no secrets.
"""

from __future__ import annotations

import argparse
import sys

import altair as alt
import pandas as pd
import streamlit as st

from daypart.demo.dashboard_data import (
    MetricsFileError,
    boundary_rows,
    default_metrics_path,
    env_rows,
    load_boundary,
    load_metrics,
    load_training,
    loss_rows,
    results_dir,
    training_rows,
)

SERIES = "#2a78d6"  # single-series bars: one hue, no legend
STATUS_LABEL = {"ok": "ok", "not_run": "not run", "failed": "FAILED"}


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--metrics", default=None)
    return ap.parse_known_args(sys.argv[1:])[0]


def main() -> None:
    st.set_page_config(page_title="Daypart results", layout="wide")
    st.title("Daypart benchmark results")
    st.caption("Synthetic SimMembers only. Not a consumer UI. Not medical advice.")

    path = _args().metrics or default_metrics_path()
    try:
        doc = load_metrics(path)
    except MetricsFileError as exc:
        st.error(str(exc))
        st.stop()

    c = st.columns(4)
    c[0].metric("Arms", len(doc.arms))
    c[1].metric("OK", doc.summary["ok"])
    c[2].metric("Not run", doc.summary["not_run"])
    c[3].metric("Failed", doc.summary["failed"])
    st.caption(
        f"runs `{doc.runs_version}` · digest `{doc.runs_digest[:12]}` · cohort seed {doc.cohort_seed} · "
        f"rules `{doc.rule_version}` · guards "
        + ", ".join(f"{k} `{v}`" for k, v in sorted(doc.guard_versions.items()))
    )

    st.subheader("All arms")
    table = pd.DataFrame(doc.arm_table())
    table["status"] = table["status"].map(STATUS_LABEL)
    st.dataframe(table, hide_index=True, use_container_width=True)

    rows = env_rows(doc)
    if rows:
        st.subheader("Environment baselines")
        env = pd.DataFrame(rows)
        env["label"] = env["policy"].str.removeprefix("always_").str.replace("_", " ")
        st.caption("Mean adherence (fraction of simulated days logged), by policy")
        chart = (
            alt.Chart(env)
            .mark_bar(color=SERIES, cornerRadiusEnd=4)
            .encode(
                x=alt.X("mean_adherence:Q", title="mean adherence", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("label:N", title=None, sort=None, axis=alt.Axis(labelLimit=180)),
                tooltip=["policy", "mean_adherence", "episode_violation_rate", "mean_return"],
            )
            .properties(height=44 * len(env) + 40)
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(
            env[["policy", "mean_adherence", "episode_violation_rate", "mean_return", "n_scenarios", "horizon"]],
            hide_index=True, use_container_width=True,
        )

    for a in doc.arms:
        if a.status != "ok":
            continue
        if a.kind == "r2_dsl":
            st.subheader(f"R2 boundary DSL · {a.name}")
            m = a.metrics or {}
            k = st.columns(4)
            k[0].metric("Recall", m.get("recall"))
            k[1].metric("False-positive rate", m.get("fp_rate"))
            k[2].metric("Missed", m.get("missed"))
            k[3].metric("False positives", m.get("false_positives"))
            per_rule = pd.DataFrame([{"rule": r, **v} for r, v in sorted((m.get("per_rule") or {}).items())])
            st.dataframe(per_rule, hide_index=True, use_container_width=True)
        elif a.kind == "i1_gate":
            m = a.metrics or {}
            st.subheader(f"I1 gate · {'passed' if m.get('passed') else 'FAILED'}")
            st.dataframe(pd.DataFrame([{"builder": b, "result": r} for b, r in sorted(m.get("builders", {}).items())]),
                         hide_index=True, use_container_width=True)

    _training_panel(results_dir(path))

    with st.expander("Per-seed detail"):
        for a in doc.arms:
            if a.kind == "env_baseline" and a.status == "ok":
                st.markdown(f"**{a.name}**")
                st.dataframe(pd.DataFrame(a.metrics["per_seed"]), hide_index=True, use_container_width=True)


def _training_panel(directory) -> None:
    """Muse Glimmer 30B QLoRA run (training.json) + the base/SFT/DPO boundary eval (boundary_eval.json)."""
    st.subheader("Training · Muse Glimmer 30B, 4-bit QLoRA on Modal")
    try:  # each file renders on its own: a bad training.json does not hide the boundary table
        record = load_training(directory)
        stages = pd.DataFrame(training_rows(record)) if record is not None else None
        losses = pd.DataFrame(loss_rows(record)) if record is not None else None
    except (MetricsFileError, KeyError, TypeError, ValueError) as exc:
        st.error(f"training.json: {exc}")
        record = None
        stages = losses = pd.DataFrame()
    if record is None and stages is None:
        st.info("No training.json committed yet.")
    elif stages is not None and not stages.empty:
        k = st.columns(len(stages) * 2)
        for i, row in enumerate(stages.itertuples()):
            k[2 * i].metric(f"{row.stage.upper()} steps", f"{row.steps:,}", help=f"{row.rows:,} rows")
            k[2 * i + 1].metric(f"{row.stage.upper()} time · peak GPU", f"{row.train_minutes} min · {row.peak_gpu_gb} GB")
        st.dataframe(stages, hide_index=True, use_container_width=True)
        if not losses.empty:
            st.caption("Training loss by step (logged every 10 steps; the run record keeps the last entries)")
            chart = (
                alt.Chart(losses)
                .mark_line(color=SERIES, point=True)
                .encode(
                    x=alt.X("step:Q", title="step"),
                    y=alt.Y("loss:Q", title="loss", scale=alt.Scale(zero=False)),
                    tooltip=["stage", "step", "loss"],
                )
                .properties(height=200)
                .facet(column=alt.Column("stage:N", title=None))
                .resolve_scale(x="independent", y="independent")
            )
            st.altair_chart(chart)
    elif record is not None:
        st.info("training.json has no sft or dpo stage.")

    st.subheader("Boundary eval · base vs SFT vs DPO (held-out prompts)")
    try:
        report = load_boundary(directory)
        table = pd.DataFrame(boundary_rows(report)) if report is not None else None
    except (MetricsFileError, KeyError, TypeError, ValueError) as exc:
        st.error(f"boundary_eval.json: {exc}")
        return
    if report is None:
        st.info("No boundary_eval.json committed yet: the base, sft and dpo arms report not run.")
        return
    st.dataframe(table, hide_index=True, use_container_width=True)
    v = st.columns(2)
    for col, key, label in ((v[0], "r3", "R3 · SFT over base by the pre-registered margin"),
                            (v[1], "r4", "R4 · DPO ≥ SFT on rubric, no boundary regression")):
        verdict = report.get(key)
        if verdict is None:
            col.metric(label, "not run")
        else:
            col.metric(label, "pass" if verdict["passed"] else "FAIL")
            col.json(verdict, expanded=False)
    st.caption(
        f"prompts `{report.get('prompt_set_version')}` · rules `{report.get('rule_version')}` · "
        f"rubric `{report.get('rubric_version')}` · margin {report.get('margin')}"
    )


main()
