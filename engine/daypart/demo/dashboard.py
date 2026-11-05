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

from daypart.demo.dashboard_data import MetricsFileError, default_metrics_path, env_rows, load_metrics

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

    with st.expander("Per-seed detail"):
        for a in doc.arms:
            if a.kind == "env_baseline" and a.status == "ok":
                st.markdown(f"**{a.name}**")
                st.dataframe(pd.DataFrame(a.metrics["per_seed"]), hide_index=True, use_container_width=True)


main()
