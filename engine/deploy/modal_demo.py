"""Hosted Daypart demo on Modal (HAC-20/HAC-24): FastAPI playground + Streamlit results dashboard.

Offline and secret-free at runtime: synthetic SimMembers only, no model keys. The image
installs the locked engine (``uv export --frozen``) and bakes the benchmark files in at
build time:

* ``metrics.json``: ``python -m daypart.eval`` over the packaged ``runs.yaml`` (every arm,
  including ``not_run``/``failed`` with reasons), unless ``docs/daypart/results/metrics.json``
  is committed, in which case that file is served as is.
* ``training.json``: the Muse Glimmer QLoRA run + boundary eval, if committed.

Deploy from ``engine/`` (CI does this on push to ``daypart/build``/``main``)::

    uv run --with modal modal deploy deploy/modal_demo.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

APP_NAME = "daypart-demo"
ENGINE = Path(__file__).resolve().parents[1]
RESULTS_SRC = ENGINE.parent / "docs" / "daypart" / "results"
RESULTS = "/app/results"
DASHBOARD_PORT = 8501

image = (
    modal.Image.from_registry("python:3.12-bookworm")  # ships git (the pinned LHM dep is a git URL)
    .pip_install("uv==0.13.7")
    .add_local_dir(ENGINE, "/app", copy=True, ignore=[".venv", "**/__pycache__", "**/.pytest_cache", "deploy"])
    .add_local_dir(RESULTS_SRC, "/app/results_src", copy=True)
)
image = image.run_commands(
    "cd /app && uv export --frozen --no-dev --group demo --no-emit-project --no-hashes -o /tmp/req.txt"
    " && uv pip install --system -r /tmp/req.txt && uv pip install --system --no-deps /app",
    f"mkdir -p {RESULTS} && cp -r /app/results_src/. {RESULTS}/",
    # Fresh harness run unless a committed metrics.json exists. Exit code 1 = some arm failed;
    # the document is still written and the dashboard shows the failure with its reason.
    f"cd /app && (test -f {RESULTS}/metrics.json || DAYPART_RESULTS_DIR={RESULTS} python -m daypart.eval --out {RESULTS}/metrics.json || test -f {RESULTS}/metrics.json)",
).env({"DAYPART_RESULTS_DIR": RESULTS})

app = modal.App(APP_NAME, image=image)


@app.function(min_containers=0, scaledown_window=600, timeout=600)
@modal.concurrent(max_inputs=50)
@modal.web_server(DASHBOARD_PORT, startup_timeout=120)
def dashboard() -> None:
    subprocess.Popen(
        [
            "python", "-m", "streamlit", "run", "/app/daypart/demo/dashboard.py",
            "--server.port", str(DASHBOARD_PORT), "--server.address", "0.0.0.0",
            "--server.headless", "true", "--browser.gatherUsageStats", "false",
            "--server.enableCORS", "false", "--server.enableXsrfProtection", "false",
            "--", "--metrics", f"{RESULTS}/metrics.json",
        ]
    )


# One container: journals are in memory per process (A&D Concurrency, T4.6). More containers
# would need journals in a modal.Dict, which is out of scope.
@app.function(min_containers=0, max_containers=1, scaledown_window=600, timeout=600)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def playground():
    from daypart.demo.app import create_app

    return create_app(cohort_seed=0, results_url=dashboard.get_web_url())
