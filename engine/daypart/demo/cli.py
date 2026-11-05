"""``daypart-demo`` CLI: ``serve`` (FastAPI playground), ``dashboard`` (Streamlit), ``members``."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def _serve(a: argparse.Namespace) -> int:
    import uvicorn

    from daypart.demo.app import create_app

    uvicorn.run(create_app(cohort_seed=a.cohort_seed), host=a.host, port=a.port)
    return 0


def _dashboard(a: argparse.Namespace) -> int:
    script = Path(__file__).with_name("dashboard.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(script), "--server.port", str(a.port),
           "--server.address", a.host, "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    if a.metrics:
        cmd += ["--", "--metrics", a.metrics]
    return subprocess.call(cmd)


def _members(a: argparse.Namespace) -> int:
    from daypart.demo.playground import Playground

    for m in Playground(a.cohort_seed).members():
        print(json.dumps(m, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="daypart-demo", description="Daypart demo (synthetic data only).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the FastAPI playground")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--cohort-seed", type=int, default=0)
    s.set_defaults(fn=_serve)
    d = sub.add_parser("dashboard", help="run the Streamlit results dashboard")
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=int, default=8501)
    d.add_argument("--metrics", default=None, help="metrics.json path (default docs/daypart/results/metrics.json)")
    d.set_defaults(fn=_dashboard)
    m = sub.add_parser("members", help="list generated members as JSON lines")
    m.add_argument("--cohort-seed", type=int, default=0)
    m.set_defaults(fn=_members)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
