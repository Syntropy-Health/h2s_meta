"""HAC-38 P1.7 (addendum G20): pyright, basic mode, gates the demo deploy in CI."""

import tomllib
from pathlib import Path

import yaml

ENGINE = Path(__file__).resolve().parents[2]
WORKFLOW = ENGINE.parent / ".github" / "workflows" / "engine.yml"


def _jobs() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]


def test_pyright_config_is_basic_over_daypart():
    cfg = tomllib.loads((ENGINE / "pyproject.toml").read_text())["tool"]["pyright"]
    assert cfg["typeCheckingMode"] == "basic"
    assert cfg["include"] == ["daypart"]
    assert cfg.get("exclude", []) == ["daypart/train/modal_app.py"]


def test_typecheck_job_runs_pinned_pyright_in_engine():
    job = _jobs()["typecheck"]
    assert job["defaults"]["run"]["working-directory"] == "engine"
    runs = " ".join(s.get("run", "") for s in job["steps"])
    assert "uv sync --locked" in runs
    assert "pyright@1." in runs and "--pythonpath .venv/bin/python" in runs


def test_deploy_needs_typecheck():
    assert "typecheck" in _jobs()["deploy-demo"]["needs"]
