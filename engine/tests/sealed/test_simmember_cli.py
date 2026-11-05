"""Sealed challenge suite — slice A: CLI (A6).

Spec: docs/daypart/contracts/CONTRACT-slice-A-simmember.md section A6 and the
determinism clause (no hash-seed dependence).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.category("integration/cli")

ENGINE_DIR = Path(__file__).resolve().parents[2]
FILES = ("profiles.jsonl", "trajectories.jsonl", "manifest.json")


def _run(out: Path, seed: int = 0, hashseed: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if hashseed is not None:
        env["PYTHONHASHSEED"] = hashseed
    return subprocess.run(
        [sys.executable, "-m", "daypart.simmember", "--seed", str(seed), "--out", str(out)],
        cwd=ENGINE_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_cli_writes_three_files(tmp_path):
    out = tmp_path / "cohort"
    out.mkdir()
    proc = _run(out)
    assert proc.returncode == 0, proc.stderr
    for name in FILES:
        assert (out / name).is_file(), name
        assert (out / name).stat().st_size > 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seed"] == 0
    assert manifest["counts"]["curated"] == 25


def test_cli_matches_library_write(tmp_path):
    from daypart.simmember.cohort import generate
    from daypart.simmember.manifest import write

    cli_out = tmp_path / "cli"
    lib_out = tmp_path / "lib"
    cli_out.mkdir()
    lib_out.mkdir()
    proc = _run(cli_out, seed=0)
    assert proc.returncode == 0, proc.stderr
    write(generate(0), lib_out)
    for name in FILES:
        assert (cli_out / name).read_bytes() == (lib_out / name).read_bytes(), name


def test_cli_independent_of_hash_seed(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    pa = _run(a, seed=5, hashseed="1")
    pb = _run(b, seed=5, hashseed="4242")
    assert pa.returncode == 0, pa.stderr
    assert pb.returncode == 0, pb.stderr
    for name in FILES:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
