"""Sealed challenge suite — E1.4 I1 gate extension (``daypart.ci.i1_check``).

Written double-blind from docs/daypart/contracts/CONTRACT-slice-E-eval-passport.md E1.4.

Categories:
  functional/i1-results     results(seed) shape and values
  functional/i1-eval-builder  "eval.scenarios" builder rows
  integration/i1-cli        python -m daypart.ci.i1_check output / exit code; lazy import

Ambiguity resolved: the contract does not fix the order of the rows from
"eval.scenarios", or whether rows are de-duplicated when several arms build the
same scenarios. Rows are compared as a set of canonical-JSON strings, and every
row must be one of the expected ones.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[2]
ORIGINAL_BUILDERS = ("simmember.training_profiles", "simmember.training_trajectories")


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _expected_rows(cohort) -> set[str]:
    """Reference: every scenario every env_baseline arm of the default runs.yaml builds."""
    import yaml

    from daypart.simmember.env import make_scenarios

    import daypart.eval as ev

    raw = yaml.safe_load(Path(ev.default_runs_path()).read_text(encoding="utf-8"))
    out: set[str] = set()
    for arm in raw["arms"]:
        if arm["kind"] != "env_baseline":
            continue
        horizon = arm.get("horizon", 14)
        for seed in arm["seeds"]:
            for s in make_scenarios(cohort, arm["n_scenarios"], seed):
                out.add(_canon(s.model_copy(update={"horizon": horizon}).model_dump(mode="json")))
    return out


def _leaky(cohort):
    return [p.model_dump(mode="json") for p in cohort.profiles if p.member_id.startswith("cur-")]


@pytest.mark.category("functional/i1-results")
class TestResults:
    def test_one_entry_per_builder(self):
        from daypart.ci.i1_check import BUILDERS, results

        r = results()
        assert isinstance(r, dict)
        assert set(r) == set(BUILDERS)
        assert set(r.values()) <= {"ok", "fail"}

    def test_all_ok_seed0(self):
        from daypart.ci.i1_check import results

        r = results(0)
        assert r == results()
        assert all(v == "ok" for v in r.values()), r
        for name in ORIGINAL_BUILDERS:
            assert r[name] == "ok"
        assert r["eval.scenarios"] == "ok"

    def test_leaky_builder_is_fail(self, monkeypatch):
        from daypart.ci import i1_check

        monkeypatch.setitem(i1_check.BUILDERS, "sealed.leaky", _leaky)
        r = i1_check.results(0)
        assert r["sealed.leaky"] == "fail"
        assert all(v == "ok" for k, v in r.items() if k != "sealed.leaky")

    def test_main_unchanged_on_leak(self, monkeypatch, capsys):
        from daypart.ci import i1_check

        monkeypatch.setitem(i1_check.BUILDERS, "sealed.leaky", _leaky)
        capsys.readouterr()
        assert i1_check.main() == 1
        cap = capsys.readouterr()
        assert "I1 FAIL sealed.leaky" in cap.err
        assert "I1 ok   eval.scenarios" in cap.out

    def test_main_ok(self, capsys):
        from daypart.ci import i1_check

        capsys.readouterr()
        assert i1_check.main() == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines == [f"I1 ok   {name}" for name in i1_check.BUILDERS]


@pytest.mark.category("functional/i1-eval-builder")
class TestEvalScenariosBuilder:
    def test_registered(self):
        from daypart.ci.i1_check import BUILDERS

        assert "eval.scenarios" in BUILDERS
        for name in ORIGINAL_BUILDERS:
            assert name in BUILDERS

    def test_rows_match_default_env_arms(self, cohort0):
        from daypart.ci.i1_check import BUILDERS

        rows = list(BUILDERS["eval.scenarios"](cohort0))
        assert rows
        got = {_canon(r) for r in rows}
        assert got == _expected_rows(cohort0)

    def test_rows_are_synthetic_and_pass_gate(self, cohort0, tmp_path):
        from daypart.ci.i1_check import BUILDERS
        from daypart.simmember.manifest import assert_no_curated, verify_manifest, write

        rows = list(BUILDERS["eval.scenarios"](cohort0))
        for r in rows:
            assert r["member"]["member_id"].startswith("syn-")
            assert r["horizon"] == 14
        write(cohort0, tmp_path)
        assert_no_curated(rows, verify_manifest(tmp_path))

    def test_covers_all_default_seeds(self, cohort0):
        from daypart.ci.i1_check import BUILDERS

        seeds = {r["seed"] for r in BUILDERS["eval.scenarios"](cohort0)}
        assert seeds == {0, 1, 2}

    def test_deterministic(self, cohort0):
        from daypart.ci.i1_check import BUILDERS

        a = [_canon(r) for r in BUILDERS["eval.scenarios"](cohort0)]
        b = [_canon(r) for r in BUILDERS["eval.scenarios"](cohort0)]
        assert a == b


@pytest.mark.category("integration/i1-cli")
class TestI1Cli:
    def test_module_prints_eval_line(self):
        proc = subprocess.run(
            [sys.executable, "-m", "daypart.ci.i1_check"],
            cwd=ENGINE,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        lines = proc.stdout.splitlines()
        assert "I1 ok   eval.scenarios" in lines
        for name in ORIGINAL_BUILDERS:
            assert f"I1 ok   {name}" in lines
        assert "FAIL" not in proc.stdout + proc.stderr

    def test_lazy_import_no_cycle(self):
        code = (
            "import sys\n"
            "import daypart.ci.i1_check as m\n"
            "assert 'daypart.eval' not in sys.modules, 'daypart.eval imported eagerly'\n"
            "assert 'eval.scenarios' in m.BUILDERS\n"
            "import daypart.eval\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], cwd=ENGINE, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr

    def test_eval_first_import_order(self):
        code = "import daypart.eval\nimport daypart.ci.i1_check as m\nassert 'eval.scenarios' in m.BUILDERS\n"
        proc = subprocess.run([sys.executable, "-c", code], cwd=ENGINE, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr
