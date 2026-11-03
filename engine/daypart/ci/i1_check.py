"""I1 CI gate — curated-25 never enter any training artifact.

Every dataset builder registers here. The gate generates the cohort, writes
its manifest, builds each training artifact and runs ``assert_no_curated``
over it. Any leak fails CI with a non-zero exit.
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from daypart.simmember.cohort import Cohort, generate
from daypart.simmember.manifest import CuratedLeakError, assert_no_curated, verify_manifest, write


def _training_profiles(cohort: Cohort) -> Iterable[Mapping[str, Any]]:
    return (p.model_dump(mode="json") for p in cohort.training_profiles())


def _training_trajectories(cohort: Cohort) -> Iterable[Mapping[str, Any]]:
    return (cohort.trajectories[p.member_id].model_dump(mode="json") for p in cohort.training_profiles())


def _eval_scenarios(cohort: Cohort) -> Iterable[Mapping[str, Any]]:
    """Every scenario the default ``runs.yaml`` env_baseline arms build for this cohort (all seeds)."""
    # lazy: daypart.eval imports this module (i1_gate arm) — no import cycle
    from daypart.eval import arm_scenarios, load_runs

    config = load_runs()
    return [
        s.model_dump(mode="json")
        for arm in config.arms
        if arm.kind == "env_baseline"
        for s in arm_scenarios(arm, cohort)
    ]


#: name -> builder(cohort) -> training rows. Add every new dataset builder (2.1 SFT, 2.3 DPO, ...).
BUILDERS: dict[str, Callable[[Cohort], Iterable[Mapping[str, Any]]]] = {
    "simmember.training_profiles": _training_profiles,
    "simmember.training_trajectories": _training_trajectories,
    "eval.scenarios": _eval_scenarios,
}


def _check(seed: int) -> Iterator[tuple[str, CuratedLeakError | None]]:
    """Yield ``(builder name, leak or None)`` for every registered builder, in order."""
    cohort = generate(seed)
    with tempfile.TemporaryDirectory() as tmp:
        write(cohort, Path(tmp))
        manifest = verify_manifest(Path(tmp))
    for name, build in BUILDERS.items():
        try:
            assert_no_curated(build(cohort), manifest)
        except CuratedLeakError as exc:
            yield name, exc
        else:
            yield name, None


def results(seed: int = 0) -> dict[str, str]:
    """Builder name -> ``"ok"`` | ``"fail"``; one entry per ``BUILDERS`` key."""
    return {name: ("fail" if exc is not None else "ok") for name, exc in _check(seed)}


def main(seed: int = 0) -> int:
    failed = False
    for name, exc in _check(seed):
        if exc is not None:
            failed = True
            print(f"I1 FAIL {name}: {exc}", file=sys.stderr)
        else:
            print(f"I1 ok   {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
