"""I1 CI gate — curated-25 never enter any training artifact.

Every dataset builder registers here. The gate generates the cohort, writes
its manifest, builds each training artifact and runs ``assert_no_curated``
over it. Any leak fails CI with a non-zero exit.
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from daypart.simmember.cohort import Cohort, generate
from daypart.simmember.manifest import CuratedLeakError, assert_no_curated, verify_manifest, write


def _training_profiles(cohort: Cohort) -> Iterable[Mapping[str, Any]]:
    return (p.model_dump(mode="json") for p in cohort.training_profiles())


def _training_trajectories(cohort: Cohort) -> Iterable[Mapping[str, Any]]:
    return (cohort.trajectories[p.member_id].model_dump(mode="json") for p in cohort.training_profiles())


_I1_MEMBERS = 12  # small fixed set of training members (<= 20) keeps the gate fast


def _realized_cache(cohort: Cohort, tmp: Path):
    from daypart.simmember.narrative import NarrativeCache, TemplateClient, realize

    cache = NarrativeCache(tmp / "narrative.jsonl")
    client = TemplateClient()
    for profile in cohort.training_profiles()[:_I1_MEMBERS]:
        realize(profile, cohort.trajectories[profile.member_id], client, cache)
    return cache


def _sft_dataset(cohort: Cohort) -> list[dict]:
    from daypart.train.sft_dataset import build_sft

    with tempfile.TemporaryDirectory() as tmp:
        return build_sft(cohort, _realized_cache(cohort, Path(tmp)))


def _dpo_pairs(cohort: Cohort) -> list[dict]:
    from daypart.simmember.narrative import TemplateClient
    from daypart.train.dpo_pairs import build_pairs, make_contexts

    with tempfile.TemporaryDirectory() as tmp:
        contexts = make_contexts(cohort, _realized_cache(cohort, Path(tmp)), max_members=_I1_MEMBERS)
    return build_pairs(contexts, TemplateClient())


#: name -> builder(cohort) -> training rows. Add every new dataset builder (2.1 SFT, 2.3 DPO, ...).
BUILDERS: dict[str, Callable[[Cohort], Iterable[Mapping[str, Any]]]] = {
    "simmember.training_profiles": _training_profiles,
    "simmember.training_trajectories": _training_trajectories,
    "train.sft_dataset": _sft_dataset,
    "train.dpo_pairs": _dpo_pairs,
}


def main(seed: int = 0) -> int:
    cohort = generate(seed)
    with tempfile.TemporaryDirectory() as tmp:
        write(cohort, Path(tmp))
        manifest = verify_manifest(Path(tmp))
    failed = False
    for name, build in BUILDERS.items():
        try:
            assert_no_curated(build(cohort), manifest)
        except CuratedLeakError as exc:
            failed = True
            print(f"I1 FAIL {name}: {exc}", file=sys.stderr)
        else:
            print(f"I1 ok   {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
