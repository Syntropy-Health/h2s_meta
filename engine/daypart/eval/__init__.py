"""eval — the Daypart eval harness (CONTRACT slice E, E1; PVR R2, A&D G2, I1).

Runs every arm of a ``runs.yaml`` config (R2 boundary corpus, synthetic env
baselines, the I1 gate, and not-yet-runnable training arms) into one
deterministic ``daypart-metrics/v1`` document.
"""

from daypart.eval.corpus import load_corpus, r2_metrics
from daypart.eval.harness import (
    arm_scenarios,
    build_scenarios,
    dumps,
    exit_code,
    main,
    run,
)
from daypart.eval.models import (
    ARM_KINDS,
    ARM_STATUSES,
    POLICIES,
    SCHEMA,
    TRAINING_ARMS,
    ArmConfig,
    CorpusRow,
    RunsConfig,
    default_runs_path,
    load_runs,
)
from daypart.eval.policies import POLICY_TEXT

__all__ = [
    "ARM_KINDS",
    "ARM_STATUSES",
    "POLICIES",
    "POLICY_TEXT",
    "SCHEMA",
    "TRAINING_ARMS",
    "ArmConfig",
    "CorpusRow",
    "RunsConfig",
    "arm_scenarios",
    "build_scenarios",
    "default_runs_path",
    "dumps",
    "exit_code",
    "load_corpus",
    "load_runs",
    "main",
    "r2_metrics",
    "run",
]
