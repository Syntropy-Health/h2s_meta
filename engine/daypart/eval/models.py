"""E1.1 runs config + E1.2 corpus row models (CONTRACT slice E)."""

from __future__ import annotations

import os
import pathlib
import re
from importlib import resources
from typing import Any, Literal

import pydantic
import yaml
from pydantic import ConfigDict, model_validator

SCHEMA = "daypart-metrics/v1"
ARM_KINDS: frozenset[str] = frozenset({"r2_dsl", "env_baseline", "i1_gate", "not_run"})
POLICIES: tuple[str, ...] = ("always_none", "always_barrier_problem_solving", "random_intent")
TRAINING_ARMS: tuple[str, ...] = ("sft", "dpo", "grpo", "rm", "jev")
ARM_STATUSES: tuple[str, ...] = ("ok", "not_run", "failed")

_ARM_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _blank(value: str | None) -> bool:
    return value is None or not value.strip()


class ArmConfig(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    kind: str  # unknown kinds are accepted here and fail at run time (E1.3)
    reason: str | None = None
    corpus: str | None = None
    policy: str | None = None
    n_scenarios: int | None = None
    seeds: tuple[int, ...] = ()
    horizon: int = 14

    @model_validator(mode="after")
    def _check(self) -> ArmConfig:
        if not _ARM_NAME.fullmatch(self.name):
            raise ValueError(f"arm name {self.name!r} must match {_ARM_NAME.pattern}")
        if not self.kind:
            raise ValueError("arm kind must be non-empty")
        if self.policy is not None and self.policy not in POLICIES:
            raise ValueError(f"policy {self.policy!r} not in {POLICIES}")
        if self.kind == "not_run" and _blank(self.reason):
            raise ValueError(f"arm {self.name!r}: kind not_run requires a non-empty reason")
        if self.kind == "r2_dsl" and _blank(self.corpus):
            raise ValueError(f"arm {self.name!r}: kind r2_dsl requires a corpus path")
        if self.kind == "env_baseline":
            if self.policy is None:
                raise ValueError(f"arm {self.name!r}: kind env_baseline requires a policy")
            if self.n_scenarios is None or self.n_scenarios < 1:
                raise ValueError(f"arm {self.name!r}: kind env_baseline requires n_scenarios >= 1")
            if not self.seeds:
                raise ValueError(f"arm {self.name!r}: kind env_baseline requires non-empty seeds")
            if len(set(self.seeds)) != len(self.seeds):
                raise ValueError(f"arm {self.name!r}: seeds must be unique")
            if self.horizon < 1:
                raise ValueError(f"arm {self.name!r}: horizon must be >= 1")
        return self


class RunsConfig(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    cohort_seed: int = 0
    arms: tuple[ArmConfig, ...]

    @model_validator(mode="after")
    def _check(self) -> RunsConfig:
        if _blank(self.version):
            raise ValueError("version must be non-empty")
        if not self.arms:
            raise ValueError("arms must be non-empty")
        names = [a.name for a in self.arms]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate arm names: {dupes}")
        return self


def default_runs_path() -> pathlib.Path:
    """The packaged default ``runs.yaml``."""
    return pathlib.Path(str(resources.files("daypart.eval").joinpath("runs.yaml")))


def load_runs(path: str | os.PathLike | None = None) -> RunsConfig:
    p = default_runs_path() if path is None else pathlib.Path(path)
    text = p.read_text(encoding="utf-8")  # FileNotFoundError propagates
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{p.name}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{p.name}: runs config must be a YAML mapping, got {type(data).__name__}")
    return RunsConfig.model_validate(data)


class CorpusRow(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    expected: Literal["violation", "compliant"]
    rule_ids: tuple[str, ...] = ()
    crisis_context: bool = False

    @model_validator(mode="after")
    def _check(self) -> CorpusRow:
        from daypart.membrane.dsl import RULE_IDS

        unknown = [r for r in self.rule_ids if r not in RULE_IDS]
        if unknown:
            raise ValueError(f"unknown rule id(s): {unknown}")
        if self.expected == "violation" and not self.rule_ids:
            raise ValueError("a violation row requires non-empty rule_ids")
        if self.expected == "compliant" and self.rule_ids:
            raise ValueError("a compliant row must have empty rule_ids")
        return self


def _coerce_row(row: Any) -> CorpusRow:
    return row if isinstance(row, CorpusRow) else CorpusRow.model_validate(row)


__all__ = [
    "ARM_KINDS",
    "ARM_STATUSES",
    "POLICIES",
    "SCHEMA",
    "TRAINING_ARMS",
    "ArmConfig",
    "CorpusRow",
    "RunsConfig",
    "default_runs_path",
    "load_runs",
]
