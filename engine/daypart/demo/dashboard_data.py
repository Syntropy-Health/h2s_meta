"""Dashboard data loader over a ``daypart-metrics/v1`` document (``daypart.eval`` output, HAC-4).

Pure (no Streamlit) so it is unit-testable. Every arm is kept, in file order,
including ``not_run`` and ``failed`` arms with their reasons (G2: all arms reported).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import pydantic
from pydantic import BaseModel, ConfigDict, model_validator

SCHEMA = "daypart-metrics/v1"
REGEN_HINT = "cd engine && uv run python -m daypart.eval --out ../docs/daypart/results/metrics.json"


class MetricsFileError(ValueError):
    """The metrics file is missing, unreadable, or not a valid ``daypart-metrics/v1`` document."""


class Arm(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    kind: str
    status: Literal["ok", "not_run", "failed"]
    reason: str | None
    metrics: dict[str, Any] | None

    @model_validator(mode="after")
    def _status_shape(self) -> "Arm":
        if self.status == "ok":
            if self.metrics is None:
                raise ValueError(f"arm {self.name!r}: status ok requires metrics")
        elif not (self.reason and self.reason.strip()):
            raise ValueError(f"arm {self.name!r}: status {self.status} requires a reason")
        return self


class MetricsDoc(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    schema_: Literal["daypart-metrics/v1"] = pydantic.Field(alias="schema")
    runs_version: str
    runs_digest: str
    cohort_seed: int
    rule_version: str
    guard_versions: dict[str, str]
    arms: tuple[Arm, ...]
    summary: dict[str, int]

    @model_validator(mode="after")
    def _summary_matches(self) -> "MetricsDoc":
        counts = {s: sum(a.status == s for a in self.arms) for s in ("ok", "not_run", "failed")}
        if dict(self.summary) != counts:
            raise ValueError(f"summary {self.summary} does not match arms {counts}")
        return self

    def arm_table(self) -> list[dict[str, str]]:
        """One row per arm, file order. ``reason`` is ``""`` for ok arms."""
        return [{"arm": a.name, "kind": a.kind, "status": a.status, "reason": a.reason or ""} for a in self.arms]


def default_metrics_path() -> Path:
    """``<repo>/docs/daypart/results/metrics.json`` (``$DAYPART_METRICS`` overrides)."""
    env = os.environ.get("DAYPART_METRICS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "docs" / "daypart" / "results" / "metrics.json"


def load_metrics(path: str | os.PathLike | None = None) -> MetricsDoc:
    p = Path(path) if path is not None else default_metrics_path()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise MetricsFileError(f"no metrics file at {p}; generate it with: {REGEN_HINT}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise MetricsFileError(f"cannot read {p}: {exc}") from None
    try:
        return MetricsDoc.model_validate(raw)
    except pydantic.ValidationError as exc:
        raise MetricsFileError(f"{p} is not a valid {SCHEMA} document: {exc}") from None


def env_rows(doc: MetricsDoc) -> list[dict[str, Any]]:
    """Headline numbers for each ``ok`` env_baseline arm, file order."""
    rows = []
    for a in doc.arms:
        if a.kind != "env_baseline" or a.status != "ok":
            continue
        m: Mapping[str, Any] = a.metrics or {}
        per_seed = m.get("per_seed") or []
        rows.append({
            "arm": a.name,
            "policy": m.get("policy"),
            "mean_adherence": m.get("mean_adherence"),
            "episode_violation_rate": m.get("episode_violation_rate"),
            "mean_return": (sum(s["mean_return"] for s in per_seed) / len(per_seed)) if per_seed else None,
            "n_scenarios": m.get("n_scenarios"),
            "horizon": m.get("horizon"),
            "seeds": list(m.get("seeds") or []),
        })
    return rows


__all__ = ["Arm", "MetricsDoc", "MetricsFileError", "REGEN_HINT", "default_metrics_path", "env_rows", "load_metrics"]
