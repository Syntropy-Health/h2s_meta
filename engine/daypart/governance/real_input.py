"""Real-input gate (AD7, A&D §4.9). Hard off in this build.

Real input needs BOTH ``REAL_INPUT_BUILD_ALLOWED`` (a code constant, ``False`` here) AND every
checklist item closed. Closing the checklist alone never enables it: changing the constant needs
a code change, a revised PVR and a new design. ``request_enable`` therefore always refuses.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict

REAL_INPUT_BUILD_ALLOWED: Final = False

ENV_VAR: Final = "DAYPART_REAL_INPUT"
_TRUTHY: Final = frozenset({"1", "on", "true"})

GATE_IDS: Final = (
    "lawful_basis",
    "consent",
    "access_control",
    "encryption",
    "retention_deletion",
    "audit",
    "incident_response",
    "processor_terms",
    "jurisdiction_review",
    "clinical_intended_use",
    "qualified_oversight",
    "regulatory_review",
)


class Gate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    label: str
    status: Literal["open", "closed"]
    owner: str  # a name, or "unassigned"
    evidence: str | None = None


class RealInputStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool
    build_allowed: bool
    open_gates: tuple[Gate, ...]
    requested: bool = False


class Refused(BaseModel):
    model_config = ConfigDict(frozen=True)
    reason: Literal["build_disallowed", "open_gates"]
    open_gates: tuple[Gate, ...]


def load_checklist(path: str | Path | None = None) -> tuple[Gate, ...]:
    """Load and validate the checklist: exactly the 12 §4.9 ids, in order."""
    if path is None:
        text = files("daypart.governance").joinpath("real_input.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict) or not isinstance(data.get("gates"), list):
        raise TypeError("real_input checklist must be a mapping with a 'gates' list")
    gates = tuple(Gate.model_validate(item) for item in data["gates"])
    ids = tuple(g.id for g in gates)
    if ids != GATE_IDS:
        raise ValueError(f"real_input checklist ids must be exactly {list(GATE_IDS)} in order; got {list(ids)}")
    return gates


def _resolve(checklist: tuple[Gate, ...] | None) -> tuple[Gate, ...]:
    return load_checklist() if checklist is None else tuple(checklist)


def status(checklist: tuple[Gate, ...] | None = None, *, requested: bool = False) -> RealInputStatus:
    gates = _resolve(checklist)
    open_gates = tuple(g for g in gates if g.status == "open")
    return RealInputStatus(
        enabled=REAL_INPUT_BUILD_ALLOWED and not open_gates,
        build_allowed=REAL_INPUT_BUILD_ALLOWED,
        open_gates=open_gates,
        requested=requested,
    )


def request_enable(checklist: tuple[Gate, ...] | None = None) -> Refused:
    """Always refuses. In this build the reason is always ``build_disallowed`` (T9.3)."""
    s = status(checklist)
    reason: Literal["build_disallowed", "open_gates"] = "build_disallowed" if not s.build_allowed else "open_gates"
    return Refused(reason=reason, open_gates=s.open_gates)


def requested_from_env(environ: Mapping[str, str] = os.environ) -> bool:
    return environ.get(ENV_VAR, "").strip().lower() in _TRUTHY


__all__ = [
    "ENV_VAR",
    "GATE_IDS",
    "REAL_INPUT_BUILD_ALLOWED",
    "Gate",
    "RealInputStatus",
    "Refused",
    "load_checklist",
    "request_enable",
    "requested_from_env",
    "status",
]
