"""R5 — the A4 concept registry (A&D §4.3, D-A5; build plan G3-G5).

One :class:`Concept` per observable in the A4 slice, loaded from ``registry_a4.json``.
Every LOINC concept is pinned to a row of ``loinc_verify_v1.csv`` (verified against
LOINC :data:`LOINC_VERSION`; T5.6). :func:`slice_ontology` grows ``core()`` by exactly
one gated :class:`~daypart.memory.ontology.OntologyUpdate` that adds the LOINC nodes not
already in core (version :data:`SLICE_VERSION`).

``plausible`` bounds are input validation only (the EventLog ``value_range``); they are
never rendered and carry no clinical meaning.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from functools import cache
from importlib import resources
from types import MappingProxyType
from typing import Literal

import pydantic
from pydantic import (
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from daypart.memory.ontology import (
    CORE_CAVEATS,
    LOINC_SYSTEM,
    Node,
    Ontology,
    OntologyUpdate,
    apply_update,
    core,
)

LOINC_VERSION = "2.83"
SLICE_VERSION = "core-v1+u1"
LOINC_NOTICE = (
    "This material contains content from LOINC® (https://loinc.org). LOINC is copyright © "
    "1995-2026, Regenstrief Institute, Inc. and the Logical Observation Identifiers Names and Codes (LOINC) "
    "Committee and is available at no cost under the license at https://loinc.org/license/. "
    "LOINC® is a registered United States trademark of Regenstrief Institute, Inc."
)
REVIEWER_STATUS = "draft, not clinically reviewed"
PROPOSER = "daypart.memory.registry"

Vertical = Literal["vitals_body", "behaviour_function", "symptoms_wellbeing"]
VERTICALS: tuple[str, ...] = ("vitals_body", "behaviour_function", "symptoms_wellbeing")

# G5: parents of the new (non-core) LOINC nodes; vitals_body defaults to dp:vital.
_PARENT_OVERRIDES = {"55423-8": "dp:activity", "41950-7": "dp:activity", "55411-3": "dp:activity",
                     "55758-7": "dp:affect"}
# Short labels for the nodes the slice adds (must pass the claim gate in apply_update).
_LABELS = {
    "8480-6": "Systolic blood pressure",
    "8462-4": "Diastolic blood pressure",
    "8302-2": "Body height",
    "39156-5": "Body mass index",
    "8310-5": "Body temperature",
    "9279-1": "Respiratory rate",
    "59408-5": "Oxygen saturation (pulse oximetry)",
    "8280-0": "Waist circumference",
    "41950-7": "Steps in 24 hours",
    "55411-3": "Exercise duration",
    "55758-7": "PHQ-2 total score",
}


class Concept(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    concept_id: str = Field(pattern=r"^chc:[a-z0-9][a-z0-9-]*$")
    code: str = Field(min_length=1)
    system: Literal["loinc", "local"]
    loinc_version: str | None
    display: str = Field(min_length=1)
    vertical: Vertical
    unit: str = Field(min_length=1)
    alt_units: Mapping[str, float] = Field(default_factory=dict)  # read-only after validation
    value_type: Literal["quantity", "integer", "ordinal"]
    plausible: tuple[float, float]
    keywords: tuple[str, ...] = Field(min_length=1)
    reviewer_status: Literal["draft, not clinically reviewed"]
    allowed_use: str = Field(min_length=1)

    @field_validator("alt_units", mode="after")
    @classmethod
    def _freeze_alt_units(cls, v: Mapping[str, float]) -> Mapping[str, float]:
        return MappingProxyType(dict(v))  # the registry is a shared cache: no caller may mutate it

    @field_serializer("alt_units")
    def _dump_alt_units(self, v: Mapping[str, float]) -> dict[str, float]:
        return dict(v)

    @model_validator(mode="after")
    def _check(self) -> Concept:
        if (self.system == "loinc") != (self.loinc_version is not None):
            raise ValueError("loinc_version must be set iff system == 'loinc'")
        if self.system == "local" and not self.code.startswith("dp:"):
            raise ValueError(f"local code {self.code!r} must start with 'dp:'")
        lo, hi = self.plausible
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"plausible {self.plausible!r} must be finite with lo < hi")
        for unit, factor in self.alt_units.items():
            if not (math.isfinite(factor) and factor > 0):
                raise ValueError(f"alt_units[{unit!r}] must be a positive finite factor")
        return self


class ConceptRegistry:
    """Read-only lookup over the A4 concepts."""

    def __init__(self, concepts: tuple[Concept, ...], gaps: tuple[str, ...], loinc_version: str) -> None:
        by_code: dict[str, Concept] = {}
        by_id: dict[str, Concept] = {}
        for c in concepts:
            if c.code in by_code or c.concept_id in by_id:
                raise ValueError(f"duplicate concept {c.concept_id!r} / {c.code!r}")
            if c.system == "loinc" and c.loinc_version != loinc_version:
                raise ValueError(f"{c.code}: loinc_version {c.loinc_version!r} != {loinc_version!r}")
            by_code[c.code] = c
            by_id[c.concept_id] = c
        self._concepts = concepts
        self._by_code = MappingProxyType(by_code)
        self._by_id = MappingProxyType(by_id)
        self.gaps = gaps
        self.loinc_version = loinc_version

    def concepts(self) -> tuple[Concept, ...]:
        return self._concepts

    def by_code(self, code: str) -> Concept | None:
        return self._by_code.get(code)

    def by_concept_id(self, concept_id: str) -> Concept | None:
        return self._by_id.get(concept_id)

    def by_vertical(self, vertical: str) -> tuple[Concept, ...]:
        return tuple(c for c in self._concepts if c.vertical == vertical)

    def for_keyword(self, keyword: str) -> Concept | None:
        """The first concept (registry order) whose keyword list contains ``keyword`` (case-insensitive)."""
        k = keyword.strip().lower()
        for c in self._concepts:
            if k in (w.lower() for w in c.keywords):
                return c
        return None


@cache
def load_registry() -> ConceptRegistry:
    raw = json.loads(resources.files("daypart.memory").joinpath("registry_a4.json").read_text("utf-8"))
    if raw["loinc_version"] != LOINC_VERSION:
        raise ValueError(f"registry loinc_version {raw['loinc_version']!r} != {LOINC_VERSION!r}")
    concepts = tuple(Concept.model_validate(c) for c in raw["concepts"])
    return ConceptRegistry(concepts, tuple(raw["gaps"]), raw["loinc_version"])


@cache
def slice_update() -> OntologyUpdate:
    base = core()
    nodes = []
    for c in load_registry().concepts():
        if c.system != "loinc" or c.code in base.nodes:
            continue
        nodes.append(
            Node(
                code=c.code,
                system=LOINC_SYSTEM,
                display=c.display,
                kind="metric",
                parent=_PARENT_OVERRIDES.get(c.code, "dp:vital"),
                lhm_domain=None,
                value_range=c.plausible,
                label=_LABELS.get(c.code, c.display),
                caveats=CORE_CAVEATS,
            )
        )
    return OntologyUpdate(
        proposer=PROPOSER,
        rationale=f"A4 concept slice: LOINC {LOINC_VERSION} codes verified 2026-11-06 not already in core-v1",
        add_nodes=tuple(nodes),
    )


@cache
def slice_ontology() -> Ontology:
    o = apply_update(core(), slice_update())
    assert o.version == SLICE_VERSION
    return o


__all__ = [
    "LOINC_NOTICE",
    "LOINC_VERSION",
    "REVIEWER_STATUS",
    "SLICE_VERSION",
    "VERTICALS",
    "Concept",
    "ConceptRegistry",
    "load_registry",
    "slice_ontology",
    "slice_update",
]
