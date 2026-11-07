"""R6 — the evidence registry (A&D §4.6, D-A6, AM-1; build plan task 6, addendum G22).

One :class:`Source` per public guideline or reference page, loaded from ``sources_v1.json``.
``supports`` says *why a measure is tracked*. A source that states a numeric range records it in
``stated_ranges`` with the page's own sentence (``statement``), so every number in a concept's
``reference_range`` (and in ``supports``) traces to its entry. :func:`load_evidence` refuses a
registry where a concept carries a range no source states, or a source states a range its concept
drops. A range is shown beside a value without interpretation (A2); ``plausible`` bounds are
validation only and never appear here.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from functools import cache
from importlib import resources
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlparse

import pydantic
from pydantic import ConfigDict, Field, field_validator, model_validator

from daypart.memory.registry import ConceptRegistry, load_registry

# AM-1 (board answer on A3, 2026-11-07). No clinician reviewed these, so the copy never says so.
VETTING = "reviewed: board sign-off 2026-11-07"
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# AM-1 / R1: no copy may claim a clinical or expert review, or "vetted" sources (spacing and hyphens folded).
_FORBIDDEN_COPY = re.compile(
    r"(?:clinic(?:al(?:ly)?|ian)|medical(?:ly)?|physician|doctor|expert)\s*(?:review|vet|approv)|\bvetted\b"
)
_PRIVATE_SUFFIXES = (
    ".local", ".internal", ".lan", ".localhost", ".corp", ".home", ".svc", ".arpa", ".test", ".invalid",
    ".example", ".intranet", ".private",
)
_IP_DNS_SUFFIXES = (".nip.io", ".sslip.io", ".xip.io")  # wildcard DNS that maps a name to an IP
_NUMERIC_HOST = re.compile(r"^(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+))*$")  # 127.1, 0x7f.0.0.1


def _numbers(text: str) -> set[float]:
    return {float(n) for n in _NUMBER.findall(text)}


def _no_forbidden_copy(text: str) -> str:
    folded = re.sub(r"[\s\-]+", " ", text.lower())
    m = _FORBIDDEN_COPY.search(folded)
    if m:
        raise ValueError(f"copy may not say {m.group(0)!r} (AM-1, R1)")
    return text


class StatedRange(pydantic.BaseModel):
    """A range as the source states it. ``statement`` is the page's own sentence and holds every bound."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    concept_id: str = Field(pattern=r"^chc:[a-z0-9][a-z0-9-]*$")
    low: float | None
    high: float | None
    unit: str = Field(min_length=1)
    population: str = Field(min_length=1)
    statement: str = Field(min_length=1, max_length=300)

    @field_validator("population", "statement")
    @classmethod
    def _copy(cls, v: str) -> str:
        return _no_forbidden_copy(v)

    @model_validator(mode="after")
    def _check(self) -> StatedRange:
        if self.low is None and self.high is None:
            raise ValueError("a stated range needs at least one bound")
        stated = _numbers(self.statement)
        for bound in (self.low, self.high):
            if bound is None:
                continue
            if not math.isfinite(bound):
                raise ValueError("stated bounds must be finite")
            if bound not in stated:
                raise ValueError(f"bound {bound} does not appear in the source statement")
        if self.low is not None and self.high is not None and not self.low < self.high:
            raise ValueError("stated low must be below high")
        return self


class Source(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str = Field(pattern=r"^ev:[a-z0-9][a-z0-9-]*$")
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    url: str
    accessed: date
    supports: str = Field(min_length=1, max_length=200)
    applies_to: tuple[str, ...] = Field(min_length=1)
    vetting: Literal["reviewed: board sign-off 2026-11-07"]
    stated_ranges: tuple[StatedRange, ...] = ()

    @field_validator("title", "publisher", "supports")
    @classmethod
    def _copy(cls, v: str) -> str:
        return _no_forbidden_copy(v)

    @field_validator("url")
    @classmethod
    def _public_https(cls, v: str) -> str:
        u = urlparse(v)
        if u.scheme != "https":
            raise ValueError("url must be a public https page")
        if u.username is not None or u.password is not None or u.port not in (None, 443):
            raise ValueError("url may not carry credentials or a port")
        host = (u.hostname or "").lower().rstrip(".")
        if "." not in host or host.endswith(_PRIVATE_SUFFIXES + _IP_DNS_SUFFIXES):
            raise ValueError("url host is not a public name")
        if _NUMERIC_HOST.match(host) or ":" in host:
            raise ValueError("url host must be a public name, not an IP address")
        return v

    @model_validator(mode="after")
    def _check(self) -> Source:
        if len(set(self.applies_to)) != len(self.applies_to):
            raise ValueError("applies_to has duplicates")
        seen: set[str] = set()
        for r in self.stated_ranges:
            if r.concept_id not in self.applies_to:
                raise ValueError(f"stated range for {r.concept_id} is not in applies_to")
            if r.concept_id in seen:
                raise ValueError(f"two stated ranges for {r.concept_id}")
            seen.add(r.concept_id)
        stated = set().union(*(_numbers(r.statement) for r in self.stated_ranges))
        if not _numbers(self.supports) <= stated:
            raise ValueError("every number in supports must appear in a stated range of this entry (AM-1)")
        return self


class EvidenceRegistry:
    """Read-only lookup over the sources."""

    def __init__(self, sources: tuple[Source, ...], concepts: ConceptRegistry | None = None) -> None:
        by_id: dict[str, Source] = {}
        for s in sources:
            if s.evidence_id in by_id:
                raise ValueError(f"duplicate evidence_id {s.evidence_id!r}")
            by_id[s.evidence_id] = s
        self.sources = sources
        self._by_id = MappingProxyType(by_id)
        self._concepts = concepts

    def by_id(self, evidence_id: str) -> Source | None:
        return self._by_id.get(evidence_id)

    def for_concept(self, code: str) -> tuple[Source, ...]:
        """Sources that apply to ``code``: a concept_id (``chc:…``) or, with a registry, its LOINC/local code."""
        concept_id = code
        if not code.startswith("chc:"):
            if self._concepts is None:
                raise ValueError("looking up a code needs a concept registry (use load_evidence())")
            c = self._concepts.by_code(code)
            if c is None:
                return ()
            concept_id = c.concept_id
        return tuple(s for s in self.sources if concept_id in s.applies_to)


def check_reference_ranges(concepts: ConceptRegistry, evidence: EvidenceRegistry) -> None:
    """Refuse any range that is not exactly what its source states, and any stated range that is dropped."""
    for c in concepts.concepts():
        rr = c.reference_range
        if rr is None:
            continue
        src = evidence.by_id(rr.source_evidence_id)
        if src is None:
            raise ValueError(f"{c.concept_id}: reference_range source {rr.source_evidence_id!r} does not resolve")
        match = [r for r in src.stated_ranges if r.concept_id == c.concept_id]
        if not match or (match[0].low, match[0].high, match[0].unit, match[0].population) != (
            rr.low, rr.high, rr.unit, rr.population
        ):
            raise ValueError(f"{c.concept_id}: reference_range is not stated by {src.evidence_id}")
    for s in evidence.sources:
        for cid in s.applies_to:
            if concepts.by_concept_id(cid) is None:
                raise ValueError(f"{s.evidence_id}: applies_to {cid!r} is not a registry concept")
        for r in s.stated_ranges:
            c = concepts.by_concept_id(r.concept_id)
            if c is None or c.reference_range is None or c.reference_range.source_evidence_id != s.evidence_id:
                raise ValueError(f"{r.concept_id}: stated by {s.evidence_id} but not carried as its reference_range")


@cache
def load_evidence() -> EvidenceRegistry:
    raw = json.loads(resources.files("daypart.evidence").joinpath("sources_v1.json").read_text("utf-8"))
    concepts = load_registry()
    evidence = EvidenceRegistry(tuple(Source.model_validate(s) for s in raw["sources"]), concepts)
    check_reference_ranges(concepts, evidence)
    return evidence


__all__ = [
    "VETTING",
    "EvidenceRegistry",
    "Source",
    "StatedRange",
    "check_reference_ranges",
    "load_evidence",
]
