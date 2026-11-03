"""D2 — the metric ontology (CONTRACT slice D, D13, I9, G3).

A frozen core registry (``core-v1``) of LOINC + local metric nodes under local
categories, plus explicit action/metric → metric linkage weights. The four LHM
``V1_DOMAINS`` map onto seed nodes (D13); LOINC nodes stay unmapped (G3).
Hierarchy lives on ``Node.parent``; edges carry weights only.

:func:`apply_update` is the only way to grow an ontology (I9): it is gated,
append-only (core nodes are never replaced or removed), returns a NEW value
object, and records an :class:`UpdateRecord` in ``history``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Literal

import pydantic
from pydantic import ConfigDict, Field, model_validator
from syntropy.lhm.metrics.claims import assert_claim_safe
from syntropy.lhm.metrics.scoring import V1_DOMAINS

from daypart.memory.events import ACTION_CODES

LOINC_SYSTEM = "http://loinc.org"
LOCAL_SYSTEM = "urn:daypart:metric"
CORE_VERSION = "core-v1"
CORE_CAVEATS: tuple[str, str] = (
    "Personal n-of-1 observation: an association in your own data, not causal.",
    "Scores carry an interval; small samples are uncertain.",
)

_LOINC_CODE = re.compile(r"^\d{1,7}-\d$")
_LOCAL_CODE = re.compile(r"^dp:[a-z_]+(\.[a-z_]+)*$")


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


class Node(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    system: Literal["http://loinc.org", "urn:daypart:metric"]
    display: str = Field(min_length=1)
    kind: Literal["category", "metric"]
    parent: str | None
    lhm_domain: str | None = None
    value_range: tuple[float, float] | None = None
    label: str = ""
    caveats: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Node:
        pattern = _LOINC_CODE if self.system == LOINC_SYSTEM else _LOCAL_CODE
        if not pattern.fullmatch(self.code):
            raise ValueError(f"code {self.code!r} invalid for system {self.system}")
        if self.lhm_domain is not None and self.lhm_domain not in V1_DOMAINS:
            raise ValueError(f"lhm_domain {self.lhm_domain!r} not in {V1_DOMAINS}")
        if self.kind == "category":
            if self.lhm_domain is not None or self.value_range is not None:
                raise ValueError("a category carries no lhm_domain or value_range")
        else:
            if self.value_range is None:
                raise ValueError("a metric node requires value_range")
            lo, hi = self.value_range
            if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                raise ValueError(f"value_range {self.value_range!r} must be finite with lo < hi")
            if self.parent is None:
                raise ValueError("a metric node requires a parent")
        return self


class Edge(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str  # a metric-node code or an ACTION_CODES code
    target: str  # a metric-node code
    weight: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)


class OntologyGateError(ValueError):
    """An :class:`OntologyUpdate` was rejected by the I9 gate."""


class UpdateRecord(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True)

    from_version: str
    to_version: str
    proposer: str
    rationale: str
    update_digest: str
    gate: Literal["passed"]


class OntologyUpdate(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    proposer: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    add_nodes: tuple[Node, ...] = ()
    set_edges: tuple[Edge, ...] = ()


class Ontology:
    """Immutable ontology value object; equal iff version, nodes, edges and history are equal."""

    __slots__ = ("_core_codes", "edges", "history", "nodes", "version")

    version: str
    nodes: Mapping[str, Node]
    edges: tuple[Edge, ...]
    history: tuple[UpdateRecord, ...]

    def __init__(
        self,
        version: str,
        nodes: Iterable[Node],
        edges: Iterable[Edge],
        history: tuple[UpdateRecord, ...] = (),
        *,
        core_codes: frozenset[str] | None = None,
    ) -> None:
        node_map = {n.code: n for n in sorted(nodes, key=lambda n: n.code)}
        edge_map = {(e.source, e.target): e for e in edges}
        set_ = object.__setattr__
        set_(self, "version", version)
        set_(self, "nodes", MappingProxyType(node_map))
        set_(self, "edges", tuple(edge_map[k] for k in sorted(edge_map)))
        set_(self, "history", tuple(history))
        set_(self, "_core_codes", frozenset(node_map) if core_codes is None else core_codes)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Ontology is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Ontology is immutable")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ontology):
            return NotImplemented
        return (
            self.version == other.version
            and dict(self.nodes) == dict(other.nodes)
            and self.edges == other.edges
            and self.history == other.history
            and self._core_codes == other._core_codes
        )

    def __hash__(self) -> int:
        return hash((self.digest(), self.history))

    def __repr__(self) -> str:
        return f"Ontology(version={self.version!r}, nodes={len(self.nodes)}, edges={len(self.edges)})"

    def node(self, code: str) -> Node:
        return self.nodes[code]

    def is_core(self, code: str) -> bool:
        return code in self._core_codes

    def metric_nodes(self) -> tuple[Node, ...]:
        return tuple(n for code, n in sorted(self.nodes.items()) if n.kind == "metric")

    def weight(self, source: str, target: str) -> float | None:
        for e in self.edges:
            if e.source == source and e.target == target:
                return e.weight
        return None

    def digest(self) -> str:
        return _sha256(
            {
                "version": self.version,
                "nodes": [n.model_dump(mode="json") for _, n in sorted(self.nodes.items())],
                "edges": [e.model_dump(mode="json") for e in self.edges],
            }
        )


# --------------------------------------------------------------------------- #
# D2.1 core registry
# --------------------------------------------------------------------------- #


def _category(code: str, display: str) -> Node:
    return Node(code=code, system=LOCAL_SYSTEM, display=display, kind="category", parent=None)


def _metric(
    code: str,
    system: str,
    display: str,
    parent: str,
    value_range: tuple[float, float],
    label: str,
    lhm_domain: str | None = None,
) -> Node:
    return Node(
        code=code,
        system=system,  # type: ignore[arg-type]
        display=display,
        kind="metric",
        parent=parent,
        lhm_domain=lhm_domain,
        value_range=value_range,
        label=label,
        caveats=CORE_CAVEATS,
    )


_L, _D = LOINC_SYSTEM, LOCAL_SYSTEM

_CORE_NODES: tuple[Node, ...] = (
    _category("dp:behavior", "Behavior"),
    _category("dp:affect", "Affect"),
    _category("dp:symptom", "Symptom"),
    _category("dp:nutrition", "Nutrition"),
    _category("dp:vital", "Vital sign"),
    _category("dp:activity", "Activity"),
    _category("dp:sleep", "Sleep"),
    _metric("dp:adherence.protocol", _D, "Protocol adherence", "dp:behavior", (0.0, 1.0),
            "Protocol adherence (logged)", "consistency"),
    _metric("dp:affect.checkin_sentiment", _D, "Check-in sentiment", "dp:affect", (-1.0, 1.0),
            "Check-in sentiment", "sentiment_trajectory"),
    _metric("dp:nutrition.food_log", _D, "Food log regularity", "dp:nutrition", (0.0, 1.0),
            "Food log regularity", "nutrition_regularity"),
    _metric("dp:symptom.severity", _D, "Symptom severity (ordinal)", "dp:symptom", (1.0, 3.0),
            "Symptom severity (self-reported)", "symptom_load"),
    _metric("72514-3", _L, "Pain severity - 0-10 verbal numeric rating [Score] - Reported",
            "dp:symptom", (0.0, 10.0), "Pain rating (0-10)"),
    _metric("44261-6", _L, "Patient Health Questionnaire 9 item (PHQ-9) total score [Reported]",
            "dp:affect", (0.0, 27.0), "PHQ-9 total score"),
    _metric("70274-6", _L, "Generalized anxiety disorder 7 item (GAD-7) total score [Reported.PHQ]",
            "dp:affect", (0.0, 21.0), "GAD-7 total score"),
    _metric("8867-4", _L, "Heart rate", "dp:vital", (20.0, 250.0), "Heart rate"),
    _metric("29463-7", _L, "Body weight", "dp:vital", (0.0, 500.0), "Body weight"),
    _metric("55423-8", _L, "Number of steps in unspecified time Pedometer", "dp:activity",
            (0.0, 100000.0), "Step count"),
    _metric("93832-4", _L, "Sleep duration", "dp:sleep", (0.0, 24.0), "Sleep duration (hours)"),
)

_CORE_EDGES: tuple[Edge, ...] = tuple(
    Edge(source=s, target=t, weight=w)
    for s, t, w in (
        ("act:reminder_sent", "dp:adherence.protocol", 0.3),
        ("act:nudge_sent", "dp:adherence.protocol", 0.2),
        ("act:goal_set", "dp:adherence.protocol", 0.2),
        ("act:education_viewed", "dp:adherence.protocol", 0.1),
        ("act:protocol_adjusted", "dp:adherence.protocol", 0.1),
        ("act:reply_sent", "dp:affect.checkin_sentiment", 0.1),
        ("93832-4", "dp:affect.checkin_sentiment", 0.2),
        ("55423-8", "dp:affect.checkin_sentiment", 0.1),
        ("72514-3", "dp:affect.checkin_sentiment", -0.3),
        ("dp:symptom.severity", "dp:affect.checkin_sentiment", -0.3),
    )
)


def core() -> Ontology:
    """The frozen core registry (``CORE_VERSION``); equal on every call."""
    return Ontology(CORE_VERSION, _CORE_NODES, _CORE_EDGES, ())


# --------------------------------------------------------------------------- #
# D2.2 gated updates (I9)
# --------------------------------------------------------------------------- #


def apply_update(ontology: Ontology, update: OntologyUpdate) -> Ontology:
    """Return a NEW ontology with ``update`` applied, or raise :class:`OntologyGateError`."""
    if not update.add_nodes and not update.set_edges:
        raise OntologyGateError("empty update")

    nodes: dict[str, Node] = dict(ontology.nodes)
    for n in update.add_nodes:
        if n.code in nodes:
            raise OntologyGateError(f"node {n.code!r} already exists (nodes are never replaced)")
        if n.parent is not None and n.parent not in nodes:
            raise OntologyGateError(f"node {n.code!r} has unknown parent {n.parent!r}")
        if n.kind == "metric":
            try:
                assert_claim_safe(n.label or n.display, n.caveats, "low", False)
            except Exception as exc:  # ClaimScopeError or anything unexpected: fail closed
                raise OntologyGateError(f"node {n.code!r} fails the claim gate: {exc}") from exc
        nodes[n.code] = n

    def is_metric(code: str) -> bool:
        node = nodes.get(code)
        return node is not None and node.kind == "metric"

    seen: set[tuple[str, str]] = set()
    for e in update.set_edges:
        if not is_metric(e.target):
            raise OntologyGateError(f"edge target {e.target!r} is not a metric node")
        if not (is_metric(e.source) or e.source in ACTION_CODES):
            raise OntologyGateError(f"edge source {e.source!r} is neither a metric node nor an action")
        key = (e.source, e.target)
        if key in seen:
            raise OntologyGateError(f"edge {key!r} appears twice in the update")
        seen.add(key)

    edges = {(e.source, e.target): e for e in ontology.edges}
    edges.update({(e.source, e.target): e for e in update.set_edges})

    n_updates = len(ontology.history) + 1
    to_version = f"{CORE_VERSION}+u{n_updates}"
    record = UpdateRecord(
        from_version=ontology.version,
        to_version=to_version,
        proposer=update.proposer,
        rationale=update.rationale,
        update_digest=_sha256(update.model_dump(mode="json")),
        gate="passed",
    )
    return Ontology(
        to_version,
        nodes.values(),
        edges.values(),
        (*ontology.history, record),
        core_codes=ontology._core_codes,
    )


__all__ = [
    "CORE_CAVEATS",
    "CORE_VERSION",
    "LOCAL_SYSTEM",
    "LOINC_SYSTEM",
    "Edge",
    "Node",
    "Ontology",
    "OntologyGateError",
    "OntologyUpdate",
    "UpdateRecord",
    "apply_update",
    "core",
]
