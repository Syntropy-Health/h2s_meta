"""D3 — per-member LHM metric state (CONTRACT slice D, I6, I7, G3, D14).

Builds one :class:`NodeState` per ontology metric node from adapter series and
``SignalEvent``s (all restricted to ``<= as_of``), scored by LHM's
``score_domain`` and surfaced only when ``is_ready`` passes with both the claim
and valence gates (I6: no number unless ready). Unmapped nodes are never ready
(G3). ``ActionEvent``s never feed a node (D14); they only yield ``alignment``.
Only the opaque ``member_id`` ever reaches the adapter (I7).
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType

from syntropy.lhm.metrics.claims import (
    CLAIMS_GUARD_VERSION,
    VALENCE_GUARD_VERSION,
    ScoreRendering,
    assert_claim_safe,
    assert_valence_safe,
)
from syntropy.lhm.metrics.scoring import DomainScore, is_ready, score_domain
from syntropy.lhm.models.mapping import OrderRecord, RegimenStartRecord, SeriesRecord
from syntropy.lhm.tools.pipeline import CandidateSignal, LhmAdapter, analyze, fetch

from daypart.memory.events import MEMBER_ID_RE, ActionEvent, EventLog, SignalEvent
from daypart.memory.ontology import Node, Ontology, core

PRIMARY_NODE = "dp:adherence.protocol"
SERIES_NODES: dict[str, str] = {
    "Protocol adherence (logged)": "dp:adherence.protocol",
    "Check-in sentiment": "dp:affect.checkin_sentiment",
}
GUARD_VERSIONS: dict[str, str] = {"claims": CLAIMS_GUARD_VERSION, "valence": VALENCE_GUARD_VERSION}

_MEMBER_ID = re.compile(MEMBER_ID_RE)

Sample = tuple[datetime, float]


@dataclass(frozen=True)
class NodeState:
    code: str
    lhm_domain: str | None
    score: DomainScore | None  # None iff lhm_domain is None
    ready: bool
    value: float | None  # == score.score iff ready, else None (I6)


@dataclass(frozen=True)
class MemberMetricState:
    member_id: str
    as_of: datetime
    ontology_version: str
    nodes: Mapping[str, NodeState]
    signals: tuple[CandidateSignal, ...]
    alignment: Mapping[str, float]
    guard_versions: Mapping[str, str]

    def ready_codes(self) -> tuple[str, ...]:
        return tuple(sorted(code for code, st in self.nodes.items() if st.ready))

    def surfaced(self) -> dict[str, float | None]:
        return {code: (st.value if st.ready else None) for code, st in self.nodes.items()}


def alignment_score(
    event: ActionEvent, ontology: Ontology, goal_nodes: Sequence[str] = (PRIMARY_NODE,)
) -> float:
    """Mean edge weight from the action's code to each goal node (missing edge → 0.0)."""
    if not goal_nodes:
        return 0.0
    return sum(ontology.weight(event.code, g) or 0.0 for g in goal_nodes) / len(goal_nodes)


def _require_aware(dt: datetime) -> None:
    if not isinstance(dt, datetime) or dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("as_of must be a timezone-aware datetime")


class _AsOfView:
    """An ``LhmAdapter`` view dropping every record/point timestamped after ``as_of``."""

    def __init__(self, adapter: LhmAdapter, as_of: datetime) -> None:
        self._adapter = adapter
        self._as_of = as_of

    async def read_orders(self, user_scope: str) -> Sequence[OrderRecord]:
        return tuple(o for o in await self._adapter.read_orders(user_scope) if o.occurred_at <= self._as_of)

    async def read_regimen_starts(self, user_scope: str) -> Sequence[RegimenStartRecord]:
        starts = await self._adapter.read_regimen_starts(user_scope)
        return tuple(s for s in starts if s.started_at <= self._as_of)

    async def read_outcome_series(self, user_scope: str) -> Sequence[SeriesRecord]:
        return tuple(
            dataclasses.replace(rec, points=tuple(p for p in rec.points if p.observed_at <= self._as_of))
            for rec in await self._adapter.read_outcome_series(user_scope)
        )


def _claim_gate(node: Node) -> bool:
    try:
        assert_claim_safe(node.label or node.display, node.caveats, "low", False)
    except Exception:  # fail closed
        return False
    return True


def _valence_gate(domain: str, score: float) -> bool:
    try:
        assert_valence_safe(ScoreRendering(domain=domain, score=score))
    except Exception:  # fail closed
        return False
    return True


def _node_state(node: Node, samples: Sequence[Sample], as_of: datetime) -> NodeState:
    if node.lhm_domain is None:
        return NodeState(node.code, None, score=None, ready=False, value=None)
    score = score_domain(node.lhm_domain, samples, as_of=as_of)
    ready = is_ready(
        score,
        node.lhm_domain,
        claim_gate_passed=_claim_gate(node),
        valence_gate_passed=_valence_gate(node.lhm_domain, score.score),
    )
    return NodeState(node.code, node.lhm_domain, score=score, ready=ready, value=score.score if ready else None)


class MemoryLhm:
    def __init__(
        self,
        adapter: LhmAdapter,
        *,
        ontology: Ontology | None = None,
        events: EventLog | None = None,
        goal_nodes: Sequence[str] = (PRIMARY_NODE,),
    ) -> None:
        self._adapter = adapter
        self._ontology = ontology if ontology is not None else core()
        self._events = events if events is not None else EventLog(self._ontology)
        self._goal_nodes = tuple(goal_nodes)

    async def metric_state(self, member_id: str, as_of: datetime) -> MemberMetricState:
        if not isinstance(member_id, str) or not _MEMBER_ID.fullmatch(member_id):
            raise ValueError("member_id must be an opaque cohort id (I7)")
        _require_aware(as_of)

        ont = self._ontology
        metric_nodes = ont.metric_nodes()
        samples: dict[str, list[Sample]] = {n.code: [] for n in metric_nodes}
        view = _AsOfView(self._adapter, as_of)

        sentiment_fn = self._adapter.sentiment_fn  # type: ignore[attr-defined]
        for rec in await view.read_outcome_series(member_id):
            code = SERIES_NODES.get(rec.metric_display)
            if code is None or code not in samples:
                continue
            for p in rec.points:
                if p.value is not None:
                    samples[code].append((p.observed_at, float(p.value)))
                elif p.note is not None:
                    samples[code].append((p.observed_at, float(sentiment_fn(p.note))))

        member_events = self._events.events(member_id, as_of=as_of)
        for e in member_events:
            if isinstance(e, SignalEvent) and e.code in samples:  # actions feed no node (D14)
                samples[e.code].append((e.occurred_at, e.value))

        nodes = {n.code: _node_state(n, samples[n.code], as_of) for n in metric_nodes}
        signals = await analyze(await fetch(view, member_id))
        alignment = {
            e.event_id: alignment_score(e, ont, self._goal_nodes)
            for e in member_events
            if isinstance(e, ActionEvent)
        }
        return MemberMetricState(
            member_id=member_id,
            as_of=as_of,
            ontology_version=ont.version,
            nodes=MappingProxyType(nodes),
            signals=tuple(signals),
            alignment=MappingProxyType(alignment),
            guard_versions=MappingProxyType(dict(GUARD_VERSIONS)),
        )


__all__ = [
    "GUARD_VERSIONS",
    "PRIMARY_NODE",
    "SERIES_NODES",
    "MemberMetricState",
    "MemoryLhm",
    "NodeState",
    "alignment_score",
]
