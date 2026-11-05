"""E2.2 — metric state → passport cards (I6 no number unless ready; fail closed)."""

from __future__ import annotations

import math
import re
from datetime import datetime

from syntropy.lhm.contracts.viz import CONTRACT_ID, VizPayload, VizPoint
from syntropy.lhm.metrics.claims import assert_claim_safe, assert_narration_safe

from daypart.memory.lhm import MemberMetricState, MemoryLhm, NodeState
from daypart.memory.ontology import Node, Ontology, core
from daypart.passport.constants import FALLBACK_TITLE, WITHHELD_MESSAGE
from daypart.passport.models import Flag, Passport

_NON_KEY = re.compile(r"[^a-z0-9]")


def metric_key(code: str) -> str:
    """``dp:adherence.protocol`` → ``dp_adherence_protocol``; ``72514-3`` → ``m_72514_3``."""
    key = _NON_KEY.sub("_", code.lower())
    return key if key[:1].isalpha() and key[:1].isascii() else f"m_{key}"


def _is_safe(text: str) -> bool:
    try:
        assert_narration_safe(text)
    except Exception:
        return False
    return True


def _title(node: Node | None) -> str:
    if node is None:
        return FALLBACK_TITLE
    text = node.label or node.display
    return text if text and _is_safe(text) else FALLBACK_TITLE


def _withheld(code: str, lhm_domain: str, title: str) -> Flag:
    return Flag(
        code=code,
        lhm_domain=lhm_domain,
        title=title,
        status="withheld",
        value=None,
        interval=None,
        caveats=(),
        message=WITHHELD_MESSAGE,
        payload=None,
    )


def _ready(code: str, lhm_domain: str, title: str, node: Node, ns: NodeState, as_of: datetime) -> Flag:
    value = ns.value
    if value is None or not math.isfinite(value):
        raise ValueError("ready node without a finite value")
    caveats = tuple(node.caveats)
    if not caveats or not all(_is_safe(c) for c in caveats):
        raise ValueError("caveats fail the narration guard")
    assert_claim_safe(title, caveats, "low", False)
    payload = VizPayload(
        contract=CONTRACT_ID,
        metric_key=metric_key(code),
        label=title,
        series=(VizPoint(ts=as_of, value=value),),
        regimen_start=None,
        confidence="low",
        caveats=caveats,
        rx_input=False,
    )
    if VizPayload.model_validate(payload.model_dump(mode="json")) != payload:
        raise ValueError("viz payload does not round-trip")
    interval = ns.score.interval if ns.score is not None else None
    return Flag(
        code=code,
        lhm_domain=lhm_domain,
        title=title,
        status="ready",
        value=value,
        interval=interval,
        caveats=caveats,
        message=None,
        payload=payload,
    )


def _flag(code: str, ns: NodeState, ontology: Ontology, as_of: datetime) -> Flag:
    assert ns.lhm_domain is not None
    node = ontology.nodes.get(code)
    title = _title(node)
    if ns.ready and node is not None:
        try:
            return _ready(code, ns.lhm_domain, title, node, ns, as_of)
        except Exception:  # fail closed: a card that cannot be gated shows no number
            pass
    return _withheld(code, ns.lhm_domain, title)


def flags_from_state(state: MemberMetricState, ontology: Ontology | None = None) -> tuple[Flag, ...]:
    """One card per mapped metric node in ``state``, sorted by code (unmapped nodes never shown)."""
    ont = ontology if ontology is not None else core()
    return tuple(
        _flag(code, ns, ont, state.as_of)
        for code, ns in sorted(state.nodes.items())
        if ns.lhm_domain is not None
    )


def passport_from_state(state: MemberMetricState, ontology: Ontology | None = None) -> Passport:
    return Passport(
        member_id=state.member_id,
        as_of=state.as_of,
        ontology_version=state.ontology_version,
        guard_versions=dict(state.guard_versions),
        flags=flags_from_state(state, ontology),
    )


class PassportRenderer:
    """Renders passport cards for a member from :class:`MemoryLhm` metric state."""

    def __init__(self, memory: MemoryLhm, ontology: Ontology | None = None) -> None:
        self._memory = memory
        self._ontology = ontology if ontology is not None else core()

    async def render(self, member_id: str, as_of: datetime) -> tuple[Flag, ...]:
        return flags_from_state(await self._memory.metric_state(member_id, as_of), self._ontology)

    async def passport(self, member_id: str, as_of: datetime) -> Passport:
        return passport_from_state(await self._memory.metric_state(member_id, as_of), self._ontology)


__all__ = ["PassportRenderer", "flags_from_state", "metric_key", "passport_from_state"]
