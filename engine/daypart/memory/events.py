"""D1 — member events (CONTRACT slice D, D14, I7).

Two event kinds: an :class:`ActionEvent` is something *done* (no value field);
a :class:`SignalEvent` is an observation of one ontology metric node. Neither
carries free text (I7) and ``member_id`` is always an opaque cohort id.

:class:`EventLog` is append-only and validates signals against an ontology.
The ``ontology`` module is imported lazily so this module has no import-time
dependency on it (import order ``events`` ← ``ontology`` ← ``lhm``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal

import pydantic
from pydantic import AwareDatetime, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from daypart.memory.ontology import Ontology

MEMBER_ID_RE = r"^(cur|syn)-\d{3}$"

ACTION_CODES: frozenset[str] = frozenset(
    {
        "act:nudge_sent",
        "act:reminder_sent",
        "act:goal_set",
        "act:protocol_adjusted",
        "act:education_viewed",
        "act:reply_sent",
    }
)


class ActionEvent(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["action"] = "action"
    event_id: str = Field(min_length=1)
    member_id: str = Field(pattern=MEMBER_ID_RE)  # opaque id only (I7)
    occurred_at: AwareDatetime
    code: str

    @field_validator("code")
    @classmethod
    def _action_code(cls, v: str) -> str:
        if v not in ACTION_CODES:
            raise ValueError(f"unknown action code {v!r}")
        return v


class SignalEvent(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["signal"] = "signal"
    event_id: str = Field(min_length=1)
    member_id: str = Field(pattern=MEMBER_ID_RE)  # opaque id only (I7)
    occurred_at: AwareDatetime
    code: str  # an ontology metric-node code (checked by EventLog)
    value: float = Field(allow_inf_nan=False)


Event = Annotated[ActionEvent | SignalEvent, Field(discriminator="kind")]
_EVENT_ADAPTER: pydantic.TypeAdapter[ActionEvent | SignalEvent] = pydantic.TypeAdapter(Event)


def parse_event(obj: Mapping[str, Any]) -> ActionEvent | SignalEvent:
    """Validate a mapping into an event, dispatching on ``kind`` (missing/unknown → ValidationError)."""
    return _EVENT_ADAPTER.validate_python(obj)


def _require_aware(dt: datetime, what: str) -> None:
    if not isinstance(dt, datetime) or dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{what} must be a timezone-aware datetime")


class EventLog:
    """Append-only per-member event store, validated against an ontology."""

    def __init__(self, ontology: Ontology | None = None) -> None:
        from daypart.memory import ontology as _ontology  # lazy: no import-time dependency

        self._ontology = ontology if ontology is not None else _ontology.core()
        self._ids: set[str] = set()
        self._by_member: dict[str, list[ActionEvent | SignalEvent]] = {}

    @property
    def ontology(self) -> Ontology:
        return self._ontology

    def append(self, event: ActionEvent | SignalEvent) -> None:
        if not isinstance(event, (ActionEvent, SignalEvent)):
            raise TypeError(f"expected ActionEvent or SignalEvent, got {type(event).__name__}")
        if event.event_id in self._ids:
            raise ValueError(f"duplicate event_id {event.event_id!r}")
        if isinstance(event, SignalEvent):
            node = self._ontology.nodes.get(event.code)
            if node is None or node.kind != "metric":
                raise ValueError(f"signal code {event.code!r} is not a metric node of the ontology")
            assert node.value_range is not None
            lo, hi = node.value_range
            if not (math.isfinite(event.value) and lo <= event.value <= hi):
                raise ValueError(f"signal value {event.value!r} outside {event.code} range [{lo}, {hi}]")
        self._ids.add(event.event_id)
        self._by_member.setdefault(event.member_id, []).append(event)

    def extend(self, events: Iterable[ActionEvent | SignalEvent]) -> None:
        for event in events:
            self.append(event)

    def events(
        self,
        member_id: str,
        *,
        as_of: datetime | None = None,
        kind: Literal["action", "signal"] | None = None,
    ) -> tuple[ActionEvent | SignalEvent, ...]:
        if as_of is not None:
            _require_aware(as_of, "as_of")
        out = [
            e
            for e in self._by_member.get(member_id, ())
            if (as_of is None or e.occurred_at <= as_of) and (kind is None or e.kind == kind)
        ]
        out.sort(key=lambda e: (e.occurred_at, e.event_id))
        return tuple(out)

    def __len__(self) -> int:
        return len(self._ids)


__all__ = [
    "ACTION_CODES",
    "MEMBER_ID_RE",
    "ActionEvent",
    "Event",
    "EventLog",
    "SignalEvent",
    "parse_event",
]
