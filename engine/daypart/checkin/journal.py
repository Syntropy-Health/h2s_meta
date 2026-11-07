"""Per-visitor journals (A&D §4.2, D-A3) and the in-memory store that owns them.

A :class:`Journal` is one visitor's clone: a synthetic training member (``syn-`` only, I1), the
slice ontology, its own append-only :class:`~daypart.memory.events.EventLog`, a ``MemoryLhm`` over
the simulated cohort plus that log, the provenance store of :class:`ObservationRecord` values, and
the pending check-ins. Nothing is persisted; :class:`JournalStore` evicts a journal after
``idle_seconds`` without a request or when more than ``max_journals`` are live (LRU) (J4).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import pydantic
from pydantic import AwareDatetime, ConfigDict

from daypart.checkin.models import CheckIn
from daypart.memory.events import EventLog
from daypart.memory.lhm import MemoryLhm
from daypart.memory.ontology import Ontology
from daypart.memory.registry import slice_ontology
from daypart.simmember.cohort import Cohort, Profile
from daypart.simmember.lhm_records import SimMemberLhmAdapter
from daypart.simmember.trajectory import START_DATE

JOURNAL_DAY_OFFSET = 28  # G9: today = START_DATE + 28 days (enough simulated history for readiness)
DEFAULT_TODAY: dt.date = START_DATE.date() + dt.timedelta(days=JOURNAL_DAY_OFFSET)
MAX_JOURNALS = 256
IDLE_SECONDS = 3600.0

_SYN_ID = re.compile(r"^syn-\d{3}$")


class JournalError(LookupError):
    """A lookup failure with an HTTP ``status`` and a top-level ``{"error": ...}`` ``body``."""

    status = 404
    error = "journal_error"

    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.body: dict[str, Any] = {"error": self.error}


class UnknownMember(JournalError):
    error = "unknown_member"


class UnknownJournal(JournalError):
    error = "unknown_journal"


class UnknownCheckin(JournalError):
    error = "unknown_checkin"


class Provenance(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_kind: Literal["text", "file_fixture", "image_fixture"]
    source_ref: str
    checkin_id: str
    method: str


class ObservationRecord(pydantic.BaseModel):
    """The R5 inspect view of one saved value, mapped or unmapped (A&D §4.2).

    ``value`` and ``unit`` are AS RECORDED: exactly what the person confirmed or typed in a
    correction (e.g. ``160`` ``"[lb_av]"``), never converted, so the passport shows their number
    back to them. The matching ``SignalEvent`` (``event_id``) carries the value converted to the
    concept's canonical unit with the registry ``alt_units`` factor (160 [lb_av] -> 72.57 kg),
    because the EventLog and every score read canonical units only. Unmapped records
    (``status="unmapped"``) have no event, no LOINC code and no LOINC version (J2).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str
    event_id: str | None
    concept_id: str | None  # None for a word with no registry concept
    loinc_code: str | None
    loinc_version: str | None
    display: str
    value: float | None
    unit: str | None
    value_type: Literal["quantity", "integer", "ordinal"] | None
    observed_at: AwareDatetime
    provenance: Provenance
    quality: Literal["confirmed", "corrected"]
    status: Literal["mapped", "unmapped"]
    reviewer_status: str
    allowed_use: str

    @pydantic.model_validator(mode="after")
    def _j2(self) -> ObservationRecord:
        if self.status == "unmapped" and (self.event_id, self.loinc_code, self.loinc_version) != (None, None, None):
            raise ValueError("an unmapped record has no event, LOINC code or LOINC version (J2)")
        if self.status == "mapped" and self.event_id is None:
            raise ValueError("a mapped record names its SignalEvent")
        return self


@dataclass(eq=False)
class Journal:
    journal_id: str
    member: Profile
    ontology: Ontology
    events: EventLog
    memory: MemoryLhm
    last_seen: float
    today: dt.date = DEFAULT_TODAY
    provenance: dict[str, ObservationRecord] = field(default_factory=dict)
    checkins: OrderedDict[str, CheckIn] = field(default_factory=OrderedDict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    event_count: int = 0
    record_count: int = 0
    checkin_count: int = 0

    def all_events(self) -> tuple[Any, ...]:
        """Every event in this journal's log (a journal holds one member's events only)."""
        events = self.events.events(self.member.member_id)
        if len(events) != len(self.events):  # pragma: no cover - J1: only review() writes, for this member
            raise RuntimeError("journal event log holds events for another member")
        return events

    def rebuild_events(self, prior: tuple[Any, ...]) -> None:
        """Last-resort rollback: replace ``events`` with a fresh log holding exactly ``prior``.

        ``EventLog`` is append-only with no rollback API, so a commit that failed mid-way is undone by
        rebuilding. ``memory`` keeps reading the same log object as ``events``.
        """
        fresh = EventLog(self.ontology)
        fresh.extend(prior)
        self.events = fresh
        self.memory._events = fresh  # MemoryLhm reads only this attribute for journal events

    # Id helpers take an offset so the service can name ids before it commits (no partial state).
    def event_id(self, n: int) -> str:
        return f"{self.journal_id[:8]}-e{n}"

    def record_id(self, n: int) -> str:
        return f"obs-{n}"

    def checkin_id(self, n: int) -> str:
        return f"{self.journal_id[:8]}-c{n}"


class JournalStore:
    """In-memory journals: create from a ``syn-`` training member, get, idle + LRU eviction."""

    def __init__(
        self,
        cohort: Cohort,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_journals: int = MAX_JOURNALS,
        idle_seconds: float = IDLE_SECONDS,
    ) -> None:
        if max_journals < 1:
            raise ValueError("max_journals must be >= 1")
        self._cohort = cohort
        self._members = {p.member_id: p for p in cohort.training_profiles()}
        # One read-only adapter over the simulated cohort, shared by every journal's MemoryLhm
        # (J5: the base series is never written; each journal's own events live in its EventLog).
        self._adapter = SimMemberLhmAdapter(cohort)
        self._clock = clock
        self.max_journals = max_journals
        self.idle_seconds = idle_seconds
        self._journals: OrderedDict[str, Journal] = OrderedDict()
        # checkin_id -> journal_id, so POST /api/checkins/{id}/review needs only the check-in id
        # (A&D API table). Entries are registered on create and dropped when the journal is evicted.
        self._checkin_journal: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._journals)

    def _drop(self, journal: Journal) -> None:
        self._journals.pop(journal.journal_id, None)
        for cid in journal.checkins:
            if self._checkin_journal.get(cid) == journal.journal_id:
                del self._checkin_journal[cid]

    def _evict_idle(self, now: float) -> None:
        for j in [j for j in self._journals.values() if now - j.last_seen > self.idle_seconds]:
            self._drop(j)

    def create(self, member_id: object) -> Journal:
        """A fresh journal for a ``syn-`` training member; curated, unknown or malformed ids -> UnknownMember."""
        if not isinstance(member_id, str) or not _SYN_ID.fullmatch(member_id) or member_id not in self._members:
            raise UnknownMember(str(member_id))
        now = self._clock()
        self._evict_idle(now)
        ontology = slice_ontology()
        events = EventLog(ontology)
        # Check-in and event ids carry the first 8 hex digits; keep that prefix unique among live journals.
        prefixes = {k[:8] for k in self._journals}
        journal_id = uuid.uuid4().hex
        while journal_id[:8] in prefixes:  # pragma: no cover - a 1-in-4-billion collision
            journal_id = uuid.uuid4().hex
        journal = Journal(
            journal_id=journal_id,
            member=self._members[member_id],
            ontology=ontology,
            events=events,
            memory=MemoryLhm(self._adapter, ontology=ontology, events=events),
            last_seen=now,
        )
        self._journals[journal.journal_id] = journal
        while len(self._journals) > self.max_journals:
            self._drop(next(iter(self._journals.values())))
        return journal

    def get(self, journal_id: object) -> Journal:
        """The live journal (touching its idle timer and LRU place), else UnknownJournal."""
        now = self._clock()
        self._evict_idle(now)
        journal = self._journals.get(journal_id) if isinstance(journal_id, str) else None
        if journal is None:
            raise UnknownJournal(str(journal_id))
        journal.last_seen = now
        self._journals.move_to_end(journal.journal_id)
        return journal

    def register_checkin(self, checkin_id: str, journal: Journal) -> None:
        """Record which live journal owns ``checkin_id`` (called once the check-in is stored)."""
        if self._journals.get(journal.journal_id) is not journal:
            raise UnknownJournal(journal.journal_id)
        self._checkin_journal[checkin_id] = journal.journal_id

    def checkin_owner(self, checkin_id: object) -> str | None:
        """The journal id that owns ``checkin_id``, or None when unknown or evicted."""
        return self._checkin_journal.get(checkin_id) if isinstance(checkin_id, str) else None

    def journal_for_checkin(self, checkin_id: object) -> Journal:
        """The live journal owning ``checkin_id``: UnknownCheckin if never registered, UnknownJournal if
        its journal has since expired (the owner is read before idle eviction runs)."""
        owner = self.checkin_owner(checkin_id)
        if owner is None:
            raise UnknownCheckin(str(checkin_id))
        return self.get(owner)


__all__ = [
    "DEFAULT_TODAY",
    "IDLE_SECONDS",
    "MAX_JOURNALS",
    "Journal",
    "JournalError",
    "JournalStore",
    "ObservationRecord",
    "Provenance",
    "UnknownCheckin",
    "UnknownJournal",
    "UnknownMember",
]
