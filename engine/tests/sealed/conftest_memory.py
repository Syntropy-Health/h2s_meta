"""Shared fixtures/helpers for the slice D (daypart.memory) sealed suites.

Imported explicitly by test_memory_events.py / test_ontology.py / test_memory_lhm.py.
Deliberately imports nothing from ``daypart.memory`` so it never masks an ImportError
in the suites themselves. Contract: docs/daypart/contracts/CONTRACT-slice-D-memory.md.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import pytest

from daypart.simmember.cohort import Cohort, generate
from daypart.simmember.lhm_records import SimMemberLhmAdapter
from daypart.simmember.trajectory import START_DATE

UTC = timezone.utc

# A spread of real cohort members (curated + synthetic, low/high ids).
SAMPLE_MEMBERS: tuple[str, ...] = ("cur-001", "cur-013", "cur-025", "syn-001", "syn-088", "syn-175")
# A syntactically valid id that is not in the generate(0) cohort.
UNKNOWN_MEMBER = "syn-999"

CORE_CAVEATS_EXPECTED: tuple[str, str] = (
    "Personal n-of-1 observation: an association in your own data, not causal.",
    "Scores carry an interval; small samples are uncertain.",
)


def day(n: float) -> datetime:
    """START_DATE + n days (tz-aware UTC)."""
    return START_DATE + timedelta(days=n)


@functools.lru_cache(maxsize=1)
def cohort0() -> Cohort:
    return generate(0)


@pytest.fixture(scope="session")
def cohort() -> Cohort:
    return cohort0()


@pytest.fixture(scope="session")
def adapter(cohort: Cohort) -> SimMemberLhmAdapter:
    return SimMemberLhmAdapter(cohort)


class SpyAdapter:
    """Wraps an LhmAdapter (+ sentiment_fn) and records every user_scope it receives."""

    def __init__(self, inner: SimMemberLhmAdapter) -> None:
        self._inner = inner
        self.scopes: list[str] = []
        self.notes: list[str] = []

    async def read_orders(self, user_scope: str):
        self.scopes.append(user_scope)
        return await self._inner.read_orders(user_scope)

    async def read_regimen_starts(self, user_scope: str):
        self.scopes.append(user_scope)
        return await self._inner.read_regimen_starts(user_scope)

    async def read_outcome_series(self, user_scope: str):
        self.scopes.append(user_scope)
        return await self._inner.read_outcome_series(user_scope)

    def sentiment_fn(self, note: str) -> float:
        self.notes.append(note)
        return self._inner.sentiment_fn(note)


class RestrictedView:
    """Reference 'adapter view restricted to records with timestamps <= as_of' (D3).

    Built by the test-author from the contract text, used only to compute the
    expected ``signals`` value.
    """

    def __init__(self, inner: SimMemberLhmAdapter, as_of: datetime) -> None:
        self._inner = inner
        self._as_of = as_of

    async def read_orders(self, user_scope: str):
        return tuple(o for o in await self._inner.read_orders(user_scope) if o.occurred_at <= self._as_of)

    async def read_regimen_starts(self, user_scope: str):
        return tuple(
            s for s in await self._inner.read_regimen_starts(user_scope) if s.started_at <= self._as_of
        )

    async def read_outcome_series(self, user_scope: str):
        out = []
        for rec in await self._inner.read_outcome_series(user_scope):
            pts = tuple(p for p in rec.points if p.observed_at <= self._as_of)
            out.append(dataclasses.replace(rec, points=pts))
        return tuple(out)

    def sentiment_fn(self, note: str) -> float:
        return self._inner.sentiment_fn(note)


def first_checkin_day(cohort: Cohort, member_id: str) -> int | None:
    days = [d.day for d in cohort.trajectories[member_id].days if d.checkin]
    return min(days) if days else None


def members(cohort: Cohort) -> Sequence[str]:
    return tuple(p.member_id for p in cohort.profiles)
