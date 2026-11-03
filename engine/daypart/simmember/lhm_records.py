"""A5 — SimMember → LHM adapter (I7, D12). No free text ever reaches LHM.

Satisfies ``syntropy.lhm.tools.pipeline.LhmAdapter``. ``subject_ref`` is the
opaque ``member_id``; check-in points carry only ``note_id`` tokens, resolved
to the day's sentiment by :meth:`SimMemberLhmAdapter.sentiment_fn`.
"""

from __future__ import annotations

from typing import Sequence

from syntropy.lhm.models.mapping import OrderRecord, RegimenStartRecord, SeriesPoint, SeriesRecord

from daypart.simmember.cohort import Cohort
from daypart.simmember.trajectory import START_DATE

NON_RX_SOURCES = frozenset({"intake", "shopify", "stripe"})

ORDER_SOURCE = "intake"
ORDER_ITEM_DISPLAY = "Daily protocol kit"
REGIMEN_DISPLAY = "Daily protocol"
ADHERENCE_DISPLAY = "Protocol adherence (logged)"
CHECKIN_DISPLAY = "Check-in sentiment"


def is_rx_for_source(source: str) -> bool:
    """``"prescription"`` → True; NON_RX_SOURCES → False; anything else → True (fail-closed)."""
    if source == "prescription":
        return True
    return source not in NON_RX_SOURCES


class SimMemberLhmAdapter:
    def __init__(self, cohort: Cohort) -> None:
        self._orders: dict[str, tuple[OrderRecord, ...]] = {}
        self._starts: dict[str, tuple[RegimenStartRecord, ...]] = {}
        self._series: dict[str, tuple[SeriesRecord, ...]] = {}
        self._sentiment: dict[str, float] = {}
        for profile in cohort.profiles:
            mid = profile.member_id
            traj = cohort.trajectories[mid]
            self._orders[mid] = (
                OrderRecord(
                    subject_ref=mid,
                    item_display=ORDER_ITEM_DISPLAY,
                    occurred_at=START_DATE,
                    is_rx=is_rx_for_source(ORDER_SOURCE),
                    code="supplement-order",
                    source=ORDER_SOURCE,
                ),
            )
            self._starts[mid] = (
                RegimenStartRecord(
                    subject_ref=mid,
                    substance_display=REGIMEN_DISPLAY,
                    started_at=START_DATE,
                    code="regimen-start",
                    dose_value=None,
                    dose_unit="",
                ),
            )
            adherence = SeriesRecord(
                subject_ref=mid,
                metric_display=ADHERENCE_DISPLAY,
                unit="",
                code="journal-metric",
                points=tuple(
                    SeriesPoint(observed_at=d.date, value=1.0 if d.adhered else 0.0)
                    for d in traj.days
                ),
            )
            checkins = SeriesRecord(
                subject_ref=mid,
                metric_display=CHECKIN_DISPLAY,
                unit="",
                code="checkin-sentiment",
                points=tuple(
                    SeriesPoint(observed_at=d.date, value=None, note=d.note_id)
                    for d in traj.days
                    if d.checkin
                ),
            )
            self._series[mid] = (adherence, checkins)
            for d in traj.days:
                if d.checkin and d.note_id is not None and d.sentiment is not None:
                    self._sentiment[d.note_id] = d.sentiment

    async def read_orders(self, user_scope: str) -> Sequence[OrderRecord]:
        return self._orders.get(user_scope, ())

    async def read_regimen_starts(self, user_scope: str) -> Sequence[RegimenStartRecord]:
        return self._starts.get(user_scope, ())

    async def read_outcome_series(self, user_scope: str) -> Sequence[SeriesRecord]:
        return self._series.get(user_scope, ())

    def sentiment_fn(self, note: str) -> float:
        """The check-in day's sentiment for a ``note_id`` token; unknown → ``KeyError``."""
        return self._sentiment[note]


__all__ = ["NON_RX_SOURCES", "SimMemberLhmAdapter", "is_rx_for_source"]
