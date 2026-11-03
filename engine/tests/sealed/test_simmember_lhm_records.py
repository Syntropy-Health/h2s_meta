"""Sealed challenge suite — slice A: LHM records adapter (I7, D12).

Spec: docs/daypart/contracts/CONTRACT-slice-A-simmember.md section A5.

Ambiguities resolved (questions for the principal):
- "one numeric SeriesRecord ... and one check-in SeriesRecord" is read as:
  exactly two series per member; the numeric one is identified by
  code == "journal-metric" with metric_display "Protocol adherence (logged)";
  the other is the check-in series. The check-in series' code/display is not
  pinned by the contract and is not asserted.
- "one point per day": point i of the numeric series is taken to fall on the
  calendar date of trajectory day i (compared by UTC date, not exact instant).
- Sentiment Observations are counted as map_series outputs whose code carries
  syntropy.lhm.models.mapping.SENTIMENT_CODE.
"""

from __future__ import annotations

import inspect
import re
from datetime import timezone

import pytest

pytestmark = pytest.mark.category("integration/lhm-records")

NOTE_RE = re.compile(r"^note:(cur|syn)-\d{3}:d\d+$")
ADHERENCE_DISPLAY = "Protocol adherence (logged)"


def _aware(dt) -> bool:
    return dt.tzinfo is not None and dt.utcoffset() is not None


@pytest.fixture(scope="module")
def adapter(cohort0):
    from daypart.simmember.lhm_records import SimMemberLhmAdapter

    return SimMemberLhmAdapter(cohort0)


@pytest.fixture(scope="module")
def all_records(adapter, cohort0):
    import asyncio

    async def collect():
        out = {}
        for p in cohort0.profiles:
            mid = p.member_id
            out[mid] = (
                list(await adapter.read_orders(mid)),
                list(await adapter.read_regimen_starts(mid)),
                list(await adapter.read_outcome_series(mid)),
            )
        return out

    return asyncio.run(collect())


def _split_series(series):
    numeric = [s for s in series if s.code == "journal-metric" and s.metric_display == ADHERENCE_DISPLAY]
    other = [s for s in series if s not in numeric]
    return numeric, other


class TestRxTable:
    pytestmark = pytest.mark.category("functional/rx-table")

    def test_non_rx_sources(self):
        from daypart.simmember.lhm_records import NON_RX_SOURCES

        assert NON_RX_SOURCES == frozenset({"intake", "shopify", "stripe"})

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("prescription", True),
            ("intake", False),
            ("shopify", False),
            ("stripe", False),
            ("unknown", True),
            ("", True),
            ("pharmacy", True),
        ],
    )
    def test_is_rx_for_source(self, source, expected):
        from daypart.simmember.lhm_records import is_rx_for_source

        assert is_rx_for_source(source) is expected


class TestAdapterShape:
    pytestmark = pytest.mark.category("functional/adapter-shape")

    def test_duck_conforms_to_lhm_adapter(self, adapter):
        for name in ("read_orders", "read_regimen_starts", "read_outcome_series"):
            meth = getattr(adapter, name, None)
            assert meth is not None and inspect.iscoroutinefunction(meth), name
        assert callable(getattr(adapter, "sentiment_fn", None))

    def test_protocol_method_names_match_lhm(self):
        from syntropy.lhm.tools.pipeline import LhmAdapter

        from daypart.simmember.lhm_records import SimMemberLhmAdapter

        for name in ("read_orders", "read_regimen_starts", "read_outcome_series"):
            assert hasattr(LhmAdapter, name)
            assert inspect.iscoroutinefunction(getattr(SimMemberLhmAdapter, name))

    async def test_unknown_scope_empty(self, adapter):
        for scope in ("syn-999", "nobody", ""):
            assert len(await adapter.read_orders(scope)) == 0
            assert len(await adapter.read_regimen_starts(scope)) == 0
            assert len(await adapter.read_outcome_series(scope)) == 0


class TestRecords:
    pytestmark = pytest.mark.category("integration/lhm-records")

    def test_record_types(self, all_records):
        from syntropy.lhm.models.mapping import OrderRecord, RegimenStartRecord, SeriesPoint, SeriesRecord

        for orders, starts, series in all_records.values():
            assert all(isinstance(o, OrderRecord) for o in orders)
            assert all(isinstance(s, RegimenStartRecord) for s in starts)
            assert all(isinstance(s, SeriesRecord) for s in series)
            for s in series:
                assert all(isinstance(pt, SeriesPoint) for pt in s.points)

    def test_subject_ref_is_member_id(self, all_records):
        for mid, (orders, starts, series) in all_records.items():
            for rec in [*orders, *starts, *series]:
                assert rec.subject_ref == mid
                assert "@" not in rec.subject_ref

    def test_all_datetimes_tz_aware(self, all_records):
        for orders, starts, series in all_records.values():
            for o in orders:
                assert _aware(o.occurred_at)
            for s in starts:
                assert _aware(s.started_at)
            for s in series:
                for pt in s.points:
                    assert _aware(pt.observed_at)

    def test_orders(self, all_records):
        from daypart.simmember.lhm_records import is_rx_for_source

        displays = set()
        for orders, _, _ in all_records.values():
            assert len(orders) >= 1
            for o in orders:
                assert o.source == "intake"
                assert o.is_rx is False
                assert o.is_rx == is_rx_for_source(o.source)
                assert isinstance(o.item_display, str) and o.item_display.strip()
                displays.add(o.item_display)
        assert len(displays) == 1, displays

    def test_order_display_claim_safe(self, all_records):
        from syntropy.lhm.metrics.claims import assert_narration_safe

        for orders, _, _ in all_records.values():
            for o in orders:
                assert_narration_safe(o.item_display)

    def test_regimen_start(self, all_records):
        from daypart.simmember.trajectory import START_DATE

        for _, starts, _ in all_records.values():
            assert len(starts) == 1
            s = starts[0]
            assert s.started_at == START_DATE
            assert s.dose_value is None
            assert s.dose_unit == ""

    def test_numeric_adherence_series(self, all_records, cohort0):
        for mid, (_, _, series) in all_records.items():
            numeric, _ = _split_series(series)
            assert len(numeric) == 1, mid
            rec = numeric[0]
            assert rec.unit == ""
            days = cohort0.trajectories[mid].days
            assert len(rec.points) == len(days)
            for pt, d in zip(rec.points, days):
                assert pt.value == (1.0 if d.adhered else 0.0)
                assert pt.observed_at.astimezone(timezone.utc).date() == d.date.astimezone(timezone.utc).date()

    def test_checkin_series(self, all_records, cohort0):
        for mid, (_, _, series) in all_records.items():
            numeric, other = _split_series(series)
            assert len(series) == 2, mid
            assert len(other) == 1, mid
            rec = other[0]
            checkins = [d for d in cohort0.trajectories[mid].days if d.checkin]
            assert [pt.note for pt in rec.points] == [d.note_id for d in checkins]
            for pt, d in zip(rec.points, checkins):
                assert pt.value is None
                assert pt.observed_at.astimezone(timezone.utc).date() == d.date.astimezone(timezone.utc).date()

    def test_every_note_is_token(self, all_records):
        n = 0
        for mid, (_, _, series) in all_records.items():
            for s in series:
                for pt in s.points:
                    if pt.note is not None:
                        n += 1
                        assert NOTE_RE.match(pt.note), pt.note
                        assert pt.note.startswith(f"note:{mid}:d")
        assert n > 0


class TestSentimentFn:
    pytestmark = pytest.mark.category("functional/sentiment")

    def test_returns_trajectory_sentiment(self, adapter, cohort0):
        n = 0
        for t in cohort0.trajectories.values():
            for d in t.days:
                if d.checkin:
                    v = adapter.sentiment_fn(d.note_id)
                    assert isinstance(v, float)
                    assert v == d.sentiment
                    n += 1
        assert n > 0

    def test_deterministic(self, adapter, cohort0, cohort0_again):
        from daypart.simmember.lhm_records import SimMemberLhmAdapter

        other = SimMemberLhmAdapter(cohort0_again)
        notes = [d.note_id for t in cohort0.trajectories.values() for d in t.days if d.checkin][:200]
        assert [adapter.sentiment_fn(x) for x in notes] == [other.sentiment_fn(x) for x in notes]

    @pytest.mark.parametrize("bad", ["note:syn-999:d0", "note:syn-001:d9999", "garbage", "", "I felt great today"])
    def test_unknown_token_keyerror(self, adapter, bad):
        with pytest.raises(KeyError):
            adapter.sentiment_fn(bad)


class TestLhmMapping:
    pytestmark = pytest.mark.category("integration/lhm-mapping")

    def test_map_all_records(self, adapter, all_records, cohort0):
        from syntropy.lhm.models.mapping import (
            SENTIMENT_CODE,
            map_order,
            map_regimen_start,
            map_series,
        )

        def is_sentiment(obs) -> bool:
            coding = obs.code.coding or []
            return any(c.code == SENTIMENT_CODE for c in coding)

        for mid, (orders, starts, series) in all_records.items():
            for o in orders:
                map_order(o)
            for s in starts:
                map_regimen_start(s)
            numeric, other = _split_series(series)
            for rec in numeric:
                obs = map_series(rec, sentiment_fn=adapter.sentiment_fn)
                assert len(obs) == len(rec.points)
                assert not any(is_sentiment(o) for o in obs)
            n_checkin = sum(d.checkin for d in cohort0.trajectories[mid].days)
            for rec in other:
                obs = map_series(rec, sentiment_fn=adapter.sentiment_fn)
                assert sum(is_sentiment(o) for o in obs) == n_checkin, mid
