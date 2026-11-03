import asyncio
import re

import pytest
from syntropy.lhm.models.mapping import map_order, map_regimen_start, map_series
from syntropy.lhm.tools.pipeline import fetch

from daypart.simmember.cohort import generate
from daypart.simmember.lhm_records import NON_RX_SOURCES, SimMemberLhmAdapter, is_rx_for_source
from daypart.simmember.trajectory import START_DATE

NOTE_RE = re.compile(r"^note:(cur|syn)-\d{3}:d\d+$")


@pytest.fixture(scope="module")
def cohort0():
    return generate(0)


def test_is_rx_for_source():
    assert is_rx_for_source("prescription") is True
    for s in NON_RX_SOURCES:
        assert is_rx_for_source(s) is False
    assert is_rx_for_source("") is True and is_rx_for_source("pharmacy") is True


def test_records_and_mapping(cohort0):
    ad = SimMemberLhmAdapter(cohort0)

    async def go():
        for p in cohort0.profiles:
            mid = p.member_id
            traj = cohort0.trajectories[mid]
            orders = await ad.read_orders(mid)
            starts = await ad.read_regimen_starts(mid)
            series = await ad.read_outcome_series(mid)
            assert len(orders) >= 1 and len(starts) == 1
            assert all(o.source == "intake" and o.is_rx is False and o.subject_ref == mid for o in orders)
            assert len({o.item_display for o in orders}) == 1
            assert starts[0].started_at == START_DATE and starts[0].dose_value is None and starts[0].dose_unit == ""
            numeric = [s for s in series if s.code == "journal-metric"]
            checkin = [s for s in series if s.code != "journal-metric"]
            assert len(numeric) == 1 and len(checkin) == 1
            assert numeric[0].metric_display == "Protocol adherence (logged)" and numeric[0].unit == ""
            assert [pt.value for pt in numeric[0].points] == [1.0 if d.adhered else 0.0 for d in traj.days]
            cdays = [d for d in traj.days if d.checkin]
            assert [pt.note for pt in checkin[0].points] == [d.note_id for d in cdays]
            assert all(pt.value is None and NOTE_RE.match(pt.note) for pt in checkin[0].points)
            for s in series:
                assert s.subject_ref == mid
                for pt in s.points:
                    assert pt.observed_at.tzinfo is not None
            map_order(orders[0])
            map_regimen_start(starts[0])
            map_series(numeric[0], sentiment_fn=ad.sentiment_fn)
            obs = map_series(checkin[0], sentiment_fn=ad.sentiment_fn)
            assert len(obs) == len(cdays)
            assert [float(o.valueQuantity.value) for o in obs] == [pytest.approx(d.sentiment) for d in cdays]
        assert await ad.read_orders("syn-999") == () or len(await ad.read_orders("syn-999")) == 0
        assert len(await ad.read_outcome_series("nobody")) == 0
        aligned = await fetch(ad, "syn-001")
        assert len(aligned) == 2

    asyncio.run(go())
    with pytest.raises(KeyError):
        ad.sentiment_fn("note:syn-001:d999")
    with pytest.raises(KeyError):
        ad.sentiment_fn("I felt great today")
