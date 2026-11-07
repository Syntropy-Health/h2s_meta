"""HAC-38 P1 task 5: per-visitor journals, the review-before-save service and its HTTP routes.

Covers T2.2 end to end, T2.3, T2.4, T3.1/T3.2 via HTTP, T4.2-T4.6, T5.3, T5.5 (P1 endpoints) and
T9.2 (fixtures only). Synthetic inputs only.
"""

import ast
import asyncio
import datetime as dt
import re
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from daypart.checkin import service
from daypart.checkin.crisis import HANDOFF_TEXT
from daypart.checkin.fixtures import MAX_UPLOAD_BYTES, fixture_bytes, manifest
from daypart.checkin.journal import (
    JournalError,
    JournalStore,
    ObservationRecord,
    UnknownJournal,
    UnknownMember,
)
from daypart.checkin.service import CheckinError
from daypart.demo.app import create_app
from daypart.memory.events import SignalEvent
from daypart.memory.registry import load_registry
from daypart.simmember.trajectory import START_DATE

MODAL_DEMO = Path(__file__).resolve().parents[2] / "deploy" / "modal_demo.py"
TODAY = START_DATE.date() + dt.timedelta(days=28)
UNLISTED_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + bytes(range(64))
PDF = b"%PDF-1.4\n%synthetic\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"
RECORD_FIELDS = {
    "record_id", "event_id", "concept_id", "loinc_code", "loinc_version", "display", "value", "unit",
    "value_type", "observed_at", "provenance", "quality", "status", "reviewer_status", "allowed_use",
}
PROVENANCE_FIELDS = {"source_kind", "source_ref", "checkin_id", "method"}


@pytest.fixture(scope="module")
def client():
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("DAYPART_REAL_INPUT", raising=False)
        with TestClient(create_app(cohort_seed=0)) as c:
            yield c


@pytest.fixture(scope="module")
def cohort(client):
    return client.app.state.playground.cohort


# --- helpers ------------------------------------------------------------------------------------


def _journal(client, member_id="syn-001"):
    r = client.post("/api/journals", json={"member_id": member_id})
    assert r.status_code == 201, r.text
    return r.json()["journal_id"]


def _j(client, jid):
    return client.app.state.journals.get(jid)


def _snapshot(j):
    return (len(j.events), dict(j.provenance), list(j.checkins.items()))


def _text(client, jid, text):
    return client.post("/api/checkins", json={"journal_id": jid, "text": text})


def _upload(client, jid, data, name="upload.bin", ctype="application/octet-stream"):
    return client.post("/api/checkins", data={"journal_id": jid}, files={"file": (name, data, ctype)})


def _review(client, cid, decisions, journal_id=None):
    """The A&D-shaped review body ``{"decisions": [...]}``; ``journal_id`` is optional."""
    body = {"decisions": decisions} if journal_id is None else {"journal_id": journal_id, "decisions": decisions}
    return client.post(f"/api/checkins/{cid}/review", json=body)


def _pending(client, text, member_id="syn-001"):
    jid = _journal(client, member_id)
    r = _text(client, jid, text)
    assert r.status_code == 201, r.text
    return jid, r.json()


def _signals(j):
    return [e for e in j.events.events(j.member.member_id, kind="signal")]


# --- journals and the store ------------------------------------------------------------------------


def test_create_journal_201_and_defaults(client):
    r = client.post("/api/journals", json={"member_id": "syn-002"})
    assert r.status_code == 201
    body = r.json()
    assert body["member_id"] == "syn-002" and body["today"] == TODAY.isoformat()
    j = _j(client, body["journal_id"])
    assert re.fullmatch(r"[0-9a-f]{32}", j.journal_id)
    assert j.today == TODAY
    assert j.ontology.version == "core-v1+u1" and j.events.ontology is j.ontology
    assert len(j.events) == 0 and j.provenance == {} and len(j.checkins) == 0


@pytest.mark.parametrize("member_id", ["cur-001", "syn-999", "nope", "syn-1", ""])
def test_create_journal_curated_unknown_or_malformed_404(client, member_id):
    r = client.post("/api/journals", json={"member_id": member_id})
    assert r.status_code == 404
    assert r.json()["error"] == "unknown_member"
    assert "detail" not in r.json()


def test_store_rejects_curated_and_unknown(cohort):
    store = JournalStore(cohort)
    for mid in ("cur-001", "cur-025", "syn-000", "syn-999", "x"):
        with pytest.raises(UnknownMember) as e:
            store.create(mid)
        assert e.value.status == 404 and e.value.body["error"] == "unknown_member"


def test_store_idle_eviction_with_injected_clock(cohort):
    now = [1000.0]
    store = JournalStore(cohort, clock=lambda: now[0], idle_seconds=3600)
    j = store.create("syn-001")
    now[0] += 3599
    assert store.get(j.journal_id) is j  # touched: last_seen moves
    now[0] += 3599
    assert store.get(j.journal_id) is j
    now[0] += 3601
    with pytest.raises(UnknownJournal) as e:
        store.get(j.journal_id)
    assert e.value.status == 404 and e.value.body["error"] == "unknown_journal"
    with pytest.raises(UnknownJournal):
        store.get("0" * 32)


def test_store_lru_eviction_at_max_journals(cohort):
    now = [0.0]
    store = JournalStore(cohort, clock=lambda: now[0], max_journals=2)
    a = store.create("syn-001")
    b = store.create("syn-002")
    c = store.create("syn-003")
    with pytest.raises(UnknownJournal):
        store.get(a.journal_id)
    assert store.get(b.journal_id) is b  # b becomes most recent
    d = store.create("syn-004")
    with pytest.raises(UnknownJournal):
        store.get(c.journal_id)
    assert store.get(b.journal_id) is b and store.get(d.journal_id) is d


def test_unknown_journal_404_on_every_journal_route(client):
    ghost = "f" * 32
    for r in (
        _text(client, ghost, "slept 6 hours"),
        _review(client, "c-1", [], journal_id=ghost),
        client.get(f"/api/journals/{ghost}/observations/obs-1"),
    ):
        assert r.status_code == 404
        assert r.json()["error"] == "unknown_journal"


# --- create_checkin -----------------------------------------------------------------------------


def test_t2_2_post_text_writes_nothing(client):
    jid = _journal(client)
    j = _j(client, jid)
    before = len(j.events)
    r = _text(client, jid, "slept 6 hours and walked 8000 steps")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "pending_review" and body["checkin_id"]
    by_code = {p["concept_code"]: p for p in body["proposals"]}
    assert by_code["93832-4"]["value"] == 6 and by_code["93832-4"]["unit"] == "h"
    assert by_code["55423-8"]["value"] == 8000
    assert len(j.events) == before == 0
    assert j.provenance == {}
    assert j.checkins[body["checkin_id"]].status == "pending_review"


@pytest.mark.parametrize("text,reason", [
    ("email me at a.person@example.org, slept 6 hours", "email"),
    ("call 555-123-4567 slept 6 hours", "phone"),
])
def test_t2_3_pii_refused_and_no_checkin(client, text, reason):
    jid = _journal(client)
    j = _j(client, jid)
    snap = _snapshot(j)
    r = _text(client, jid, text)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "looks_like_personal_data"
    assert reason in body["reasons"]
    assert body["warning"] == "Do not enter real health data."
    assert _snapshot(j) == snap and len(j.checkins) == 0


def test_t2_4_crisis_text_handoff_no_checkin_no_events(client):
    jid = _journal(client)
    j = _j(client, jid)
    r = _text(client, jid, "slept 3 hours, I want to end my life")
    assert r.status_code == 200
    assert r.json() == {"status": "handoff", "text": HANDOFF_TEXT}
    assert len(j.checkins) == 0 and len(j.events) == 0 and j.provenance == {}


@pytest.mark.parametrize("payload,status,error", [
    ({"text": ""}, 422, "empty_checkin"),
    ({"text": "   \n\t"}, 422, "empty_checkin"),
    ({}, 422, "empty_checkin"),
    ({"text": "hello there"}, 422, "nothing_extracted"),
    ({"text": "slept 6 hours " + "x" * 500}, 422, "text_too_long"),
    ({"text": "slept 6 hours", "fixture_id": "demo-scale-photo"}, 422, None),
    ({"fixture_id": "demo-no-such-fixture"}, 404, "unknown_fixture"),
])
def test_rejections_leave_journal_unchanged(client, payload, status, error):
    jid = _journal(client)
    j = _j(client, jid)
    snap = _snapshot(j)
    r = client.post("/api/checkins", json={"journal_id": jid, **payload})
    assert r.status_code == status, r.text
    body = r.json()
    assert "error" in body and "detail" not in body
    if error is not None:
        assert body["error"] == error
    if error == "nothing_extracted":
        assert "slept 7 hours" in body["hint"]
    assert _snapshot(j) == snap


def test_text_of_exactly_500_chars_is_accepted(client):
    jid = _journal(client)
    text = "slept 6 hours" + " " * 487
    assert len(text) == 500
    assert _text(client, jid, text).status_code == 201


@pytest.mark.parametrize("fixture_id", sorted(manifest()))
def test_t3_1_each_fixture_id_and_upload_201_and_correctable(client, fixture_id):
    jid = _journal(client)
    r = client.post("/api/checkins", json={"journal_id": jid, "fixture_id": fixture_id})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["proposals"] and all(p["method"] == "fixture-sidecar-v1" for p in body["proposals"])
    decisions = []
    for p in body["proposals"]:
        if p["concept_code"] is None:
            decisions.append({"proposal_id": p["proposal_id"], "action": "save_unmapped"})
        else:
            concept = load_registry().by_code(p["concept_code"])
            decisions.append({"proposal_id": p["proposal_id"], "action": "correct",
                              "value": p["value"], "unit": concept.unit})
    rr = _review(client, body["checkin_id"], decisions)
    assert rr.status_code == 200, rr.text

    data, media = fixture_bytes(fixture_id)
    jid2 = _journal(client)
    up = _upload(client, jid2, data, name="renamed.bin", ctype=media)
    assert up.status_code == 201, up.text
    assert [p["source_ref"] for p in up.json()["proposals"]] == [p["source_ref"] for p in body["proposals"]]


def test_fixture_routes(client):
    r = client.get("/api/fixtures")
    assert r.status_code == 200
    ids = [f["fixture_id"] for f in r.json()]
    assert ids == list(manifest())
    for fid in ids:
        rb = client.get(f"/api/fixtures/{fid}")
        data, media = fixture_bytes(fid)
        assert rb.status_code == 200 and rb.content == data
        assert rb.headers["content-type"].split(";")[0] == media.split(";")[0]
    missing = client.get("/api/fixtures/demo-nope")
    assert missing.status_code == 404 and missing.json()["error"] == "unknown_fixture"


@pytest.mark.parametrize("data,status,error", [
    (UNLISTED_PNG, 422, "not_a_demo_fixture"),
    (b"\x89PNG\r\n\x1a\n" + b"\x00" * (MAX_UPLOAD_BYTES + 1 - 8), 413, "too_large"),
    (PDF, 415, "unsupported_type"),
])
def test_t3_2_uploads_via_http_leave_journal_unchanged(client, data, status, error):
    jid = _journal(client)
    j = _j(client, jid)
    snap = _snapshot(j)
    r = _upload(client, jid, data, name="scale-photo.png", ctype="image/png")
    assert r.status_code == status, r.text
    assert r.json()["error"] == error
    if error == "not_a_demo_fixture":
        assert r.json()["message"] == "Only the bundled demo files can be used."
    assert _snapshot(j) == snap


def test_stream_cap_on_content_length_and_chunked_body(client):
    jid = _journal(client)
    j = _j(client, jid)
    snap = _snapshot(j)
    big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (MAX_UPLOAD_BYTES + 64 * 1024)
    r = _upload(client, jid, big)
    assert r.status_code == 413 and r.json()["error"] == "too_large"

    def chunks():
        for _ in range(40):
            yield b"a" * 32 * 1024

    r2 = client.post("/api/checkins", content=chunks(),
                     headers={"content-type": "multipart/form-data; boundary=xyz"})
    assert r2.status_code == 413 and r2.json()["error"] == "too_large"
    assert _snapshot(j) == snap


def test_t9_2_real_input_env_still_fixtures_only(monkeypatch):
    monkeypatch.setenv("DAYPART_REAL_INPUT", "1")
    with TestClient(create_app(cohort_seed=0)) as c:
        mode = c.get("/api/mode").json()["real_input"]
        assert mode["requested"] is True and mode["enabled"] is False
        jid = _journal(c)
        r = _upload(c, jid, UNLISTED_PNG, ctype="image/png")
        assert r.status_code == 422 and r.json()["error"] == "not_a_demo_fixture"
        ok = c.post("/api/checkins", json={"journal_id": jid, "fixture_id": "demo-scale-photo"})
        assert ok.status_code == 201


# --- review -----------------------------------------------------------------------------------------


def test_t4_2_confirm_blocking_refused_nothing_written(client):
    jid, ck = _pending(client, "slept 6 hours and weighed 160")
    j = _j(client, jid)
    snap = _snapshot(j)
    r = _review(client, ck["checkin_id"], [
        {"proposal_id": "p-1", "action": "confirm"},
        {"proposal_id": "p-2", "action": "confirm"},
    ])
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "unresolved_flags"
    assert body["proposals"] == [{"proposal_id": "p-2", "flags": ["unit_assumed"]}]
    assert _snapshot(j) == snap and j.checkins[ck["checkin_id"]].status == "pending_review"


def test_t4_2_correct_with_valid_values_200(client):
    jid, ck = _pending(client, "slept 30 hours")
    r = _review(client, ck["checkin_id"], [{"proposal_id": "p-1", "action": "correct", "value": 7}])
    assert r.status_code == 200, r.text
    j = _j(client, jid)
    (event,) = _signals(j)
    assert event.code == "93832-4" and event.value == 7
    (rec,) = j.provenance.values()
    assert rec.quality == "corrected" and rec.status == "mapped"


def test_contradiction_cleared_by_discarding_one_side(client):
    jid, ck = _pending(client, "slept 6 hours and slept 7 hours")
    bad = _review(client, ck["checkin_id"], [
        {"proposal_id": "p-1", "action": "confirm"},
        {"proposal_id": "p-2", "action": "correct", "value": 6},
    ])
    assert bad.status_code == 422 and bad.json()["error"] == "unresolved_flags"
    ok = _review(client, ck["checkin_id"], [
        {"proposal_id": "p-1", "action": "confirm"},
        {"proposal_id": "p-2", "action": "discard"},
    ])
    assert ok.status_code == 200, ok.text
    assert ok.json()["discarded"] == ["p-2"]


def test_t4_2_save_unmapped_no_signal_event(client):
    jid, ck = _pending(client, "headache")
    r = _review(client, ck["checkin_id"], [{"proposal_id": "p-1", "action": "save_unmapped"}])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["written"] == [] and len(body["records"]) == 1
    j = _j(client, jid)
    assert len(j.events) == 0
    rec = j.provenance[body["records"][0]]
    assert rec.status == "unmapped" and rec.event_id is None and rec.loinc_code is None


def test_save_unmapped_allowed_on_blocking_known_concept(client):
    jid, ck = _pending(client, "weighed 160")
    r = _review(client, ck["checkin_id"], [{"proposal_id": "p-1", "action": "save_unmapped"}])
    assert r.status_code == 200
    j = _j(client, jid)
    assert len(j.events) == 0
    (rec,) = j.provenance.values()
    assert rec.status == "unmapped" and rec.loinc_code is None and rec.loinc_version is None
    assert rec.value == 160 and rec.unit == "kg"


def test_t4_3_discard_leaves_no_trace(client):
    jid, ck = _pending(client, "slept 6 hours and walked 8000 steps")
    r = _review(client, ck["checkin_id"], [
        {"proposal_id": "p-1", "action": "confirm"},
        {"proposal_id": "p-2", "action": "discard"},
    ])
    assert r.status_code == 200
    j = _j(client, jid)
    assert all(e.code != "55423-8" for e in _signals(j))
    assert all(rec.concept_id != "chc:steps" for rec in j.provenance.values())
    assert all(rec.value != 8000 for rec in j.provenance.values())


def test_t4_4_confirm_one_of_two_then_409(client):
    jid, ck = _pending(client, "slept 6 hours and walked 8000 steps")
    j = _j(client, jid)
    decisions = [{"proposal_id": "p-1", "action": "confirm"}, {"proposal_id": "p-2", "action": "discard"}]
    r = _review(client, ck["checkin_id"], decisions)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "saved" and len(body["written"]) == 1 and body["discarded"] == ["p-2"]
    assert len(_signals(j)) == 1
    (event,) = _signals(j)
    assert event.event_id == body["written"][0] == f"{jid[:8]}-e1"
    assert event.member_id == "syn-001"
    assert event.occurred_at == dt.datetime.combine(TODAY, dt.time(12), tzinfo=dt.UTC)
    again = _review(client, ck["checkin_id"], decisions)
    assert again.status_code == 409 and again.json()["error"] == "already_saved"
    assert len(j.events) == 1 and j.checkins[ck["checkin_id"]].status == "saved"


async def test_t4_5_concurrent_double_review_one_write(cohort):
    store = JournalStore(cohort)
    j = store.create("syn-005")
    ck = service.create_checkin(j, text="slept 6 hours")
    decisions = [{"proposal_id": "p-1", "action": "confirm"}]
    results = await asyncio.gather(
        service.review(j, ck["checkin_id"], decisions),
        service.review(j, ck["checkin_id"], decisions),
        return_exceptions=True,
    )
    oks = [r for r in results if isinstance(r, dict)]
    errs = [r for r in results if isinstance(r, CheckinError)]
    assert len(oks) == 1 and len(errs) == 1
    assert errs[0].status == 409 and errs[0].body["error"] == "already_saved"
    assert len(j.events) == 1 and len(j.provenance) == 1


def test_review_unknown_checkin_or_other_journal_404(client):
    jid, ck = _pending(client, "slept 6 hours")
    other = _journal(client)
    decisions = [{"proposal_id": "p-1", "action": "confirm"}]
    for j_, cid in ((jid, "nope"), (other, ck["checkin_id"]), (None, "nope")):
        r = _review(client, cid, decisions, journal_id=j_)
        assert r.status_code == 404 and r.json()["error"] == "unknown_checkin"
    assert _j(client, jid).checkins[ck["checkin_id"]].status == "pending_review"


def test_review_spec_body_without_journal_id_resolves_the_journal(client):
    jid, ck = _pending(client, "slept 6 hours")
    r = client.post(f"/api/checkins/{ck['checkin_id']}/review",
                    json={"decisions": [{"proposal_id": "p-1", "action": "confirm"}]})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "saved" and len(_signals(_j(client, jid))) == 1


def test_review_optional_journal_id_that_matches_is_accepted(client):
    jid, ck = _pending(client, "slept 6 hours")
    r = _review(client, ck["checkin_id"], [{"proposal_id": "p-1", "action": "confirm"}], journal_id=jid)
    assert r.status_code == 200, r.text


def test_store_checkin_map_registered_and_dropped_on_eviction(cohort):
    now = [0.0]
    store = JournalStore(cohort, clock=lambda: now[0], idle_seconds=10, max_journals=1)
    a = store.create("syn-001")
    ck = service.create_checkin(a, text="slept 6 hours")
    store.register_checkin(ck["checkin_id"], a)
    assert store.journal_for_checkin(ck["checkin_id"]) is a
    now[0] += 11  # idle: resolving the check-in finds its journal gone
    with pytest.raises(UnknownJournal):
        store.journal_for_checkin(ck["checkin_id"])
    b = store.create("syn-002")
    ckb = service.create_checkin(b, text="slept 6 hours")
    store.register_checkin(ckb["checkin_id"], b)
    store.create("syn-003")  # LRU evicts b and its check-in entry
    with pytest.raises(JournalError) as e:
        store.journal_for_checkin(ckb["checkin_id"])
    assert e.value.status == 404 and e.value.body["error"] in ("unknown_checkin", "unknown_journal")
    assert store.checkin_owner(ck["checkin_id"]) is None and store.checkin_owner(ckb["checkin_id"]) is None


@pytest.mark.parametrize("decisions,field", [
    ([{"proposal_id": "p-1", "action": "confirm"}], "missing"),
    ([{"proposal_id": "p-1", "action": "confirm"}, {"proposal_id": "p-1", "action": "discard"},
      {"proposal_id": "p-2", "action": "confirm"}], "duplicate"),
    ([{"proposal_id": "p-1", "action": "confirm"}, {"proposal_id": "p-2", "action": "confirm"},
      {"proposal_id": "p-9", "action": "confirm"}], "unknown"),
])
def test_missing_duplicate_unknown_decisions_refused(client, decisions, field):
    jid, ck = _pending(client, "slept 6 hours and walked 8000 steps")
    j = _j(client, jid)
    snap = _snapshot(j)
    r = _review(client, ck["checkin_id"], decisions)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "missing_decision"
    assert body[field]
    assert _snapshot(j)[:2] == snap[:2] and j.checkins[ck["checkin_id"]].status == "pending_review"


def test_lb_correction_records_as_entered_and_converts_event(client):
    jid, ck = _pending(client, "weighed 160")
    r = _review(client, ck["checkin_id"], [
        {"proposal_id": "p-1", "action": "correct", "unit": "[lb_av]"},
    ])
    assert r.status_code == 200, r.text
    j = _j(client, jid)
    (event,) = _signals(j)
    assert event.code == "29463-7"
    assert event.value == pytest.approx(160 * 0.45359237)
    assert event.value == pytest.approx(72.57, abs=0.01)
    rec = j.provenance[r.json()["records"][0]]
    assert rec.value == 160 and rec.unit == "[lb_av]" and rec.quality == "corrected"
    assert rec.event_id == event.event_id


@pytest.mark.parametrize("correction,reason", [
    ({"value": 600, "unit": "kg"}, "value_out_of_range"),
    ({"value": -1, "unit": "kg"}, "value_out_of_range"),
    ({"value": 2000, "unit": "[lb_av]"}, "value_out_of_range"),
    ({"unit": "stone"}, "unit_not_allowed"),
    ({"unit": "kg", "date": (TODAY + dt.timedelta(days=1)).isoformat()}, "date_outside_window"),
    ({"unit": "kg", "date": (TODAY - dt.timedelta(days=29)).isoformat()}, "date_outside_window"),
])
def test_invalid_correction_refused_without_bounds(client, correction, reason):
    jid, ck = _pending(client, "weighed 160")
    j = _j(client, jid)
    snap = _snapshot(j)
    r = _review(client, ck["checkin_id"], [{"proposal_id": "p-1", "action": "correct", **correction}])
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "invalid_correction"
    assert body["proposals"] == [{"proposal_id": "p-1", "reasons": [reason]}]
    assert "500" not in r.text
    _assert_no_bounds(body)
    assert _snapshot(j)[:2] == snap[:2]


def test_correction_at_window_edge_accepted(client):
    jid, ck = _pending(client, "weighed 160")
    r = _review(client, ck["checkin_id"], [{
        "proposal_id": "p-1", "action": "correct", "unit": "kg",
        "date": (TODAY - dt.timedelta(days=28)).isoformat(),
    }])
    assert r.status_code == 200, r.text
    (event,) = _signals(_j(client, jid))
    assert event.occurred_at.date() == TODAY - dt.timedelta(days=28)


def test_correct_on_unknown_concept_is_invalid(client):
    jid, ck = _pending(client, "headache")
    r = _review(client, ck["checkin_id"], [{"proposal_id": "p-1", "action": "correct", "value": 3}])
    assert r.status_code == 422 and r.json()["error"] == "invalid_correction"


def test_event_log_value_error_maps_to_invalid_correction_without_message(client, monkeypatch):
    jid, ck = _pending(client, "slept 6 hours and walked 8000 steps")
    j = _j(client, jid)

    def boom(event):
        raise ValueError("internal detail range [0.0, 100000.0]")

    monkeypatch.setattr(j.events, "append", boom)
    r = _review(client, ck["checkin_id"], [
        {"proposal_id": "p-1", "action": "confirm"}, {"proposal_id": "p-2", "action": "confirm"},
    ])
    assert r.status_code == 422 and r.json()["error"] == "invalid_correction"
    assert "internal detail" not in r.text and "100000" not in r.text
    monkeypatch.undo()
    assert len(j.events) == 0 and j.provenance == {}
    assert j.checkins[ck["checkin_id"]].status == "pending_review"


# --- observation records (T5.3) ------------------------------------------------------------------------


def test_t5_3_observation_records_confirmed_and_unmapped(client):
    jid, ck = _pending(client, "slept 6 hours and headache")
    r = _review(client, ck["checkin_id"], [
        {"proposal_id": "p-1", "action": "confirm"},
        {"proposal_id": "p-2", "action": "save_unmapped"},
    ])
    assert r.status_code == 200, r.text
    rec_ids = r.json()["records"]
    assert len(rec_ids) == 2 and all(re.fullmatch(r"obs-\d+", x) for x in rec_ids)
    mapped = client.get(f"/api/journals/{jid}/observations/{rec_ids[0]}")
    unmapped = client.get(f"/api/journals/{jid}/observations/{rec_ids[1]}")
    assert mapped.status_code == unmapped.status_code == 200
    m, u = mapped.json(), unmapped.json()
    for rec in (m, u):
        assert set(rec) == RECORD_FIELDS
        assert set(rec["provenance"]) == PROVENANCE_FIELDS
        assert rec["provenance"]["checkin_id"] == ck["checkin_id"]
        assert rec["provenance"]["method"] == "rules-v1"
        assert rec["reviewer_status"] == "draft, not clinically reviewed"
        assert rec["allowed_use"]
    assert m["status"] == "mapped" and m["quality"] == "confirmed"
    assert m["loinc_code"] == "93832-4" and m["loinc_version"] == "2.83"
    assert m["event_id"] == r.json()["written"][0]
    assert m["value"] == 6 and m["unit"] == "h" and m["concept_id"] == "chc:sleep-duration"
    assert m["observed_at"].startswith(TODAY.isoformat() + "T12:00:00")
    assert u["status"] == "unmapped" and u["loinc_code"] is None and u["event_id"] is None
    assert u["loinc_version"] is None
    missing = client.get(f"/api/journals/{jid}/observations/obs-999")
    assert missing.status_code == 404 and missing.json()["error"] == "unknown_record"


def test_observation_record_model_is_frozen():
    fields = set(ObservationRecord.model_fields)
    assert fields == RECORD_FIELDS
    assert ObservationRecord.model_config.get("frozen") is True


# --- T4.6: one container -----------------------------------------------------------------------------


def _kw(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name:
            return ast.literal_eval(k.value)
    return None


def test_t4_6_playground_runs_in_one_container():
    tree = ast.parse(MODAL_DEMO.read_text(encoding="utf-8"))
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "playground"]
    calls = [d for d in fn.decorator_list if isinstance(d, ast.Call)]
    names = [ast.unparse(c.func) for c in calls]
    function = calls[names.index("app.function")]
    assert _kw(function, "max_containers") == 1
    concurrent = calls[names.index("modal.concurrent")]
    assert _kw(concurrent, "max_inputs") == 50


# --- T5.5: plausible bounds never in a response ----------------------------------------------------------

_NUMBER_TOKEN = re.compile(r"(?<![\w.\-])-?\d+(?:\.\d+)?(?![\w.\-])")
# "value" is the member's own recorded number and "snippet" echoes the member's own input; both
# may coincide with a bound by chance (e.g. "slept 6 hours"), so they are not leak channels.
_EXEMPT_KEYS = frozenset({"value", "snippet"})


def _bound_tokens() -> set[str]:
    out: set[str] = set()
    for c in load_registry().concepts():
        for b in c.plausible:
            out.add(f"{b:g}")
            out.add(repr(float(b)))
    return out


BOUNDS = _bound_tokens()
# A registry display name is a LOINC long common name ("... 2 item (PHQ-2) total score"): its
# digits are part of the name, not a bound.
_REGISTRY_DISPLAYS = frozenset(c.display for c in load_registry().concepts())


def _assert_no_bounds(obj, key=None):
    if key in _EXEMPT_KEYS or (key == "display" and obj in _REGISTRY_DISPLAYS):
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert not set(_NUMBER_TOKEN.findall(str(k))) & BOUNDS, k
            _assert_no_bounds(v, k)
    elif isinstance(obj, list):
        for v in obj:
            _assert_no_bounds(v, key)
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, (int, float)):
        assert f"{obj:g}" not in BOUNDS and repr(float(obj)) not in BOUNDS, (key, obj)
    else:
        hits = set(_NUMBER_TOKEN.findall(str(obj))) & BOUNDS
        assert not hits, (key, obj, hits)


def test_t5_5_no_plausible_bound_in_any_p1_response(client):
    bodies = []

    def keep(r):
        bodies.append(r.json())
        return r

    keep(client.post("/api/journals", json={"member_id": "cur-001"}))
    jr = keep(client.post("/api/journals", json={"member_id": "syn-007"}))
    jid = jr.json()["journal_id"]
    keep(client.get("/api/fixtures"))
    keep(client.get("/api/fixtures/demo-nope"))
    for payload in ({"text": "weighed 160 and slept 30 hours and headache"}, {"text": ""},
                    {"text": "hello there"}, {"text": "x" * 501},
                    {"text": "email a.person@example.org"}, {"text": "I want to end my life"},
                    {"fixture_id": "demo-nope"}, {"text": "slept 6 hours", "fixture_id": "demo-bp-cuff"}):
        keep(client.post("/api/checkins", json={"journal_id": jid, **payload}))
    for fid in manifest():
        keep(client.post("/api/checkins", json={"journal_id": jid, "fixture_id": fid}))
    for data in (UNLISTED_PNG, PDF, b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_UPLOAD_BYTES):
        keep(_upload(client, jid, data))
    keep(_text(client, "f" * 32, "slept 6 hours"))

    ck = _text(client, jid, "weighed 160 and slept 30 hours and headache").json()
    cid = ck["checkin_id"]
    for decisions in (
        [{"proposal_id": "p-1", "action": "confirm"}],
        [{"proposal_id": "p-1", "action": "confirm"}, {"proposal_id": "p-2", "action": "confirm"},
         {"proposal_id": "p-3", "action": "confirm"}],
        [{"proposal_id": "p-1", "action": "correct", "value": 600, "unit": "kg"},
         {"proposal_id": "p-2", "action": "correct", "value": 25},
         {"proposal_id": "p-3", "action": "correct", "value": 1}],
        [{"proposal_id": "p-1", "action": "correct", "unit": "kg", "date": "2020-01-01"},
         {"proposal_id": "p-2", "action": "correct", "value": 7, "unit": "furlong"},
         {"proposal_id": "p-3", "action": "save_unmapped"}],
    ):
        keep(_review(client, cid, decisions))
    ok = keep(_review(client, cid, [
        {"proposal_id": "p-1", "action": "correct", "unit": "[lb_av]"},
        {"proposal_id": "p-2", "action": "correct", "value": 7},
        {"proposal_id": "p-3", "action": "save_unmapped"},
    ]))
    assert ok.status_code == 200, ok.text
    keep(_review(client, cid, []))
    keep(_review(client, "nope", []))
    for rid in ok.json()["records"]:
        keep(client.get(f"/api/journals/{jid}/observations/{rid}"))
    keep(client.get(f"/api/journals/{jid}/observations/obs-999"))

    assert len(bodies) > 25
    for body in bodies:
        _assert_no_bounds(body)
    # the invalid-correction bodies carried per-proposal reasons
    assert any(isinstance(b, dict) and b.get("error") == "invalid_correction" for b in bodies)


def test_signal_events_are_free_text_free(client):
    jid, ck = _pending(client, "slept 6 hours")
    _review(client, ck["checkin_id"], [{"proposal_id": "p-1", "action": "confirm"}])
    (event,) = _signals(_j(client, jid))
    assert isinstance(event, SignalEvent)
    assert set(event.model_dump()) == {"kind", "event_id", "member_id", "occurred_at", "code", "value"}
