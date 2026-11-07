"""Sealed: review before saving (R4) and the R5 inspect view — T4.1-T4.5, T5.3.

Spec: A&D HAC-33 4.1 (Proposal, CheckIn state machine, confidence rules, every flag blocking,
flag re-run after corrections), 4.2 (ObservationRecord, J1, J2), API table
(``POST /api/checkins/{id}/review``, ``GET /api/journals/{id}/observations/{record_id}``),
"Failure handling" (missing_decision, unresolved_flags, invalid_correction, already_saved,
contradiction, relative dates, unknown word, no partial write, concurrency), T4.1-T4.5, T5.3;
AM-1 item 1 (``reviewer_status`` label); plan G9 (correction date window [today-28, today]).

T4.2 asks for "a fixture input producing" each flag kind. The A&D names the trigger for most
kinds but not the exact wording, so each kind has a short list of candidate inputs written from
the A&D rules (plus every bundled fixture); the first one that raises the flag is used, and the
test fails if none does.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pytest

from tests.sealed.conftest_chc import (
    FLAG_KINDS,
    OBSERVATION_FIELDS,
    PROVENANCE_FIELDS,
    REVIEWED_LABEL,
    body,
    by_concept,
    chc_app,  # noqa: F401 - fixture
    checkin_ok,
    client,  # noqa: F401 - fixture
    concepts_by_code,
    discard_all_but,
    find_observations,
    fixture_manifest,
    has_error,
    new_journal,
    observed_cards,
    passport_or_skip,
    plausible_bounds,
    post_fixture,
    post_text,
    review,
    valid_correction,
    writes,  # noqa: F401 - fixture
)

pytestmark = pytest.mark.category("integration/chc-review")

T22_TEXT = "slept 6 hours and walked 8000 steps"
SLEEP, STEPS, SYS, DIA = "93832-4", "55423-8", "8480-6", "8462-4"

FLAG_CANDIDATES: dict[str, tuple[str, ...]] = {
    # 4.1: keyword + number, unit taken from the concept default
    "unit_assumed": ("heart rate 72", "my heart rate was 72 today", "weight 70", "slept 7"),
    # a number with a unit the concept does not allow
    "unit_mismatch": ("weight 70 cm", "slept 6 kg", "heart rate 72 kg", "body temperature 37 mmHg"),
    # Failure handling: a weekday name or "last week" is ambiguous
    "ambiguous_date": ("slept 7 hours on monday", "slept 7 hours last week", "walked 8000 steps on Friday"),
    # Failure handling: systolic <= diastolic, or two values for one concept
    "contradiction": (
        "blood pressure 80/120 mmHg",
        "systolic 80 mmHg diastolic 120 mmHg",
        "slept 6 hours and slept 8 hours",
        "weight 70 kg and weight 90 kg",
    ),
    # Failure handling: an unknown word ("headache")
    "unknown_code": ("headache", "bad headache today", "headache 6/10"),
    # registry plausible bounds (validation only)
    "out_of_plausible_range": ("heart rate 900 /min", "slept 30 hours", "walked 9000000 steps", "weight 900 kg"),
    # keyword without a number
    "missing_value": ("slept badly today", "slept", "I walked today", "pain today"),
}


def _all_flag_inputs():
    for kind, texts in FLAG_CANDIDATES.items():
        for t in texts:
            yield kind, ("text", t)


def _checkin_for_flag(client, kind: str):
    """(journal, checkin, target proposal) for the first candidate input raising ``kind``."""
    tried = []
    sources = [("text", t) for t in FLAG_CANDIDATES[kind]]
    try:
        sources += [("fixture", fid) for fid in sorted(fixture_manifest())]
    except Exception:  # noqa: BLE001 - fixtures are a fallback source only
        pass
    for src, value in sources:
        j = new_journal(client)
        r = post_text(client, j["journal_id"], value) if src == "text" else post_fixture(client, j["journal_id"], value)
        tried.append(f"{src}:{value!r}->{r.status_code}")
        if r.status_code != 201:
            continue
        c = r.json()
        hits = [p for p in c["proposals"] if kind in p.get("flags", [])]
        if hits:
            return j, c, hits[0]
    pytest.fail(f"no input raised flag {kind!r}; tried {tried}")


# --------------------------------------------------------------------------- #
# T4.1  every proposal carries snippet, confidence, unit, date, flags
# --------------------------------------------------------------------------- #


def _assert_proposal_shape(p: dict) -> None:
    for key in ("proposal_id", "concept_code", "display", "value", "unit", "date", "source_kind",
                "source_ref", "snippet", "method", "confidence", "flags", "blocking"):
        assert key in p, f"proposal lacks {key!r}: {p}"
    assert isinstance(p["snippet"], str) and p["snippet"].strip(), p
    assert len(p["snippet"]) <= 80, p
    assert p["confidence"] in ("high", "medium", "low"), p
    assert p["source_kind"] in ("text", "file_fixture", "image_fixture"), p
    assert p["method"] in ("rules-v1", "fixture-sidecar-v1"), p
    flags = list(p["flags"])
    assert set(flags) <= set(FLAG_KINDS), p
    # every flag is blocking; no soft flags (4.1)
    assert p["blocking"] is bool(flags), p
    if p["unit"] is None:
        assert {"missing_value", "unit_assumed", "unknown_code"} & set(flags), p
    if p["date"] is None:
        assert "ambiguous_date" in flags, p
    if p["value"] is None:
        assert {"missing_value", "unknown_code"} & set(flags), p
    if p["concept_code"] is None:
        assert "unknown_code" in flags, p
    if p["method"] == "rules-v1":
        if p["confidence"] == "medium":
            assert "unit_assumed" in flags, p
        if p["confidence"] == "high":
            assert p["value"] is not None and p["unit"] is not None and "unit_assumed" not in flags, p


class TestT41ProposalShape:
    def test_t4_1_t22_text(self, client):
        """T4.1: the T2.2 proposals carry every field; both are high confidence with no flags."""
        j = new_journal(client)
        c = checkin_ok(post_text(client, j["journal_id"], T22_TEXT))
        ids = [p["proposal_id"] for p in c["proposals"]]
        assert len(ids) == len(set(ids))
        for p in c["proposals"]:
            _assert_proposal_shape(p)
        for code in (SLEEP, STEPS):
            (p,) = by_concept(c["proposals"], code)
            assert p["confidence"] == "high", p
            assert p["flags"] == [] and p["blocking"] is False, p
            assert p["date"] == j["today"], p

    @pytest.mark.parametrize(("kind", "src"), list(_all_flag_inputs()), ids=lambda x: str(x))
    def test_t4_1_candidate_inputs(self, client, kind, src):
        """T4.1: every proposal from every candidate input carries the T4.1 fields."""
        j = new_journal(client)
        r = post_text(client, j["journal_id"], src[1])
        if r.status_code != 201:
            pytest.skip(f"{src[1]!r} produced no check-in ({r.status_code})")
        for p in r.json()["proposals"]:
            _assert_proposal_shape(p)

    def test_t4_1_fixtures(self, client):
        """T4.1: every proposal from every bundled fixture carries the T4.1 fields."""
        for fid in sorted(fixture_manifest()):
            j = new_journal(client)
            c = checkin_ok(post_fixture(client, j["journal_id"], fid))
            for p in c["proposals"]:
                _assert_proposal_shape(p)

    def test_t4_1_relative_dates_resolve(self, client):
        """Failure handling: "today"/"yesterday" resolve to journal dates (not ambiguous)."""
        j = new_journal(client)
        today = date.fromisoformat(j["today"])
        c = checkin_ok(post_text(client, j["journal_id"], "slept 7 hours yesterday"))
        (p,) = by_concept(c["proposals"], SLEEP)
        assert p["date"] == (today - timedelta(days=1)).isoformat(), p
        assert "ambiguous_date" not in p["flags"]


# --------------------------------------------------------------------------- #
# T4.2  each flag kind blocks confirm; correct / save_unmapped resolves it
# --------------------------------------------------------------------------- #


class TestT42Flags:
    @pytest.mark.parametrize("kind", FLAG_KINDS)
    def test_t4_2_flag_is_blocking_and_confirm_refused(self, client, writes, kind):
        """T4.2: an input producing the flag -> blocking=true; confirm -> 422 unresolved_flags,
        nothing written; the check-in stays pending (all-or-nothing)."""
        j, c, target = _checkin_for_flag(client, kind)
        assert target["blocking"] is True
        writes.clear()
        decisions = discard_all_but(c["proposals"], {target["proposal_id"]: {"action": "confirm"}})
        r = review(client, c["checkin_id"], decisions)
        assert r.status_code == 422, r.text
        assert has_error(r, "unresolved_flags"), body(r)
        assert len(writes) == 0
        assert find_observations(client, j["journal_id"], limit=10) == []
        # still pending: an all-discard review now succeeds (no 409)
        r2 = review(client, c["checkin_id"], discard_all_but(c["proposals"], {}))
        assert r2.status_code == 200, r2.text
        assert len(writes) == 0

    @pytest.mark.parametrize("kind", [k for k in FLAG_KINDS if k != "unknown_code"])
    def test_t4_2_correct_with_valid_values_saves(self, client, writes, kind):
        """T4.2: `correct` with valid values -> 200 and exactly the corrected values written."""
        concepts = concepts_by_code()
        j, c, target = _checkin_for_flag(client, kind)
        keep: dict[str, dict] = {}
        if kind == "contradiction" and by_concept(c["proposals"], SYS) and by_concept(c["proposals"], DIA):
            for code, v in ((SYS, 120), (DIA, 80)):
                for p in by_concept(c["proposals"], code):
                    keep[p["proposal_id"]] = {**valid_correction(p, j["today"], concepts), "value": v}
                    break
        else:
            assert target["concept_code"] in concepts, target
            keep[target["proposal_id"]] = valid_correction(target, j["today"], concepts)
        writes.clear()
        r = review(client, c["checkin_id"], discard_all_but(c["proposals"], keep))
        assert r.status_code == 200, (kind, keep, r.text)
        out = r.json()
        assert out["status"] == "saved"
        assert len(out["written"]) == len(keep)
        assert len(writes.signals) == len(keep)
        assert {e.event_id for e in writes.signals} == set(out["written"])
        corrected_values = sorted(float(d["value"]) for d in keep.values())
        assert sorted(float(e.value) for e in writes.signals) == corrected_values

    def test_t4_2_save_unmapped_stores_no_signal(self, client, writes):
        """T4.2 / J2: `save_unmapped` -> stored with status="unmapped" and no SignalEvent."""
        j, c, target = _checkin_for_flag(client, "unknown_code")
        writes.clear()
        r = review(client, c["checkin_id"], discard_all_but(c["proposals"], {target["proposal_id"]: {"action": "save_unmapped"}}))
        assert r.status_code == 200, r.text
        assert r.json()["written"] == [] or all(w is None for w in r.json()["written"])
        assert len(writes.signals) == 0
        recs = find_observations(client, j["journal_id"])
        assert len(recs) == 1, recs
        assert recs[0]["status"] == "unmapped"
        assert recs[0]["event_id"] is None

    def test_t4_2_unit_assumed_cleared_by_explicit_unit(self, client, writes):
        """4.1: unit_assumed clears when `correct` states a unit explicitly."""
        concepts = concepts_by_code()
        j, c, target = _checkin_for_flag(client, "unit_assumed")
        fix = {"action": "correct", "value": target["value"], "unit": concepts[target["concept_code"]]["unit"], "date": j["today"]}
        lo, hi = plausible_bounds(concepts[target["concept_code"]])
        if fix["value"] is None or not lo <= fix["value"] <= hi:
            fix = valid_correction(target, j["today"], concepts)
        r = review(client, c["checkin_id"], discard_all_but(c["proposals"], {target["proposal_id"]: fix}))
        assert r.status_code == 200, r.text
        assert len(writes.signals) == 1

    def test_t4_2_contradiction_flags_both_sides(self, client):
        """Failure handling: systolic <= diastolic (or two values for one concept) -> both get
        a contradiction flag."""
        j, c, target = _checkin_for_flag(client, "contradiction")
        flagged = [p for p in c["proposals"] if "contradiction" in p["flags"]]
        assert len(flagged) >= 2, c["proposals"]
        assert all(p["blocking"] for p in flagged)

    def test_t4_2_correcting_one_side_clears_both(self, client, writes):
        """4.1: flags are re-run over the post-correction set; correcting one side of a sys/dia
        contradiction so they no longer conflict lets both be saved."""
        concepts = concepts_by_code()
        j = new_journal(client)
        hit = None
        for text in ("blood pressure 80/120 mmHg", "systolic 80 mmHg diastolic 120 mmHg"):
            r = post_text(client, j["journal_id"], text)
            if r.status_code == 201:
                c = r.json()
                s, d = by_concept(c["proposals"], SYS), by_concept(c["proposals"], DIA)
                if s and d and "contradiction" in s[0]["flags"] and "contradiction" in d[0]["flags"]:
                    hit = (c, s[0], d[0])
                    break
        if hit is None:
            pytest.fail("no candidate produced a sys/dia contradiction")
        c, s, d = hit
        keep = {
            s["proposal_id"]: {**valid_correction(s, j["today"], concepts), "value": 140},
            d["proposal_id"]: {**valid_correction(d, j["today"], concepts), "value": d["value"]},
        }
        writes.clear()
        r = review(client, c["checkin_id"], discard_all_but(c["proposals"], keep))
        assert r.status_code == 200, r.text
        assert len(writes.signals) == 2


class TestT42ReviewValidation:
    def _clean(self, client):
        j = new_journal(client)
        c = checkin_ok(post_text(client, j["journal_id"], T22_TEXT))
        (sleep,) = by_concept(c["proposals"], SLEEP)
        return j, c, sleep

    def test_missing_decision(self, client, writes):
        """Failure handling: a review missing a decision -> 422 missing_decision, nothing written."""
        j, c, sleep = self._clean(client)
        writes.clear()
        r = review(client, c["checkin_id"], [{"proposal_id": sleep["proposal_id"], "action": "confirm"}])
        assert r.status_code == 422, r.text
        assert has_error(r, "missing_decision"), body(r)
        assert len(writes) == 0
        r2 = review(client, c["checkin_id"], discard_all_but(c["proposals"], {}))
        assert r2.status_code == 200, r2.text

    @pytest.mark.parametrize("bad", ["unit", "value_high", "value_low", "date_future", "date_old"])
    def test_invalid_correction(self, client, writes, bad):
        """Failure handling: a correction with a unit not allowed, a value outside the
        plausibility bounds, or a date outside [today-28, today] -> 422 invalid_correction;
        the whole review is refused."""
        concepts = concepts_by_code()
        j, c, sleep = self._clean(client)
        concept = concepts[SLEEP]
        lo, hi = plausible_bounds(concept)
        today = date.fromisoformat(j["today"])
        fix = valid_correction(sleep, j["today"], concepts)
        if bad == "unit":
            fix["unit"] = "kg"
        elif bad == "value_high":
            fix["value"] = hi + max(1.0, abs(hi))
        elif bad == "value_low":
            fix["value"] = lo - max(1.0, abs(lo))
        elif bad == "date_future":
            fix["date"] = (today + timedelta(days=1)).isoformat()
        else:
            fix["date"] = (today - timedelta(days=29)).isoformat()
        writes.clear()
        r = review(client, c["checkin_id"], discard_all_but(c["proposals"], {sleep["proposal_id"]: fix}))
        assert r.status_code == 422, r.text
        assert has_error(r, "invalid_correction"), body(r)
        assert len(writes) == 0
        assert find_observations(client, j["journal_id"], limit=10) == []

    def test_correction_window_edges_accepted(self, client, writes):
        """Plan G9: the window [today-28, today] is inclusive at both ends."""
        concepts = concepts_by_code()
        for days in (0, 28):
            j, c, sleep = self._clean(client)
            fix = valid_correction(sleep, j["today"], concepts)
            fix["date"] = (date.fromisoformat(j["today"]) - timedelta(days=days)).isoformat()
            r = review(client, c["checkin_id"], discard_all_but(c["proposals"], {sleep["proposal_id"]: fix}))
            assert r.status_code == 200, (days, r.text)

    def test_unknown_checkin_is_404(self, client):
        """API table: review of an unknown check-in -> 404."""
        j, c, sleep = self._clean(client)
        r = review(client, "no-such-checkin", [])
        assert r.status_code == 404, r.text


# --------------------------------------------------------------------------- #
# T4.3  discard -> absent from EventLog, provenance and passport
# --------------------------------------------------------------------------- #


class TestT43Discard:
    def test_t4_3_discarded_value_absent(self, client, writes):
        """T4.3: confirm sleep, discard steps -> steps absent from EventLog and provenance."""
        j = new_journal(client)
        c = checkin_ok(post_text(client, j["journal_id"], T22_TEXT))
        (sleep,) = by_concept(c["proposals"], SLEEP)
        (steps,) = by_concept(c["proposals"], STEPS)
        writes.clear()
        r = review(client, c["checkin_id"], discard_all_but(c["proposals"], {sleep["proposal_id"]: {"action": "confirm"}}))
        assert r.status_code == 200, r.text
        assert steps["proposal_id"] in r.json()["discarded"]
        assert all(e.code != STEPS for e in writes.signals)
        recs = find_observations(client, j["journal_id"])
        assert all(rec.get("loinc_code") != STEPS for rec in recs), recs
        assert all(rec.get("value") != 8000 for rec in recs), recs

    def test_t4_3_discard_all_writes_nothing(self, client, writes):
        """T4.3: discarding every proposal saves the check-in and writes nothing."""
        j = new_journal(client)
        c = checkin_ok(post_text(client, j["journal_id"], T22_TEXT))
        writes.clear()
        r = review(client, c["checkin_id"], discard_all_but(c["proposals"], {}))
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["status"] == "saved"
        assert out["written"] == []
        assert sorted(out["discarded"]) == sorted(p["proposal_id"] for p in c["proposals"])
        assert len(writes) == 0
        assert find_observations(client, j["journal_id"]) == []

    def test_t4_3_discarded_value_absent_from_passport(self, client):
        """T4.3: a discarded value never reaches the passport (no observed card)."""
        j = new_journal(client)
        c = checkin_ok(post_text(client, j["journal_id"], T22_TEXT))
        assert review(client, c["checkin_id"], discard_all_but(c["proposals"], {})).status_code == 200
        pp = passport_or_skip(client, j["journal_id"])
        reg = concepts_by_code()
        ids = {reg[k]["concept_id"] for k in (SLEEP, STEPS) if k in reg}
        assert observed_cards(pp, {SLEEP, STEPS}, ids) == []


# --------------------------------------------------------------------------- #
# T4.4  confirm 1 of 2 -> exactly 1 SignalEvent; second review -> 409
# --------------------------------------------------------------------------- #


class TestT44ConfirmOne:
    def test_t4_4_one_write_then_409(self, client, writes):
        """T4.4: confirm 1 of 2 -> exactly 1 SignalEvent written; a second review -> 409
        already_saved and nothing more written."""
        j = new_journal(client)
        c = checkin_ok(post_text(client, j["journal_id"], T22_TEXT))
        (sleep,) = by_concept(c["proposals"], SLEEP)
        writes.clear()
        decisions = discard_all_but(c["proposals"], {sleep["proposal_id"]: {"action": "confirm"}})
        r = review(client, c["checkin_id"], decisions)
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["status"] == "saved"
        assert len(out["written"]) == 1
        assert len(writes.signals) == 1 and len(writes) == 1
        ev = writes.signals[0]
        assert ev.event_id == out["written"][0]
        assert ev.code == SLEEP
        assert ev.value == 6
        assert ev.member_id == j["member_id"]
        assert ev.occurred_at.date().isoformat() == j["today"]

        r2 = review(client, c["checkin_id"], decisions)
        assert r2.status_code == 409, r2.text
        assert has_error(r2, "already_saved"), body(r2)
        assert len(writes) == 1

    def test_t4_4_confirmed_value_reaches_passport(self, client):
        """SC9 (positive side): after confirmation the value is an observed card."""
        j = new_journal(client)
        c = checkin_ok(post_text(client, j["journal_id"], T22_TEXT))
        (sleep,) = by_concept(c["proposals"], SLEEP)
        r = review(client, c["checkin_id"], discard_all_but(c["proposals"], {sleep["proposal_id"]: {"action": "confirm"}}))
        assert r.status_code == 200
        pp = passport_or_skip(client, j["journal_id"])
        reg = concepts_by_code()
        cards = observed_cards(pp, {SLEEP}, {reg[SLEEP]["concept_id"]})
        assert len(cards) == 1, pp.get("cards")


# --------------------------------------------------------------------------- #
# T4.5  concurrent double review -> one 200, one 409, one write
# --------------------------------------------------------------------------- #


class TestT45Concurrency:
    def test_t4_5_concurrent_double_review(self, chc_app, writes):
        """T4.5: two concurrent reviews of one check-in -> exactly one 200 and one 409, and one
        review's writes."""
        from fastapi.testclient import TestClient

        with TestClient(chc_app) as c:
            for _ in range(5):
                j = new_journal(c)
                ci = checkin_ok(post_text(c, j["journal_id"], T22_TEXT))
                (sleep,) = by_concept(ci["proposals"], SLEEP)
                decisions = discard_all_but(ci["proposals"], {sleep["proposal_id"]: {"action": "confirm"}})
                writes.clear()
                barrier = threading.Barrier(2)

                def go():
                    barrier.wait()
                    return review(c, ci["checkin_id"], decisions)

                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda _: go(), range(2)))
                codes = sorted(r.status_code for r in results)
                assert codes == [200, 409], [(r.status_code, r.text) for r in results]
                loser = next(r for r in results if r.status_code == 409)
                assert has_error(loser, "already_saved"), body(loser)
                assert len(writes.signals) == 1


# --------------------------------------------------------------------------- #
# T5.3  inspect view for a confirmed and an unmapped value
# --------------------------------------------------------------------------- #


def _assert_record_fields(rec: dict) -> None:
    for key in OBSERVATION_FIELDS:
        assert key in rec, f"ObservationRecord lacks {key!r}: {rec}"
    for key in PROVENANCE_FIELDS:
        assert key in rec["provenance"], f"provenance lacks {key!r}: {rec}"
    assert str(rec["record_id"]).startswith("obs-"), rec
    assert rec["quality"] in ("confirmed", "corrected"), rec
    assert rec["status"] in ("mapped", "unmapped"), rec
    assert rec["reviewer_status"] == REVIEWED_LABEL, rec  # AM-1 item 1
    assert isinstance(rec["allowed_use"], str) and rec["allowed_use"].strip(), rec


class TestT53InspectView:
    def test_t5_3_confirmed_value(self, client):
        """T5.3: GET observations/{record_id} returns every R5 field for a confirmed value."""
        import daypart.memory.registry as reg_mod

        concepts = concepts_by_code()
        j = new_journal(client)
        c = checkin_ok(post_text(client, j["journal_id"], T22_TEXT))
        (sleep,) = by_concept(c["proposals"], SLEEP)
        r = review(client, c["checkin_id"], discard_all_but(c["proposals"], {sleep["proposal_id"]: {"action": "confirm"}}))
        assert r.status_code == 200, r.text
        (event_id,) = r.json()["written"]
        recs = [x for x in find_observations(client, j["journal_id"]) if x.get("event_id") == event_id]
        assert len(recs) == 1, find_observations(client, j["journal_id"])
        rec = recs[0]
        _assert_record_fields(rec)
        direct = client.get(f"/api/journals/{j['journal_id']}/observations/{rec['record_id']}")
        assert direct.status_code == 200 and direct.json() == rec
        assert rec["status"] == "mapped"
        assert rec["quality"] == "confirmed"
        assert rec["loinc_code"] == SLEEP
        assert rec["loinc_version"] == reg_mod.LOINC_VERSION
        assert rec["concept_id"] == concepts[SLEEP]["concept_id"]
        assert rec["value"] == 6
        assert rec["unit"] == "h"
        assert rec["value_type"] == concepts[SLEEP]["value_type"]
        assert str(rec["observed_at"]).startswith(j["today"]), rec
        prov = rec["provenance"]
        assert prov["source_kind"] == "text"
        assert prov["checkin_id"] == c["checkin_id"]
        assert prov["method"] == "rules-v1"
        assert c["checkin_id"] in str(prov["source_ref"])

    def test_t5_3_corrected_value_quality(self, client):
        """4.2: a corrected value's record has quality="corrected" and the corrected value."""
        concepts = concepts_by_code()
        j = new_journal(client)
        c = checkin_ok(post_text(client, j["journal_id"], T22_TEXT))
        (sleep,) = by_concept(c["proposals"], SLEEP)
        fix = {**valid_correction(sleep, j["today"], concepts), "value": 7}
        r = review(client, c["checkin_id"], discard_all_but(c["proposals"], {sleep["proposal_id"]: fix}))
        assert r.status_code == 200, r.text
        (event_id,) = r.json()["written"]
        (rec,) = [x for x in find_observations(client, j["journal_id"]) if x.get("event_id") == event_id]
        _assert_record_fields(rec)
        assert rec["quality"] == "corrected"
        assert rec["value"] == 7

    def test_t5_3_unmapped_value(self, client, writes):
        """T5.3: a save_unmapped value -> status="unmapped", loinc_code=null, event_id=null, and
        every R5 field present."""
        j, c, target = _checkin_for_flag(client, "unknown_code")
        writes.clear()
        r = review(client, c["checkin_id"], discard_all_but(c["proposals"], {target["proposal_id"]: {"action": "save_unmapped"}}))
        assert r.status_code == 200, r.text
        recs = find_observations(client, j["journal_id"])
        assert len(recs) == 1, recs
        rec = recs[0]
        _assert_record_fields(rec)
        assert rec["status"] == "unmapped"
        assert rec["loinc_code"] is None
        assert rec["event_id"] is None
        assert rec["provenance"]["checkin_id"] == c["checkin_id"]
        assert len(writes.signals) == 0

    def test_t5_3_unknown_journal_or_record_is_404(self, client):
        """API table: unknown journal or record -> 404."""
        j = new_journal(client)
        assert client.get(f"/api/journals/{j['journal_id']}/observations/obs-999").status_code == 404
        assert client.get(f"/api/journals/{'f' * 32}/observations/obs-1").status_code == 404

    def test_t5_3_records_are_per_journal(self, client):
        """D-A3: a record saved in one journal is not visible from another journal."""
        j1, j2 = new_journal(client), new_journal(client)
        c = checkin_ok(post_text(client, j1["journal_id"], T22_TEXT))
        (sleep,) = by_concept(c["proposals"], SLEEP)
        review(client, c["checkin_id"], discard_all_but(c["proposals"], {sleep["proposal_id"]: {"action": "confirm"}}))
        assert find_observations(client, j1["journal_id"])
        assert find_observations(client, j2["journal_id"]) == []
