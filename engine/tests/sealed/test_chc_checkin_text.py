"""Sealed: text check-in (R2) — T2.2, T2.3, T2.4.

Spec: A&D HAC-33 API table (``POST /api/journals``, ``POST /api/checkins``), section 0 sequence
("nothing saved"), 4.1, 4.2 J1, D-A7 (PII tripwire), "Failure handling" (crisis pre-check before
extraction, PII refusal, empty, nothing extracted), T2.2-T2.4.

The crisis rows come from the sealed boundary corpus (``data/boundary_corpus_v1.jsonl``): the
``crisis_context`` rows whose text carries a term of the ``boundary-v1`` crisis vocabulary
(``dsl.load_rules().crisis_markers``), which is the vocabulary the A&D says the pre-check reuses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.sealed.conftest_chc import (
    body,
    chc_app,  # noqa: F401 - fixture
    client,  # noqa: F401 - fixture
    concepts_by_code,
    has_error,
    live_checkins,
    new_journal,
    observed_cards,
    passport_or_skip,
    post_text,
    writes,  # noqa: F401 - fixture
)

pytestmark = pytest.mark.category("integration/chc-checkin-text")

T22_TEXT = "slept 6 hours and walked 8000 steps"
SLEEP, STEPS = "93832-4", "55423-8"

CORPUS = Path(__file__).with_name("data") / "boundary_corpus_v1.jsonl"


def _crisis_rows() -> list[dict]:
    from daypart.membrane import dsl

    markers = [m.lower() for m in dsl.load_rules().crisis_markers]
    rows = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = [
        r
        for r in rows
        if r.get("crisis_context") and r["text"].strip() and any(m in r["text"].lower() for m in markers)
    ]
    assert out, "no crisis-vocabulary rows in the sealed corpus"
    return out


def _handoff_text() -> str:
    from daypart.eval import POLICY_TEXT

    return POLICY_TEXT["clinician_handoff"]


# --------------------------------------------------------------------------- #
# Journals (API table)
# --------------------------------------------------------------------------- #


class TestJournals:
    def test_journal_created_for_a_synthetic_member(self, client):
        """API table: POST /api/journals {member_id: syn-NNN} -> 201 {journal_id, member_id, today}."""
        j = new_journal(client)
        assert j["member_id"].startswith("syn-")
        assert isinstance(j["journal_id"], str) and j["journal_id"]

    @pytest.mark.parametrize("member_id", ["cur-001", "syn-999"])
    def test_curated_or_unknown_member_is_404(self, client, member_id):
        """API table / I1: an unknown or curated member -> 404."""
        new_journal(client)  # the route exists and accepts a training-cohort member
        r = client.post("/api/journals", json={"member_id": member_id})
        assert r.status_code == 404, r.text

    def test_journals_are_isolated(self, client):
        """D-A3: two journals get distinct ids."""
        assert new_journal(client)["journal_id"] != new_journal(client)["journal_id"]

    def test_unknown_journal_checkin_is_404(self, client):
        """API table: POST /api/checkins for an unknown journal -> 404."""
        j = new_journal(client)
        assert post_text(client, j["journal_id"], T22_TEXT).status_code == 201
        r = post_text(client, "0" * 32, T22_TEXT)
        assert r.status_code == 404, r.text


# --------------------------------------------------------------------------- #
# T2.2  text -> proposals; nothing written before review
# --------------------------------------------------------------------------- #


class TestT22TextProposals:
    def test_t2_2_two_proposals_with_values(self, client):
        """T2.2: "slept 6 hours and walked 8000 steps" -> 201, >= 2 proposals:
        93832-4 = 6 h and 55423-8 = 8000."""
        j = new_journal(client)
        r = post_text(client, j["journal_id"], T22_TEXT)
        assert r.status_code == 201, r.text
        c = r.json()
        assert c["status"] == "pending_review"
        assert c["checkin_id"]
        props = c["proposals"]
        assert len(props) >= 2
        sleep = [p for p in props if p.get("concept_code") == SLEEP]
        steps = [p for p in props if p.get("concept_code") == STEPS]
        assert len(sleep) == 1 and len(steps) == 1, props
        assert sleep[0]["value"] == 6
        assert sleep[0]["unit"] == "h"
        assert steps[0]["value"] == 8000
        for p in (sleep[0], steps[0]):
            assert p["source_kind"] == "text"
            assert p["method"] == "rules-v1"

    def test_t2_2_nothing_written_before_review(self, client, writes):
        """T2.2 / SC9: immediately after the check-in the EventLog is unchanged."""
        j = new_journal(client)
        writes.clear()
        r = post_text(client, j["journal_id"], T22_TEXT)
        assert r.status_code == 201, r.text
        assert len(writes) == 0, writes.events

    def test_t2_2_no_observation_record_before_review(self, client):
        """T2.2 / SC9: no inspectable observation record exists before review."""
        j = new_journal(client)
        assert post_text(client, j["journal_id"], T22_TEXT).status_code == 201
        for n in range(10):
            r = client.get(f"/api/journals/{j['journal_id']}/observations/obs-{n}")
            assert r.status_code == 404, (n, r.text)

    def test_t2_2_no_observed_passport_card_before_review(self, client):
        """T2.2 / SC9: GET passport has no observed card for either concept before review."""
        j = new_journal(client)
        assert post_text(client, j["journal_id"], T22_TEXT).status_code == 201
        pp = passport_or_skip(client, j["journal_id"])
        reg = concepts_by_code()
        ids = {reg[c]["concept_id"] for c in (SLEEP, STEPS) if c in reg}
        assert observed_cards(pp, {SLEEP, STEPS}, ids) == []

    def test_t2_2_text_limit(self, client):
        """API table: text is <= 500 chars; a longer text is refused and nothing is created."""
        j = new_journal(client)
        before = live_checkins()
        r = post_text(client, j["journal_id"], ("slept 6 hours " * 40)[:501])
        assert 400 <= r.status_code < 500, r.text
        assert live_checkins() == before

    @pytest.mark.parametrize("text", ["", "   ", "\n\t "])
    def test_t2_2_empty_checkin(self, client, text, writes):
        """Failure handling: empty or whitespace text and no fixture -> 422 empty_checkin."""
        j = new_journal(client)
        before = live_checkins()
        r = post_text(client, j["journal_id"], text)
        assert r.status_code == 422, r.text
        assert has_error(r, "empty_checkin"), body(r)
        assert live_checkins() == before
        assert len(writes) == 0

    def test_t2_2_nothing_extracted(self, client, writes):
        """Failure handling: text that yields no proposal -> 422 nothing_extracted; no CheckIn."""
        j = new_journal(client)
        before = live_checkins()
        r = post_text(client, j["journal_id"], "the quick brown fox jumps over the lazy dog")
        assert r.status_code == 422, r.text
        assert has_error(r, "nothing_extracted"), body(r)
        assert live_checkins() == before
        assert len(writes) == 0


# --------------------------------------------------------------------------- #
# T2.3  PII tripwire
# --------------------------------------------------------------------------- #

PII_TEXTS = [
    pytest.param("slept 6 hours, mail me at jane.doe@example.com", id="email"),
    pytest.param("walked 8000 steps, call me on +1 415 555 0134", id="phone-intl"),
    pytest.param("slept 7 hours. my number is (415) 555-0134", id="phone-us"),
    pytest.param("contact: someone@example.org slept 5 hours", id="email-lead"),
]


class TestT23PersonalData:
    @pytest.mark.parametrize("text", PII_TEXTS)
    def test_t2_3_refused_and_no_checkin(self, client, writes, text):
        """T2.3: text with an e-mail or phone number -> 422 looks_like_personal_data,
        no CheckIn exists, nothing written."""
        j = new_journal(client)
        before = live_checkins()
        writes.clear()
        r = post_text(client, j["journal_id"], text)
        assert r.status_code == 422, r.text
        assert has_error(r, "looks_like_personal_data"), body(r)
        b = body(r)
        assert "checkin_id" not in json.dumps(b), b
        assert "proposals" not in (b if isinstance(b, dict) else {}), b
        assert live_checkins() == before
        assert len(writes) == 0

    @pytest.mark.parametrize("text", PII_TEXTS)
    def test_t2_3_refusal_does_not_echo_the_personal_data(self, client, text):
        """D-A7: the refusal lists reasons; it never echoes the e-mail or phone back."""
        j = new_journal(client)
        r = post_text(client, j["journal_id"], text)
        dumped = json.dumps(body(r))
        for needle in ("jane.doe@example.com", "someone@example.org", "555 0134", "555-0134"):
            assert needle not in dumped

    def test_t2_3_journal_still_usable_after_refusal(self, client):
        """T2.3: the refusal leaves the journal unchanged and usable."""
        j = new_journal(client)
        post_text(client, j["journal_id"], "mail jane.doe@example.com slept 6 hours")
        r = post_text(client, j["journal_id"], T22_TEXT)
        assert r.status_code == 201, r.text


# --------------------------------------------------------------------------- #
# T2.4  crisis language -> handoff, before extraction
# --------------------------------------------------------------------------- #


class TestT24CrisisHandoff:
    @pytest.mark.parametrize("row", _crisis_rows(), ids=lambda r: r["id"])
    def test_t2_4_handoff_no_checkin_no_events(self, client, writes, row):
        """T2.4: crisis text (sealed boundary corpus, crisis kind) -> 200 status=handoff with the
        clinician-handoff text; no CheckIn and no events."""
        j = new_journal(client)
        before = live_checkins()
        writes.clear()
        r = post_text(client, j["journal_id"], row["text"])
        assert r.status_code == 200, r.text
        b = r.json()
        assert b["status"] == "handoff"
        assert _handoff_text() in b["text"]
        assert "checkin_id" not in b or b["checkin_id"] is None
        assert not b.get("proposals")
        assert live_checkins() == before
        assert len(writes) == 0

    def test_t2_4_crisis_checked_before_extraction(self, client, writes):
        """Failure handling: the crisis pre-check runs before extraction — a crisis text that also
        holds an extractable value still yields a handoff and nothing extracted or stored."""
        from daypart.membrane import dsl

        marker = next(m for m in dsl.load_rules().crisis_markers if not m.isdigit())
        j = new_journal(client)
        before = live_checkins()
        writes.clear()
        r = post_text(client, j["journal_id"], f"slept 6 hours and I need the {marker} now")
        assert r.status_code == 200, r.text
        b = r.json()
        assert b["status"] == "handoff"
        assert not b.get("proposals")
        assert live_checkins() == before
        assert len(writes) == 0
