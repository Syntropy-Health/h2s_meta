"""Check-in service (A&D §4.1-§4.2, failure table): create a pending check-in, then review it.

Framework-free. Every refusal raises :class:`CheckinError`, which carries an HTTP ``status`` and a
``body`` with ``"error"`` at the top level; the HTTP layer only serialises it.

* :func:`create_checkin` runs gate -> shape checks -> crisis pre-check -> PII tripwire -> extract
  (text), or fixture lookup / upload intake. It stores a ``pending_review`` CheckIn and writes
  nothing to ``events`` or ``provenance`` (J1). A refusal leaves the journal exactly as it was.
* :func:`review` is all-or-nothing under the journal lock: validate every decision, apply the
  corrections, re-run flags over the kept set, build and pre-check every SignalEvent, and only
  then write events, records and the ``saved`` status together. New events are dry-run on a
  throwaway EventLog seeded with the journal's events; a commit that still fails mid-way rebuilds
  the log from the prior events and is a 500 with nothing written.

No message here ever names a ``plausible`` bound (T5.5): reasons are fixed codes.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import pydantic
from pydantic import ConfigDict, Field

from daypart.checkin import fixtures
from daypart.checkin.crisis import HANDOFF_TEXT, is_crisis
from daypart.checkin.extract_text import detect_flags, extract
from daypart.checkin.journal import Journal, ObservationRecord, Provenance
from daypart.checkin.models import CheckIn, Proposal
from daypart.checkin.tripwire import scan
from daypart.governance import real_input
from daypart.memory.events import EventLog, SignalEvent
from daypart.memory.registry import REVIEWER_STATUS, Concept, load_registry

MAX_TEXT_CHARS = 500
# Hard bound on the raw text, checked before any normalisation (NFKC can expand one character to
# 18, so normalising an unbounded body blocks the loop). Generous (4x) so crisis wording in a
# slightly long text still reaches the crisis pre-check and gets the handoff.
MAX_RAW_TEXT_CHARS = 4 * MAX_TEXT_CHARS
WINDOW_DAYS = 28
PII_WARNING = "Do not enter real health data."
NOTHING_EXTRACTED_HINT = "try e.g. 'slept 7 hours'"
NOT_A_DEMO_FIXTURE_MESSAGE = "Only the bundled demo files can be used."
UNMAPPED_ALLOWED_USE = "demo display of self-reported values; no interpretation"
UPLOAD_MESSAGES: Mapping[str, str] = {
    "too_large": "The file is too large for the demo.",
    "unsupported_type": "Only PNG, JPEG, CSV and JSON demo files can be used.",
    "not_a_demo_fixture": NOT_A_DEMO_FIXTURE_MESSAGE,
}
_UPLOAD_STATUS: Mapping[str, int] = {"too_large": 413, "unsupported_type": 415, "not_a_demo_fixture": 422}

Action = Literal["confirm", "correct", "discard", "save_unmapped"]


class CheckinError(Exception):
    """A refusal: ``status`` (HTTP) and ``body`` (``{"error": code, ...}``)."""

    def __init__(self, status: int, error: str, **extra: Any) -> None:
        super().__init__(error)
        self.status = status
        self.body: dict[str, Any] = {"error": error, **extra}


class Decision(pydantic.BaseModel):
    """One review decision. ``value``/``unit``/``date`` are read only for ``correct``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_id: str = Field(min_length=1)
    action: Action
    value: float | None = None
    unit: str | None = None
    date: dt.date | None = None

    def corrected_fields(self) -> frozenset[str]:
        """The fields this ``correct`` decision actually provides (non-null)."""
        return frozenset(f for f in ("value", "unit", "date") if getattr(self, f) is not None)


# --- create ---------------------------------------------------------------------------------------


def _demo_only(real_input_requested: bool) -> None:
    # AD7: real input is hard off. The request flag changes nothing: fixtures stay allow-listed.
    s = real_input.status(requested=real_input_requested)
    if s.enabled:  # unreachable while REAL_INPUT_BUILD_ALLOWED is False; never trust it silently
        raise RuntimeError("real input must never be enabled in this build")


def create_checkin(
    journal: Journal,
    *,
    text: str | None = None,
    fixture_id: str | None = None,
    upload: bytes | None = None,
    real_input_requested: bool = False,
) -> dict[str, Any]:
    """A pending check-in (``{"checkin_id","status":"pending_review","proposals"}``) or a handoff.

    The crisis handoff is ``{"status":"handoff","text": HANDOFF_TEXT}`` (HTTP 200) and creates
    nothing. Everything else that is not a pending check-in raises :class:`CheckinError`.
    """
    _demo_only(real_input_requested)
    if text is not None and not isinstance(text, str):
        raise CheckinError(422, "invalid_request")
    has_text = text is not None and text.strip() != ""
    sources = sum((has_text, fixture_id is not None, upload is not None))
    if sources == 0:
        raise CheckinError(422, "empty_checkin")
    if sources > 1:
        raise CheckinError(422, "one_source_only", message="Send text, a demo file or a demo upload, not more than one.")

    n = journal.checkin_count + 1
    checkin_id = journal.checkin_id(n)
    proposals: tuple[Proposal, ...]
    if has_text:
        assert text is not None
        if len(text) > MAX_RAW_TEXT_CHARS:  # before is_crisis: never normalise an unbounded body
            raise CheckinError(422, "text_too_long")
        # Crisis language gets the handoff even in an over-long text: safety before validation.
        if is_crisis(text):
            return {"status": "handoff", "text": HANDOFF_TEXT}
        if len(text) > MAX_TEXT_CHARS:
            raise CheckinError(422, "text_too_long")
        reasons = scan(text)
        if reasons:
            raise CheckinError(422, "looks_like_personal_data", reasons=list(reasons), warning=PII_WARNING)
        proposals = extract(text, journal.today, checkin_id=checkin_id)
        if not proposals:
            raise CheckinError(422, "nothing_extracted", hint=NOTHING_EXTRACTED_HINT)
    else:
        if upload is not None:
            try:
                fixture_id = fixtures.intake_upload(upload)
            except fixtures.FixtureError as e:
                code = e.error
                raise CheckinError(_UPLOAD_STATUS.get(code, 422), code, message=UPLOAD_MESSAGES.get(code, "")) from None
        assert fixture_id is not None
        try:
            proposals = fixtures.proposals_for(fixture_id, journal.today, checkin_id=checkin_id)
        except fixtures.UnknownFixture:
            raise CheckinError(422, "not_a_demo_fixture", message=NOT_A_DEMO_FIXTURE_MESSAGE) from None

    checkin = CheckIn(
        checkin_id=checkin_id,
        journal_id=journal.journal_id,
        created_at=dt.datetime.now(dt.UTC),
        status="pending_review",
        proposals=proposals,
    )
    journal.checkins[checkin_id] = checkin
    journal.checkin_count = n
    return {
        "checkin_id": checkin_id,
        "status": "pending_review",
        "proposals": [p.model_dump(mode="json") for p in proposals],
    }


# --- review ---------------------------------------------------------------------------------------


def _allowed_units(concept: Concept) -> dict[str, float]:
    return {concept.unit: 1.0, **dict(concept.alt_units)}


def _parse_decisions(decisions: Iterable[Decision | Mapping[str, Any]]) -> list[Decision]:
    out = []
    try:
        for d in decisions:
            out.append(d if isinstance(d, Decision) else Decision.model_validate(d))
    except (pydantic.ValidationError, TypeError):
        raise CheckinError(422, "invalid_request") from None
    return out


def _check_decision_set(proposals: Sequence[Proposal], decisions: Sequence[Decision]) -> None:
    ids = [p.proposal_id for p in proposals]
    given = [d.proposal_id for d in decisions]
    seen: set[str] = set()
    duplicate = []
    for pid in given:
        if pid in seen and pid not in duplicate:
            duplicate.append(pid)
        seen.add(pid)
    missing = [pid for pid in ids if pid not in seen]
    unknown = [pid for pid in dict.fromkeys(given) if pid not in ids]
    if missing or unknown or duplicate:
        raise CheckinError(422, "missing_decision", missing=missing, unknown=unknown, duplicate=duplicate)


def _correction_reasons(journal: Journal, p: Proposal, d: Decision) -> list[str]:
    concept = load_registry().by_code(p.concept_code) if p.concept_code is not None else None
    if concept is None:
        return ["unknown_concept"]  # nothing to map to: save_unmapped or discard instead
    reasons = []
    allowed = _allowed_units(concept)
    unit = d.unit if d.unit is not None else p.unit
    if d.unit is not None and d.unit not in allowed:
        reasons.append("unit_not_allowed")
    if d.value is not None:
        factor = allowed.get(unit, 1.0) if unit is not None else 1.0
        lo, hi = concept.plausible
        if not (math.isfinite(d.value) and lo <= d.value * factor <= hi):
            reasons.append("value_out_of_range")
    if d.date is not None and not (journal.today - dt.timedelta(days=WINDOW_DAYS) <= d.date <= journal.today):
        reasons.append("date_outside_window")
    return reasons


def _apply(p: Proposal, d: Decision) -> Proposal:
    changes: dict[str, Any] = {}
    if d.value is not None:
        changes["value"] = float(d.value)
    if d.unit is not None:
        changes["unit"] = d.unit
    if d.date is not None:
        changes["date"] = d.date
    return p.with_changes(**changes) if changes else p


def _noon_utc(day: dt.date) -> dt.datetime:
    return dt.datetime.combine(day, dt.time(12), tzinfo=dt.UTC)


def _dry_run(journal: Journal, prior: Sequence[Any]) -> EventLog:
    """A throwaway ``EventLog`` on the journal's ontology, seeded with the journal's events.

    Appending the new events here first runs exactly the checks ``EventLog.append`` makes
    (duplicate ids, metric codes, value ranges), so the real log is only touched once all pass.
    """
    probe = EventLog(journal.ontology)
    probe.extend(prior)
    return probe


def _invalid(pids: Iterable[str], reason: str) -> CheckinError:
    return CheckinError(422, "invalid_correction", proposals=[{"proposal_id": pid, "reasons": [reason]} for pid in pids])


async def review(
    journal: Journal, checkin_id: str, decisions: Iterable[Decision | Mapping[str, Any]]
) -> dict[str, Any]:
    """Save a pending check-in all-or-nothing.

    Returns ``{"status":"saved","written":[event_id],"discarded":[proposal_id],"records":[record_id]}``.
    """
    async with journal.lock:
        checkin = journal.checkins.get(checkin_id) if isinstance(checkin_id, str) else None
        if checkin is None:
            raise CheckinError(404, "unknown_checkin")
        if checkin.status == "saved":
            raise CheckinError(409, "already_saved")
        parsed = _parse_decisions(decisions)
        proposals = checkin.proposals
        _check_decision_set(proposals, parsed)
        by_id = {d.proposal_id: d for d in parsed}

        bad = []
        for p in proposals:
            d = by_id[p.proposal_id]
            if d.action == "correct":
                reasons = _correction_reasons(journal, p, d)
                if reasons:
                    bad.append({"proposal_id": p.proposal_id, "reasons": reasons})
        if bad:
            raise CheckinError(422, "invalid_correction", proposals=bad)

        # Corrections first, then flags re-run over the kept set (contradiction is a set property).
        # Discarded and save_unmapped proposals are out of the set: they never become events.
        edited = [_apply(p, by_id[p.proposal_id]) if by_id[p.proposal_id].action == "correct" else p
                  for p in proposals]
        kept = [p for p in edited if by_id[p.proposal_id].action in ("confirm", "correct")]
        corrected = {p.proposal_id: by_id[p.proposal_id].corrected_fields()
                     for p in kept if by_id[p.proposal_id].action == "correct"}
        reflagged = {p.proposal_id: p for p in detect_flags(kept, corrected)}
        unresolved = [{"proposal_id": pid, "flags": list(p.flags)} for pid, p in reflagged.items() if p.blocking]
        if unresolved:
            raise CheckinError(422, "unresolved_flags", proposals=unresolved)

        reg = load_registry()
        events: list[SignalEvent] = []
        records: list[ObservationRecord] = []
        written_pids: list[str] = []
        discarded: list[str] = []
        n_event, n_record = journal.event_count, journal.record_count
        prior = journal.all_events()
        probe = _dry_run(journal, prior)
        for p in edited:
            d = by_id[p.proposal_id]
            if d.action == "discard":
                discarded.append(p.proposal_id)  # T4.3: no event, no record, nothing else
                continue
            concept = reg.by_code(p.concept_code) if p.concept_code is not None else None
            day = p.date if p.date is not None else journal.today
            observed_at = _noon_utc(day)
            event_id = None
            if d.action in ("confirm", "correct"):
                assert concept is not None and p.unit is not None and p.value is not None  # no flags left
                n_event += 1
                event_id = journal.event_id(n_event)
                try:
                    # The event carries the canonical unit: as-recorded value x the alt_units factor.
                    event = SignalEvent(
                        event_id=event_id,
                        member_id=journal.member.member_id,
                        occurred_at=observed_at,
                        code=concept.code,
                        value=p.value * _allowed_units(concept)[p.unit],
                    )
                except pydantic.ValidationError:
                    raise _invalid([p.proposal_id], "value_not_recordable") from None
                try:
                    probe.append(event)
                except ValueError:
                    raise _invalid([p.proposal_id], "value_not_recordable") from None
                events.append(event)
                written_pids.append(p.proposal_id)
            n_record += 1
            mapped = event_id is not None
            records.append(
                ObservationRecord(
                    record_id=journal.record_id(n_record),
                    event_id=event_id,
                    concept_id=concept.concept_id if concept is not None else None,
                    loinc_code=concept.code if mapped and concept is not None and concept.system == "loinc" else None,
                    loinc_version=concept.loinc_version if mapped and concept is not None else None,
                    display=concept.display if concept is not None else p.display,
                    value=p.value,
                    unit=p.unit,
                    value_type=concept.value_type if concept is not None else None,
                    observed_at=observed_at,
                    provenance=Provenance(
                        source_kind=p.source_kind, source_ref=p.source_ref, checkin_id=checkin_id, method=p.method
                    ),
                    quality="corrected" if d.action == "correct" else "confirmed",
                    status="mapped" if mapped else "unmapped",
                    reviewer_status=concept.reviewer_status if concept is not None else REVIEWER_STATUS,
                    allowed_use=concept.allowed_use if concept is not None else UNMAPPED_ALLOWED_USE,
                )
            )
        saved = checkin.mark_saved()

        # Commit. Every event already passed a dry run on a log seeded with the journal's events.
        appended = 0
        try:
            for event in events:
                journal.events.append(event)
                appended += 1
        except Exception:  # noqa: BLE001 - whatever failed, the journal must be left as it was
            if appended == 0 and len(journal.events) == len(prior):
                raise _invalid(written_pids, "value_not_recordable") from None
            # A&D failure table: a 500 means nothing was written. There is no rollback API, so
            # rebuild the log from the prior events; records and the saved status were not written.
            journal.rebuild_events(prior)
            raise CheckinError(500, "write_failed") from None
        for rec in records:
            journal.provenance[rec.record_id] = rec
        journal.event_count, journal.record_count = n_event, n_record
        journal.checkins[checkin_id] = saved
        return {
            "status": "saved",
            "written": [e.event_id for e in events],
            "discarded": discarded,
            "records": [r.record_id for r in records],
        }


def observation(journal: Journal, record_id: str) -> dict[str, Any]:
    rec = journal.provenance.get(record_id) if isinstance(record_id, str) else None
    if rec is None:
        raise CheckinError(404, "unknown_record")
    return rec.model_dump(mode="json")


__all__ = [
    "MAX_RAW_TEXT_CHARS",
    "MAX_TEXT_CHARS",
    "NOTHING_EXTRACTED_HINT",
    "NOT_A_DEMO_FIXTURE_MESSAGE",
    "PII_WARNING",
    "UNMAPPED_ALLOWED_USE",
    "UPLOAD_MESSAGES",
    "CheckinError",
    "Decision",
    "create_checkin",
    "observation",
    "review",
]
