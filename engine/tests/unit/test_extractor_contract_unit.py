"""HAC-38 P1.7: the typed extractor boundary (CTO comment 859317b9, addendum G21).

Every extractor (rules-v1 text, fixture sidecars, any future model-backed one) hands the service
validated :class:`Proposal` models. Malformed output is refused before anything reaches the
journal. Synthetic inputs only; no live model.
"""

import datetime as dt

import pytest

from daypart.checkin import fixtures, service
from daypart.checkin.extract_text import extract
from daypart.checkin.extractor import ExtractorOutputError, validate_proposals
from daypart.checkin.journal import JournalStore
from daypart.checkin.models import Proposal
from daypart.checkin.service import CheckinError
from daypart.simmember.cohort import generate
from daypart.simmember.trajectory import START_DATE

TODAY = START_DATE.date() + dt.timedelta(days=28)


def _good() -> dict:
    return {
        "proposal_id": "p-1",
        "concept_code": "93832-4",
        "display": "Sleep duration",
        "value": 6.0,
        "unit": "h",
        "date": TODAY.isoformat(),
        "source_kind": "text",
        "source_ref": "c-1",
        "snippet": "slept 6 hours",
        "method": "rules-v1",
        "confidence": "high",
        "flags": [],
    }


def test_valid_dicts_and_models_pass_as_proposals():
    out = validate_proposals([_good(), Proposal.model_validate({**_good(), "proposal_id": "p-2"})])
    assert all(type(p) is Proposal for p in out)
    assert [p.proposal_id for p in out] == ["p-1", "p-2"]
    assert isinstance(out, tuple)


def test_real_extractor_output_passes_unchanged():
    props = extract("slept 6 hours and walked 8000 steps", TODAY, checkin_id="c-1")
    assert validate_proposals(props) == props


@pytest.mark.parametrize(
    "bad",
    [
        {**_good(), "risk_score": 0.9},  # extra field: extra="forbid"
        {k: v for k, v in _good().items() if k != "confidence"},  # missing field
        {**_good(), "flags": ["looks_high"]},  # not a Flag
        {**_good(), "value": "six"},  # wrong type
        {**_good(), "flags": ["unit_assumed"], "blocking": False},  # blocking disagrees with flags
        "slept 6 hours",  # not an object
        None,
    ],
)
def test_malformed_item_is_refused(bad):
    with pytest.raises(ExtractorOutputError):
        validate_proposals([bad])


def test_model_construct_bypass_is_revalidated():
    forged = Proposal.model_construct(**{**_good(), "flags": ("unit_assumed",), "blocking": False})
    with pytest.raises(ExtractorOutputError):
        validate_proposals([forged])


def test_duplicate_proposal_ids_are_refused():
    with pytest.raises(ExtractorOutputError):
        validate_proposals([_good(), _good()])


@pytest.mark.parametrize("bad", [None, "p-1", _good()])
def test_output_must_be_a_sequence(bad):
    with pytest.raises(ExtractorOutputError):
        validate_proposals(bad)


def test_error_message_carries_no_input_values():
    with pytest.raises(ExtractorOutputError) as e:
        validate_proposals([{**_good(), "snippet": "secret-snippet", "risk_score": 0.9}])
    assert "secret-snippet" not in str(e.value)


# --- the service refuses malformed output before the journal ------------------------------------


@pytest.fixture(scope="module")
def cohort():
    return generate(0, n_synthetic=5, days=42)


@pytest.fixture
def journal(cohort):
    return JournalStore(cohort).create("syn-001")


def _assert_untouched(j):
    assert j.checkins == {} and j.checkin_count == 0 and len(j.events) == 0 and j.provenance == {}


def test_malformed_text_extractor_output_never_reaches_the_journal(journal, monkeypatch):
    monkeypatch.setattr(service, "extract", lambda *a, **k: ({**_good(), "risk_score": 1},))
    with pytest.raises(CheckinError) as e:
        service.create_checkin(journal, text="slept 6 hours")
    assert e.value.status == 500 and e.value.body == {"error": "extractor_output_invalid"}
    _assert_untouched(journal)


def test_malformed_fixture_extractor_output_never_reaches_the_journal(journal, monkeypatch):
    fid = next(iter(fixtures.manifest()))
    forged = Proposal.model_construct(**{**_good(), "source_kind": "file_fixture", "method": "x", "blocking": True})
    monkeypatch.setattr(fixtures, "proposals_for", lambda *a, **k: (forged,))
    with pytest.raises(CheckinError) as e:
        service.create_checkin(journal, fixture_id=fid)
    assert e.value.status == 500 and e.value.body == {"error": "extractor_output_invalid"}
    _assert_untouched(journal)
