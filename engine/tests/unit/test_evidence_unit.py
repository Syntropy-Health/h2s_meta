"""Unit tests for the evidence registry (A&D §4.6, D-A6, AM-1; R6 T6.3 data side; plan task 6, G22)."""

import json
import re
from datetime import date
from importlib import resources
from urllib.parse import urlparse

import pydantic
import pytest

from daypart.evidence.registry import (
    VETTING,
    EvidenceRegistry,
    Source,
    StatedRange,
    check_reference_ranges,
    load_evidence,
)
from daypart.memory.registry import ConceptRegistry, ReferenceRange, load_registry

VETTING_LABEL = "reviewed: board sign-off 2026-11-07"  # AM-1
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def raw_sources() -> dict:
    text = resources.files("daypart.evidence").joinpath("sources_v1.json").read_text("utf-8")
    return json.loads(text)


def numbers(text: str) -> set[float]:
    return {float(n) for n in NUMBER.findall(text)}


def a_source(**over) -> dict:
    base = {
        "evidence_id": "ev:test",
        "title": "Test page",
        "publisher": "Test publisher",
        "url": "https://example.org/page",
        "accessed": "2026-11-07",
        "supports": "Tracked because it is a routine vital sign.",
        "applies_to": ["chc:heart-rate"],
        "vetting": VETTING_LABEL,
        "stated_ranges": [],
    }
    base.update(over)
    return base


# --- the shipped data ---------------------------------------------------------------------------


def test_vetting_label_is_the_am1_label_and_never_clinical():
    assert VETTING == VETTING_LABEL
    ev = load_evidence()
    assert ev.sources()
    for s in ev.sources():
        assert s.vetting == VETTING_LABEL
        for text in (s.title, s.publisher, s.supports):
            low = text.lower()
            assert "clinically reviewed" not in low and "vetted source" not in low


def test_every_source_is_a_public_https_page_with_a_short_supports():
    for s in load_evidence().sources():
        u = urlparse(s.url)
        assert u.scheme == "https" and u.hostname and "." in u.hostname
        assert not u.hostname.endswith((".local", ".internal", ".lan")), s.url
        assert len(s.supports) <= 200
        assert isinstance(s.accessed, date)


def test_evidence_ids_are_unique_and_applies_to_resolves():
    reg = load_registry()
    data = raw_sources()["sources"]
    ids = [s["evidence_id"] for s in data]
    assert len(ids) == len(set(ids))
    for s in load_evidence().sources():
        assert s.applies_to
        for cid in s.applies_to:
            assert reg.by_concept_id(cid) is not None, cid


def test_every_number_in_supports_traces_to_a_stated_range_of_its_entry():
    for s in load_evidence().sources():
        stated = set().union(*(numbers(r.statement) for r in s.stated_ranges)) if s.stated_ranges else set()
        assert numbers(s.supports) <= stated, s.evidence_id


def test_every_stated_number_appears_in_its_statement():
    for s in load_evidence().sources():
        for r in s.stated_ranges:
            for bound in (r.low, r.high):
                if bound is not None:
                    assert bound in numbers(r.statement), (s.evidence_id, bound)


def test_every_reference_range_resolves_to_its_source_and_numbers():
    reg, ev = load_registry(), load_evidence()
    ranged = [c for c in reg.concepts() if c.reference_range is not None]
    assert ranged, "AM-1: at least one concept carries a sourced range"
    for c in ranged:
        rr = c.reference_range
        assert rr is not None
        src = ev.by_id(rr.source_evidence_id)
        assert src is not None, rr.source_evidence_id
        assert c.concept_id in src.applies_to
        stated = [r for r in src.stated_ranges if r.concept_id == c.concept_id]
        assert len(stated) == 1
        st = stated[0]
        assert (st.low, st.high, st.unit, st.population) == (rr.low, rr.high, rr.unit, rr.population)
        entry_numbers = numbers(st.statement)
        for bound in (rr.low, rr.high):
            if bound is not None:
                assert bound in entry_numbers


def test_every_stated_range_is_carried_by_its_concept():
    reg = load_registry()
    for s in load_evidence().sources():
        for r in s.stated_ranges:
            c = reg.by_concept_id(r.concept_id)
            assert c is not None and c.reference_range is not None
            assert c.reference_range.source_evidence_id == s.evidence_id


def test_a_concept_without_a_source_has_no_reference_range():
    reg, ev = load_registry(), load_evidence()
    unsourced = [c for c in reg.concepts() if not ev.for_concept(c.concept_id)]
    assert unsourced, "at least one concept has no source (exercises suppression, T6.3)"
    for c in unsourced:
        assert c.reference_range is None, c.concept_id


def test_plausible_bounds_are_not_reference_ranges():
    for c in load_registry().concepts():
        rr = c.reference_range
        if rr is not None:
            assert (rr.low, rr.high) != tuple(c.plausible)


def test_scores_carry_no_range():
    # PHQ/GAD totals: no severity band (A&D §4.3); pain: no band either.
    reg = load_registry()
    for cid in ("chc:phq9-total", "chc:gad7-total", "chc:phq2-total", "chc:pain-severity"):
        c = reg.by_concept_id(cid)
        assert c is not None and c.reference_range is None


# --- lookup -------------------------------------------------------------------------------------


def test_for_concept_accepts_concept_id_or_code():
    reg, ev = load_registry(), load_evidence()
    hr = reg.by_concept_id("chc:heart-rate")
    assert hr is not None
    by_id = ev.for_concept("chc:heart-rate")
    assert by_id and ev.for_concept(hr.code) == by_id
    assert all("chc:heart-rate" in s.applies_to for s in by_id)
    assert ev.for_concept("chc:no-such") == ()
    assert ev.for_concept("0000-0") == ()


def test_by_id_and_load_is_cached():
    ev = load_evidence()
    assert load_evidence() is ev
    first = ev.sources()[0]
    assert ev.by_id(first.evidence_id) is first
    assert ev.by_id("ev:missing") is None


# --- validation ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "over",
    [
        {"vetting": "clinically reviewed"},
        {"vetting": "agent draft, not clinically reviewed"},
        {"url": "http://example.org/page"},
        {"url": "https://localhost/page"},
        {"url": "https://10.0.0.1/page"},
        {"url": "https://host.internal/page"},
        {"supports": "x" * 201},
        {"supports": ""},
        {"applies_to": []},
        {"evidence_id": "test"},
        {"extra": 1},
        {"stated_ranges": [{"concept_id": "chc:heart-rate", "low": 60, "high": 100, "unit": "/min",
                            "population": "adults", "statement": "Pulse: 60 to 101 per minute"}]},
        {"stated_ranges": [{"concept_id": "chc:steps", "low": 1, "high": 2, "unit": "{steps}",
                            "population": "adults", "statement": "1 to 2"}]},
        {"supports": "A heart rate of 72 is common."},
    ],
)
def test_source_rejects(over):
    with pytest.raises(pydantic.ValidationError):
        Source.model_validate(a_source(**over))


def test_source_accepts_a_traced_range():
    s = Source.model_validate(a_source(
        supports="Resting pulse of 60 to 100 is the stated adult range.",
        stated_ranges=[{"concept_id": "chc:heart-rate", "low": 60, "high": 100, "unit": "/min",
                        "population": "adults at rest", "statement": "Pulse: 60 to 100 beats per minute"}],
    ))
    assert s.stated_ranges[0] == StatedRange(concept_id="chc:heart-rate", low=60, high=100, unit="/min",
                                             population="adults at rest",
                                             statement="Pulse: 60 to 100 beats per minute")


def test_registry_rejects_duplicate_ids():
    s = Source.model_validate(a_source())
    with pytest.raises(ValueError, match="duplicate"):
        EvidenceRegistry((s, s))


def _registry_with(rr: ReferenceRange | None) -> ConceptRegistry:
    """The shipped concepts with only heart-rate's range set (to ``rr``); every other range cleared."""
    base = load_registry()
    concepts = tuple(
        c.model_copy(update={"reference_range": rr if c.concept_id == "chc:heart-rate" else None})
        for c in base.concepts()
    )
    return ConceptRegistry(concepts, base.gaps, base.loinc_version)


def test_check_reference_ranges_refuses_an_unsourced_range():
    ev = EvidenceRegistry((Source.model_validate(a_source()),))
    rr = ReferenceRange(low=60, high=100, unit="/min", population="adults", source_evidence_id="ev:test")
    with pytest.raises(ValueError, match="chc:heart-rate"):
        check_reference_ranges(_registry_with(rr), ev)


def test_check_reference_ranges_refuses_an_unresolved_source():
    ev = EvidenceRegistry((Source.model_validate(a_source()),))
    rr = ReferenceRange(low=60, high=100, unit="/min", population="adults", source_evidence_id="ev:gone")
    with pytest.raises(ValueError, match="ev:gone"):
        check_reference_ranges(_registry_with(rr), ev)


def test_check_reference_ranges_refuses_a_dropped_stated_range():
    ev = EvidenceRegistry((Source.model_validate(a_source(stated_ranges=[
        {"concept_id": "chc:heart-rate", "low": 60, "high": 100, "unit": "/min",
         "population": "adults", "statement": "60 to 100"}])),))
    with pytest.raises(ValueError, match="chc:heart-rate"):
        check_reference_ranges(_registry_with(None), ev)


def test_check_reference_ranges_accepts_a_matching_range():
    ev = EvidenceRegistry((Source.model_validate(a_source(stated_ranges=[
        {"concept_id": "chc:heart-rate", "low": 60, "high": 100, "unit": "/min",
         "population": "adults", "statement": "60 to 100"}])),))
    rr = ReferenceRange(low=60, high=100, unit="/min", population="adults", source_evidence_id="ev:test")
    reg = _registry_with(rr)
    check_reference_ranges(reg, ev)


# --- QG P2.1 ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost./x", "https://foo.local./x", "https://10.0.0.1./x", "https://127.1/x",
        "https://0x7f.0.0.1/x", "https://0177.0.0.1/x", "https://a.svc/x", "https://k.default.svc.cluster.local/x",
        "https://h.home.arpa/x", "https://foo.test/x", "https://foo.invalid/x", "https://foo.example/x",
        "https://127.0.0.1.nip.io/x", "https://10.0.0.1.sslip.io/x", "https://foo.lan/x", "https://foo.corp/x",
        "https://foo.home/x", "https://foo.localhost/x", "https://foo.LOCAL/x", "https://[::1]/x",
        "https://8.8.8.8/x", "https://169.254.169.254/x",
    ],
)
def test_f7_url_must_be_a_public_name(url):
    with pytest.raises(pydantic.ValidationError, match="public"):
        Source.model_validate(a_source(url=url))


def test_f7_shipped_urls_pass_the_hardened_validator():
    for s in load_evidence().sources():
        Source.model_validate({**s.model_dump(mode="json"), "url": s.url})
        assert urlparse(s.url).hostname in {"medlineplus.gov", "www.nhlbi.nih.gov", "www.who.int"}


@pytest.mark.parametrize("url", ["https://u:p@example.org/x", "https://u@example.org/x", "https://example.org:8443/x"])
def test_b2_url_has_no_credentials_or_port(url):
    with pytest.raises(pydantic.ValidationError, match="credentials or a port"):
        Source.model_validate(a_source(url=url))


def test_b2_url_query_and_port_443_are_allowed():
    assert Source.model_validate(a_source(url="https://example.org:443/x?lang=en")).url.endswith("lang=en")


HR_RANGE = {"concept_id": "chc:heart-rate", "low": 60, "high": 100, "unit": "/min",
            "population": "adults", "statement": "Pulse: 60 to 100 beats per minute"}


@pytest.mark.parametrize("text", ["Clinically-reviewed", "clinically  reviewed", "vetted  sources",
                                  "medically reviewed", "clinician reviewed", "expert-vetted", "doctor approved"])
@pytest.mark.parametrize("field", ["population", "statement"])
def test_b3_stated_range_copy_is_checked(field, text):
    bad = {**HR_RANGE, field: f"{text} 60 to 100" if field == "statement" else text}
    with pytest.raises(pydantic.ValidationError, match="copy may not say"):
        StatedRange.model_validate(bad)


def test_b5_for_concept_takes_code_by_keyword_as_the_ad_names_it():
    ev = load_evidence()
    assert ev.for_concept(code="8867-4") == ev.for_concept("chc:heart-rate") != ()


def test_b6_a_code_without_a_concept_registry_is_refused_not_silently_empty():
    bare = EvidenceRegistry((Source.model_validate(a_source()),))
    assert bare.for_concept("chc:heart-rate")  # concept ids need no registry
    with pytest.raises(ValueError, match="concept registry"):
        bare.for_concept("8867-4")


def test_b9_sources_is_a_read_only_method_like_concepts():
    ev = load_evidence()
    with pytest.raises(AttributeError):
        ev.sources = ()  # type: ignore[misc]
    assert isinstance(ev.sources(), tuple) and ev.sources()
