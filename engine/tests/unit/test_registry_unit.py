"""Unit tests for the A4 concept registry (A&D §4.3, R5: T5.1, T5.2, T5.6; plan G3-G5)."""

import csv
import json
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

import pydantic
import pytest

from daypart.memory.events import EventLog, SignalEvent
from daypart.memory.ontology import CORE_CAVEATS, LOINC_SYSTEM, apply_update, core
from daypart.memory.registry import (
    LOINC_NOTICE,
    LOINC_VERSION,
    REVIEWER_STATUS,
    SLICE_VERSION,
    Concept,
    load_registry,
    slice_ontology,
    slice_update,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_LOINC = {"72514-3", "44261-6", "70274-6", "8867-4", "29463-7", "55423-8", "93832-4"}
NEW_LOINC = {
    "8480-6", "8462-4", "8302-2", "39156-5", "8310-5", "9279-1", "59408-5", "8280-0",
    "41950-7", "55411-3", "55758-7",
}
VERTICALS = ("vitals_body", "behaviour_function", "symptoms_wellbeing")
REVIEWER = "reviewed: board sign-off 2026-11-07"  # AM-1 (board answer on A3)
T0 = datetime(2026, 1, 1, 12, tzinfo=UTC)


def csv_rows():
    text = resources.files("daypart.memory").joinpath("loinc_verify_v1.csv").read_text("utf-8")
    return list(csv.DictReader(text.splitlines()))


# ---------------------------------------------------------------- T5.1


def test_slice_ontology_is_one_update_over_core():
    before = core().digest()
    o = slice_ontology()
    assert o == apply_update(core(), slice_update())
    assert core().digest() == before
    assert o.version == SLICE_VERSION == "core-v1+u1"
    assert len(o.history) == 1
    rec = o.history[0]
    assert rec.from_version == "core-v1" and rec.proposer == "daypart.memory.registry"
    assert slice_ontology() is o  # cached


def test_slice_update_adds_exactly_the_non_core_loinc_codes():
    upd = slice_update()
    codes = {n.code for n in upd.add_nodes}
    assert len(upd.add_nodes) == len(codes) == 11
    assert codes == NEW_LOINC
    reg_loinc = {c.code for c in load_registry().concepts() if c.system == "loinc"}
    core_codes = set(core().nodes)
    assert codes == reg_loinc - core_codes
    assert reg_loinc & core_codes == CORE_LOINC
    assert upd.set_edges == ()


def test_slice_nodes_parents_and_caveats():
    o = slice_ontology()
    activity = {"41950-7", "55411-3"}
    for code in NEW_LOINC:
        n = o.node(code)
        assert n.system == LOINC_SYSTEM and n.kind == "metric"
        assert n.lhm_domain is None and n.caveats == CORE_CAVEATS and n.label
        assert not o.is_core(code)
        if code in activity:
            assert n.parent == "dp:activity"
        elif code == "55758-7":
            assert n.parent == "dp:affect"
        else:
            assert n.parent == "dp:vital"
    reg = load_registry()
    vitals = {c.code for c in reg.by_vertical("vitals_body")}
    assert all(o.node(c).parent == "dp:vital" for c in vitals)


def test_plausible_equals_slice_value_range():
    o = slice_ontology()
    for c in load_registry().concepts():
        assert o.node(c.code).value_range == c.plausible, c.code


def test_core_codes_reuse_core_value_range():
    reg, o = load_registry(), core()
    for c in reg.concepts():
        if c.code in o.nodes:
            assert c.plausible == o.node(c.code).value_range


def test_eventlog_accepts_new_code_on_slice_only():
    ev = SignalEvent(event_id="e1", member_id="syn-001", occurred_at=T0, code="8480-6", value=120)
    log = EventLog(slice_ontology())
    log.append(ev)
    assert log.events("syn-001") == (ev,)
    with pytest.raises(ValueError):
        EventLog(core()).append(ev)
    with pytest.raises(ValueError):
        EventLog(slice_ontology()).append(ev.model_copy(update={"value": 1e9}))


# ---------------------------------------------------------------- T5.2


def test_every_concept_has_required_fields():
    reg = load_registry()
    concepts = reg.concepts()
    assert len(concepts) == 20
    assert len({c.concept_id for c in concepts}) == len({c.code for c in concepts}) == 20
    for c in concepts:
        assert c.concept_id.startswith("chc:") and c.code and c.display and c.unit
        assert c.value_type in {"quantity", "integer", "ordinal"}
        assert c.reviewer_status == REVIEWER
        assert c.allowed_use and c.keywords
        assert c.vertical in VERTICALS
        if c.system == "loinc":
            assert c.loinc_version == LOINC_VERSION == "2.83"
        else:
            assert c.system == "local" and c.loinc_version is None and c.code.startswith("dp:")
        assert reg.by_code(c.code) is c and reg.by_concept_id(c.concept_id) is c
    for v in VERTICALS:
        assert len(reg.by_vertical(v)) >= 4
    assert len(reg.by_vertical("vitals_body")) == 10
    n_loinc = sum(c.system == "loinc" for c in concepts)
    assert 15 <= n_loinc <= 25 and n_loinc == 18
    assert set(reg.gaps) >= {"labs", "conditions", "medications", "care history", "preventive care"}


def test_phq_gad_allowed_use_has_no_interpretation():
    reg = load_registry()
    for code in ("44261-6", "70274-6", "55758-7"):
        assert reg.by_code(code).allowed_use == (
            "display of a synthetic self-report score; no severity band, no screening interpretation"
        )
    assert reg.by_code("8867-4").allowed_use == (
        "demo display of self-reported or simulated values; no interpretation"
    )


def test_keyword_lookup():
    reg = load_registry()
    assert reg.for_keyword("slept").code == "93832-4"
    assert reg.for_keyword("Steps").code == "55423-8"
    assert reg.for_keyword("bpm").code == "8867-4"
    assert reg.for_keyword("nothing-here") is None


def test_concept_model_validation():
    good = load_registry().by_code("8480-6").model_dump()
    Concept(**good)
    with pytest.raises(pydantic.ValidationError):
        Concept(**{**good, "reviewer_status": "clinically reviewed"})
    with pytest.raises(pydantic.ValidationError):
        Concept(**{**good, "reviewer_status": "draft, not clinically reviewed"})  # superseded by AM-1
    with pytest.raises(pydantic.ValidationError):
        Concept(**{**good, "loinc_version": None})
    with pytest.raises(pydantic.ValidationError):
        Concept(**{**good, "system": "local", "code": "dp:x"})  # local must not carry a version
    with pytest.raises(pydantic.ValidationError):
        Concept(**{**good, "plausible": (5.0, 5.0)})
    with pytest.raises(pydantic.ValidationError):
        Concept(**{**good, "extra_field": 1})
    c = load_registry().by_code("8480-6")
    with pytest.raises(pydantic.ValidationError):
        c.display = "x"  # type: ignore[misc]


def test_alt_units_are_positive_factors():
    reg = load_registry()
    assert reg.by_code("29463-7").alt_units["[lb_av]"] == pytest.approx(0.45359237)
    assert reg.by_code("8302-2").alt_units["[in_i]"] == pytest.approx(2.54)
    for c in reg.concepts():
        assert all(f > 0 for f in c.alt_units.values())


# ---------------------------------------------------------------- T5.6


def test_loinc_concepts_match_verify_csv():
    rows = csv_rows()
    by_code = {r["code"]: r for r in rows}
    assert len(rows) == len(by_code) == 18
    assert all(set(r) == {"code", "display", "loinc_version", "verified_on", "source"} for r in rows)
    assert all(r["loinc_version"] == LOINC_VERSION for r in rows)
    loinc = [c for c in load_registry().concepts() if c.system == "loinc"]
    for c in loinc:
        row = by_code[c.code]
        assert (c.code, c.display, c.loinc_version) == (row["code"], row["display"], row["loinc_version"])
    assert {c.code for c in loinc} == set(by_code)


def test_failed_code_is_dropped():
    assert "69725-0" not in {r["code"] for r in csv_rows()}
    assert load_registry().by_code("69725-0") is None
    assert "69725-0" not in slice_ontology().nodes


def test_registry_json_records_loinc_version():
    import json

    raw = json.loads(resources.files("daypart.memory").joinpath("registry_a4.json").read_text("utf-8"))
    assert raw["loinc_version"] == LOINC_VERSION


def test_attribution_carries_loinc_notice():
    text = (REPO_ROOT / "ATTRIBUTION.md").read_text("utf-8")
    assert "LOINC is copyright © 1995-2026, Regenstrief Institute, Inc." in text
    assert "https://loinc.org/license/" in text
    assert LOINC_NOTICE in text


# --- QG P1.1 bug-exposing tests ---------------------------------------------------------------


def test_loinc_notice_carries_the_regenstrief_year_range():
    # F3: Regenstrief's notice wording includes the "1995-<year>" copyright range.
    repo_root = Path(__file__).resolve().parents[3]
    expected = "LOINC is copyright © 1995-2026, Regenstrief Institute, Inc."
    assert expected in LOINC_NOTICE
    assert expected in (repo_root / "ATTRIBUTION.md").read_text(encoding="utf-8")


def test_cached_concept_alt_units_cannot_be_mutated():
    # F1: the registry is a shared cache; one caller must not change another's alt_units.
    c = load_registry().by_code("29463-7")
    with pytest.raises(TypeError):
        c.alt_units["x"] = 1.0  # type: ignore[index]
    assert "x" not in load_registry().by_code("29463-7").alt_units
    assert c.model_dump(mode="json")["alt_units"]["[lb_av]"] == pytest.approx(0.45359237)


# ---------------------------------------------------------------- AM-1: reference ranges


def test_am1_label_is_honest():
    assert REVIEWER_STATUS == REVIEWER
    assert "clinically" not in REVIEWER_STATUS.casefold() and "clinician" not in REVIEWER_STATUS.casefold()


def test_am1_every_concept_carries_reference_range_null_without_a_source():
    # AM-1 item 2: a range is set only where a sources_v1.json entry states it (P2 task 6); the key is
    # always present in the dumped concept. The tracing itself is pinned in test_evidence_unit.py.
    from daypart.evidence.registry import load_evidence

    ev = load_evidence()
    for c in load_registry().concepts():
        assert "reference_range" in c.model_dump()
        if c.reference_range is not None:
            assert ev.by_id(c.reference_range.source_evidence_id) is not None
        if not ev.for_concept(c.concept_id):
            assert c.reference_range is None and c.model_dump()["reference_range"] is None


def test_am1_reference_range_model():
    good = load_registry().by_code("8480-6").model_dump()
    rr = {"low": 90, "high": 120, "unit": "mm[Hg]", "population": "adults", "source_evidence_id": "ev-x"}
    c = Concept(**{**good, "reference_range": rr})
    assert c.reference_range is not None and c.reference_range.low == 90 and c.reference_range.unit == "mm[Hg]"
    assert c.model_dump()["reference_range"] == {**rr, "low": 90.0, "high": 120.0}
    for bad in (
        {**rr, "low": 120, "high": 90},  # low must be below high
        {**rr, "low": float("nan")},
        {**rr, "unit": "mmHg"},  # must be the concept's unit or one of its alt_units
        {**rr, "population": ""},
        {**rr, "source_evidence_id": ""},  # AM-1: a range without a source is not allowed
        {k: v for k, v in rr.items() if k != "source_evidence_id"},
        {**rr, "extra": 1},
    ):
        with pytest.raises(pydantic.ValidationError):
            Concept(**{**good, "reference_range": bad})
    one_sided = Concept(**{**good, "reference_range": {**rr, "low": None}})
    assert one_sided.reference_range.low is None  # a guideline may state only an upper bound
    with pytest.raises(pydantic.ValidationError):
        Concept(**{**good, "reference_range": {**rr, "low": None, "high": None}})


def test_am1_registry_json_states_reference_range_explicitly():
    raw = json.loads(resources.files("daypart.memory").joinpath("registry_a4.json").read_text("utf-8"))
    concepts = raw["concepts"] if isinstance(raw, dict) else raw
    assert len(concepts) == 20
    for c in concepts:
        # the key is always explicit; null unless a source states it (traced in test_evidence_unit.py)
        assert "reference_range" in c, c["code"]
        rr = c["reference_range"]
        assert rr is None or rr["source_evidence_id"].startswith("ev:"), c["code"]
        assert c["reviewer_status"] == REVIEWER


def test_am1_reference_range_edges():
    weight = load_registry().by_code("29463-7").model_dump()
    rr = {"low": 100, "high": 200, "unit": "[lb_av]", "population": "adults", "source_evidence_id": "ev-x"}
    assert Concept(**{**weight, "reference_range": rr}).reference_range.unit == "[lb_av]"  # an alt unit
    for bad in ({**rr, "unit": "cm"}, {**rr, "low": 150, "high": 150}, {**rr, "high": float("inf")}):
        with pytest.raises(pydantic.ValidationError):
            Concept(**{**weight, "reference_range": bad})
    upper_open = Concept(**{**weight, "reference_range": {**rr, "high": None}}).reference_range
    assert upper_open.low == 100 and upper_open.high is None
    with pytest.raises(pydantic.ValidationError):
        upper_open.low = 1  # type: ignore[misc]
