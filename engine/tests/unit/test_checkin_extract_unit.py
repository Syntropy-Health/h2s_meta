"""P1 task 3 — text check-in core (A&D §4.1, failure table; T2.2-T2.4, T4.1, T4.2).

Pure functions only: rules-v1 extraction, the seven blocking flags, the PII tripwire and the
crisis pre-check. Synthetic inputs only.
"""

from __future__ import annotations

import datetime as dt
import time
from datetime import timedelta

import pydantic
import pytest

from daypart.checkin.crisis import HANDOFF_TEXT, is_crisis
from daypart.checkin.extract_text import detect_flags, extract
from daypart.checkin.models import FLAGS, METHOD_FIXTURE, METHOD_TEXT, CheckIn, Proposal
from daypart.checkin.tripwire import scan
from daypart.demo.safety import safe_text
from daypart.eval import POLICY_TEXT
from daypart.membrane import dsl
from daypart.simmember.trajectory import START_DATE

TODAY = (START_DATE + timedelta(days=28)).date()  # G9: 2026-09-29, a Tuesday


def _by_code(props: tuple[Proposal, ...], code: str | None) -> list[Proposal]:
    return [p for p in props if p.concept_code == code]


def _one(props: tuple[Proposal, ...], code: str | None) -> Proposal:
    hits = _by_code(props, code)
    assert len(hits) == 1, (code, props)
    return hits[0]


# ---------------------------------------------------------------- T2.2 extraction


def test_t2_2_sleep_and_steps_extracted_high_confidence() -> None:
    props = extract("slept 6 hours and walked 8000 steps", TODAY, checkin_id="c-1")
    assert len(props) >= 2
    sleep = _one(props, "93832-4")
    steps = _one(props, "55423-8")
    assert (sleep.value, sleep.unit) == (6.0, "h")
    assert steps.value == 8000.0
    for p in (sleep, steps):
        assert p.confidence == "high"
        assert p.flags == ()
        assert p.blocking is False
        assert p.method == METHOD_TEXT == "rules-v1"
        assert p.source_kind == "text"
        assert p.date == TODAY
        assert p.source_ref.startswith("c-1:")
    assert [p.proposal_id for p in props] == [f"p-{i}" for i in range(1, len(props) + 1)]
    assert sleep.proposal_id == "p-1"  # text order


def test_source_ref_is_the_char_span_and_snippet_comes_from_it() -> None:
    text = "slept 6 hours and walked 8000 steps"
    for p in extract(text, TODAY, checkin_id="c-9"):
        cid, span = p.source_ref.split(":")
        start, end = (int(x) for x in span.split("-"))
        assert cid == "c-9"
        assert 0 <= start < end <= len(text)
        assert p.snippet == text[start:end][:80]
        assert len(p.snippet) <= 80


def test_extraction_is_deterministic() -> None:
    text = "bp 120/80 mmhg, weight 72.5 kg yesterday, pain 4/10"
    assert extract(text, TODAY) == extract(text, TODAY)


def test_blood_pressure_pair_yields_systolic_and_diastolic() -> None:
    props = extract("bp 120/80 mmHg", TODAY)
    sys_, dia = _one(props, "8480-6"), _one(props, "8462-4")
    assert (sys_.value, sys_.unit, sys_.confidence, sys_.flags) == (120.0, "mm[Hg]", "high", ())
    assert (dia.value, dia.unit, dia.confidence, dia.flags) == (80.0, "mm[Hg]", "high", ())
    bare = extract("120/80", TODAY)
    assert _one(bare, "8480-6").value == 120.0
    assert _one(bare, "8462-4").value == 80.0


@pytest.mark.parametrize(
    ("text", "code", "value", "unit"),
    [
        ("weight 72.5 kg", "29463-7", 72.5, "kg"),
        ("weighed 160 lbs", "29463-7", 160.0, "[lb_av]"),
        ("heart rate 64 bpm", "8867-4", 64.0, "/min"),
        ("pulse 70/min", "8867-4", 70.0, "/min"),
        ("temperature 37.2 c", "8310-5", 37.2, "Cel"),
        ("spo2 97%", "59408-5", 97.0, "%"),
        ("pain 4/10", "72514-3", 4.0, "{score}"),
        ("pain 3 out of 10", "72514-3", 3.0, "{score}"),
        ("exercised 30 minutes", "55411-3", 30.0, "min"),
        ("slept 6.5 hrs", "93832-4", 6.5, "h"),
        ("height 170 cm", "8302-2", 170.0, "cm"),
    ],
)
def test_unit_lexicon(text: str, code: str, value: float, unit: str) -> None:
    p = _one(extract(text, TODAY), code)
    assert (p.value, p.unit, p.confidence, p.flags) == (value, unit, "high", ())


def test_dates_resolve() -> None:
    assert _one(extract("slept 6 hours today", TODAY), "93832-4").date == TODAY
    assert _one(extract("slept 6 hours yesterday", TODAY), "93832-4").date == TODAY - timedelta(days=1)
    past = _one(extract("slept 6 hours on sep 20", TODAY), "93832-4")
    assert (past.date, past.flags) == (dt.date(2026, 9, 20), ())
    past2 = _one(extract("slept 6 hours on 20 september", TODAY), "93832-4")
    assert (past2.date, past2.value) == (dt.date(2026, 9, 20), 6.0)


def test_number_in_a_date_is_not_a_value() -> None:
    p = _one(extract("weight on 2026-09-20 was 70 kg", TODAY), "29463-7")
    assert (p.value, p.unit, p.date) == (70.0, "kg", dt.date(2026, 9, 20))


def test_text_without_any_concept_yields_nothing() -> None:
    assert extract("had a lovely walk in the park", TODAY) == ()
    assert extract("   ", TODAY) == ()


# ---------------------------------------------------------------- T4.1 field completeness


@pytest.mark.parametrize(
    "text",
    [
        "slept 6 hours and walked 8000 steps",
        "slept 6",
        "my weight",
        "headache",
        "temperature 101 °F",
        "bp 80/120",
        "slept 6 hours on monday",
        "heart rate 900 bpm, spo2 97%, pain 4/10, nausea",
    ],
)
def test_t4_1_every_proposal_is_complete(text: str) -> None:
    props = extract(text, TODAY)
    assert props
    for p in props:
        assert p.snippet
        assert p.confidence in ("high", "medium", "low")
        assert p.unit is not None or {"missing_value", "unit_assumed"} & set(p.flags)
        assert p.date is not None
        assert isinstance(p.flags, tuple)
        assert set(p.flags) <= set(FLAGS)
        assert p.blocking is bool(p.flags)


# ---------------------------------------------------------------- T4.2 one input per flag


@pytest.mark.parametrize(
    ("text", "flag", "code"),
    [
        pytest.param("temperature 101 °F", "unit_mismatch", "8310-5", id="unit_mismatch"),
        pytest.param("slept 6", "unit_assumed", "93832-4", id="unit_assumed"),
        pytest.param("slept 6 hours on monday", "ambiguous_date", "93832-4", id="ambiguous_date-weekday"),
        pytest.param("slept 6 hours last week", "ambiguous_date", "93832-4", id="ambiguous_date-last-week"),
        pytest.param("slept 6 hours on dec 3", "ambiguous_date", "93832-4", id="ambiguous_date-future"),
        pytest.param("slept 6 hours and slept 7 hours", "contradiction", "93832-4", id="contradiction-duplicate"),
        pytest.param("bp 80/120", "contradiction", "8480-6", id="contradiction-bp"),
        pytest.param("headache", "unknown_code", None, id="unknown_code"),
        pytest.param("heart rate 900 bpm", "out_of_plausible_range", "8867-4", id="out_of_plausible_range"),
        pytest.param("my weight", "missing_value", "29463-7", id="missing_value"),
    ],
)
def test_t4_2_each_flag_is_produced_and_blocking(text: str, flag: str, code: str | None) -> None:
    props = extract(text, TODAY)
    hits = [p for p in props if p.concept_code == code and flag in p.flags]
    assert hits, props
    assert all(p.blocking is True for p in hits)


def test_t4_2_every_flag_kind_covered() -> None:
    inputs = [
        "temperature 101 °F", "slept 6", "slept 6 hours on monday", "bp 80/120",
        "headache", "heart rate 900 bpm", "my weight",
    ]
    seen = {f for t in inputs for p in extract(t, TODAY) for f in p.flags}
    assert seen == set(FLAGS)
    assert len(FLAGS) == 7


def test_unit_assumed_is_medium_with_default_unit() -> None:
    p = _one(extract("slept 6", TODAY), "93832-4")
    assert (p.value, p.unit, p.confidence, p.flags) == (6.0, "h", "medium", ("unit_assumed",))


def test_unit_mismatch_is_low_and_skips_range() -> None:
    p = _one(extract("temperature 101 °F", TODAY), "8310-5")
    assert p.confidence == "low"
    assert p.flags == ("unit_mismatch",)


def test_unknown_word_has_no_code() -> None:
    p = _one(extract("headache", TODAY), None)
    assert p.display == "headache"
    assert p.value is None
    assert "unknown_code" in p.flags


def test_missing_value_has_no_value() -> None:
    p = _one(extract("my weight", TODAY), "29463-7")
    assert p.value is None and p.unit is None and p.confidence == "low"


def test_both_sides_of_a_contradiction_flagged() -> None:
    dup = _by_code(extract("slept 6 hours and slept 7 hours", TODAY), "93832-4")
    assert len(dup) == 2 and all("contradiction" in p.flags for p in dup)
    bp = extract("bp 80/120", TODAY)
    assert "contradiction" in _one(bp, "8480-6").flags
    assert "contradiction" in _one(bp, "8462-4").flags


def test_ambiguous_weekday_still_carries_a_date() -> None:
    p = _one(extract("slept 6 hours on monday", TODAY), "93832-4")
    assert p.date is not None and p.date <= TODAY


# ---------------------------------------------------------------- detect_flags (review re-run)


def test_detect_flags_clears_contradiction_after_correction() -> None:
    props = extract("slept 6 hours and slept 7 hours", TODAY)
    second = props[1]
    fixed = (props[0], second.model_copy(update={"concept_code": "55411-3", "unit": "h"}))
    out = detect_flags(fixed)
    assert all(p.flags == () and p.blocking is False for p in out)


def test_detect_flags_clears_bp_contradiction_when_corrected() -> None:
    props = extract("bp 80/120 mmhg", TODAY)
    assert all(p.flags == ("contradiction",) for p in props)
    swapped = tuple(
        p.model_copy(update={"value": 120.0 if p.concept_code == "8480-6" else 80.0}) for p in props
    )
    assert all(p.flags == () for p in detect_flags(swapped))


def test_detect_flags_unit_assumed_cleared_only_for_corrected_ids() -> None:
    props = extract("slept 6", TODAY)
    assert detect_flags(props)[0].flags == ("unit_assumed",)
    out = detect_flags(props, corrected={props[0].proposal_id: frozenset({"unit"})})
    assert out[0].flags == () and out[0].blocking is False


def test_detect_flags_ambiguous_date_cleared_by_correction() -> None:
    props = extract("slept 6 hours on monday", TODAY)
    fixed = (props[0].model_copy(update={"date": TODAY}),)
    assert detect_flags(fixed)[0].flags == ("ambiguous_date",)
    assert detect_flags(fixed, corrected={"p-1": frozenset({"date"})})[0].flags == ()


def test_detect_flags_reraises_range_and_unit() -> None:
    p = _one(extract("heart rate 64 bpm", TODAY), "8867-4")
    assert detect_flags((p.model_copy(update={"value": 900.0}),))[0].flags == ("out_of_plausible_range",)
    assert detect_flags((p.model_copy(update={"unit": "kg"}),))[0].flags == ("unit_mismatch",)
    assert detect_flags((p.model_copy(update={"value": None}),))[0].flags == ("missing_value",)
    w = _one(extract("weight 150 lb", TODAY), "29463-7")
    assert w.flags == () and detect_flags((w,))[0].flags == ()  # alt unit converted before range


def test_detect_flags_fixes_out_of_range_when_corrected() -> None:
    p = _one(extract("heart rate 900 bpm", TODAY), "8867-4")
    assert detect_flags((p.model_copy(update={"value": 90.0}),))[0].flags == ()


def test_detect_flags_keeps_unknown_code() -> None:
    props = extract("headache", TODAY)
    assert "unknown_code" in detect_flags(props)[0].flags


# ---------------------------------------------------------------- models


def _proposal(**kw: object) -> Proposal:
    base: dict[str, object] = {
        "proposal_id": "p-1", "concept_code": "93832-4", "display": "Sleep duration", "value": 6.0,
        "unit": "h", "date": TODAY, "source_kind": "text", "source_ref": "c-0:0-13",
        "snippet": "slept 6 hours", "method": METHOD_TEXT, "confidence": "high", "flags": (),
    }
    base.update(kw)
    return Proposal.model_validate(base)


def test_proposal_blocking_invariant() -> None:
    assert _proposal().blocking is False
    assert _proposal(flags=("unit_assumed",)).blocking is True
    with pytest.raises(pydantic.ValidationError):
        _proposal(flags=("unit_assumed",), blocking=False)
    with pytest.raises(pydantic.ValidationError):
        _proposal(flags=(), blocking=True)
    with pytest.raises(pydantic.ValidationError):
        _proposal(flags=("nonsense",))
    with pytest.raises(pydantic.ValidationError):
        _proposal(extra_field=1)
    dumped = _proposal(flags=("contradiction",)).model_dump(mode="json")
    assert dumped["blocking"] is True
    assert Proposal.model_validate(dumped).blocking is True


def test_proposal_is_frozen_and_snippet_capped() -> None:
    p = _proposal()
    with pytest.raises(pydantic.ValidationError):
        p.value = 7.0  # type: ignore[misc]
    with pytest.raises(pydantic.ValidationError):
        _proposal(snippet="x" * 81)


def test_methods() -> None:
    assert METHOD_FIXTURE == "fixture-sidecar-v1"
    assert _proposal(method=METHOD_FIXTURE, source_kind="image_fixture").method == METHOD_FIXTURE


def test_checkin_status_transition_one_way() -> None:
    c = CheckIn(
        checkin_id="c-1", journal_id="j-1", created_at=dt.datetime(2026, 9, 29, 12, tzinfo=dt.UTC),
        status="pending_review", proposals=(_proposal(),),
    )
    saved = c.mark_saved()
    assert saved.status == "saved" and c.status == "pending_review"
    assert saved.proposals == c.proposals
    with pytest.raises(ValueError):
        saved.mark_saved()
    with pytest.raises(pydantic.ValidationError):
        CheckIn(checkin_id="c-1", journal_id="j-1", created_at=c.created_at, status="discarded", proposals=())


# ---------------------------------------------------------------- T2.3 PII tripwire


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("reach me at sam.example@example.org", "email"),
        ("call 555-123-4567 tomorrow", "phone"),
        ("my number is (555) 123 4567", "phone"),
        ("+44 20 7946 0000 any time", "phone"),
        ("ssn 123-45-6789", "national_id"),
        ("id 123456789", "national_id"),
        ("my name is Sam", "self_name"),
        ("Hi, call me Alex", "self_name"),
        ("I am called Robin", "self_name"),
        ("date of birth 1990-01-01", "date_of_birth"),
        ("DOB 01/02/1990", "date_of_birth"),
        ("I was born on 3 March 1985", "date_of_birth"),
        ("my birthday is 12 may", "date_of_birth"),
    ],
)
def test_t2_3_tripwire_fires(text: str, reason: str) -> None:
    assert reason in scan(text)


@pytest.mark.parametrize(
    "text",
    [
        "slept 6 hours and walked 8000 steps",
        "walked 12000 steps",
        "bp 120/80 mmHg",
        "slept 6.5 hours",
        "weight on 2026-11-03 was 72.5 kg",
        "pain 4/10, heart rate 64 bpm, spo2 97%",
        "temperature 37.2 c on sep 20",
        "walked 10000 steps and 12000 steps",
        "",
    ],
)
def test_t2_3_tripwire_quiet_on_measurements(text: str) -> None:
    assert scan(text) == ()


def test_tripwire_reasons_are_known_codes() -> None:
    reasons = scan("my name is Sam, sam@example.org, 555-123-4567, dob 1990-01-01, 123-45-6789")
    assert set(reasons) == {"email", "phone", "national_id", "self_name", "date_of_birth"}
    assert len(reasons) == len(set(reasons))


# ---------------------------------------------------------------- T2.4 crisis pre-check


@pytest.mark.parametrize(
    "text",
    [
        "I want to die",
        "I've been thinking about suicide",
        "feeling suicidal today",
        "I might kill myself",
        "I want to end my life",
        "I'm better off dead",
        "there is no reason to live",
        "I have been self-harming, self harm again",
        "I want to hurt myself",
        "I overdosed last night",
        "I can’t go on like this",
        "I cannot go on",
        "I KEEP CUTTING MYSELF",
    ],
)
def test_t2_4_crisis_detected(text: str) -> None:
    assert is_crisis(text) is True


@pytest.mark.parametrize(
    "text",
    ["slept 6 hours and walked 8000 steps", "pain 4/10", "I'm dying to try the new trail", "headache", ""],
)
def test_t2_4_crisis_quiet_on_neutral(text: str) -> None:
    assert is_crisis(text) is False


def test_handoff_text_passes_both_gates() -> None:
    assert HANDOFF_TEXT.startswith(POLICY_TEXT["clinician_handoff"])
    assert "emergency services" in HANDOFF_TEXT
    assert safe_text(HANDOFF_TEXT) == HANDOFF_TEXT
    assert dsl.check(HANDOFF_TEXT, crisis_context=True).passed is True


# ---------------------------------------------------------------- QG P1.3 fixes

_PATHOLOGICAL = [
    pytest.param(("dob" + " is" * 166 + "x")[:500], id="dob-repeated-is"),
    pytest.param(("born on" + " in" * 166 + "x")[:500], id="born-repeated-in"),
    pytest.param(("date of birth" + " was" * 125 + "!")[:500], id="dob-repeated-was"),
    pytest.param(("dob" + ": is -" * 100)[:500], id="dob-separators"),
    pytest.param(("1 " * 250)[:500], id="digits-spaces"),
    pytest.param(("1-" * 249 + "x")[:500], id="digits-dashes"),
    pytest.param(("1." * 249 + "x")[:500], id="digits-dots"),
    pytest.param(("12/" * 167)[:500], id="digits-slashes"),
    pytest.param(("my name " * 63)[:500], id="repeated-my-name"),
    pytest.param(("a." * 250)[:499] + "@", id="email-ish"),
]


@pytest.mark.parametrize("text", _PATHOLOGICAL)
def test_p1_regexes_are_linear_on_pathological_input(text: str) -> None:
    assert len(text) <= 500
    t0 = time.perf_counter()
    scan(text)
    assert time.perf_counter() - t0 < 0.2
    t0 = time.perf_counter()
    extract(text, TODAY)
    assert time.perf_counter() - t0 < 0.2


@pytest.mark.parametrize(
    "text",
    ["I was born 12 May 1985", "born 3/4/1990", "born on 3 March 1985", "born in 1985", "Born: 1990-01-01"],
)
def test_p3_born_without_on_or_in_is_date_of_birth(text: str) -> None:
    assert "date_of_birth" in scan(text)


@pytest.mark.parametrize(
    "text",
    [
        "my name's John",
        "my name’s John",
        "I'm called Jane",
        "I’m called Jane",
        "name: John Smith",
        "I'm John Smith",
        "I’m John Smith and slept 6 hours",
        "Hi, I'm Sam.",
        "I'm Sam",
        "MY NAME IS SAM",
        "mу nаme is Sam",  # Cyrillic у and а
        "ｍｙ ｎａｍｅ ｉｓ Sam",  # fullwidth
        "call me Alex",
        "Call me Аlex",  # Cyrillic capital А
    ],
)
def test_p5_self_name_fires_after_normalisation(text: str) -> None:
    assert "self_name" in scan(text)


@pytest.mark.parametrize(
    "text",
    [
        "I'm tired",
        "I'm tired today, slept 6 hours",
        "I’m feeling Better",
        "please call me tomorrow",
        "call me back",
        "call me later",
        "Call me when you can",
    ],
)
def test_p5_self_name_quiet_on_ordinary_phrases(text: str) -> None:
    assert "self_name" not in scan(text)


@pytest.mark.parametrize("text", ["DOB 01/02/1990", "ＤＯＢ 1990-01-01", "dоb: 1990-01-01", "BORN 3/4/1990"])
def test_p5_date_of_birth_after_normalisation(text: str) -> None:
    assert "date_of_birth" in scan(text)


@pytest.mark.parametrize("text", ["ssn 123 45 6789", "my id is 123.45.6789.", "123-45-6789", "ssn:123 45 6789, thanks"])
def test_p4_ssn_with_spaces_or_dots_is_national_id(text: str) -> None:
    assert "national_id" in scan(text)


@pytest.mark.parametrize(
    "text",
    [
        "bp 120 80",
        "bp 120 80 pulse 64",
        "bp 120 80 8000 steps",
        "bp 120/80 and 8000 steps",
        "120 80 64 bpm",
        "weight 72.5 kg, temp 36.6",
        "123 45 67",
    ],
)
def test_p4_ssn_quiet_on_measurement_sequences(text: str) -> None:
    assert "national_id" not in scan(text)


@pytest.mark.parametrize(
    "text",
    [
        # die
        "I wanted to die last night", "I wanna die", "I'm going to die by my own hand", "I'm gonna die, I want it",
        # end
        "thinking of ending my life", "I ended my life once almost", "I want to end it all",
        # kill
        "I killed myself almost", "thought about killing my self", "I want to kill my self",
        # not live
        "I don't want to live anymore", "I do not want to live", "I don’t want to live",
        # dead
        "I wish I was dead", "I wish I were dead",
        # pills
        "I took too many pills", "I have taken too many pills", "I might take too many pills",
        "I overdose sometimes", "overdosing again", "I OVERDOSED",
        # self harm, any dash or space
        "self–harm again", "self—harm", "self - harm", "self  harm", "self-harmed yesterday",
        # suicid*
        "suicidality is back", "Suicide", "SUICIDAL thoughts",
    ],
)
def test_p2_crisis_families_detected(text: str) -> None:
    assert is_crisis(text) is True


@pytest.mark.parametrize(
    "text",
    [
        # passive ideation (screening wording: wish to be dead, go to sleep and not wake up)
        "I wish I could go to sleep and not wake up", "sometimes I hope I don't wake up",
        "I want to disappear forever", "I don't want to be here anymore", "I don't want to exist",
        "everyone would be better off without me", "I'd be better off gone",
        "there's no point in living", "life isn't worth living", "what's the point of living",
        # active ideation and method / plan
        "I've been having thoughts of killing myself", "thinking about taking my own life",
        "I want to take my life", "I want to hang myself", "I'm going to jump off the bridge",
        "I have a plan to end it", "I'm planning to end it tonight", "I'm going to end it",
        "I want to cut myself", "I tried to kill myself last year", "I've attempted suicide",
        "I'm not safe right now", "I can't keep myself safe", "I'm going to hurt myself tonight",
        "I can't take it anymore, I want to die",
    ],
)
def test_am_crisis_screening_families_detected(text: str) -> None:
    assert is_crisis(text) is True


@pytest.mark.parametrize(
    "text",
    ["dead tired after work", "reading about harm reduction", "the meeting will end it early",
     "I'm planning to end the call at 5", "jumped off the curb", "I cut my hair", "point of view", "my rent is overdue", "self-harmony yoga class",
     "pain 4/10", "I'm dying to try the new trail", "the battery will die soon", "end of my shift"],
)
def test_p2_crisis_quiet_on_lookalikes(text: str) -> None:
    assert is_crisis(text) is False


@pytest.mark.parametrize("text", ["my bp was 120/80.", "bp 120/80, slept 6 hours", "(bp 120/80)", "bp 120/80!"])
def test_p7_bp_pair_before_sentence_punctuation(text: str) -> None:
    props = extract(text, TODAY)
    assert _one(props, "8480-6").value == 120.0
    assert _one(props, "8462-4").value == 80.0


def test_p7_bp_pair_still_rejects_decimals() -> None:
    props = extract("bp 120/80.5", TODAY)
    assert not _by_code(props, "8462-4")


def test_p8_bare_keyword_beside_another_concepts_value_is_missing_value() -> None:
    props = extract("weight 72 kg heart rate high", TODAY)
    w = _one(props, "29463-7")
    assert (w.value, w.unit, w.flags) == (72.0, "kg", ())
    hr = _one(props, "8867-4")
    assert hr.value is None and "missing_value" in hr.flags and hr.blocking is True


@pytest.mark.parametrize("text", ["walked 8000 steps today", "walked 8000 steps", "bp 120/80 systolic"])
def test_p8_bare_keyword_of_the_bound_quantity_stays_silent(text: str) -> None:
    props = extract(text, TODAY)
    assert not [p for p in props if "missing_value" in p.flags], props


def test_p9_date_only_correction_keeps_unit_assumed() -> None:
    props = extract("slept 6 on monday", TODAY)
    assert props[0].flags == ("unit_assumed", "ambiguous_date")
    fixed = (props[0].model_copy(update={"date": TODAY}),)
    assert detect_flags(fixed, corrected={"p-1": frozenset({"date"})})[0].flags == ("unit_assumed",)
    assert detect_flags(fixed, corrected={"p-1": frozenset({"value"})})[0].flags == ("unit_assumed", "ambiguous_date")
    assert detect_flags(fixed, corrected={"p-1": frozenset({"unit"})})[0].flags == ("ambiguous_date",)
    assert detect_flags(fixed, corrected={"p-2": frozenset({"unit", "date"})})[0].flags == ("unit_assumed", "ambiguous_date")
    assert detect_flags(fixed, corrected={"p-1": frozenset({"unit", "date"})})[0].flags == ()


def test_p11_with_changes_revalidates_the_blocking_invariant() -> None:
    p = _proposal(flags=("unit_assumed",))
    assert p.blocking is True
    cleared = p.with_changes(flags=())
    assert cleared.flags == () and cleared.blocking is False
    assert p.blocking is True  # the original is untouched
    assert p.with_changes(value=7.0).blocking is True
    assert _proposal().with_changes(flags=("contradiction",)).blocking is True
    with pytest.raises(pydantic.ValidationError):
        p.with_changes(flags=(), blocking=True)
    with pytest.raises(pydantic.ValidationError):
        p.with_changes(blocking=False)
    with pytest.raises(pydantic.ValidationError):
        p.with_changes(confidence="certain")


def test_p11_mark_saved_revalidates() -> None:
    c = CheckIn(
        checkin_id="c-1", journal_id="j-1", created_at=dt.datetime(2026, 9, 29, 12, tzinfo=dt.UTC),
        status="pending_review", proposals=(_proposal(flags=("unit_assumed",)),),
    )
    saved = c.mark_saved()
    assert type(saved) is CheckIn and saved.status == "saved"
    assert saved.proposals == c.proposals and saved.proposals[0].blocking is True
    bad = Proposal.model_construct(**{**_proposal().model_dump(), "flags": ("unit_assumed",), "blocking": False})
    forged = CheckIn.model_construct(**{**c.model_dump(), "proposals": (bad,)})
    with pytest.raises(pydantic.ValidationError):
        forged.mark_saved()  # model_copy would have carried the broken invariant into "saved"


# ---------------------------------------------------------------- QG-P1-H1: journal window on extracted dates


def test_extracted_date_before_window_is_ambiguous() -> None:
    (p,) = extract("weight 70 kg on 2025-01-01", TODAY)
    assert p.date == dt.date(2025, 1, 1)
    assert p.flags == ("ambiguous_date",) and p.blocking


def test_extracted_date_at_window_edge_is_not_flagged() -> None:
    edge = TODAY - timedelta(days=28)
    (p,) = extract(f"weight 70 kg on {edge.isoformat()}", TODAY)
    assert p.flags == ()


def test_detect_flags_window_needs_today() -> None:
    (p,) = extract("weight 70 kg", TODAY)
    old = p.with_changes(date=TODAY - timedelta(days=29))
    assert detect_flags((old,))[0].flags == ()  # no today: no window check
    assert detect_flags((old,), today=TODAY)[0].flags == ("ambiguous_date",)
    corrected = {p.proposal_id: frozenset({"date"})}
    assert detect_flags((old,), corrected, today=TODAY)[0].flags == ("ambiguous_date",)
    assert detect_flags((old.with_changes(date=TODAY),), corrected, today=TODAY)[0].flags == ()
