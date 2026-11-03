"""Sealed challenge suite — slice B boundary DSL (``daypart.membrane.dsl``).

Category: functional/membrane-dsl
Spec: docs/daypart/contracts/CONTRACT-slice-B-membrane.md sections B1, B2.

Written double-blind against the contract only. Ambiguities resolved here
(questions for the principal if the implementer disagrees):

* ``Violation.match`` for ``crisis.missing_handoff`` has no natural matched
  substring; only its type (``str``) is asserted.
* The shape of ``RulePack`` is unspecified; only that ``load_rules`` returns
  without error for "boundary-v1" and raises ``FileNotFoundError`` otherwise.
* ``half`` as a number word is not exercised (its natural phrasing,
  "half a pill", puts an article between number and unit).
* ``lhm.imperative_dose`` is only exercised with digit quantities (LHM's
  structure is digits-only; B2 does not say whether number words count).
* ``dosing.imperative`` is only exercised with base verb forms plus the
  explicit "stop taking"/"start taking" phrases.
* ``g`` as a unit requires a word boundary: "2 glasses" is not a dose.
"""

from __future__ import annotations

import dataclasses
import importlib.resources

import pytest
import yaml

from daypart.membrane import dsl
from daypart.membrane.dsl import (
    RULE_IDS,
    RULE_VERSION,
    Verdict,
    Violation,
    check,
    load_rules,
)

ALL_RULES = frozenset(
    {
        "dosing.unit_quantity",
        "dosing.imperative",
        "diagnosis.claim",
        "treatment.claim",
        "substance.rx_named",
        "lhm.banned_lexicon",
        "lhm.imperative_dose",
        "crisis.missing_handoff",
    }
)

D10_PASS = "Try setting your evening reminder 15 minutes earlier tonight."
D10_FAIL = "You should take 400 mg of magnesium."

COMPLIANT_EXAMPLES = [
    "Nice work logging three days in a row — want to keep the streak going?",
    "What got in the way yesterday? We can plan around it together.",
    "Would a reminder at 8pm fit your evening better?",
    "Your clinician is the right person for questions about your protocol.",
    "It's okay to miss a day. Want to pick it back up tomorrow morning?",
]


def fired(text: str, **kw) -> set[str]:
    return {v.rule_id for v in check(text, **kw).violations}


# --------------------------------------------------------------------------- #
# B1 — rule pack / module surface
# --------------------------------------------------------------------------- #


class TestRulePack:
    """functional/membrane-dsl :: rule pack + module surface (B1)."""

    def test_rule_version_constant(self):
        assert RULE_VERSION == "boundary-v1"

    def test_rule_ids_exactly_b2(self):
        assert isinstance(RULE_IDS, frozenset)
        assert RULE_IDS == ALL_RULES

    def test_load_rules_default(self):
        assert load_rules() is not None

    def test_load_rules_explicit_version(self):
        assert load_rules("boundary-v1") is not None

    @pytest.mark.parametrize("bad", ["boundary-v0", "boundary-v2", "nope", "boundary-v1-extra"])
    def test_load_rules_unknown_version(self, bad):
        with pytest.raises(FileNotFoundError):
            load_rules(bad)

    def test_rule_pack_shipped_as_package_data(self):
        res = importlib.resources.files("daypart.membrane").joinpath("rules/boundary-v1.yaml")
        assert res.is_file()
        data = yaml.safe_load(res.read_text(encoding="utf-8"))
        assert data["version"] == "boundary-v1"

    def test_dataclasses_frozen(self):
        v = check(D10_FAIL)
        assert dataclasses.is_dataclass(v) and isinstance(v, Verdict)
        with pytest.raises(dataclasses.FrozenInstanceError):
            v.passed = True  # type: ignore[misc]
        viol = v.violations[0]
        assert isinstance(viol, Violation)
        with pytest.raises(dataclasses.FrozenInstanceError):
            viol.rule_id = "x"  # type: ignore[misc]

    def test_module_exposes_check(self):
        assert dsl.check is check


# --------------------------------------------------------------------------- #
# Verdict semantics
# --------------------------------------------------------------------------- #

VERDICT_PROBES = [
    "",
    D10_PASS,
    D10_FAIL,
    "You logged 500 mg and asked about metformin.",
    "This routine will cure everything.",
    *COMPLIANT_EXAMPLES,
]


class TestVerdict:
    """functional/membrane-dsl :: verdict semantics."""

    @pytest.mark.parametrize("text", VERDICT_PROBES)
    def test_passed_iff_no_violations(self, text):
        v = check(text)
        assert isinstance(v.violations, tuple)
        assert v.passed is (len(v.violations) == 0)

    @pytest.mark.parametrize("text", VERDICT_PROBES)
    def test_rule_version_field(self, text):
        assert check(text).rule_version == "boundary-v1"
        assert check(text, crisis_context=True).rule_version == "boundary-v1"

    @pytest.mark.parametrize("text", VERDICT_PROBES)
    def test_violation_ids_in_rule_ids(self, text):
        for crisis in (False, True):
            for viol in check(text, crisis_context=crisis).violations:
                assert viol.rule_id in RULE_IDS
                assert isinstance(viol.match, str)

    def test_match_is_from_normalized_text(self):
        text = "You LOGGED 400 MG Yesterday."
        v = check(text)
        hits = [x for x in v.violations if x.rule_id == "dosing.unit_quantity"]
        assert hits
        for h in hits:
            assert h.match
            assert h.match in text.casefold()
            assert "400" in h.match

    def test_multiple_rules_fire(self):
        ids = fired("You logged 500 mg yesterday and you probably have diabetes; metformin cures it.")
        assert {
            "dosing.unit_quantity",
            "diagnosis.claim",
            "substance.rx_named",
            "treatment.claim",
            "lhm.banned_lexicon",
        } <= ids

    def test_determinism(self):
        texts = [*VERDICT_PROBES, "TAKE 500MG", "met​formin", "mеtformin"]
        for t in texts:
            for crisis in (False, True):
                a = check(t, crisis_context=crisis)
                b = check(t, crisis_context=crisis)
                assert a == b

    def test_crisis_context_keyword_only(self):
        with pytest.raises(TypeError):
            check("hello", True)  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# D10 pair + compliant examples + empty string
# --------------------------------------------------------------------------- #


class TestContractExamples:
    """functional/membrane-dsl :: D10 pair, compliant examples, empty string."""

    def test_d10_pass(self):
        assert check(D10_PASS).passed is True

    def test_d10_fail(self):
        v = check(D10_FAIL)
        assert v.passed is False
        assert "dosing.unit_quantity" in {x.rule_id for x in v.violations}

    @pytest.mark.parametrize("text", COMPLIANT_EXAMPLES)
    def test_compliant_examples_pass(self, text):
        v = check(text)
        assert v.passed is True, [x.rule_id for x in v.violations]
        assert v.violations == ()

    def test_empty_passes(self):
        v = check("")
        assert v.passed is True
        assert v.violations == ()

    def test_empty_crisis_fires_only_handoff(self):
        v = check("", crisis_context=True)
        assert v.passed is False
        assert {x.rule_id for x in v.violations} == {"crisis.missing_handoff"}


# --------------------------------------------------------------------------- #
# B2 per-rule positives and negatives
# --------------------------------------------------------------------------- #

UNIT_QUANTITY_POS = [
    "You logged 500 mg yesterday.",
    "You logged 400mg yesterday.",
    "The label lists 1,000 IU per serving.",
    "You noted 2.5 ml last night.",
    "You logged 250 mcg this morning.",
    "You logged 50 µg this morning.",
    "You logged 50 ug this morning.",
    "You logged 3 g at lunch.",
    "You logged four hundred milligrams.",
    "You noted two capsules with dinner.",
    "You noted a couple of softgels.",
    "You noted ten drops in water.",
    "You noted 2 scoops after the gym.",
    "You noted 1 tablet at breakfast.",
    "You noted three pills at breakfast.",
    "You noted 5 units before dinner.",
    "You noted 10 international units.",
    "You noted 20 milliliters.",
    "You noted 20 millilitres.",
    "You noted 100 micrograms.",
    "You noted 2 grams.",
    "You noted one thousand IU.",
    "YOU LOGGED 500 MG YESTERDAY.",
]
UNIT_QUANTITY_NEG = [
    "Would a reminder at 8pm fit better?",
    "Try it 15 minutes earlier.",
    "That's three days in a row.",
    "You drank 2 glasses of water today.",
    "Your 7am walk is on the calendar.",
    "You hit 4,000 steps before lunch.",
    "Give it a couple of days and check back.",
    "You have 3 goals this week.",
]

IMPERATIVE_POS = [
    "Remember to take your medication tonight.",
    "Increase your dose next week.",
    "You could lower the dosage.",
    "Double your supplements tomorrow.",
    "Skip your meds tonight.",
    "Halve the pills on rest days.",
    "Adjust your prescription this week.",
    "Stop taking it before bed.",
    "Start taking it in the morning.",
    "Just double up tomorrow.",
    "Take an extra one tonight.",
    "Take extra on busy days.",
    "Take more if you feel tired.",
    "Take less on rest days.",
]
IMPERATIVE_NEG = [
    "Take a moment to breathe before bed.",
    "Want to change your reminder time?",
    "Start your walk whenever it suits you.",
    "Did you log your evening routine?",
    "You can skip the check-in today.",
    "Increase your step goal if it feels easy.",
]

DIAGNOSIS_POS = [
    "You probably have diabetes.",
    "You might have sleep apnea.",
    "You may have an iron deficiency.",
    "You are suffering from insomnia.",
    "You are developing hypertension.",
    "You likely have a thyroid disorder.",
    "This sounds like a symptom of depression.",
    "That could be a sign of something bigger.",
    "It looks like signs of burnout.",
    "This indicates symptoms of stress.",
    "You are prediabetic.",
    "You are diabetic.",
    "You are hypertensive.",
    "You are depressed.",
    "You are anemic.",
    "You are deficient in magnesium.",
    "Your diagnosis looks clear.",
    "That reading looks diagnostic.",
]
CONDITIONS = [
    "diabetes",
    "prediabetes",
    "hypertension",
    "depression",
    "anxiety disorder",
    "anemia",
    "deficiency",
    "insomnia",
    "sleep apnea",
    "hypothyroidism",
    "infection",
    "disorder",
    "disease",
    "syndrome",
    "cancer",
    "arrhythmia",
]
DIAGNOSIS_NEG = [
    "You have a busy week ahead.",
    "You are doing a great job.",
    "That is a lot to juggle.",
    "This is a big step forward.",
    "You have three days logged.",
    "You are developing a solid habit.",
]

TREATMENT_POS = [
    "This routine will fix your sleep.",
    "Magnesium can heal your gut.",
    "That could reverse the trend.",
    "Walking should treat the cramps.",
    "Use this to fix your energy.",
    "It is a treatment for headaches.",
    "Ginger is a remedy for nausea.",
    "Yoga is a therapy for stress.",
    "Breathing exercises are a cure for stress.",
    "This tea heals inflammation.",
    "Walking cures everything.",
    "This stretch treats back pain.",
    "Sleep reverses the damage.",
]
TREATMENT_NEG = [
    "Your reminder will fire at 8pm.",
    "You can change it anytime.",
    "Want to adjust the reminder?",
    "Let's pick a time that works.",
]

RX = [
    "metformin",
    "rapamycin",
    "sirolimus",
    "semaglutide",
    "ozempic",
    "wegovy",
    "tirzepatide",
    "mounjaro",
    "insulin",
    "statin",
    "statins",
    "atorvastatin",
    "rosuvastatin",
    "warfarin",
    "testosterone",
    "hgh",
    "sertraline",
    "fluoxetine",
    "adderall",
    "modafinil",
    "levothyroxine",
    "prednisone",
    "antibiotic",
    "antibiotics",
]
RX_NEG = [
    "Restating your goal out loud helps.",
    "Your morning supplement check-in is ready.",
    "Keep your streak going this week.",
]

BANNED_POS = [
    "This could improve your evenings.",
    "Nice improvement this week!",
    "Treat yourself tonight.",
    "A cure-all routine.",
    "Let's prevent missed days.",
    "Prevention starts with a plan.",
    "The trend reversed this week.",
    "The efficacy of this habit is high.",
    "This is clinically proven.",
    "A clinical-grade plan.",
]
BANNED_NEG = [
    "Your clinician is the right person to ask.",
    "Your data is stored securely.",
    "A weekend retreat could be a nice reset.",
]
ADVICE_ONLY = [
    "We recommend a reminder at 8pm.",
    "You should try an earlier reminder.",
    "We advise keeping your phone nearby.",
    "This is helpful advice for busy weeks.",
]

IMPERATIVE_DOSE_POS = [
    "Take 500 mg tonight.",
    "Start 200mg tomorrow morning.",
    "Keep taking 2 g after meals.",
    "You should start with 1000 IU daily.",
    "Stop the 50 mcg once you wake.",
    "She takes 5 ml with breakfast.",
    "TAKE 500 MG TONIGHT.",
    "Ta​ke 500 mg tonight.",
    "Tаke 500 mg tonight.",
]
_FAR = (
    "Take your time with the evening routine, the reminders, the walking plan, "
    "the journaling habit and the water goal you set, and then you logged 500 mg"
)
IMPERATIVE_DOSE_NEG = [
    "You logged 500 mg yesterday.",
    "Take a walk. You logged 500 mg earlier.",
    _FAR,
]

CRISIS_SILENT = [
    "I hear you. You're not alone in this.",
    "Let's breathe slowly together for a moment.",
    "That sounds really heavy.",
]
CRISIS_HANDOFFS = [
    "Please call or text 988 now. You're not alone.",
    "Please contact emergency services right away.",
    "You can reach a crisis line any time, day or night.",
    "The Crisis Text Line is there for you right now.",
    "If you're in danger, call 911.",
    "Please call your local emergency number now.",
]


class TestDosingUnitQuantity:
    """functional/membrane-dsl :: dosing.unit_quantity."""

    @pytest.mark.parametrize("text", UNIT_QUANTITY_POS)
    def test_positive(self, text):
        assert "dosing.unit_quantity" in fired(text)

    @pytest.mark.parametrize("text", UNIT_QUANTITY_NEG)
    def test_negative(self, text):
        assert "dosing.unit_quantity" not in fired(text)

    def test_past_tense_stricter_than_lhm(self):
        v = check("You logged 500 mg yesterday.")
        assert v.passed is False


class TestDosingImperative:
    """functional/membrane-dsl :: dosing.imperative."""

    @pytest.mark.parametrize("text", IMPERATIVE_POS)
    def test_positive(self, text):
        assert "dosing.imperative" in fired(text)

    @pytest.mark.parametrize("text", IMPERATIVE_NEG)
    def test_negative(self, text):
        assert "dosing.imperative" not in fired(text)


class TestDiagnosisClaim:
    """functional/membrane-dsl :: diagnosis.claim."""

    @pytest.mark.parametrize("text", DIAGNOSIS_POS)
    def test_positive(self, text):
        assert "diagnosis.claim" in fired(text)

    @pytest.mark.parametrize("cond", CONDITIONS)
    def test_conditions_list(self, cond):
        assert "diagnosis.claim" in fired(f"You could have {cond}.")

    @pytest.mark.parametrize("cond", CONDITIONS)
    def test_condition_within_four_words(self, cond):
        # condition is the 3rd word after "have" — within 4 under any reading
        assert "diagnosis.claim" in fired(f"You might have a mild {cond} today.")

    @pytest.mark.parametrize("text", DIAGNOSIS_NEG)
    def test_negative(self, text):
        assert "diagnosis.claim" not in fired(text)


class TestTreatmentClaim:
    """functional/membrane-dsl :: treatment.claim."""

    @pytest.mark.parametrize("text", TREATMENT_POS)
    def test_positive(self, text):
        assert "treatment.claim" in fired(text)

    @pytest.mark.parametrize("text", TREATMENT_NEG)
    def test_negative(self, text):
        assert "treatment.claim" not in fired(text)


class TestSubstanceRxNamed:
    """functional/membrane-dsl :: substance.rx_named."""

    @pytest.mark.parametrize("name", RX)
    def test_each_listed_substance(self, name):
        assert "substance.rx_named" in fired(f"Have you asked your care team about {name}?")

    @pytest.mark.parametrize("text", RX_NEG)
    def test_negative_word_boundary(self, text):
        assert "substance.rx_named" not in fired(text)


class TestBannedLexicon:
    """functional/membrane-dsl :: lhm.banned_lexicon (D10)."""

    @pytest.mark.parametrize("text", BANNED_POS)
    def test_positive(self, text):
        assert "lhm.banned_lexicon" in fired(text)

    def test_every_lhm_stem_fires(self):
        from syntropy.lhm.metrics.claims import BANNED_LEXICON

        for stem in BANNED_LEXICON:
            assert "lhm.banned_lexicon" in fired(f"A note about {stem} today."), stem

    @pytest.mark.parametrize("text", BANNED_NEG)
    def test_negative(self, text):
        assert "lhm.banned_lexicon" not in fired(text)

    @pytest.mark.parametrize("text", ADVICE_ONLY)
    def test_advice_lexicon_not_in_dsl(self, text):
        v = check(text)
        assert v.passed is True, [x.rule_id for x in v.violations]


class TestLhmImperativeDose:
    """functional/membrane-dsl :: lhm.imperative_dose."""

    @pytest.mark.parametrize("text", IMPERATIVE_DOSE_POS)
    def test_positive(self, text):
        assert "lhm.imperative_dose" in fired(text)

    @pytest.mark.parametrize("text", IMPERATIVE_DOSE_NEG)
    def test_negative(self, text):
        ids = fired(text)
        assert "lhm.imperative_dose" not in ids
        assert "dosing.unit_quantity" in ids


class TestCrisisMissingHandoff:
    """functional/membrane-dsl :: crisis.missing_handoff."""

    @pytest.mark.parametrize("text", CRISIS_SILENT)
    def test_fires_only_with_crisis_context(self, text):
        assert fired(text, crisis_context=True) == {"crisis.missing_handoff"}
        assert check(text).passed is True
        assert check(text, crisis_context=False).passed is True

    @pytest.mark.parametrize("text", CRISIS_HANDOFFS)
    def test_handoff_markers_satisfy(self, text):
        v = check(text, crisis_context=True)
        assert v.passed is True, [x.rule_id for x in v.violations]

    @pytest.mark.parametrize("text", [*VERDICT_PROBES, *UNIT_QUANTITY_POS[:3]])
    def test_never_fires_without_crisis_context(self, text):
        assert "crisis.missing_handoff" not in fired(text)
        assert "crisis.missing_handoff" not in fired(text, crisis_context=False)

    def test_handoff_does_not_mask_other_rules(self):
        ids = fired("Call 911 if needed, and you logged 500 mg.", crisis_context=True)
        assert "crisis.missing_handoff" not in ids
        assert "dosing.unit_quantity" in ids

    def test_handoff_marker_case_insensitive(self):
        assert check("CALL 911 NOW.", crisis_context=True).passed is True


# --------------------------------------------------------------------------- #
# Normalization robustness
# --------------------------------------------------------------------------- #

NORMALIZATION_CASES = [
    # (text, expected rule_id)
    ("YOU LOGGED 400 MG.", "dosing.unit_quantity"),
    ("ASK ABOUT METFORMIN.", "substance.rx_named"),
    ("THIS WILL IMPROVE THINGS.", "lhm.banned_lexicon"),
    ("YOU PROBABLY HAVE DIABETES.", "diagnosis.claim"),
    ("You logged 400mg.", "dosing.unit_quantity"),
    ("You logged 2.5ml.", "dosing.unit_quantity"),
    ("You logged 1,000iu.", "dosing.unit_quantity"),
    ("You logged 250MCG.", "dosing.unit_quantity"),
    ("You logged four hundred milligrams.", "dosing.unit_quantity"),
    ("You logged Four Hundred Milligrams.", "dosing.unit_quantity"),
    ("You noted two capsules.", "dosing.unit_quantity"),
    ("You noted a couple of tablets.", "dosing.unit_quantity"),
    ("You noted twenty drops.", "dosing.unit_quantity"),
    ("You noted fifty micrograms.", "dosing.unit_quantity"),
    ("Ask about met​formin.", "substance.rx_named"),
    ("Ask about oz‌empic.", "substance.rx_named"),
    ("This will im‍prove things.", "lhm.banned_lexicon"),
    ("This will pre﻿vent lapses.", "lhm.banned_lexicon"),
    ("You logged 400 m​g.", "dosing.unit_quantity"),
    ("Your dia​gnosis is clear.", "diagnosis.claim"),
    ("Ask about mеtformin.", "substance.rx_named"),  # Cyrillic e
    ("This is a сure.", "lhm.banned_lexicon"),  # Cyrillic s -> c
    ("This will іmprove things.", "lhm.banned_lexicon"),  # Cyrillic i
    ("You probably have diаbetes.", "diagnosis.claim"),  # Cyrillic a
    ("Ask about аdderall.", "substance.rx_named"),  # Cyrillic a
    ("Ask about ozεmpic.", "substance.rx_named"),  # Greek epsilon
    ("Ask about ｍｅｔｆｏｒｍｉｎ.", "substance.rx_named"),  # fullwidth
    ("Ask about met̲formin.", "substance.rx_named"),  # combining low line
    ("TАKE 500 MG.", "lhm.imperative_dose"),  # Cyrillic capital A (casefold -> а -> a)
]


class TestNormalization:
    """functional/membrane-dsl :: normalization robustness."""

    @pytest.mark.parametrize("text,rule", NORMALIZATION_CASES)
    def test_obfuscation_still_fires(self, text, rule):
        assert rule in fired(text)


# --------------------------------------------------------------------------- #
# Negated disclaimer excision
# --------------------------------------------------------------------------- #

DISCLAIMER_PASS = [
    "This is not intended to diagnose, treat, cure, or prevent any disease.",
    "Not meant to treat or cure anything, just a nudge to log tonight.",
    "This is not medical advice.",
    "This is not a diagnosis.",
    "Not a clinical recommendation, just a reminder for 8pm.",
    "This is NOT INTENDED TO DIAGNOSE, TREAT, CURE, OR PREVENT ANY DISEASE.",
]
DISCLAIMER_SMUGGLED = [
    ("This is not medical advice — you logged 400 mg of magnesium.", "dosing.unit_quantity"),
    ("This is not medical advice: take 400 mg of magnesium tonight.", "lhm.imperative_dose"),
    ("Not intended to diagnose or treat anything; magnesium cures cramps.", "treatment.claim"),
    ("Not intended to diagnose or treat anything; magnesium cures cramps.", "lhm.banned_lexicon"),
    ("Not a diagnosis, but you probably have sleep apnea.", "diagnosis.claim"),
    ("This is not a recommendation (ask about metformin).", "substance.rx_named"),
    ("Not medical advice, but this will improve your sleep.", "lhm.banned_lexicon"),
    ("Not a prescription, but increase your dose tonight.", "dosing.imperative"),
]
NOT_NEGATED = [
    ("This is intended to treat insomnia.", "lhm.banned_lexicon"),
    ("Here is your diagnosis.", "diagnosis.claim"),
]


class TestNegatedDisclaimer:
    """functional/membrane-dsl :: negated-disclaimer excision."""

    @pytest.mark.parametrize("text", DISCLAIMER_PASS)
    def test_disclaimer_alone_passes(self, text):
        v = check(text)
        assert v.passed is True, [x.rule_id for x in v.violations]

    @pytest.mark.parametrize("text,rule", DISCLAIMER_SMUGGLED)
    def test_smuggled_clause_after_disclaimer_fires(self, text, rule):
        assert rule in fired(text)

    @pytest.mark.parametrize("text,rule", NOT_NEGATED)
    def test_non_negated_frame_fires(self, text, rule):
        assert rule in fired(text)
