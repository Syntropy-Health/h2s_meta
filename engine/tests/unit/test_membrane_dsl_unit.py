"""Implementer unit tests for daypart.membrane.dsl (CONTRACT slice B, B1/B2/B4)."""

from __future__ import annotations

import pytest

from daypart.membrane import dsl
from daypart.membrane.dsl import RULE_IDS, RULE_VERSION, Verdict, Violation, check, load_rules

# --------------------------------------------------------------------------- #
# obfuscation helpers
# --------------------------------------------------------------------------- #

_CYR = {"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "x": "х", "y": "у"}
_GRK = {"o": "ο", "i": "ι", "a": "α", "k": "κ", "t": "τ"}


def zw(s: str) -> str:
    """Insert zero-width space / joiner between every letter."""
    return "​".join(s) if len(s) < 4 else "".join(c + ("‍" if c.isalpha() else "") for c in s)


def cyr(s: str) -> str:
    return "".join(_CYR.get(c, c) for c in s)


def grk(s: str) -> str:
    return "".join(_GRK.get(c, c) for c in s)


def variants(s: str) -> list[str]:
    return [s, s.upper(), zw(s), cyr(s), grk(s), cyr(s).upper()]


def fired(text: str, **kw) -> set[str]:
    return {v.rule_id for v in check(text, **kw).violations}


# --------------------------------------------------------------------------- #
# B1 surface
# --------------------------------------------------------------------------- #


def test_version_and_ids():
    assert RULE_VERSION == "boundary-v1"
    assert RULE_IDS == {
        "dosing.unit_quantity",
        "dosing.imperative",
        "diagnosis.claim",
        "treatment.claim",
        "substance.rx_named",
        "lhm.banned_lexicon",
        "lhm.imperative_dose",
        "crisis.missing_handoff",
    }
    pack = load_rules()
    assert pack.version == "boundary-v1"
    assert pack.rule_ids == RULE_IDS


@pytest.mark.parametrize("bad", ["boundary-v0", "boundary-v2", "../boundary-v1", "", "nope"])
def test_unknown_version(bad):
    with pytest.raises(FileNotFoundError):
        load_rules(bad)


def test_pack_lists_contain_contract_minimums():
    pack = load_rules()
    for c in [
        "diabetes", "prediabetes", "hypertension", "depression", "anxiety disorder", "anemia",
        "deficiency", "insomnia", "sleep apnea", "hypothyroidism", "infection", "disorder",
        "disease", "syndrome", "cancer", "arrhythmia",
    ]:
        assert c in pack.conditions
    for s in [
        "metformin", "rapamycin", "sirolimus", "semaglutide", "ozempic", "wegovy", "tirzepatide",
        "mounjaro", "insulin", "statin", "atorvastatin", "rosuvastatin", "warfarin", "testosterone",
        "hgh", "sertraline", "fluoxetine", "adderall", "modafinil", "levothyroxine", "prednisone",
        "antibiotic",
    ]:
        assert s in pack.rx_substances


def test_banned_lexicon_is_imported_not_copied():
    import pathlib

    from syntropy.lhm.metrics.claims import BANNED_LEXICON

    yaml_text = (pathlib.Path(dsl.__file__).parent / "rules" / "boundary-v1.yaml").read_text()
    assert "efficacy" not in yaml_text  # stems come from LHM at runtime
    for stem in BANNED_LEXICON:
        assert "lhm.banned_lexicon" in fired(f"this word {stem}s here")


def test_verdict_shape_and_determinism():
    v = check("You should take 400 mg of magnesium.")
    assert isinstance(v, Verdict) and v.rule_version == "boundary-v1"
    assert v.passed is False and all(isinstance(x, Violation) for x in v.violations)
    assert all(x.rule_id in RULE_IDS for x in v.violations)
    assert check("You should take 400 mg of magnesium.") == v
    assert check("") == Verdict(True, (), "boundary-v1")


def test_d10_pair():
    assert check("Try setting your evening reminder 15 minutes earlier tonight.").passed is True
    v = check("You should take 400 mg of magnesium.")
    assert v.passed is False
    assert "dosing.unit_quantity" in {x.rule_id for x in v.violations}


def test_match_is_from_normalized_text():
    v = check("TAKE 400​MG")
    assert Violation("dosing.unit_quantity", "400mg") in v.violations


# --------------------------------------------------------------------------- #
# Violations — every rule, with obfuscations
# --------------------------------------------------------------------------- #

VIOLATIONS: dict[str, list[str]] = {
    "dosing.unit_quantity": [
        "You logged 500 mg yesterday.",
        "Magnesium 400mg before bed works for many people.",
        "Try four hundred milligrams of magnesium.",
        "A couple of capsules with dinner is typical.",
        "Have 2.5 g of creatine with breakfast.",
        "Use 1,000 IU of vitamin D.",
        "Ten drops under the tongue.",
        "One scoop in your shake.",
        "Half a tablet at night.",
        "Two extra softgels this week.",
        "5 ml of the syrup.",
        "200 mcg of B12.",
        "Fifty micrograms is a common amount.",
        "Twenty-five units at dinner.",
        "Take 2 pills after lunch.",
        "1/2 tablet in the evening.",
        "3 international units",
        "250 µg",
    ],
    "dosing.imperative": [
        "Increase your dose this week.",
        "Did you take your meds? Take your dose now.",
        "Skip your medication tonight.",
        "Stop taking it if you feel off.",
        "Double up tomorrow to catch up.",
        "Take an extra one if you missed yesterday.",
        "Take more if it feels easy.",
        "Halve the dosage for a few days.",
        "Lower your meds while traveling.",
        "Adjust your prescription before the trip.",
        "Start taking it in the mornings instead.",
        "Change your supplements around.",
        "Take less on weekends.",
        "Reduce the pills on rest days.",
    ],
    "diagnosis.claim": [
        "You probably have prediabetes.",
        "You might have a vitamin D deficiency.",
        "You are developing insulin resistance.",
        "You're suffering from insomnia.",
        "You likely have mild sleep apnea.",
        "This is a sign of hypothyroidism.",
        "That sounds like a symptom of anemia.",
        "It could be a sign of something serious.",
        "You are prediabetic.",
        "You're hypertensive based on these numbers.",
        "You are deficient in iron.",
        "Your logs point to a diagnosis.",
        "Let me diagnose what's going on.",
        "These are classic symptoms of depression.",
        "This indicates symptoms of stress.",
        "That looks like a sign of stress.",
        "It means a symptom of fatigue.",
        "It sounds like signs of burnout.",
        "That could be a symptom of poor sleep.",
    ],
    "treatment.claim": [
        "Magnesium will fix your sleep.",
        "This routine can heal your gut.",
        "Fasting is a remedy for fatigue.",
        "Walking heals the body.",
        "Cold showers could fix your energy crash.",
        "Turmeric is a natural therapy for joint pain.",
        "Breathing exercises are a cure for stress.",
        "This stack reverses aging.",
        "We can fix that by moving the reminder.",
        "This will fix your schedule problems for good.",
        "Rest days to heal faster.",
    ],
    "substance.rx_named": [
        "Metformin is popular for longevity.",
        "Have you asked about rapamycin?",
        "Ozempic users often skip meals.",
        "Your statins log is up to date.",
        "Insulin timing matters.",
        "Some people use HGH.",
        "antibiotics can upset your stomach",
        "wegovy",
        "Talk about tirzepatide with them.",
    ],
    "lhm.banned_lexicon": [
        "This will improve your sleep.",
        "Great for treating fatigue.",
        "Clinically shown.",
        "It helps prevent crashes.",
        "Strong efficacy.",
        "Treat yourself tonight!",
        "Reverse the trend.",
    ],
    "lhm.imperative_dose": [
        "Take 500 mg with food.",
        "Start at 200mg and see.",
        "Stopping the 5 g scoop is fine.",
        "Keep taking your magnesium, around 400 mg.",
        "Take four hundred mg tonight.",
        "Started 10 ml yesterday.",
    ],
}


@pytest.mark.parametrize(
    "rule_id,text",
    [(rid, t) for rid, ts in VIOLATIONS.items() for t in ts],
)
def test_violation_fires_with_obfuscations(rule_id, text):
    for variant in variants(text):
        assert rule_id in fired(variant), (rule_id, variant)


def test_combining_marks_stripped():
    assert "substance.rx_named" in fired("ḿetförmin")


def test_negated_disclaimer_excised_but_rest_scanned():
    ok = "These notes are not intended to diagnose, treat, cure, or prevent any disease."
    assert check(ok).passed, check(ok)
    assert check(ok.upper()).passed
    assert check("This is not intended to diagnose, treat, cure, or prevent any disease.").passed
    assert check("THIS IS NOT INTENDED TO DIAGNOSE, TREAT, CURE, OR PREVENT ANY DISEASE.").passed
    assert check("It's not meant to diagnose or treat disease.").passed
    assert check("This is not medical advice — your clinician decides.").passed
    assert check("Not a diagnosis, just an observation from your logs.").passed
    v = fired("This is not medical advice — take 300mg tonight.")
    assert {"dosing.unit_quantity", "lhm.imperative_dose"} <= v
    v = fired("Not intended to diagnose, treat, cure, or prevent — magnesium will cure insomnia.")
    assert "treatment.claim" in v
    v = fired("Not a diagnosis, but you probably have prediabetes.")
    assert "diagnosis.claim" in v


# --------------------------------------------------------------------------- #
# crisis
# --------------------------------------------------------------------------- #


def test_crisis_missing_handoff():
    msg = "I'm really sorry you're going through this. I'm here with you."
    assert check(msg).passed
    assert fired(msg, crisis_context=True) == {"crisis.missing_handoff"}
    assert fired("", crisis_context=True) == {"crisis.missing_handoff"}
    assert check("", crisis_context=False).passed
    for ok in [
        "Please call or text 988 right now.",
        "Please contact emergency services.",
        "The crisis line is open 24/7.",
        "You can reach the Crisis Text Line by texting HOME to 741741.",
        "If you're in danger, call 911.",
        "Please call your local emergency number.",
        "Please c​all 9​11.",
        "CALL 988",
    ]:
        assert check(ok, crisis_context=True).passed, ok
    assert not check("Your room number is 1988.", crisis_context=True).passed


# --------------------------------------------------------------------------- #
# Compliant adherence copy — must pass (FP budget)
# --------------------------------------------------------------------------- #

COMPLIANT = [
    "Try setting your evening reminder 15 minutes earlier tonight.",
    "Nice work logging three days in a row — want to keep the streak going?",
    "What got in the way yesterday? We can plan around it together.",
    "Would a reminder at 8pm fit your evening better?",
    "Your clinician is the right person for questions about your protocol.",
    "It's okay to miss a day. Want to pick it back up tomorrow morning?",
    "You have a busy week — want me to shift the reminder to 7am?",
    "Five days logged this week. That's real momentum.",
    "Missing one day doesn't erase the progress you've made.",
    "Would 9:30pm work better than 8pm for your check-in?",
    "Let's try a 10-minute window after breakfast.",
    "Two reminders a day might feel like a lot. Want to drop to one?",
    "You logged every morning this week — nice consistency.",
    "That's a sign of real commitment.",
    "Showing up after a rough week is a sign of resilience.",
    "A couple of days off is normal. Ready to restart today?",
    "Twenty minutes before bed could be a good slot for your routine.",
    "Want to pair your routine with brushing your teeth?",
    "Your streak is at twelve days. Keep it going!",
    "It sounds like mornings are hectic. Would evenings be easier?",
    "We can move the reminder to after lunch if that suits you.",
    "Did you take your meds this morning?",
    "Do you take your supplements with breakfast or dinner?",
    "Let's fix the schedule so it matches your shift work.",
    "Would it help to keep the bottle next to your coffee maker?",
    "Questions about your protocol are best for your care team.",
    "I can share your log with your clinician before the appointment.",
    "You've checked in 4 out of 7 days. Want to aim for 5 next week?",
    "Half the battle is remembering. A sticky note might help.",
    "Take your time — there's no rush to respond.",
    "How did the 6:45am reminder feel this week?",
    "Traveling this weekend? Want to pause reminders for 3 days?",
    "One small step: set out your things the night before.",
    "Your next check-in is Thursday at 10am.",
    "Thanks for being honest about the missed days. That helps us plan.",
    "Want a gentle nudge at 8:15 instead of 8:00?",
    "You have three reminders scheduled for tomorrow.",
    "It's been two weeks since you started — how's it going?",
    "Ninety days is a big milestone. Proud of the effort!",
    "Would a weekly summary every Sunday be useful?",
    "Let's find a routine that fits your evenings.",
    "If something feels off, your clinician is the right person to ask.",
    "Bring this week's log to your appointment on Friday.",
    "Your care team can answer questions about side effects.",
    "Should I move tomorrow's reminder to 7:30?",
    "Want me to set up two pill reminders, morning and evening?",
    "Let's keep it simple: one reminder after dinner.",
    "You checked in 30 days this quarter. That's something.",
    "Busy day? A 2-minute check-in still counts.",
    "What's one thing that would make tomorrow easier?",
    "You hit your goal six days out of seven.",
    "Would a reminder on your phone or your watch work better?",
    "Mornings at 6am are early — want to try 6:30?",
    "No pressure. Pick it back up whenever you're ready.",
    "Weekends are tricky for a lot of people. Want a weekend-only reminder?",
    "Keeping things by the toothbrush is a common trick.",
    "Nice! Eight days in a row now.",
    "Want to celebrate the streak with something small you enjoy?",
    "Your logs show evenings work best for you.",
    "Let's plan for the trip: pack everything the night before.",
    "Shift work makes timing hard. Want reminders tied to your shift start?",
    "It's normal for routines to wobble during the holidays.",
    "Do you want a reminder 30 minutes after you wake up?",
    "You're doing great showing up, even on hard days.",
    "Would texting a friend for accountability help?",
    "A one-week reset could be a nice fresh start.",
    "Your clinician mentioned checking in monthly — want a calendar reminder?",
    "Missed yesterday? No worries, today is a new day.",
    "Want to log it now or later tonight?",
    "Setting a phone alarm at 9pm might help.",
    "Fourteen days logged — that's two full weeks!",
    "Let's look at what worked last month.",
    "You have a lot on your plate. Small steps count.",
    "It sounds like the reminder came at a bad time. Let's move it.",
    "Want to try stacking it with your morning coffee?",
    "I noticed you checked in late three nights this week. Is the timing off?",
    "We can try a different time for a few days and see.",
    "How about a quick check-in at lunch?",
    "Your streak reset, but your total is still 40 days. Nice.",
    "Is there anything getting in the way that I can help plan around?",
    "Would it help to set a reminder on weekends only?",
    "Ask your pharmacist or clinician about anything protocol-related.",
    "It's a good idea to keep your care team in the loop.",
    "Want me to send a summary to share at your next visit?",
    "You've logged 100 days total. That's a lot of showing up!",
    "Let's aim for a 3-day streak to start.",
    "Hydration check: a glass of water with breakfast?",
    "Eight glasses of water is a nice goal for your routine log.",
    "Your 10,000-step goal is already logged today.",
    "You slept about 7 hours last night according to your log.",
    "A 5-minute wind-down might make evenings smoother.",
    "Reminders at 7am and 7pm — does that still work?",
    "I'll check in again in two days.",
    "How did the new spot for your organizer work out?",
    "You picked it back up after a break. That matters.",
    "Do you want fewer notifications this week?",
    "This is not medical advice — your clinician decides on your protocol.",
    "Life got busy, and that's okay. Ready when you are.",
    "Want to set a recurring reminder for refills a week ahead?",
    "Don't forget to pick up your prescription from the pharmacy on Monday.",
    "You've done the evening check-in 11 times this month.",
    "Would a reminder 1 hour before bed fit better?",
    "Mornings at 5:45 sound tough. Want to try a later time?",
    "Keep a log for one week and we'll review together.",
    "Want to change your medication reminder to 8:30pm?",
    "Let's skip the supplement reminder on your travel days.",
    "Thirty seconds to log is all it takes.",
    "That's a sign of how far you've come.",
    "Your routine has held for four weeks now.",
    "Want to try a different reminder sound?",
    "You set a reminder for 6pm, but you've been logging at 9pm. Should we switch?",
    "One missed day out of ten is still a strong run.",
    "If you're unsure about anything protocol-related, check with your clinician first.",
    "Want a reminder to refill before you run out?",
    "Can I ask what made yesterday hard?",
    "Your clinician can help with any questions about timing or amounts.",
    "Great job sticking with it for a whole month.",
    "Want to try a buddy system for the weekends?",
]


def test_compliant_corpus_false_positive_rate():
    assert len(COMPLIANT) >= 100
    fps = [(t, check(t).violations) for t in COMPLIANT if not check(t).passed]
    assert fps == []


def test_scan_stays_fast_on_adversarial_input():
    import time

    for s in ["one " * 5000, "take " + "a" * 20000, "you have " * 3000, "x" * 50000]:
        t0 = time.perf_counter()
        check(s)
        assert time.perf_counter() - t0 < 1.0
