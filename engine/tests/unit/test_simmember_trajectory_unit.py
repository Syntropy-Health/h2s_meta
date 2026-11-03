from datetime import timedelta

import numpy as np
import pydantic
import pytest

from daypart.simmember.cohort import generate
from daypart.simmember.dynamics import INTENTS, STATES, transition
from daypart.simmember.trajectory import START_DATE, DayRecord, Trajectory, sentiment_for


@pytest.fixture(scope="module")
def cohort0():
    return generate(0)


def _stats(cohort):
    seen, lapse_recover, adh = set(), False, {s: [0, 0] for s in STATES}
    for tr in cohort.trajectories.values():
        lapsed_seen = False
        for d in tr.days:
            seen.add(d.state)
            adh[d.state][0] += d.adhered
            adh[d.state][1] += 1
            lapsed_seen |= d.state == "lapsed"
            lapse_recover |= lapsed_seen and d.state == "recovering"
    return seen, lapse_recover, adh


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 11, 99])
def test_cohort_level_guarantees_hold_robustly(seed):
    seen, lapse_recover, adh = _stats(generate(seed))
    assert seen == set(STATES)
    assert lapse_recover
    eng = adh["engaged"][0] / adh["engaged"][1]
    lap = adh["lapsed"][0] / adh["lapsed"][1]
    assert eng - lap > 0.5  # far from marginal


def test_record_invariants(cohort0):
    for mid, tr in cohort0.trajectories.items():
        assert tr.member_id == mid and len(tr.days) == 42
        assert tr.days[0].state == "engaged"
        for i, d in enumerate(tr.days):
            assert d.day == i and d.date == START_DATE + timedelta(days=i)
            assert d.date.tzinfo is not None
            if d.checkin:
                assert -1.0 <= d.sentiment <= 1.0 and d.note_id == f"note:{mid}:d{i}"
            else:
                assert d.sentiment is None and d.note_id is None
        for prev, nxt in zip(tr.days, tr.days[1:]):
            if prev.state == "lapsed":
                assert nxt.state in ("lapsed", "recovering")


def test_dayrecord_validation():
    ok = dict(day=2, date=START_DATE + timedelta(days=2), state="engaged", adhered=True,
              checkin=True, sentiment=0.2, note_id="note:syn-001:d2")
    DayRecord(**ok)
    for patch in ({"sentiment": None}, {"note_id": None}, {"sentiment": 1.5},
                  {"note_id": "free text note"}, {"note_id": "note:syn-001:d3"},
                  {"date": START_DATE}, {"date": (START_DATE + timedelta(days=2)).replace(tzinfo=None)},
                  {"state": "happy"}, {"checkin": False}, {"extra": 1}):
        with pytest.raises(pydantic.ValidationError):
            DayRecord(**(ok | patch))
    with pytest.raises(pydantic.ValidationError):
        Trajectory(member_id="syn-002", days=(DayRecord(**(ok | {"day": 0, "date": START_DATE,
                                                                  "note_id": "note:syn-001:d0"})),))


def test_sentiment_is_deterministic_and_bounded():
    assert sentiment_for(0.5, "engaged", 0.3) == sentiment_for(0.5, "engaged", 0.3)
    assert sentiment_for(1.0, "engaged", 50.0) == 1.0
    assert sentiment_for(-1.0, "lapsed", -50.0) == -1.0
    assert sentiment_for(0.0, "engaged", 0.0) > sentiment_for(0.0, "lapsed", 0.0)


def test_transition_chain_rules(cohort0):
    lat = cohort0.training_profiles()[0].latents
    rng = np.random.default_rng(0)
    for _ in range(2000):
        nxt, adhered = transition(lat, "lapsed", "none", rng)
        assert nxt in ("lapsed", "recovering") and isinstance(adhered, bool)
        nxt, _ = transition(lat, "recovering", "none", rng)
        assert nxt in ("engaged", "recovering", "lapsed")
    with pytest.raises(ValueError):
        transition(lat, "engaged", "nag", rng)
    with pytest.raises(ValueError):
        transition(lat, "bored", "none", rng)


def test_transition_draws_fixed_and_seeded(cohort0):
    lat = cohort0.training_profiles()[0].latents
    for intent in INTENTS:
        a, b = np.random.default_rng(5), np.random.default_rng(5)
        assert transition(lat, "strained", intent, a) == transition(lat, "strained", intent, b)
        assert a.random() == b.random()


def test_risky_latents_slip_more(cohort0):
    syn = cohort0.training_profiles()
    hi = [p.latents for p in syn if p.latents.motivation == "high" and p.latents.engagement_pattern == "steady"]
    lo = [p.latents for p in syn if p.latents.motivation == "low" and p.latents.engagement_pattern in ("fading", "erratic")]
    assert hi and lo

    def slip_rate(lats):
        rng, n, slips = np.random.default_rng(1), 0, 0
        for lat in lats:
            for _ in range(500):
                nxt, _ = transition(lat, "engaged", "none", rng)
                slips += nxt != "engaged"
                n += 1
        return slips / n

    assert slip_rate(lo) > slip_rate(hi) + 0.1


def _rollout_mean(syn, intent, seed_base):
    out = []
    for k in range(200):
        lat = syn[k % len(syn)].latents
        rng, state, adh = np.random.default_rng(seed_base + k), "engaged", 0
        for _ in range(14):
            state, a = transition(lat, state, intent, rng)
            adh += a
        out.append(adh / 14)
    return float(np.mean(out))


def test_barrier_problem_solving_beats_none_robustly(cohort0):
    syn = cohort0.training_profiles()
    for base in (0, 10_000, 20_000):
        # independent rng streams per policy — not relying on common random numbers
        assert _rollout_mean(syn, "barrier_problem_solving", base) > _rollout_mean(syn, "none", base + 5_000) + 0.1


def test_bps_raises_leave_probability(cohort0):
    syn = cohort0.training_profiles()
    for state, leave in (("strained", "engaged"), ("lapsed", "recovering")):
        rates = {}
        for intent in ("none", "barrier_problem_solving"):
            rng, hits, n = np.random.default_rng(3), 0, 0
            for p in syn:
                for _ in range(40):
                    nxt, _ = transition(p.latents, state, intent, rng)
                    hits += nxt == leave
                    n += 1
            rates[intent] = hits / n
        assert rates["barrier_problem_solving"] > rates["none"] + 0.1
