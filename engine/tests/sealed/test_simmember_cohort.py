"""Sealed challenge suite — slice A: latents, profiles, cohort, trajectories (R1).

Spec: docs/daypart/contracts/CONTRACT-slice-A-simmember.md sections A1-A3.

Ambiguities resolved (questions for the principal):
- "affect_baseline rounded to 3 decimals": tested on GENERATED profiles only;
  the constructor is not required to round arbitrary input.
- "lower motivation / fading / erratic make strained/lapsed more likely": the
  exact probabilities are implementation-defined, so only the hard state-machine
  constraints and the cohort-wide properties stated in A3 are asserted.
- The "recovering moves to engaged, recovering or lapsed" constraint is asserted
  as a hard rule (recovering never goes directly to strained).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.category("functional/cohort")

SPEC_AXES = {
    "motivation": ("low", "moderate", "high"),
    "barrier_type": ("time", "cost", "forgetfulness", "side_effect_concern", "social", "low_energy"),
    "escalation_trigger": ("none", "symptom_flare", "mood_drop", "missed_streak", "conflicting_info"),
    "age_band": ("18-29", "30-44", "45-59", "60-74", "75+"),
    "literacy": ("low", "medium", "high"),
    "engagement_pattern": ("steady", "weekend_dip", "burst", "fading", "erratic"),
}

VALID_LATENTS = dict(
    motivation="moderate",
    barrier_type="time",
    escalation_trigger="none",
    age_band="30-44",
    literacy="medium",
    engagement_pattern="steady",
    affect_baseline=0.25,
)


# --------------------------------------------------------------------------- #
# A1 — Latents schema
# --------------------------------------------------------------------------- #


class TestLatentsSchema:
    pytestmark = pytest.mark.category("functional/schema")

    def test_axes_match_spec(self):
        from daypart.simmember.latents import AXES

        assert {k: tuple(v) for k, v in AXES.items()} == SPEC_AXES

    def test_min_per_level_is_two(self):
        from daypart.simmember.latents import MIN_PER_LEVEL

        assert MIN_PER_LEVEL == 2

    def test_valid_latents_construct(self):
        from daypart.simmember.latents import Latents

        lat = Latents(**VALID_LATENTS)
        assert lat.motivation == "moderate"
        assert lat.affect_baseline == pytest.approx(0.25)

    @pytest.mark.parametrize("bound", [-1.0, 1.0, 0.0])
    def test_affect_baseline_bounds_inclusive(self, bound):
        from daypart.simmember.latents import Latents

        Latents(**{**VALID_LATENTS, "affect_baseline": bound})

    @pytest.mark.parametrize("bad", [1.001, -1.001, 2.0, -5.0])
    def test_affect_baseline_out_of_range_rejected(self, bad):
        from daypart.simmember.latents import Latents

        with pytest.raises(ValidationError):
            Latents(**{**VALID_LATENTS, "affect_baseline": bad})

    @pytest.mark.parametrize("axis", list(SPEC_AXES))
    def test_unknown_axis_level_rejected(self, axis):
        from daypart.simmember.latents import Latents

        with pytest.raises(ValidationError):
            Latents(**{**VALID_LATENTS, axis: "not-a-level"})

    @pytest.mark.parametrize("axis", list(SPEC_AXES))
    def test_every_axis_level_accepted(self, axis):
        from daypart.simmember.latents import Latents

        for level in SPEC_AXES[axis]:
            assert getattr(Latents(**{**VALID_LATENTS, axis: level}), axis) == level

    def test_extra_field_forbidden(self):
        from daypart.simmember.latents import Latents

        with pytest.raises(ValidationError):
            Latents(**VALID_LATENTS, name="Alice")

    def test_missing_field_rejected(self):
        from daypart.simmember.latents import Latents

        kw = dict(VALID_LATENTS)
        kw.pop("literacy")
        with pytest.raises(ValidationError):
            Latents(**kw)

    def test_latents_frozen(self):
        from daypart.simmember.latents import Latents

        lat = Latents(**VALID_LATENTS)
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            lat.motivation = "high"


# --------------------------------------------------------------------------- #
# A2 — Profile schema
# --------------------------------------------------------------------------- #


class TestProfileSchema:
    pytestmark = pytest.mark.category("functional/schema")

    def _lat(self):
        from daypart.simmember.latents import Latents

        return Latents(**VALID_LATENTS)

    @pytest.mark.parametrize("mid,cohort", [("cur-001", "curated"), ("syn-042", "synthetic")])
    def test_valid_profile(self, mid, cohort):
        from daypart.simmember.cohort import Profile

        p = Profile(member_id=mid, cohort=cohort, latents=self._lat())
        assert p.member_id == mid and p.cohort == cohort

    @pytest.mark.parametrize("mid,cohort", [("cur-001", "synthetic"), ("syn-001", "curated")])
    def test_cohort_prefix_disagreement_rejected(self, mid, cohort):
        from daypart.simmember.cohort import Profile

        with pytest.raises(ValidationError):
            Profile(member_id=mid, cohort=cohort, latents=self._lat())

    @pytest.mark.parametrize(
        "mid", ["cur-1", "cur-0001", "syn-01", "abc-001", "CUR-001", "cur_001", "", "alice@example.com"]
    )
    def test_bad_member_id_rejected(self, mid):
        from daypart.simmember.cohort import Profile

        cohort = "synthetic" if mid.startswith("syn") else "curated"
        with pytest.raises(ValidationError):
            Profile(member_id=mid, cohort=cohort, latents=self._lat())

    def test_bad_cohort_literal_rejected(self):
        from daypart.simmember.cohort import Profile

        with pytest.raises(ValidationError):
            Profile(member_id="syn-001", cohort="other", latents=self._lat())

    @pytest.mark.parametrize("field", ["name", "email", "bio", "first_name", "phone"])
    def test_identity_fields_forbidden(self, field):
        from daypart.simmember.cohort import Profile

        with pytest.raises(ValidationError):
            Profile(member_id="syn-001", cohort="synthetic", latents=self._lat(), **{field: "x"})

    def test_profile_has_only_contract_fields(self):
        from daypart.simmember.cohort import Profile

        assert set(Profile.model_fields) == {"member_id", "cohort", "latents"}

    def test_latents_has_only_contract_fields(self):
        from daypart.simmember.latents import Latents

        assert set(Latents.model_fields) == set(SPEC_AXES) | {"affect_baseline"}


# --------------------------------------------------------------------------- #
# A2 — Curated table
# --------------------------------------------------------------------------- #


class TestCuratedProfiles:
    pytestmark = pytest.mark.category("functional/curated")

    def test_exactly_25_in_id_order(self):
        from daypart.simmember.cohort import CURATED_PROFILES

        assert len(CURATED_PROFILES) == 25
        assert [p.member_id for p in CURATED_PROFILES] == [f"cur-{i:03d}" for i in range(1, 26)]
        assert all(p.cohort == "curated" for p in CURATED_PROFILES)

    def test_curated_covers_every_axis_level(self):
        from daypart.simmember.cohort import CURATED_PROFILES

        for axis, levels in SPEC_AXES.items():
            seen = Counter(getattr(p.latents, axis) for p in CURATED_PROFILES)
            for level in levels:
                assert seen[level] >= 1, f"curated table missing {axis}={level}"

    def test_curated_affect_baseline_in_range(self):
        from daypart.simmember.cohort import CURATED_PROFILES

        for p in CURATED_PROFILES:
            assert -1.0 <= p.latents.affect_baseline <= 1.0


# --------------------------------------------------------------------------- #
# A2 — Cohort generation
# --------------------------------------------------------------------------- #


class TestCohortGeneration:
    pytestmark = pytest.mark.category("functional/cohort")

    def test_counts(self, cohort0):
        assert len(cohort0.profiles) == 200
        assert sum(p.cohort == "curated" for p in cohort0.profiles) == 25
        assert sum(p.cohort == "synthetic" for p in cohort0.profiles) == 175

    def test_seed_recorded(self, cohort0, cohort1):
        assert cohort0.seed == 0
        assert cohort1.seed == 1

    def test_ids_and_ordering(self, cohort0):
        ids = [p.member_id for p in cohort0.profiles]
        expected = [f"cur-{i:03d}" for i in range(1, 26)] + [f"syn-{i:03d}" for i in range(1, 176)]
        assert ids == expected
        for p in cohort0.profiles:
            assert p.cohort == ("curated" if p.member_id.startswith("cur-") else "synthetic")

    def test_curated_first_equals_table(self, cohort0):
        from daypart.simmember.cohort import CURATED_PROFILES

        assert tuple(cohort0.profiles[:25]) == tuple(CURATED_PROFILES)

    def test_training_and_heldout_split(self, cohort0):
        train = cohort0.training_profiles()
        held = cohort0.heldout_profiles()
        assert len(train) == 175 and len(held) == 25
        assert all(p.cohort == "synthetic" and p.member_id.startswith("syn-") for p in train)
        assert all(p.cohort == "curated" and p.member_id.startswith("cur-") for p in held)
        assert tuple(train) == tuple(cohort0.profiles[25:])
        assert tuple(held) == tuple(cohort0.profiles[:25])

    def test_min_per_level_across_200(self, cohort0):
        from daypart.simmember.latents import MIN_PER_LEVEL

        for axis, levels in SPEC_AXES.items():
            seen = Counter(getattr(p.latents, axis) for p in cohort0.profiles)
            for level in levels:
                assert seen[level] >= MIN_PER_LEVEL, f"{axis}={level} occurs {seen[level]}x"

    @pytest.mark.parametrize("seed", [1, 7])
    def test_min_per_level_other_seeds(self, seed, cohort1):
        from daypart.simmember.cohort import generate
        from daypart.simmember.latents import MIN_PER_LEVEL

        c = cohort1 if seed == 1 else generate(seed)
        for axis, levels in SPEC_AXES.items():
            seen = Counter(getattr(p.latents, axis) for p in c.profiles)
            for level in levels:
                assert seen[level] >= MIN_PER_LEVEL

    @pytest.mark.parametrize("which", ["cohort0", "cohort1"])
    def test_curated_latents_disjoint_from_synthetic(self, which, request):
        c = request.getfixturevalue(which)
        curated = {p.latents for p in c.heldout_profiles()}
        for p in c.training_profiles():
            assert p.latents not in curated, f"{p.member_id} duplicates curated latents"

    def test_generated_affect_baseline_rounded_and_in_range(self, cohort0):
        for p in cohort0.profiles:
            ab = p.latents.affect_baseline
            assert -1.0 <= ab <= 1.0
            assert round(ab, 3) == ab

    def test_curated_identical_across_seeds(self, cohort0, cohort1):
        assert cohort0.heldout_profiles() == cohort1.heldout_profiles()

    def test_synthetic_differs_across_seeds(self, cohort0, cohort1):
        a = [p.latents for p in cohort0.training_profiles()]
        b = [p.latents for p in cohort1.training_profiles()]
        assert a != b

    def test_generate_twice_equal(self, cohort0, cohort0_again):
        assert cohort0.profiles == cohort0_again.profiles
        assert cohort0.trajectories == cohort0_again.trajectories
        assert cohort0 == cohort0_again

    def test_write_byte_identical(self, cohort0, cohort0_again, tmp_path):
        from daypart.simmember.manifest import write

        d1, d2 = tmp_path / "a", tmp_path / "b"
        d1.mkdir()
        d2.mkdir()
        write(cohort0, d1)
        write(cohort0_again, d2)
        for name in ("profiles.jsonl", "trajectories.jsonl", "manifest.json"):
            assert (d1 / name).read_bytes() == (d2 / name).read_bytes(), name

    def test_small_cohort_parameters(self):
        from daypart.simmember.cohort import generate

        c = generate(3, n_synthetic=10, days=5)
        assert len(c.profiles) == 35
        assert [p.member_id for p in c.training_profiles()] == [f"syn-{i:03d}" for i in range(1, 11)]
        assert set(c.trajectories) == {p.member_id for p in c.profiles}
        for t in c.trajectories.values():
            assert len(t.days) == 5
        curated = {p.latents for p in c.heldout_profiles()}
        assert not any(p.latents in curated for p in c.training_profiles())


# --------------------------------------------------------------------------- #
# A3 — Trajectories
# --------------------------------------------------------------------------- #

STATES = ("engaged", "strained", "lapsed", "recovering")


class TestTrajectoryStructure:
    pytestmark = pytest.mark.category("functional/trajectory")

    def test_constants(self):
        from daypart.simmember.trajectory import START_DATE, STATES as S

        assert tuple(S) == STATES
        assert START_DATE == datetime(2026, 9, 1, tzinfo=timezone.utc)
        assert START_DATE.tzinfo is not None

    def test_one_trajectory_per_profile(self, cohort0):
        assert set(cohort0.trajectories) == {p.member_id for p in cohort0.profiles}
        assert len(cohort0.trajectories) == 200
        for mid, t in cohort0.trajectories.items():
            assert t.member_id == mid

    def test_days_contiguous_default_42(self, cohort0):
        for t in cohort0.trajectories.values():
            assert [d.day for d in t.days] == list(range(42))

    def test_dates_tz_aware_and_aligned(self, cohort0):
        from daypart.simmember.trajectory import START_DATE

        for t in cohort0.trajectories.values():
            for d in t.days:
                assert d.date.tzinfo is not None and d.date.utcoffset() is not None
                assert d.date == START_DATE + timedelta(days=d.day)

    def test_day0_engaged(self, cohort0):
        for t in cohort0.trajectories.values():
            assert t.days[0].state == "engaged"

    def test_states_valid(self, cohort0):
        for t in cohort0.trajectories.values():
            for d in t.days:
                assert d.state in STATES
                assert isinstance(d.adhered, bool) and isinstance(d.checkin, bool)

    def test_sentiment_and_note_iff_checkin(self, cohort0):
        for t in cohort0.trajectories.values():
            for d in t.days:
                if d.checkin:
                    assert d.sentiment is not None
                    assert -1.0 <= d.sentiment <= 1.0
                    assert d.note_id == f"note:{t.member_id}:d{d.day}"
                else:
                    assert d.sentiment is None
                    assert d.note_id is None

    def test_some_checkins_exist(self, cohort0):
        assert any(d.checkin for t in cohort0.trajectories.values() for d in t.days)

    def test_lapsed_only_to_lapsed_or_recovering(self, cohort0):
        for t in cohort0.trajectories.values():
            for a, b in zip(t.days, t.days[1:]):
                if a.state == "lapsed":
                    assert b.state in ("lapsed", "recovering"), (t.member_id, a.day, b.state)

    def test_recovering_never_to_strained(self, cohort0):
        for t in cohort0.trajectories.values():
            for a, b in zip(t.days, t.days[1:]):
                if a.state == "recovering":
                    assert b.state in ("engaged", "recovering", "lapsed"), (t.member_id, a.day, b.state)

    def test_state_machine_holds_other_seed(self, cohort1):
        for t in cohort1.trajectories.values():
            assert t.days[0].state == "engaged"
            for a, b in zip(t.days, t.days[1:]):
                if a.state == "lapsed":
                    assert b.state in ("lapsed", "recovering")
                if a.state == "recovering":
                    assert b.state in ("engaged", "recovering", "lapsed")

    def test_no_free_text_in_trajectories(self, cohort0):
        import re

        tok = re.compile(r"^note:(cur|syn)-\d{3}:d\d+$")
        for t in cohort0.trajectories.values():
            for d in t.days:
                if d.note_id is not None:
                    assert tok.match(d.note_id)
            assert set(type(t).model_fields) == {"member_id", "days"}

    def test_dayrecord_extra_forbidden(self, cohort0):
        from daypart.simmember.trajectory import DayRecord

        d = next(iter(cohort0.trajectories.values())).days[0]
        with pytest.raises(ValidationError):
            DayRecord(**d.model_dump(), note="free text")


class TestTrajectoryCohortProperties:
    pytestmark = pytest.mark.category("functional/trajectory-dynamics")

    def test_every_state_occurs(self, cohort0):
        seen = {d.state for t in cohort0.trajectories.values() for d in t.days}
        assert seen == set(STATES)

    def test_some_member_lapses_then_recovers(self, cohort0):
        def lapse_then_recover(t):
            lapsed_seen = False
            for d in t.days:
                if d.state == "lapsed":
                    lapsed_seen = True
                elif d.state == "recovering" and lapsed_seen:
                    return True
            return False

        assert any(lapse_then_recover(t) for t in cohort0.trajectories.values())

    def test_engaged_adherence_exceeds_lapsed(self, cohort0):
        eng = [d.adhered for t in cohort0.trajectories.values() for d in t.days if d.state == "engaged"]
        lap = [d.adhered for t in cohort0.trajectories.values() for d in t.days if d.state == "lapsed"]
        assert eng and lap
        assert sum(eng) / len(eng) > sum(lap) / len(lap)
