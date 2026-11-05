from collections import Counter

import pydantic
import pytest

from daypart.simmember.cohort import CURATED_PROFILES, Profile, generate, synthetic_latents
from daypart.simmember.latents import AXES, MIN_PER_LEVEL, Latents


def _lat(**kw):
    base = dict(motivation="low", barrier_type="time", escalation_trigger="none", age_band="18-29",
                literacy="low", engagement_pattern="steady", affect_baseline=0.1234)
    return Latents(**(base | kw))


def test_latents_validation():
    assert _lat().affect_baseline == 0.123
    with pytest.raises(pydantic.ValidationError):
        _lat(motivation="extreme")
    for bad in (1.01, -1.5, float("nan")):
        with pytest.raises(pydantic.ValidationError):
            _lat(affect_baseline=bad)
    with pytest.raises(pydantic.ValidationError):
        Latents(**_lat().model_dump(), name="x")
    with pytest.raises(pydantic.ValidationError):
        _lat().motivation = "high"  # frozen


def test_profile_id_rules():
    lat = _lat()
    assert Profile(member_id="syn-001", cohort="synthetic", latents=lat)
    for mid, coh in [("cur-001", "synthetic"), ("syn-001", "curated"), ("syn-1", "synthetic"),
                     ("abc-001", "synthetic"), ("syn-0001", "synthetic")]:
        with pytest.raises(pydantic.ValidationError):
            Profile(member_id=mid, cohort=coh, latents=lat)
    with pytest.raises(pydantic.ValidationError):
        Profile(member_id="syn-001", cohort="synthetic", latents=lat, email="a@b.c")


def test_curated_table_shape_and_coverage():
    assert len(CURATED_PROFILES) == 25
    assert [p.member_id for p in CURATED_PROFILES] == [f"cur-{i:03d}" for i in range(1, 26)]
    assert all(p.cohort == "curated" for p in CURATED_PROFILES)
    assert len({p.latents for p in CURATED_PROFILES}) == 25
    for axis, levels in AXES.items():
        seen = {getattr(p.latents, axis) for p in CURATED_PROFILES}
        assert seen == set(levels), axis
    assert set(Profile.model_fields) == {"member_id", "cohort", "latents"}


@pytest.mark.parametrize("seed", [0, 1, 7, 123, -5, 2**40])
def test_min_per_level_and_disjoint(seed):
    c = generate(seed, days=1)
    assert len(c.profiles) == 200
    for axis, levels in AXES.items():
        cnt = Counter(getattr(p.latents, axis) for p in c.profiles)
        assert min(cnt[lvl] for lvl in levels) >= MIN_PER_LEVEL, axis
        syn_cnt = Counter(getattr(p.latents, axis) for p in c.training_profiles())
        assert min(syn_cnt[lvl] for lvl in levels) >= MIN_PER_LEVEL, axis
    curated = {p.latents for p in CURATED_PROFILES}
    assert not any(p.latents in curated for p in c.training_profiles())


def test_collision_resample(monkeypatch):
    import daypart.simmember.cohort as cohort_mod

    # Force a collision: make the first synthetic draw equal a curated latent's categoricals
    # and affect; the generator must resample rather than emit a curated latent.
    lats = synthetic_latents(0, 20)
    curated = {p.latents for p in CURATED_PROFILES}
    assert not set(lats) & curated
    target = CURATED_PROFILES[0].latents
    seq = iter([target.affect_baseline] * 3 + [0.5] * 100)
    monkeypatch.setattr(cohort_mod, "_draw_affect", lambda rng: next(seq))
    monkeypatch.setattr(
        cohort_mod, "AXES", {k: (getattr(target, k),) for k in AXES}
    )
    out = cohort_mod.synthetic_latents(0, 1)
    assert out[0] != target and out[0].affect_baseline == 0.5


def test_cohort_order_split_and_determinism():
    a, b = generate(0), generate(0)
    assert a == b
    ids = [p.member_id for p in a.profiles]
    assert ids == [f"cur-{i:03d}" for i in range(1, 26)] + [f"syn-{i:03d}" for i in range(1, 176)]
    assert a.heldout_profiles() == CURATED_PROFILES
    assert all(p.cohort == "synthetic" for p in a.training_profiles())
    assert set(a.trajectories) == set(ids)
    c = generate(1)
    assert [p.latents for p in c.training_profiles()] != [p.latents for p in a.training_profiles()]
    assert c.heldout_profiles() == a.heldout_profiles()


def test_n_synthetic_and_days_params():
    c = generate(3, n_synthetic=12, days=5)
    assert len(c.training_profiles()) == 12
    assert all(len(t.days) == 5 for t in c.trajectories.values())
