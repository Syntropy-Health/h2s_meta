"""A2 — profiles, the fixed Curated-25 held-out table, and cohort generation.

Profiles carry NO identity (I7): an opaque ``member_id``, the cohort label,
and synthetic latents. Curated-25 is a hand-written, seed-independent table
covering every level of every categorical axis; it is the held-out set and
must never reach training (I1, enforced by :mod:`.manifest`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pydantic
from pydantic import ConfigDict, model_validator

from daypart.simmember.dynamics import seed_words
from daypart.simmember.latents import AXES, MIN_PER_LEVEL, Latents
from daypart.simmember.trajectory import Trajectory, generate_trajectory

_ID_RE = re.compile(r"^(cur|syn)-\d{3}$")
_PREFIX_COHORT = {"cur": "curated", "syn": "synthetic"}


class Profile(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    member_id: str
    cohort: Literal["curated", "synthetic"]
    latents: Latents

    @model_validator(mode="after")
    def _id_matches_cohort(self) -> "Profile":
        if not _ID_RE.match(self.member_id):
            raise ValueError(f"member_id {self.member_id!r} must match ^(cur|syn)-\\d{{3}}$")
        if _PREFIX_COHORT[self.member_id[:3]] != self.cohort:
            raise ValueError(f"cohort {self.cohort!r} disagrees with id {self.member_id!r}")
        return self


# --------------------------------------------------------------------------- #
# Curated-25 — hand-drafted, diverse, covers every level of every axis.
# (motivation, barrier_type, escalation_trigger, age_band, literacy, engagement_pattern, affect)
# --------------------------------------------------------------------------- #
_CURATED_ROWS: tuple[tuple[str, str, str, str, str, str, float], ...] = (
    ("low", "time", "none", "18-29", "low", "steady", -0.2),
    ("moderate", "cost", "symptom_flare", "30-44", "medium", "weekend_dip", 0.1),
    ("high", "forgetfulness", "mood_drop", "45-59", "high", "burst", 0.45),
    ("low", "side_effect_concern", "missed_streak", "60-74", "medium", "fading", -0.5),
    ("moderate", "social", "conflicting_info", "75+", "low", "erratic", 0.0),
    ("high", "low_energy", "none", "30-44", "high", "steady", 0.3),
    ("moderate", "time", "mood_drop", "60-74", "high", "fading", -0.15),
    ("low", "forgetfulness", "symptom_flare", "75+", "medium", "weekend_dip", -0.35),
    ("high", "cost", "missed_streak", "18-29", "low", "erratic", 0.6),
    ("moderate", "side_effect_concern", "none", "45-59", "medium", "burst", 0.05),
    ("low", "social", "mood_drop", "30-44", "high", "steady", -0.65),
    ("high", "low_energy", "conflicting_info", "60-74", "low", "weekend_dip", 0.2),
    ("moderate", "forgetfulness", "missed_streak", "18-29", "medium", "fading", -0.05),
    ("low", "time", "conflicting_info", "45-59", "medium", "erratic", -0.4),
    ("high", "side_effect_concern", "symptom_flare", "75+", "high", "steady", 0.15),
    ("moderate", "low_energy", "none", "18-29", "high", "burst", 0.35),
    ("low", "cost", "none", "60-74", "low", "fading", -0.75),
    ("high", "social", "symptom_flare", "45-59", "medium", "weekend_dip", 0.5),
    ("moderate", "time", "missed_streak", "30-44", "low", "burst", -0.1),
    ("low", "low_energy", "mood_drop", "75+", "low", "erratic", -0.55),
    ("high", "forgetfulness", "conflicting_info", "30-44", "medium", "fading", 0.25),
    ("moderate", "cost", "mood_drop", "45-59", "low", "steady", -0.25),
    ("high", "side_effect_concern", "none", "18-29", "medium", "erratic", 0.7),
    ("low", "social", "missed_streak", "60-74", "high", "burst", -0.3),
    ("moderate", "low_energy", "symptom_flare", "75+", "medium", "weekend_dip", 0.0),
)

_AXIS_NAMES = tuple(AXES.keys())

CURATED_PROFILES: tuple[Profile, ...] = tuple(
    Profile(
        member_id=f"cur-{i:03d}",
        cohort="curated",
        latents=Latents(**dict(zip(_AXIS_NAMES, row[:6])), affect_baseline=row[6]),
    )
    for i, row in enumerate(_CURATED_ROWS, start=1)
)


@dataclass(frozen=True)
class Cohort:
    seed: int
    profiles: tuple[Profile, ...]
    trajectories: dict[str, Trajectory]

    def training_profiles(self) -> tuple[Profile, ...]:
        """Synthetic only — the only profiles allowed into training (I1)."""
        return tuple(p for p in self.profiles if p.cohort == "synthetic")

    def heldout_profiles(self) -> tuple[Profile, ...]:
        """Curated only — the held-out set."""
        return tuple(p for p in self.profiles if p.cohort == "curated")


def _draw_affect(rng: np.random.Generator) -> float:
    return round(float(np.clip(rng.normal(0.0, 0.4), -1.0, 1.0)), 3)


def synthetic_latents(seed: int, n: int) -> tuple[Latents, ...]:
    """Seeded synthetic latents, disjoint from Curated-25.

    Coverage: for each axis independently, the first ``MIN_PER_LEVEL * len(levels)``
    draws are a seeded shuffle of each level repeated ``MIN_PER_LEVEL`` times
    (stratified), the rest uniform. So whenever ``n >= MIN_PER_LEVEL * 6`` every
    level of every axis appears ≥ MIN_PER_LEVEL times among the synthetics alone,
    for ANY seed. On a collision with a curated latent the affect baseline is
    resampled — this always breaks equality and keeps the stratification intact.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    rng = np.random.default_rng(seed_words(seed, "latents"))
    columns: dict[str, list[str]] = {}
    for axis, levels in AXES.items():
        strat = [lvl for lvl in levels for _ in range(MIN_PER_LEVEL)]
        strat = [strat[j] for j in rng.permutation(len(strat))][:n]
        rest = [levels[j] for j in rng.integers(0, len(levels), size=max(0, n - len(strat)))]
        columns[axis] = strat + rest
    curated = {p.latents for p in CURATED_PROFILES}
    out: list[Latents] = []
    for i in range(n):
        cats = {axis: columns[axis][i] for axis in _AXIS_NAMES}
        lat = Latents(**cats, affect_baseline=_draw_affect(rng))
        while lat in curated:  # resample on collision (I1 disjointness)
            lat = Latents(**cats, affect_baseline=_draw_affect(rng))
        out.append(lat)
    return tuple(out)


def generate(seed: int, *, n_synthetic: int = 175, days: int = 42) -> Cohort:
    """Curated-25 (seed-independent) + ``n_synthetic`` seeded synthetics, with trajectories."""
    if n_synthetic > 999:
        raise ValueError("n_synthetic must be <= 999 (3-digit ids)")
    synthetic = tuple(
        Profile(member_id=f"syn-{i:03d}", cohort="synthetic", latents=lat)
        for i, lat in enumerate(synthetic_latents(seed, n_synthetic), start=1)
    )
    profiles = CURATED_PROFILES + synthetic
    trajectories = {
        p.member_id: generate_trajectory(p.member_id, p.latents, seed, days) for p in profiles
    }
    return Cohort(seed=seed, profiles=profiles, trajectories=trajectories)


__all__ = ["CURATED_PROFILES", "Cohort", "Profile", "generate", "synthetic_latents"]
