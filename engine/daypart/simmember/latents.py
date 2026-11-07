"""A1 — SimMember latent axes and the ``Latents`` model (PVR R1, A&D D12).

Latents are the hidden, synthetic drivers of a simulated member's behaviour.
They carry no identity (I7): six categorical axes plus a continuous affect
baseline. Everything here is pure; sampling lives in :mod:`.cohort`.
"""

from __future__ import annotations

import pydantic
from pydantic import ConfigDict, field_validator

AXES: dict[str, tuple[str, ...]] = {
    "motivation": ("low", "moderate", "high"),
    "barrier_type": ("time", "cost", "forgetfulness", "side_effect_concern", "social", "low_energy"),
    "escalation_trigger": ("none", "symptom_flare", "mood_drop", "missed_streak", "conflicting_info"),
    "age_band": ("18-29", "30-44", "45-59", "60-74", "75+"),
    "literacy": ("low", "medium", "high"),
    "engagement_pattern": ("steady", "weekend_dip", "burst", "fading", "erratic"),
}

#: In a generated 200-member cohort every level of every axis occurs at least this often.
MIN_PER_LEVEL = 2


class Latents(pydantic.BaseModel):
    """Hidden behavioural drivers of one simulated member (no identity fields)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    motivation: str
    barrier_type: str
    escalation_trigger: str
    age_band: str
    literacy: str
    engagement_pattern: str
    affect_baseline: float

    @field_validator(*AXES.keys())
    @classmethod
    def _level_of_axis(cls, value: str, info: pydantic.ValidationInfo) -> str:
        assert info.field_name is not None  # always set for a field validator
        levels = AXES[info.field_name]
        if value not in levels:
            raise ValueError(f"{info.field_name}={value!r} is not one of {levels}")
        return value

    @field_validator("affect_baseline")
    @classmethod
    def _affect_in_range(cls, value: float) -> float:
        if not (-1.0 <= value <= 1.0):  # also rejects NaN
            raise ValueError(f"affect_baseline={value!r} must be within [-1.0, 1.0]")
        return round(float(value), 3)


__all__ = ["AXES", "MIN_PER_LEVEL", "Latents"]
