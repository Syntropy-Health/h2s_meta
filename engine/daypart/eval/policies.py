"""E1.3 — the fixed baseline policies for the ``env_baseline`` arm."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from daypart.simmember.dynamics import INTENTS

_MASK64 = (1 << 64) - 1

#: One boundary-safe message per intent. Every entry passes ``dsl.check`` and
#: ``assert_narration_safe`` (unit-tested), so a baseline never triggers a violation.
POLICY_TEXT: Mapping[str, str] = MappingProxyType(
    {
        "none": "Hi, hope your day is going okay.",
        "reminder": "A quick nudge: today's routine is ready whenever you are.",
        "encouragement": "Every day you show up counts. Nice work keeping at it.",
        "barrier_problem_solving": (
            "What got in the way of your routine lately? Let's find one small change that fits your day."
        ),
        "celebrate_streak": "You have kept your routine going several days in a row. That is worth celebrating.",
        "check_in_question": "How are you feeling about your routine this week?",
        "clinician_handoff": "If anything about your routine feels off, your care team is glad to talk it through with you.",
    }
)

_FIXED_INTENT = {"always_none": "none", "always_barrier_problem_solving": "barrier_problem_solving"}


def random_intent_rng(seed: int, scenario_id: str, day: int) -> np.random.Generator:
    """An rng seeded only from ``(seed, scenario_id, day)`` — independent of run order."""
    sid = int.from_bytes(hashlib.sha256(scenario_id.encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng([int(seed) & _MASK64, sid, int(day) & _MASK64])


def choose(policy: str, seed: int, scenario_id: str, day: int) -> tuple[str, str]:
    """``(text, intent)`` for ``policy`` on ``day`` of ``scenario_id`` under ``seed``."""
    if policy in _FIXED_INTENT:
        intent = _FIXED_INTENT[policy]
    elif policy == "random_intent":
        intent = INTENTS[int(random_intent_rng(seed, scenario_id, day).integers(len(INTENTS)))]
    else:
        raise ValueError(f"unknown policy {policy!r}")
    return POLICY_TEXT[intent], intent


__all__ = ["POLICY_TEXT", "choose", "random_intent_rng"]
