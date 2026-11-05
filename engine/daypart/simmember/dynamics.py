"""C1 — the shared latent-driven Markov chain (A&D D16; slice A3 semantics).

One function, :func:`transition`, is used both to generate the slice-A
baseline trajectories (``intent="none"``) and to advance the aviary env, so
the two can never drift apart.

Chain over ``STATES`` (see :mod:`.trajectory`)::

    engaged    -> engaged | strained | lapsed
    strained   -> engaged | strained | lapsed
    lapsed     -> lapsed  | recovering            (never straight back to engaged)
    recovering -> engaged | recovering | lapsed

Coupling design: every call consumes EXACTLY two uniforms from ``rng`` (one
for adherence, one for the transition), whatever the path. Outcomes are
ordered best-to-worst along the unit interval, and every intent only raises
the "improve" mass / adherence probability and shrinks the "slip" mass versus
``"none"``. So under common random numbers an intent is never worse than the
baseline — the directional effect is structural, not a statistical accident.
"""

from __future__ import annotations

import numpy as np

from daypart.simmember.latents import Latents

STATES = ("engaged", "strained", "lapsed", "recovering")

INTENTS = (
    "none",
    "reminder",
    "encouragement",
    "barrier_problem_solving",
    "celebrate_streak",
    "check_in_question",
    "clinician_handoff",
)

_MOTIVATION_RISK = {"low": 0.40, "moderate": 0.22, "high": 0.08}
_PATTERN_RISK = {"steady": 0.0, "weekend_dip": 0.05, "burst": 0.07, "fading": 0.18, "erratic": 0.15}
_LITERACY_RISK = {"low": 0.05, "medium": 0.02, "high": 0.0}

#: Baseline P(adhered today | state).
_BASE_ADHERENCE = {"engaged": 0.90, "strained": 0.55, "recovering": 0.62, "lapsed": 0.08}

#: Barrier types that structured problem-solving addresses most directly.
_PS_RESPONSIVE = {"time": 0.12, "cost": 0.12, "side_effect_concern": 0.14, "social": 0.10,
                  "low_energy": 0.10, "forgetfulness": 0.05}


_STREAMS = {"latents": 0, "trajectory": 1, "observation": 2, "scenarios": 3, "env": 4}
_MASK64 = (1 << 64) - 1


def seed_words(seed: int, stream: str, *extra: int) -> list[int]:
    """Non-negative SeedSequence entropy words for ``(seed, stream, *extra)``.

    Negative seeds are folded to their 64-bit two's complement, so every int
    seed is accepted and distinct seeds stay distinct.
    """
    return [int(seed) & _MASK64, _STREAMS[stream], *(int(e) & _MASK64 for e in extra)]


def member_key(member_id: str) -> tuple[int, int]:
    """``("cur-007") -> (0, 7)``, ``("syn-012") -> (1, 12)`` — the per-member rng key."""
    prefix, _, num = member_id.partition("-")
    return (0 if prefix == "cur" else 1), int(num)


def member_rng(seed: int, stream: str, member_id: str, *extra: int) -> np.random.Generator:
    """The rng derived from ``(seed, member index[, extra...])`` for one stream."""
    return np.random.default_rng(seed_words(seed, stream, *member_key(member_id), *extra))


def risk(latents: Latents) -> float:
    """Latent slip propensity in [0, 1]: lower motivation / fading / erratic → higher."""
    r = (
        _MOTIVATION_RISK[latents.motivation]
        + _PATTERN_RISK[latents.engagement_pattern]
        + _LITERACY_RISK[latents.literacy]
        + (0.03 if latents.escalation_trigger != "none" else 0.0)
        + max(0.0, -latents.affect_baseline) * 0.06
    )
    return min(1.0, max(0.0, r))


def _intent_effect(latents: Latents, state: str, intent: str) -> tuple[float, float, float]:
    """(improve_bonus, slip_factor, adherence_bonus) for ``intent`` — all non-harmful."""
    if intent == "none":
        return 0.0, 1.0, 0.0
    off_track = state != "engaged"
    if intent == "barrier_problem_solving":
        fit = _PS_RESPONSIVE[latents.barrier_type]
        return 0.14 + fit, 0.65, (0.10 + fit if off_track else 0.03)
    if intent == "reminder":
        forgetful = latents.barrier_type == "forgetfulness"
        return (0.06 if forgetful else 0.02), 0.90, (0.18 if forgetful else 0.05)
    if intent == "encouragement":
        low_mot = latents.motivation == "low"
        return (0.10 if low_mot else 0.05), 0.90, (0.06 if off_track else 0.02)
    if intent == "celebrate_streak":
        return (0.0, 0.75, 0.04) if state == "engaged" else (0.02, 1.0, 0.0)
    if intent == "check_in_question":
        return 0.03, 0.95, 0.02
    if intent == "clinician_handoff":
        if latents.escalation_trigger != "none":
            return 0.10, 0.80, 0.04
        return 0.0, 1.0, 0.0
    raise ValueError(f"unknown intent {intent!r}; expected one of {INTENTS}")


def _clip(p: float) -> float:
    return min(1.0, max(0.0, p))


def transition(
    latents: Latents, state: str, intent: str, rng: np.random.Generator
) -> tuple[str, bool]:
    """Advance one day: returns ``(next_state, adhered_today)``.

    ``adhered_today`` is drawn from the CURRENT state (the day being lived);
    ``next_state`` is tomorrow's state. Unknown ``intent``/``state`` → ``ValueError``.
    """
    if intent not in INTENTS:
        raise ValueError(f"unknown intent {intent!r}; expected one of {INTENTS}")
    if state not in STATES:
        raise ValueError(f"unknown state {state!r}; expected one of {STATES}")

    r = risk(latents)
    improve, slip, adh_bonus = _intent_effect(latents, state, intent)
    u_adh, u_next = rng.random(2)

    p_adh = _clip(_BASE_ADHERENCE[state] - (0.10 * r if state == "engaged" else 0.05 * r) + adh_bonus)
    adhered = bool(u_adh < min(p_adh, 0.98))

    # Outcomes laid out best → worst on [0, 1): [improve | stay | slip].
    if state == "engaged":
        p_up = 0.0
        p_down_strained = (0.05 + 0.35 * r) * slip
        p_down_lapsed = (0.01 + 0.06 * r) * slip
        if u_next < 1.0 - p_down_strained - p_down_lapsed:
            nxt = "engaged"
        elif u_next < 1.0 - p_down_lapsed:
            nxt = "strained"
        else:
            nxt = "lapsed"
    elif state == "strained":
        p_up = _clip(0.25 * (1.0 - r) + 0.05 + improve)
        p_down = (0.08 + 0.30 * r) * slip
        p_down = min(p_down, 1.0 - p_up)
        if u_next < p_up:
            nxt = "engaged"
        elif u_next < 1.0 - p_down:
            nxt = "strained"
        else:
            nxt = "lapsed"
    elif state == "lapsed":
        p_up = _clip(0.10 + 0.15 * (1.0 - r) + improve)
        nxt = "recovering" if u_next < p_up else "lapsed"
    else:  # recovering
        p_up = _clip(0.25 + 0.20 * (1.0 - r) + improve)
        p_down = (0.05 + 0.20 * r) * slip
        p_down = min(p_down, 1.0 - p_up)
        if u_next < p_up:
            nxt = "engaged"
        elif u_next < 1.0 - p_down:
            nxt = "recovering"
        else:
            nxt = "lapsed"
    return nxt, adhered


__all__ = ["INTENTS", "STATES", "member_key", "member_rng", "risk", "seed_words", "transition"]
