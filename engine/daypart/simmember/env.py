"""C2/C3 — scenarios and the aviary ``SimMemberEnv`` (A&D D16, R6, I1).

The policy speaks to a simulated member through exactly one tool,
``send_message(text, intent)``. Each honoured call:

1. runs the boundary DSL (``daypart.membrane.dsl.check``, imported lazily) —
   a failing message ends the episode with reward ``-1.0`` (R6);
2. normalises an unknown ``intent`` to ``"none"`` (with a note in the reply);
3. advances one day through the shared :func:`~.dynamics.transition` with an
   rng seeded from ``(scenario.seed, member index, day)``; reward is
   ``1.0`` if the member adhered that day else ``0.0``.

Everything is deterministic: no wall clock, no uuid, no global random state.
"""

from __future__ import annotations

import hashlib
import importlib
from typing import Any, Sequence

import numpy as np
import pydantic
from aviary.core import (
    Environment,
    Frame,
    Message,
    Tool,
    ToolCall,
    ToolRequestMessage,
    ToolResponseMessage,
)
from pydantic import ConfigDict, field_validator, model_validator

from daypart.simmember.cohort import CURATED_PROFILES, Cohort, Profile
from daypart.simmember.dynamics import INTENTS, member_rng, seed_words, transition
from daypart.simmember.manifest import CuratedLeakError, canonical_json

SEND_MESSAGE = "send_message"
_CURATED_LATENTS = frozenset(p.latents for p in CURATED_PROFILES)


def _assert_not_curated(profile: Profile) -> None:
    if (
        profile.cohort == "curated"
        or profile.member_id.startswith("cur-")
        or profile.latents in _CURATED_LATENTS
    ):
        raise CuratedLeakError(
            f"curated member {profile.member_id} may not be used in a training scenario (I1)"
        )


class Scenario(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str
    member: Profile
    seed: int
    horizon: int = 14

    @model_validator(mode="before")
    @classmethod
    def _default_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and "scenario_id" not in data:
            member = data.get("member")
            mid = member.member_id if isinstance(member, Profile) else (member or {}).get("member_id")
            if mid is not None and "seed" in data:
                data = {**data, "scenario_id": f"scn-{data['seed']}-{mid}"}
        return data

    @field_validator("member")
    @classmethod
    def _member_not_curated(cls, member: Profile) -> Profile:
        _assert_not_curated(member)  # CuratedLeakError propagates as-is (not a ValidationError)
        return member

    @model_validator(mode="after")
    def _check(self) -> "Scenario":
        expected = f"scn-{self.seed}-{self.member.member_id}"
        if self.scenario_id != expected:
            raise ValueError(f"scenario_id must be {expected!r}, got {self.scenario_id!r}")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        return self


def make_scenarios(cohort: Cohort, n: int, seed: int) -> list[Scenario]:
    """``n`` distinct synthetic members drawn (seeded) from ``cohort.training_profiles()``."""
    pool = cohort.training_profiles()
    if n < 0 or n > len(pool):
        raise ValueError(f"n={n} must be within [0, {len(pool)}] (synthetic pool size)")
    rng = np.random.default_rng(seed_words(seed, "scenarios"))
    picks = rng.permutation(len(pool))[:n]
    return [Scenario(scenario_id=f"scn-{seed}-{pool[i].member_id}", member=pool[i], seed=seed)
            for i in picks.tolist()]


def scenario_manifest(scenarios: Sequence[Scenario]) -> dict:
    return {
        "scenarios": {
            s.scenario_id: hashlib.sha256(
                canonical_json(s.model_dump(mode="json")).encode("utf-8")
            ).hexdigest()
            for s in scenarios
        },
        "count": len(scenarios),
    }


class EnvState(pydantic.BaseModel):
    day: int
    member_state: str
    adherence: list[bool]
    done: bool
    violations: list[str]


def send_message(text: str, intent: str) -> str:
    """Send one message to the member for today.

    Args:
        text: The message text shown to the member. Must pass the boundary rules.
        intent: The message intent, one of: none, reminder, encouragement,
            barrier_problem_solving, celebrate_streak, check_in_question, clinician_handoff.
    """
    return "queued"


_STATE_PHRASE = {
    "engaged": "on track and engaged",
    "strained": "under some strain",
    "lapsed": "to have lapsed from the routine",
    "recovering": "to be getting back on track",
}


class SimMemberEnv(Environment[EnvState]):
    """One simulated (synthetic-only) member over ``scenario.horizon`` days."""

    def __init__(self, scenario: Scenario) -> None:
        if not isinstance(scenario, Scenario):
            raise TypeError("SimMemberEnv requires a Scenario")
        _assert_not_curated(scenario.member)
        self.scenario = scenario
        self.tools = [Tool.from_function(send_message)]
        self.state = self._initial_state()

    @staticmethod
    def _initial_state() -> EnvState:
        return EnvState(day=0, member_state="engaged", adherence=[], done=False, violations=[])

    @property
    def member_id(self) -> str:
        return self.scenario.member.member_id

    async def get_id(self) -> str:
        return self.scenario.scenario_id

    async def reset(self) -> tuple[list[Message], list[Tool]]:
        self.state = self._initial_state()
        self.tools = [Tool.from_function(send_message)]
        lat = self.scenario.member.latents
        content = (
            f"Simulated member {self.member_id}, day 0 of {self.scenario.horizon}. "
            f"Current status: {_STATE_PHRASE[self.state.member_state]}. "
            f"Motivation: {lat.motivation}. Main barrier: {lat.barrier_type.replace('_', ' ')}. "
            f"Health literacy: {lat.literacy}. Engagement pattern: "
            f"{lat.engagement_pattern.replace('_', ' ')}. "
            f"Send one message per day with {SEND_MESSAGE}(text, intent)."
        )
        return [Message(content=content)], self.tools

    def _reply(self, call: ToolCall, content: str) -> ToolResponseMessage:
        return ToolResponseMessage.from_call(call, content=content)

    async def step(self, action: ToolRequestMessage) -> tuple[list[Message], float, bool, bool]:
        if self.state.done:
            raise RuntimeError("episode is done; call reset() before stepping again")
        if not isinstance(action, ToolRequestMessage) or not action.tool_calls:
            return self.default_no_tool_calls_response

        honoured_pos: int | None = None
        replies: list[str | None] = []
        for pos, call in enumerate(action.tool_calls):
            if call.function.name != SEND_MESSAGE:
                replies.append(f"Invalid tool call: {call.function.name}")
            elif honoured_pos is not None:
                replies.append(
                    "Error: only one send_message call per step is honoured; this one was ignored."
                )
            else:
                honoured_pos = pos
                replies.append(None)

        reward, done, truncated = 0.0, False, False
        if honoured_pos is not None:
            call = action.tool_calls[honoured_pos]
            replies[honoured_pos], reward, done, truncated = self._send(call.function.arguments)
        msgs: list[Message] = [
            self._reply(call, reply or "") for call, reply in zip(action.tool_calls, replies)
        ]
        return msgs, reward, done, truncated

    def _send(self, args: dict[str, Any]) -> tuple[str, float, bool, bool]:
        text = args.get("text")
        if not isinstance(text, str):
            return "Error: send_message requires a string 'text' argument.", 0.0, False, False

        dsl = importlib.import_module("daypart.membrane.dsl")
        verdict = dsl.check(text)
        if not verdict.passed:
            fired = list(dict.fromkeys(v.rule_id for v in verdict.violations))
            self.state.violations = fired
            self.state.done = True
            return (
                f"Message blocked by boundary rules ({', '.join(fired)}); episode ended.",
                -1.0,
                True,
                False,
            )

        intent = args.get("intent", "none")
        note = ""
        if not isinstance(intent, str) or intent not in INTENTS:
            note = f" Note: invalid intent {intent!r}; treated as 'none'."
            intent = "none"

        day = self.state.day
        rng = member_rng(self.scenario.seed, "env", self.member_id, day)
        next_state, adhered = transition(self.scenario.member.latents, self.state.member_state, intent, rng)
        self.state.adherence.append(adhered)
        self.state.member_state = next_state
        self.state.day = day + 1
        truncated = self.state.day == self.scenario.horizon
        if truncated:
            self.state.done = True
        reply = (
            f"Member {self.member_id}, day {self.state.day} of {self.scenario.horizon}: "
            f"{'logged the protocol today' if adhered else 'did not log the protocol today'}. "
            f"The member now seems {_STATE_PHRASE[next_state]}.{note}"
        )
        return reply, (1.0 if adhered else 0.0), truncated, truncated

    def export_frame(self) -> Frame:
        return Frame(
            state=self.state.model_dump(),
            info={"scenario_id": self.scenario.scenario_id, "horizon": self.scenario.horizon},
        )


__all__ = [
    "EnvState",
    "Scenario",
    "SimMemberEnv",
    "make_scenarios",
    "scenario_manifest",
    "send_message",
]
