"""Playground core: generated members, stepped ``SimMemberEnv`` episodes, membrane verdicts, passports.

Framework-free so the FastAPI layer stays thin. Synthetic members only (I1):
curated (``cur-``) ids are never listed and never resolvable.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from aviary.core import ToolCall, ToolRequestMessage

from daypart.eval import POLICY_TEXT
from daypart.membrane import dsl
from daypart.membrane.post import post
from daypart.memory.lhm import MemoryLhm
from daypart.passport import PassportRenderer
from daypart.simmember.cohort import Cohort, Profile, generate
from daypart.simmember.dynamics import INTENTS
from daypart.simmember.env import SEND_MESSAGE, Scenario, SimMemberEnv
from daypart.simmember.lhm_records import SimMemberLhmAdapter
from daypart.simmember.trajectory import START_DATE

from daypart.demo.safety import WITHHELD_DRAFT, safe_text

MAX_EPISODES = 256  # in-memory LRU cap; the playground is single-operator and offline
MAX_HORIZON = 42


class UnknownMember(LookupError):
    pass


class UnknownEpisode(LookupError):
    pass


class EpisodeDone(RuntimeError):
    pass


@dataclass
class Episode:
    episode_id: str
    scenario: Scenario
    env: SimMemberEnv
    observation: str


def _member_view(p: Profile) -> dict[str, Any]:
    lat = p.latents
    return {
        "member_id": p.member_id,
        "motivation": lat.motivation,
        "barrier_type": lat.barrier_type,
        "literacy": lat.literacy,
        "engagement_pattern": lat.engagement_pattern,
        "age_band": lat.age_band,
    }


class Playground:
    def __init__(self, cohort_seed: int = 0) -> None:
        self.cohort_seed = cohort_seed
        self.cohort: Cohort = generate(cohort_seed)
        self._members: dict[str, Profile] = {p.member_id: p for p in self.cohort.training_profiles()}
        self._renderer = PassportRenderer(MemoryLhm(SimMemberLhmAdapter(self.cohort)))
        self._episodes: OrderedDict[str, Episode] = OrderedDict()

    # -- members ---------------------------------------------------------------------------
    def members(self) -> list[dict[str, Any]]:
        return [_member_view(p) for p in self._members.values()]

    def member(self, member_id: str) -> Profile:
        p = self._members.get(member_id)
        if p is None:  # unknown, malformed, or curated (never in _members)
            raise UnknownMember(member_id)
        return p

    @staticmethod
    def intents() -> list[dict[str, str]]:
        return [{"intent": i, "default_text": POLICY_TEXT[i]} for i in INTENTS]

    # -- episodes --------------------------------------------------------------------------
    async def start(self, member_id: str, *, seed: int = 0, horizon: int = 14) -> dict[str, Any]:
        scenario = Scenario(member=self.member(member_id), seed=seed, horizon=horizon)  # pyright: ignore[reportCallIssue]  # scenario_id is filled by a before-validator
        env = SimMemberEnv(scenario)
        msgs, _tools = await env.reset()
        ep = Episode(uuid.uuid4().hex, scenario, env, safe_text(msgs[0].content))
        self._episodes[ep.episode_id] = ep
        while len(self._episodes) > MAX_EPISODES:
            self._episodes.popitem(last=False)
        return self._episode_view(ep)

    def _get(self, episode_id: str) -> Episode:
        ep = self._episodes.get(episode_id)
        if ep is None:
            raise UnknownEpisode(episode_id)
        self._episodes.move_to_end(episode_id)
        return ep

    def _episode_view(self, ep: Episode) -> dict[str, Any]:
        return {
            "episode_id": ep.episode_id,
            "scenario_id": ep.scenario.scenario_id,
            "member_id": ep.scenario.member.member_id,
            "seed": ep.scenario.seed,
            "horizon": ep.scenario.horizon,
            "observation": ep.observation,
            "state": ep.env.state.model_dump(mode="json"),
        }

    def episode(self, episode_id: str) -> dict[str, Any]:
        return self._episode_view(self._get(episode_id))

    async def step(self, episode_id: str, intent: str, text: str | None = None) -> dict[str, Any]:
        """Run the draft through the membrane (layer 0 + DSL), then step the env with it.

        The env is the benchmark (R6): a DSL violation scores -1 and ends the episode,
        so the draft is stepped even when the membrane escalates. The draft itself is
        only echoed back when the membrane passes it.
        """
        ep = self._get(episode_id)
        if ep.env.state.done:
            raise EpisodeDone(episode_id)
        if intent not in INTENTS:
            raise ValueError(f"unknown intent {intent!r}")
        draft = POLICY_TEXT[intent] if text is None else text
        verdict = post(draft)
        # post() stops at layer 0; run the DSL on its own so its rule hits are always shown.
        dsl_verdict = dsl.check(draft)
        call = ToolCall.from_name(SEND_MESSAGE, text=draft, intent=intent)
        msgs, reward, done, truncated = await ep.env.step(ToolRequestMessage(tool_calls=[call]))
        reply = safe_text(msgs[0].content if msgs else "")
        return {
            "episode_id": ep.episode_id,
            "intent": intent,
            "draft": draft if verdict.decision == "pass" else WITHHELD_DRAFT,
            "membrane": verdict.trace,
            "dsl_hits": sorted({v.rule_id for v in dsl_verdict.violations}),
            "reward": float(reward),
            "done": bool(done),
            "truncated": bool(truncated),
            "reply": reply,
            "state": ep.env.state.model_dump(mode="json"),
        }

    # -- passport --------------------------------------------------------------------------
    async def passport(self, member_id: str, day: int) -> dict[str, Any]:
        self.member(member_id)
        p = await self._renderer.passport(member_id, START_DATE + timedelta(days=day))
        out = p.model_dump(mode="json")
        # Flag validation already enforces the narration guard; re-gate on the way out (fail closed).
        for f in out["flags"]:
            f["title"] = safe_text(f["title"])
            if f["message"] is not None:
                f["message"] = safe_text(f["message"])
            f["caveats"] = [safe_text(c) for c in f["caveats"]]
        out["day"] = day
        return out


__all__ = ["EpisodeDone", "MAX_EPISODES", "MAX_HORIZON", "Playground", "UnknownEpisode", "UnknownMember"]
