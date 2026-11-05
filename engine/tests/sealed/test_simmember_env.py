"""Sealed challenge suite — slice C: dynamics + aviary SimMemberEnv.

Spec: docs/daypart/contracts/CONTRACT-slice-C-env.md (C1-C3); DSL interface from
CONTRACT-slice-B-membrane.md (`daypart.membrane.dsl.check`).

Ambiguities resolved (questions for the principal):
- Directional effect test (C1): rollouts call `transition` directly, start in
  "engaged" (A3 day-0 state), use synthetic profile i % 175 for rollout i, and
  seed each rollout with numpy.random.default_rng(i), using the same seed for
  both policies (common random numbers). The assertion is strict:
  mean adherence under barrier_problem_solving > under none.
- "Same chain semantics as A3" is asserted as hard constraints only:
  lapsed -> {lapsed, recovering}; recovering -> {engaged, recovering, lapsed};
  next state always in STATES; adhered is a bool.
- make_scenarios: sampling is taken to be without replacement (implied by
  "n larger than the synthetic pool -> ValueError"), so member ids are unique.
  Scenario.seed vs the make_scenarios seed argument is not pinned; only
  scenario_id == f"scn-{scenario.seed}-{member_id}" is asserted.
- scenario_manifest hash input ("canonical json") is not pinned beyond being
  canonical; tested for determinism, shape, 64-hex values, and distinctness.
- "invalid intent treated as none": asserted behaviourally (identical rewards and
  state to an env driven with intent "none") plus the tool response mentioning
  "invalid" (case-insensitive).
- Two send_message calls in one step: the day advances exactly once and two
  ToolResponseMessages come back (one honoured, one error).
- After a DSL violation, `truncated` is not asserted (not pinned).
- The exact observation wording is implementation-defined; only the presence of
  the member_id and the absence of any other member id are checked.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from aviary.core import Environment, ToolCall, ToolRequestMessage, ToolResponseMessage

pytestmark = pytest.mark.category("integration/env")

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
COMPLIANT = [
    "Would a reminder at 8pm fit your evening better?",
    "What got in the way yesterday? We can plan around it together.",
    "Nice work logging three days in a row — want to keep the streak going?",
    "It's okay to miss a day. Want to pick it back up tomorrow morning?",
    "Try setting your evening reminder 15 minutes earlier tonight.",
]
VIOLATING = "You should take 400 mg of magnesium."
MEMBER_ID_RE = re.compile(r"\b(?:cur|syn)-\d{3}\b")
CURATED_RE = re.compile(r"\bcur-\d{3}\b")

_call_counter = 0


def _send(text: str, intent: str, n_calls: int = 1) -> ToolRequestMessage:
    global _call_counter
    calls = []
    for _ in range(n_calls):
        _call_counter += 1
        calls.append(ToolCall.from_name("send_message", id=f"call{_call_counter}", text=text, intent=intent))
    return ToolRequestMessage(tool_calls=calls)


def _contents(msgs) -> list[str]:
    return [m.content or "" for m in msgs]


def _scenario(profile, seed: int = 0, horizon: int | None = None):
    from daypart.simmember.env import Scenario

    kw = dict(scenario_id=f"scn-{seed}-{profile.member_id}", member=profile, seed=seed)
    if horizon is not None:
        kw["horizon"] = horizon
    return Scenario(**kw)


@pytest.fixture(scope="module")
def syn_profile(cohort0):
    return cohort0.training_profiles()[0]


@pytest.fixture(scope="module")
def cur_profile(cohort0):
    return cohort0.heldout_profiles()[0]


@pytest.fixture()
async def env(syn_profile):
    from daypart.simmember.env import SimMemberEnv

    e = SimMemberEnv(_scenario(syn_profile))
    obs, tools = await e.reset()
    return e


# --------------------------------------------------------------------------- #
# C1 — dynamics
# --------------------------------------------------------------------------- #


class TestDynamics:
    pytestmark = pytest.mark.category("functional/dynamics")

    def test_intents_constant(self):
        from daypart.simmember.dynamics import INTENTS as I

        assert tuple(I) == INTENTS

    def test_transition_constraints(self, cohort0):
        from daypart.simmember.dynamics import transition

        for idx, p in enumerate(cohort0.training_profiles()[::5]):
            for state in STATES:
                for intent in INTENTS:
                    rng = np.random.default_rng(1000 * idx + 7)
                    for _ in range(3):
                        nxt, adh = transition(p.latents, state, intent, rng)
                        assert nxt in STATES
                        assert isinstance(adh, (bool, np.bool_))
                        if state == "lapsed":
                            assert nxt in ("lapsed", "recovering"), (p.member_id, intent, nxt)
                        if state == "recovering":
                            assert nxt in ("engaged", "recovering", "lapsed"), (p.member_id, intent, nxt)

    def test_transition_deterministic_given_rng(self, cohort0):
        from daypart.simmember.dynamics import transition

        p = cohort0.training_profiles()[10]
        for state in STATES:
            for intent in INTENTS:
                a = [transition(p.latents, state, intent, np.random.default_rng(42)) for _ in range(2)]
                assert a[0] == a[1]

    @pytest.mark.parametrize("bad", ["bogus", "", "None", "REMINDER", "barrier problem solving"])
    def test_unknown_intent_value_error(self, cohort0, bad):
        from daypart.simmember.dynamics import transition

        p = cohort0.training_profiles()[0]
        with pytest.raises(ValueError):
            transition(p.latents, "engaged", bad, np.random.default_rng(0))

    def test_barrier_problem_solving_beats_none(self, cohort0):
        from daypart.simmember.dynamics import transition

        synth = cohort0.training_profiles()

        def mean_adherence(intent: str) -> float:
            total = 0
            n = 0
            for i in range(200):
                lat = synth[i % len(synth)].latents
                rng = np.random.default_rng(i)
                state = "engaged"
                for _ in range(14):
                    state, adh = transition(lat, state, intent, rng)
                    total += int(bool(adh))
                    n += 1
            return total / n

        bps = mean_adherence("barrier_problem_solving")
        none = mean_adherence("none")
        assert bps > none, (bps, none)


# --------------------------------------------------------------------------- #
# C2 — scenarios
# --------------------------------------------------------------------------- #


class TestScenarios:
    pytestmark = pytest.mark.category("functional/scenarios")

    def test_make_scenarios_synthetic_only(self, cohort0):
        from daypart.simmember.env import Scenario, make_scenarios

        train_ids = {p.member_id for p in cohort0.training_profiles()}
        scns = make_scenarios(cohort0, 40, 0)
        assert len(scns) == 40
        for s in scns:
            assert isinstance(s, Scenario)
            assert s.member.member_id in train_ids
            assert s.member.cohort == "synthetic"
            assert s.scenario_id == f"scn-{s.seed}-{s.member.member_id}"
            assert s.horizon == 14
        assert len({s.member.member_id for s in scns}) == 40

    def test_make_scenarios_full_pool_no_curated(self, cohort0):
        from daypart.simmember.env import make_scenarios

        scns = make_scenarios(cohort0, 175, 3)
        assert len(scns) == 175
        assert not any(CURATED_RE.search(s.model_dump_json()) for s in scns)

    def test_make_scenarios_deterministic(self, cohort0, cohort0_again):
        from daypart.simmember.env import make_scenarios

        assert make_scenarios(cohort0, 25, 9) == make_scenarios(cohort0, 25, 9)
        assert make_scenarios(cohort0, 25, 9) == make_scenarios(cohort0_again, 25, 9)

    def test_make_scenarios_too_many(self, cohort0):
        from daypart.simmember.env import make_scenarios

        with pytest.raises(ValueError):
            make_scenarios(cohort0, 176, 0)

    def test_make_scenarios_too_many_small_pool(self):
        from daypart.simmember.cohort import generate
        from daypart.simmember.env import make_scenarios

        c = generate(0, n_synthetic=10, days=3)
        assert len(make_scenarios(c, 10, 1)) == 10
        with pytest.raises(ValueError):
            make_scenarios(c, 11, 1)

    def test_scenario_default_horizon(self, syn_profile):
        assert _scenario(syn_profile).horizon == 14

    def test_curated_scenario_raises(self, cur_profile):
        from daypart.simmember.manifest import CuratedLeakError

        with pytest.raises(CuratedLeakError):
            _scenario(cur_profile)

    def test_curated_env_raises(self, cur_profile):
        from daypart.simmember.env import Scenario, SimMemberEnv
        from daypart.simmember.manifest import CuratedLeakError

        scn = Scenario.model_construct(
            scenario_id=f"scn-0-{cur_profile.member_id}", member=cur_profile, seed=0, horizon=14
        )
        with pytest.raises(CuratedLeakError):
            SimMemberEnv(scn)

    def test_scenario_manifest(self, cohort0):
        import json

        from daypart.simmember.env import make_scenarios, scenario_manifest

        scns = make_scenarios(cohort0, 12, 4)
        m1 = scenario_manifest(scns)
        m2 = scenario_manifest(make_scenarios(cohort0, 12, 4))
        assert m1 == m2
        assert json.dumps(m1, sort_keys=True, separators=(",", ":")) == json.dumps(
            m2, sort_keys=True, separators=(",", ":")
        )
        assert m1["count"] == 12
        assert set(m1["scenarios"]) == {s.scenario_id for s in scns}
        hashes = list(m1["scenarios"].values())
        assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes)
        assert len(set(hashes)) == len(hashes)


# --------------------------------------------------------------------------- #
# C3 — environment
# --------------------------------------------------------------------------- #


class TestReset:
    pytestmark = pytest.mark.category("integration/env-reset")

    async def test_reset_shape(self, syn_profile):
        from daypart.simmember.env import SimMemberEnv

        e = SimMemberEnv(_scenario(syn_profile))
        assert isinstance(e, Environment)
        obs, tools = await e.reset()
        assert len(obs) == 1
        content = obs[0].content or ""
        assert syn_profile.member_id in content
        assert set(MEMBER_ID_RE.findall(content)) == {syn_profile.member_id}
        assert not CURATED_RE.search(content)
        assert len(tools) == 1
        tool = tools[0]
        assert tool.info.name == "send_message"
        props = tool.info.get_properties()
        assert set(props) == {"text", "intent"}
        assert props["text"].get("type") == "string"
        assert props["intent"].get("type") == "string"

    async def test_reset_state(self, env):
        st = env.state
        assert st.day == 0
        assert st.member_state in STATES
        assert st.adherence == []
        assert st.done is False
        assert st.violations == []


class TestStep:
    pytestmark = pytest.mark.category("integration/env-step")

    async def test_compliant_step(self, env, syn_profile):
        obs, reward, done, truncated = await env.step(_send(COMPLIANT[0], "reminder"))
        assert reward in (0.0, 1.0)
        assert done is False and truncated is False
        assert env.state.day == 1
        assert len(env.state.adherence) == 1
        assert reward == (1.0 if env.state.adherence[-1] else 0.0)
        assert env.state.member_state in STATES
        assert env.state.violations == []
        assert any(isinstance(m, ToolResponseMessage) for m in obs)
        for c in _contents(obs):
            assert not CURATED_RE.search(c)

    async def test_dsl_violation_ends_episode(self, env):
        obs, reward, done, truncated = await env.step(_send(VIOLATING, "encouragement"))
        assert reward == -1.0
        assert done is True
        assert env.state.done is True
        assert env.state.violations
        assert "dosing.unit_quantity" in env.state.violations
        with pytest.raises(RuntimeError):
            await env.step(_send(COMPLIANT[0], "reminder"))

    async def test_dsl_violation_mid_episode(self, env):
        for text in COMPLIANT[:3]:
            _, r, d, _ = await env.step(_send(text, "encouragement"))
            assert d is False and r in (0.0, 1.0)
        _, reward, done, _ = await env.step(_send(VIOLATING, "reminder"))
        assert reward == -1.0 and done is True
        assert "dosing.unit_quantity" in env.state.violations

    async def test_invalid_intent_handled(self, env):
        obs, reward, done, truncated = await env.step(_send(COMPLIANT[1], "bogus_intent"))
        assert reward in (0.0, 1.0)
        assert done is False
        assert env.state.day == 1
        assert any("invalid" in c.lower() for c in _contents(obs))

    async def test_invalid_intent_equivalent_to_none(self, syn_profile):
        from daypart.simmember.env import SimMemberEnv

        a = SimMemberEnv(_scenario(syn_profile, seed=11))
        b = SimMemberEnv(_scenario(syn_profile, seed=11))
        await a.reset()
        await b.reset()
        for _ in range(8):
            _, ra, da, ta = await a.step(_send(COMPLIANT[2], "not-an-intent"))
            _, rb, db, tb = await b.step(_send(COMPLIANT[2], "none"))
            assert (ra, da, ta) == (rb, db, tb)
        assert a.state.adherence == b.state.adherence
        assert a.state.member_state == b.state.member_state
        assert a.state.day == b.state.day

    async def test_no_tool_calls(self, env):
        obs, reward, done, truncated = await env.step(ToolRequestMessage(tool_calls=[]))
        assert reward == 0.0
        assert done is False
        assert env.state.day == 0
        assert env.state.adherence == []

    async def test_extra_calls_ignored(self, env):
        obs, reward, done, truncated = await env.step(_send(COMPLIANT[0], "reminder", n_calls=2))
        assert env.state.day == 1
        assert len(env.state.adherence) == 1
        assert reward in (0.0, 1.0)
        assert sum(isinstance(m, ToolResponseMessage) for m in obs) == 2

    async def test_truncation_at_horizon(self, syn_profile):
        from daypart.simmember.env import SimMemberEnv

        e = SimMemberEnv(_scenario(syn_profile, horizon=3))
        await e.reset()
        results = []
        for i in range(3):
            _, r, d, t = await e.step(_send(COMPLIANT[i], INTENTS[i + 1]))
            results.append((r, d, t))
        assert [(d, t) for _, d, t in results] == [(False, False), (False, False), (True, True)]
        assert all(r in (0.0, 1.0) for r, _, _ in results)
        assert e.state.day == 3
        assert len(e.state.adherence) == 3
        assert e.state.done is True
        assert e.state.violations == []
        with pytest.raises(RuntimeError):
            await e.step(_send(COMPLIANT[0], "reminder"))

    async def test_default_horizon_14(self, env):
        for i in range(13):
            _, _, d, t = await env.step(_send(COMPLIANT[i % len(COMPLIANT)], INTENTS[i % len(INTENTS)]))
            assert d is False and t is False
        _, _, d, t = await env.step(_send(COMPLIANT[0], "none"))
        assert d is True and t is True
        assert env.state.day == 14
        assert len(env.state.adherence) == 14
        with pytest.raises(RuntimeError):
            await env.step(_send(COMPLIANT[0], "none"))

    async def test_rewards_match_adherence(self, env):
        rewards = []
        for i in range(10):
            _, r, _, _ = await env.step(_send(COMPLIANT[i % len(COMPLIANT)], "barrier_problem_solving"))
            rewards.append(r)
        assert rewards == [1.0 if a else 0.0 for a in env.state.adherence]


class TestDeterminismAndFrame:
    pytestmark = pytest.mark.category("integration/env-determinism")

    async def _drive(self, scenario, actions):
        from daypart.simmember.env import SimMemberEnv

        e = SimMemberEnv(scenario)
        obs, _ = await e.reset()
        trace = [("reset", tuple(_contents(obs)))]
        for text, intent in actions:
            o, r, d, t = await e.step(_send(text, intent))
            trace.append((r, d, t, tuple(_contents(o))))
            if d:
                break
        return trace, e.state.model_dump()

    async def test_two_envs_identical(self, cohort0):
        from daypart.simmember.env import make_scenarios

        actions = [(COMPLIANT[i % len(COMPLIANT)], (INTENTS + ("zzz",))[i % 8]) for i in range(14)]
        for scn in make_scenarios(cohort0, 3, 5):
            a = await self._drive(scn, actions)
            b = await self._drive(scn, actions)
            assert a == b

    async def test_two_envs_identical_with_violation(self, syn_profile):
        scn = _scenario(syn_profile, seed=2)
        actions = [(COMPLIANT[0], "reminder"), (COMPLIANT[1], "encouragement"), (VIOLATING, "none")]
        a = await self._drive(scn, actions)
        b = await self._drive(scn, actions)
        assert a == b
        assert a[0][-1][0] == -1.0

    async def test_export_frame(self, env):
        await env.step(_send(COMPLIANT[0], "reminder"))
        frame = env.export_frame()
        st = frame.state
        if hasattr(st, "model_dump"):
            st = st.model_dump()
        assert isinstance(st, dict)
        assert {"day", "member_state", "adherence"} <= set(st)
        assert st["day"] == env.state.day == 1
        assert st["member_state"] == env.state.member_state
        assert list(st["adherence"]) == list(env.state.adherence)
