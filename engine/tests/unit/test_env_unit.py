import sys
import types
from dataclasses import dataclass

import pytest
from aviary.core import Message, ToolCall, ToolRequestMessage, ToolResponseMessage

from daypart.simmember.cohort import CURATED_PROFILES, generate
from daypart.simmember.dynamics import STATES
from daypart.simmember.env import EnvState, Scenario, SimMemberEnv, make_scenarios, scenario_manifest
from daypart.simmember.manifest import CuratedLeakError


@dataclass(frozen=True)
class _Violation:
    rule_id: str
    match: str


@dataclass(frozen=True)
class _Verdict:
    passed: bool
    violations: tuple
    rule_version: str = "boundary-v1"


def _fake_check(text, *, crisis_context=False):
    if "400 mg" in text:
        v = (_Violation("dosing.unit_quantity", "400 mg"), _Violation("dosing.unit_quantity", "400 mg"))
        return _Verdict(False, v)
    return _Verdict(True, ())


@pytest.fixture(autouse=True)
def fake_dsl(monkeypatch):
    mod = types.ModuleType("daypart.membrane.dsl")
    mod.check = _fake_check
    monkeypatch.setitem(sys.modules, "daypart.membrane.dsl", mod)


@pytest.fixture(scope="module")
def cohort0():
    return generate(0, days=3)


def _send(text="Hi there, how is today going?", intent="reminder", cid="c1"):
    return ToolRequestMessage(tool_calls=[ToolCall.from_name("send_message", id=cid, text=text, intent=intent)])


def test_make_scenarios(cohort0):
    a = make_scenarios(cohort0, 10, seed=3)
    assert a == make_scenarios(cohort0, 10, seed=3)
    assert a != make_scenarios(cohort0, 10, seed=4)
    assert len({s.member.member_id for s in a}) == 10
    assert all(s.member.cohort == "synthetic" and s.scenario_id == f"scn-3-{s.member.member_id}" for s in a)
    assert all(s.horizon == 14 and s.seed == 3 for s in a)
    assert len(make_scenarios(cohort0, 175, seed=0)) == 175
    with pytest.raises(ValueError):
        make_scenarios(cohort0, 176, seed=0)
    m = scenario_manifest(a)
    assert m == scenario_manifest(make_scenarios(cohort0, 10, seed=3))
    assert m["count"] == 10 and set(m["scenarios"]) == {s.scenario_id for s in a}


def test_curated_member_rejected(cohort0):
    with pytest.raises(CuratedLeakError):
        Scenario(scenario_id="scn-0-cur-001", member=CURATED_PROFILES[0], seed=0)
    with pytest.raises(CuratedLeakError):
        Scenario(member=CURATED_PROFILES[1], seed=0)
    ok = make_scenarios(cohort0, 1, seed=0)[0]
    bad = ok.model_construct(scenario_id="scn-0-cur-001", member=CURATED_PROFILES[0], seed=0, horizon=14)
    with pytest.raises(CuratedLeakError):
        SimMemberEnv(bad)


async def test_reset_and_tool(cohort0):
    sc = make_scenarios(cohort0, 1, seed=0)[0]
    env = SimMemberEnv(sc)
    obs, tools = await env.reset()
    assert len(obs) == 1 and isinstance(obs[0], Message)
    assert sc.member.member_id in obs[0].content and "cur-" not in obs[0].content
    assert [t.info.name for t in tools] == ["send_message"]
    props = tools[0].info.parameters.properties
    assert set(props) == {"text", "intent"}
    assert props["text"]["type"] == "string" and props["intent"]["type"] == "string"
    assert env.state == EnvState(day=0, member_state="engaged", adherence=[], done=False, violations=[])
    assert await env.get_id() == sc.scenario_id


async def test_full_episode_truncates(cohort0):
    sc = make_scenarios(cohort0, 1, seed=0)[0]
    env = SimMemberEnv(sc)
    await env.reset()
    for d in range(sc.horizon):
        msgs, reward, done, trunc = await env.step(_send(cid=f"c{d}"))
        assert len(msgs) == 1 and isinstance(msgs[0], ToolResponseMessage)
        assert msgs[0].tool_call_id == f"c{d}"
        assert reward in (0.0, 1.0) and reward == float(env.state.adherence[-1])
        assert env.state.member_state in STATES and env.state.day == d + 1
        assert done == trunc == (d + 1 == sc.horizon)
    assert env.state.done and len(env.state.adherence) == sc.horizon
    frame = env.export_frame()
    assert {"day", "member_state", "adherence"} <= set(frame.state)
    with pytest.raises(RuntimeError):
        await env.step(_send())


async def test_dsl_violation_ends_episode(cohort0):
    env = SimMemberEnv(make_scenarios(cohort0, 1, seed=0)[0])
    await env.reset()
    await env.step(_send())
    msgs, reward, done, trunc = await env.step(_send(text="Take 400 mg tonight", cid="bad"))
    assert reward == -1.0 and done is True and trunc is False
    assert env.state.done and env.state.violations == ["dosing.unit_quantity"]
    assert env.state.day == 1
    with pytest.raises(RuntimeError):
        await env.step(_send())


async def test_invalid_intent_treated_as_none(cohort0):
    sc = make_scenarios(cohort0, 1, seed=0)[0]
    a, b = SimMemberEnv(sc), SimMemberEnv(sc)
    await a.reset()
    await b.reset()
    for d in range(5):
        ra = await a.step(_send(intent="bribe", cid=f"x{d}"))
        rb = await b.step(_send(intent="none", cid=f"x{d}"))
        assert "invalid intent" in ra[0][0].content.lower()
        assert ra[1:] == rb[1:]
    assert a.state == b.state


async def test_no_tool_calls_and_extra_calls(cohort0):
    env = SimMemberEnv(make_scenarios(cohort0, 1, seed=0)[0])
    await env.reset()
    msgs, reward, done, trunc = await env.step(ToolRequestMessage(tool_calls=[]))
    assert (reward, done, trunc) == (0.0, False, False) and env.state.day == 0
    req = ToolRequestMessage(tool_calls=[
        ToolCall.from_name("send_message", id="a", text="Hello", intent="none"),
        ToolCall.from_name("send_message", id="b", text="Hello again", intent="none"),
        ToolCall.from_name("other_tool", id="c"),
    ])
    msgs, reward, done, trunc = await env.step(req)
    assert [m.tool_call_id for m in msgs] == ["a", "b", "c"]
    assert "ignored" in msgs[1].content.lower() and "invalid" in msgs[2].content.lower()
    assert env.state.day == 1


async def _drive(sc, actions):
    env = SimMemberEnv(sc)
    await env.reset()
    trace = []
    for i, (text, intent) in enumerate(actions):
        if env.state.done:
            break
        msgs, r, d, t = await env.step(_send(text=text, intent=intent, cid=f"k{i}"))
        trace.append(([m.content for m in msgs], r, d, t))
    return trace, env.state


async def test_determinism(cohort0):
    sc = make_scenarios(cohort0, 2, seed=9)[1]
    actions = [("Hi", "reminder"), ("Hey", "barrier_problem_solving"), ("Yo", "none")] * 5
    t1, s1 = await _drive(sc, actions)
    t2, s2 = await _drive(sc, actions)
    assert t1 == t2 and s1 == s2


async def test_env_bps_beats_none_over_scenarios():
    cohort = generate(0, days=1)
    scenarios = make_scenarios(cohort, 175, seed=0)
    means = {}
    for intent in ("none", "barrier_problem_solving"):
        total = 0.0
        for sc in scenarios:
            trace, _ = await _drive(sc, [("Hello", intent)] * sc.horizon)
            total += sum(r for _, r, _, _ in trace) / sc.horizon
        means[intent] = total / len(scenarios)
    assert means["barrier_problem_solving"] > means["none"] + 0.1
