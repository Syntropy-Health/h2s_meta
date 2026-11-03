"""Sealed challenge suite — slice D3: daypart.memory.lhm (I6, I7, G3, D14).

Contract: docs/daypart/contracts/CONTRACT-slice-D-memory.md §D3, §D3.1 (fixtures from
slice A: generate(0) cohort + SimMemberLhmAdapter).

Categories: functional/lhm-alignment, functional/lhm-state-shape, functional/lhm-scoring,
functional/lhm-readiness (I6), functional/lhm-d14 (sealed property), functional/lhm-g3,
functional/lhm-identity (I7), integration/lhm-signals, functional/lhm-determinism.

Ambiguities recorded for the principal (resolved as noted; nothing invented):
- Expected DomainScores are recomputed here with syntropy.lhm `score_domain` over exactly
  the samples D3 enumerates (same pinned LHM) — no score numbers are hard-coded.
- "Adapter view restricted to records with timestamps <= as_of": whether a SeriesRecord
  whose points are ALL dropped is kept (empty) or removed is unspecified. The reference
  comparison for `signals` therefore only uses (member, as_of) pairs where every series
  keeps >= 1 point, plus the "before any record → ()" case.
- I6 at START+4 is asserted for all 200 members (n_effective 5 < LHM N_MIN 7 makes
  readiness impossible); START+41 uses the contract's >= 95% threshold.
- Mapping fields are compared via dict(...) so any read-only Mapping type is accepted.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta, timezone

import pytest

from daypart.memory.events import ACTION_CODES, ActionEvent, EventLog, SignalEvent
from daypart.memory.lhm import (
    GUARD_VERSIONS,
    PRIMARY_NODE,
    SERIES_NODES,
    MemberMetricState,
    MemoryLhm,
    NodeState,
    alignment_score,
)
from daypart.memory.ontology import (
    CORE_CAVEATS,
    LOCAL_SYSTEM,
    Edge,
    Node,
    OntologyUpdate,
    apply_update,
    core,
)
from daypart.simmember.trajectory import START_DATE
from syntropy.lhm.metrics.claims import (
    CLAIMS_GUARD_VERSION,
    VALENCE_GUARD_VERSION,
    ClaimScopeError,
    ScoreRendering,
    assert_valence_safe,
)
from syntropy.lhm.metrics.scoring import DomainScore, is_ready, score_domain
from syntropy.lhm.tools.pipeline import analyze, fetch
from tests.sealed.conftest_memory import (  # noqa: F401  (fixtures)
    SAMPLE_MEMBERS,
    UNKNOWN_MEMBER,
    RestrictedView,
    SpyAdapter,
    adapter,
    cohort,
    cohort0,
    day,
    first_checkin_day,
    members,
)

UTC = timezone.utc
CHECKIN = "dp:affect.checkin_sentiment"
UNMAPPED_CORE = ("29463-7", "44261-6", "55423-8", "70274-6", "72514-3", "8867-4", "93832-4")
MAPPED_CORE = {
    "dp:adherence.protocol": "consistency",
    "dp:affect.checkin_sentiment": "sentiment_trajectory",
    "dp:nutrition.food_log": "nutrition_regularity",
    "dp:symptom.severity": "symptom_load",
}


async def state(adp, mid, as_of, *, events=None, ontology=None, goal_nodes=None) -> MemberMetricState:
    kw = {}
    if ontology is not None:
        kw["ontology"] = ontology
    if events is not None:
        kw["events"] = events
    if goal_nodes is not None:
        kw["goal_nodes"] = goal_nodes
    return await MemoryLhm(adp, **kw).metric_state(mid, as_of)


def act(eid, mid, at, code):
    return ActionEvent(event_id=eid, member_id=mid, occurred_at=at, code=code)


def sig(eid, mid, at, code, value):
    return SignalEvent(event_id=eid, member_id=mid, occurred_at=at, code=code, value=value)


def valence_ok(score: DomainScore) -> bool:
    try:
        assert_valence_safe(ScoreRendering(domain=score.domain, score=score.score))
    except ClaimScopeError:
        return False
    return True


def contract_samples(adp, cohort_, mid, as_of):
    """The per-node samples D3 enumerates, from the adapter series only (no events)."""
    out = {"dp:adherence.protocol": [], CHECKIN: []}
    traj = cohort_.trajectories.get(mid)
    if traj is None:
        return out
    for d in traj.days:
        if d.date <= as_of:
            out["dp:adherence.protocol"].append((d.date, 1.0 if d.adhered else 0.0))
            if d.checkin:
                out[CHECKIN].append((d.date, adp.sentiment_fn(d.note_id)))
    return out


# --------------------------------------------------------------------------- #
# functional/lhm-alignment
# --------------------------------------------------------------------------- #


def test_constants():
    assert PRIMARY_NODE == "dp:adherence.protocol"
    assert SERIES_NODES == {
        "Protocol adherence (logged)": "dp:adherence.protocol",
        "Check-in sentiment": "dp:affect.checkin_sentiment",
    }
    assert dict(GUARD_VERSIONS) == {"claims": CLAIMS_GUARD_VERSION, "valence": VALENCE_GUARD_VERSION}


@pytest.mark.parametrize(
    "code,expected",
    [
        ("act:reminder_sent", 0.3),
        ("act:nudge_sent", 0.2),
        ("act:goal_set", 0.2),
        ("act:education_viewed", 0.1),
        ("act:protocol_adjusted", 0.1),
        ("act:reply_sent", 0.0),  # no edge to the default goal → 0.0
    ],
)
def test_alignment_score_default_goal(code, expected):
    e = act("a", "syn-001", day(3), code)
    assert alignment_score(e, core()) == pytest.approx(expected)
    assert alignment_score(e, core(), (PRIMARY_NODE,)) == pytest.approx(expected)


def test_alignment_score_empty_goals_is_zero():
    for code in ACTION_CODES:
        assert alignment_score(act("a", "syn-001", day(3), code), core(), ()) == 0.0


def test_alignment_score_mean_over_goals():
    o = core()
    goals = (PRIMARY_NODE, CHECKIN)
    assert alignment_score(act("a", "syn-001", day(3), "act:reminder_sent"), o, goals) == pytest.approx(0.15)
    assert alignment_score(act("a", "syn-001", day(3), "act:reply_sent"), o, goals) == pytest.approx(0.05)
    assert alignment_score(act("a", "syn-001", day(3), "act:reply_sent"), o, (CHECKIN,)) == pytest.approx(0.1)
    # a goal with no edge contributes 0.0 to the mean
    assert alignment_score(act("a", "syn-001", day(3), "act:reminder_sent"), o,
                           (PRIMARY_NODE, "8867-4", "93832-4")) == pytest.approx(0.1)


def test_alignment_score_follows_reweighted_ontology():
    o1 = apply_update(core(), OntologyUpdate(proposer="p", rationale="r", set_edges=(
        Edge(source="act:reminder_sent", target=PRIMARY_NODE, weight=-0.6),)))
    assert alignment_score(act("a", "syn-001", day(3), "act:reminder_sent"), o1) == pytest.approx(-0.6)
    assert alignment_score(act("a", "syn-001", day(3), "act:reminder_sent"), core()) == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# functional/lhm-state-shape
# --------------------------------------------------------------------------- #


async def test_state_shape(adapter):
    as_of = day(20)
    st = await state(adapter, "syn-001", as_of)
    assert isinstance(st, MemberMetricState)
    assert st.member_id == "syn-001"
    assert st.as_of == as_of
    assert st.ontology_version == "core-v1"
    assert dict(st.guard_versions) == dict(GUARD_VERSIONS)
    o = core()
    assert set(st.nodes) == {n.code for n in o.metric_nodes()}
    for code, ns in st.nodes.items():
        assert isinstance(ns, NodeState)
        assert ns.code == code
        assert ns.lhm_domain == o.node(code).lhm_domain
        assert (ns.score is None) == (ns.lhm_domain is None)
        if ns.score is not None:
            assert isinstance(ns.score, DomainScore)
            assert ns.score.domain == ns.lhm_domain
    assert dict(st.alignment) == {}
    assert isinstance(st.signals, tuple)


async def test_unmapped_core_nodes_never_scored(adapter):
    st = await state(adapter, "cur-001", day(41))
    for code in UNMAPPED_CORE:
        ns = st.nodes[code]
        assert ns == NodeState(code, None, score=None, ready=False, value=None)


async def test_ready_codes_and_surfaced(adapter):
    for mid in SAMPLE_MEMBERS:
        for d in (4, 20, 41):
            st = await state(adapter, mid, day(d))
            rc = st.ready_codes()
            assert isinstance(rc, tuple)
            assert list(rc) == sorted(c for c, n in st.nodes.items() if n.ready)
            sf = st.surfaced()
            assert set(sf) == set(st.nodes)
            for code, ns in st.nodes.items():
                if ns.ready:
                    assert sf[code] == ns.value == ns.score.score
                else:
                    assert ns.value is None
                    assert sf[code] is None


async def test_naive_as_of_rejected(adapter):
    with pytest.raises(ValueError):
        await state(adapter, "syn-001", datetime(2026, 9, 20))


async def test_unknown_member(adapter):
    log = EventLog()
    log.append(act("u-a", UNKNOWN_MEMBER, day(3), "act:goal_set"))
    st = await state(adapter, UNKNOWN_MEMBER, day(41), events=log)
    assert st.member_id == UNKNOWN_MEMBER
    assert st.signals == ()
    assert set(st.nodes) == {n.code for n in core().metric_nodes()}
    for ns in st.nodes.values():
        assert ns.ready is False and ns.value is None
    assert st.ready_codes() == ()
    assert all(v is None for v in st.surfaced().values())
    assert dict(st.alignment) == {"u-a": pytest.approx(0.2)}


async def test_as_of_before_any_record(adapter):
    st = await state(adapter, "cur-002", START_DATE - timedelta(days=1))
    assert st.signals == ()
    assert all(not ns.ready and ns.value is None for ns in st.nodes.values())


# --------------------------------------------------------------------------- #
# functional/lhm-scoring
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mid", SAMPLE_MEMBERS)
@pytest.mark.parametrize("d", [4, 13, 27, 41, 70])
async def test_mapped_scores_follow_contract_samples(adapter, cohort, mid, d):
    as_of = day(d) + timedelta(hours=6)
    st = await state(adapter, mid, as_of)
    samples = contract_samples(adapter, cohort, mid, as_of)
    for code, dom in MAPPED_CORE.items():
        expected = score_domain(dom, samples.get(code, []), as_of=as_of)
        ns = st.nodes[code]
        assert ns.lhm_domain == dom
        assert ns.score == expected, (mid, d, code)
        exp_ready = is_ready(expected, dom, claim_gate_passed=True, valence_gate_passed=valence_ok(expected))
        assert ns.ready is exp_ready, (mid, d, code)
        assert ns.value == (expected.score if exp_ready else None)


async def test_samples_exclude_points_after_as_of(adapter, cohort):
    # as_of just before a day boundary must exclude that day's series point
    mid = "syn-010"
    as_of = day(20) - timedelta(microseconds=1)
    st = await state(adapter, mid, as_of)
    n_days = sum(1 for dd in cohort.trajectories[mid].days if dd.date <= as_of)
    assert n_days == 20
    assert st.nodes[PRIMARY_NODE].score.n_effective == 20


async def test_empty_mapped_nodes_not_ready(adapter):
    st = await state(adapter, "syn-050", day(41))
    for code in ("dp:nutrition.food_log", "dp:symptom.severity"):
        ns = st.nodes[code]
        assert ns.score is not None and ns.score.n_effective == 0
        assert ns.ready is False and ns.value is None


# --------------------------------------------------------------------------- #
# functional/lhm-readiness (I6 / fail-closed)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def cohort_states():
    c = cohort0()
    from daypart.simmember.lhm_records import SimMemberLhmAdapter

    lhm = MemoryLhm(SimMemberLhmAdapter(c))

    async def run():
        out = {}
        for mid in members(c):
            out[(mid, 4)] = await lhm.metric_state(mid, day(4))
            out[(mid, 41)] = await lhm.metric_state(mid, day(41))
        return out

    return asyncio.run(run())


def test_i6_five_day_members_not_ready(cohort_states):
    for mid in members(cohort0()):
        ns = cohort_states[(mid, 4)].nodes[PRIMARY_NODE]
        assert ns.ready is False, mid
        assert ns.value is None, mid
        assert cohort_states[(mid, 4)].surfaced()[PRIMARY_NODE] is None


def test_i6_forty_two_day_members_ready(cohort_states):
    ms = members(cohort0())
    assert len(ms) == 200
    ready = 0
    for mid in ms:
        ns = cohort_states[(mid, 41)].nodes[PRIMARY_NODE]
        if ns.ready:
            ready += 1
            assert ns.value == ns.score.score
            assert isinstance(ns.value, float) and math.isfinite(ns.value)
            assert 0.0 <= ns.value <= 100.0
            assert cohort_states[(mid, 41)].surfaced()[PRIMARY_NODE] == ns.value
    assert ready >= 190, ready


def test_i6_no_number_unless_ready(cohort_states):
    for st in cohort_states.values():
        sf = st.surfaced()
        for code, ns in st.nodes.items():
            if not ns.ready:
                assert ns.value is None, (st.member_id, code)
                assert sf[code] is None
            else:
                assert ns.score is not None and ns.value == ns.score.score
        for code in UNMAPPED_CORE:
            assert st.nodes[code].ready is False and sf[code] is None


# --------------------------------------------------------------------------- #
# functional/lhm-d14 (sealed property)
# --------------------------------------------------------------------------- #

_CODES = sorted(ACTION_CODES)
_PLUS9 = timezone(timedelta(hours=9))
_MINUS4 = timezone(timedelta(hours=-4))


def action_stream(mid: str, as_of: datetime, tag: str):
    """A varied, deterministic stream of ActionEvents: every code, many times
    (before START, inside and outside the LHM window, exactly at as_of, after as_of,
    sub-second offsets, non-UTC offsets), plus noise for other members."""
    times = [
        START_DATE - timedelta(days=30),
        START_DATE,
        START_DATE + timedelta(hours=1),
        as_of - timedelta(days=40),
        as_of - timedelta(days=27, hours=3),
        as_of - timedelta(days=3, minutes=7),
        as_of - timedelta(microseconds=1),
        as_of,
        as_of + timedelta(microseconds=1),
        as_of + timedelta(days=5),
        (as_of - timedelta(days=2)).astimezone(_PLUS9),
        (as_of - timedelta(days=9, hours=11)).astimezone(_MINUS4),
    ]
    out = []
    i = 0
    for t in times:
        for j, code in enumerate(_CODES):
            # several events per (time, code) for some codes; same instant repeated
            for k in range(1 + (j % 3)):
                out.append(act(f"{tag}-{mid}-{i}", mid, t + timedelta(seconds=k), code))
                i += 1
    # 30 consecutive daily reminders, like a real engine schedule
    for n in range(30):
        out.append(act(f"{tag}-{mid}-daily-{n}", mid, as_of - timedelta(days=n, hours=2), "act:reminder_sent"))
    other = "syn-174" if mid != "syn-174" else "syn-173"
    out.append(act(f"{tag}-other-{mid}", other, as_of - timedelta(days=1), "act:goal_set"))
    return out


D14_CASES = [(mid, d) for mid in ("cur-001", "cur-020", "syn-001", "syn-099", "syn-175", UNKNOWN_MEMBER)
             for d in (0, 4, 15, 41, 90)]


@pytest.mark.parametrize("mid,d", D14_CASES)
async def test_d14_actions_never_change_nodes_or_signals(adapter, mid, d):
    as_of = day(d) + timedelta(hours=13)
    base = await state(adapter, mid, as_of)
    stream = action_stream(mid, as_of, "x")
    log = EventLog()
    log.extend(stream)
    with_actions = await state(adapter, mid, as_of, events=log)
    assert dict(with_actions.nodes) == dict(base.nodes)
    assert with_actions.signals == base.signals
    assert with_actions.ready_codes() == base.ready_codes()
    assert with_actions.surfaced() == base.surfaced()
    # only alignment changes, and exactly per the contract
    o = core()
    expected = {e.event_id: alignment_score(e, o) for e in stream if e.member_id == mid and e.occurred_at <= as_of}
    assert dict(with_actions.alignment) == pytest.approx(expected)
    assert dict(base.alignment) == {}


@pytest.mark.parametrize("mid", ["cur-003", "syn-042"])
async def test_d14_actions_on_top_of_signals(adapter, mid):
    as_of = day(41)
    sigs = [sig(f"s-{mid}-{n}", mid, as_of - timedelta(days=n, hours=5), PRIMARY_NODE, float(n % 2))
            for n in range(10)]
    sigs += [sig(f"h-{mid}-{n}", mid, as_of - timedelta(days=n, hours=1), "8867-4", 60.0 + n) for n in range(10)]
    only = EventLog()
    only.extend(sigs)
    both = EventLog()
    both.extend(sigs)
    both.extend(action_stream(mid, as_of, "y"))
    a = await state(adapter, mid, as_of, events=only)
    b = await state(adapter, mid, as_of, events=both)
    assert dict(a.nodes) == dict(b.nodes)
    assert a.signals == b.signals
    assert set(a.alignment) == set()
    assert len(b.alignment) > 0


async def test_d14_custom_goals_and_extended_ontology(adapter):
    ext = apply_update(core(), OntologyUpdate(proposer="p", rationale="r", set_edges=(
        Edge(source="act:reply_sent", target=PRIMARY_NODE, weight=1.0),
        Edge(source="act:nudge_sent", target=CHECKIN, weight=-1.0),
    )))
    as_of = day(30)
    stream = action_stream("syn-020", as_of, "z")
    log = EventLog(ext)
    log.extend(stream)
    goals = (PRIMARY_NODE, CHECKIN)
    base = await state(adapter, "syn-020", as_of, ontology=ext, events=EventLog(ext), goal_nodes=goals)
    st = await state(adapter, "syn-020", as_of, ontology=ext, events=log, goal_nodes=goals)
    assert dict(st.nodes) == dict(base.nodes) and st.signals == base.signals
    expected = {e.event_id: alignment_score(e, ext, goals) for e in stream
                if e.member_id == "syn-020" and e.occurred_at <= as_of}
    assert dict(st.alignment) == pytest.approx(expected)
    assert st.ontology_version == "core-v1+u1"


# --------------------------------------------------------------------------- #
# integration/lhm-signals (SignalEvents feed nodes; adapter view feeds `signals`)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k", [1, 3, 9])
async def test_signal_events_change_n_effective_by_in_window_count(adapter, k):
    mid, as_of = "cur-004", day(41)
    base = await state(adapter, mid, as_of)
    log = EventLog()
    for n in range(k):  # in window (last 28 calendar days, <= as_of), off the series' midnight stamps
        log.append(sig(f"in-{n}", mid, as_of - timedelta(days=n + 1, hours=-7), PRIMARY_NODE, 1.0))
    log.append(sig("at-as-of", mid, as_of, PRIMARY_NODE, 0.0))  # inclusive
    log.append(sig("too-old", mid, as_of - timedelta(days=35), PRIMARY_NODE, 1.0))  # outside window
    log.append(sig("future", mid, as_of + timedelta(days=1), PRIMARY_NODE, 1.0))  # after as_of
    log.append(sig("chk", mid, as_of - timedelta(hours=3), CHECKIN, 0.5))
    st = await state(adapter, mid, as_of, events=log)
    assert st.nodes[PRIMARY_NODE].score.n_effective == base.nodes[PRIMARY_NODE].score.n_effective + k + 1
    assert st.nodes[CHECKIN].score.n_effective == base.nodes[CHECKIN].score.n_effective + 1
    assert st.signals == base.signals
    for code in UNMAPPED_CORE + ("dp:nutrition.food_log", "dp:symptom.severity"):
        assert st.nodes[code] == base.nodes[code]


async def test_signal_events_only_member_scored_per_contract(adapter):
    mid, as_of = UNKNOWN_MEMBER, day(41)
    log = EventLog()
    pts = []
    for n in range(28):
        t = as_of - timedelta(days=n)
        v = float(n % 2)
        log.append(sig(f"p-{n}", mid, t, PRIMARY_NODE, v))
        pts.append((t, v))
    # signals for another member must not leak in
    log.append(sig("other", "syn-002", as_of, PRIMARY_NODE, 1.0))
    st = await state(adapter, mid, as_of, events=log)
    expected = score_domain("consistency", pts, as_of=as_of)
    assert st.nodes[PRIMARY_NODE].score == expected
    assert st.nodes[PRIMARY_NODE].ready is is_ready(expected, "consistency", claim_gate_passed=True,
                                                    valence_gate_passed=valence_ok(expected))
    assert st.signals == ()


async def test_unmapped_core_nodes_never_ready_with_abundant_signals(adapter):
    mid, as_of = "syn-033", day(60)
    log = EventLog()
    for n in range(60):
        t = as_of - timedelta(days=n)
        log.append(sig(f"sl-{n}", mid, t, "93832-4", 7.0 + (n % 3) * 0.5))
        log.append(sig(f"hr-{n}", mid, t, "8867-4", 60.0 + n % 7))
        log.append(sig(f"st-{n}", mid, t, "55423-8", 5000.0 + 100 * (n % 5)))
        log.append(sig(f"pn-{n}", mid, t, "72514-3", float(n % 4)))
    st = await state(adapter, mid, as_of, events=log)
    for code in ("93832-4", "8867-4", "55423-8", "72514-3"):
        assert st.nodes[code] == NodeState(code, None, score=None, ready=False, value=None)
        assert st.surfaced()[code] is None
        assert code not in st.ready_codes()


def _reference_signals(adp, mid, as_of):
    async def run():
        return await analyze(await fetch(RestrictedView(adp, as_of), mid))

    return run()


@pytest.mark.parametrize("mid", SAMPLE_MEMBERS)
@pytest.mark.parametrize("d", [12, 20, 41, 100])
async def test_signals_match_restricted_adapter_view(adapter, cohort, mid, d):
    fc = first_checkin_day(cohort, mid)
    if fc is None or fc > d:
        pytest.skip("restricted view would drop every check-in point (unspecified shape)")
    as_of = day(d) + timedelta(hours=2)
    st = await state(adapter, mid, as_of)
    assert st.signals == await _reference_signals(adapter, mid, as_of)


async def test_signals_full_history_equals_unrestricted(adapter):
    as_of = day(365)
    for mid in ("cur-007", "syn-120"):
        st = await state(adapter, mid, as_of)
        assert st.signals == await analyze(await fetch(adapter, mid))


async def test_signals_depend_on_as_of(adapter, cohort):
    mid = next(m for m in members(cohort) if (first_checkin_day(cohort, m) or 99) <= 3)
    early = await state(adapter, mid, day(12))
    late = await state(adapter, mid, day(41))
    assert early.signals == await _reference_signals(adapter, mid, day(12))
    assert late.signals == await _reference_signals(adapter, mid, day(41))


# --------------------------------------------------------------------------- #
# functional/lhm-g3 (via ontology extension)
# --------------------------------------------------------------------------- #


async def test_g3_via_extension(adapter):
    mapped = Node(code="dp:behavior.walk_logged", system=LOCAL_SYSTEM, display="Walk logged", kind="metric",
                  parent="dp:behavior", lhm_domain="consistency", value_range=(0.0, 1.0),
                  label="Walk logged", caveats=CORE_CAVEATS)
    unmapped = Node(code="dp:behavior.stretch_logged", system=LOCAL_SYSTEM, display="Stretch logged",
                    kind="metric", parent="dp:behavior", lhm_domain=None, value_range=(0.0, 1.0),
                    label="Stretch logged", caveats=CORE_CAVEATS)
    ext = apply_update(core(), OntologyUpdate(proposer="test-author", rationale="g3", add_nodes=(mapped, unmapped)))
    assert ext.version == "core-v1+u1"
    as_of = day(41) + timedelta(hours=20)
    for mid in ("syn-061", UNKNOWN_MEMBER):
        log = EventLog(ext)
        for n in range(28):
            t = as_of - timedelta(days=n, hours=1)
            v = 1.0 if n % 2 == 0 else 0.0
            log.append(sig(f"w-{mid}-{n}", mid, t, mapped.code, v))
            log.append(sig(f"s-{mid}-{n}", mid, t, unmapped.code, v))
        st = await state(adapter, mid, as_of, ontology=ext, events=log)
        assert st.ontology_version == "core-v1+u1"
        assert set(st.nodes) == {n.code for n in ext.metric_nodes()}
        m = st.nodes[mapped.code]
        assert m.lhm_domain == "consistency"
        assert m.score is not None and m.score.n_effective == 28
        assert m.ready is True
        assert m.value == m.score.score and 0.0 <= m.value <= 100.0
        assert mapped.code in st.ready_codes()
        u = st.nodes[unmapped.code]
        assert u == NodeState(unmapped.code, None, score=None, ready=False, value=None)
        assert st.surfaced()[unmapped.code] is None


async def test_default_events_log_uses_given_ontology(adapter):
    ext = apply_update(core(), OntologyUpdate(proposer="p", rationale="r", add_nodes=(
        Node(code="dp:behavior.walk_logged", system=LOCAL_SYSTEM, display="Walk logged", kind="metric",
             parent="dp:behavior", lhm_domain="consistency", value_range=(0.0, 1.0), label="Walk logged",
             caveats=CORE_CAVEATS),)))
    st = await state(adapter, "syn-001", day(20), ontology=ext)
    assert st.ontology_version == "core-v1+u1"
    ns = st.nodes["dp:behavior.walk_logged"]
    assert ns.ready is False and ns.value is None and ns.score.n_effective == 0
    assert dict(st.alignment) == {}


# --------------------------------------------------------------------------- #
# functional/lhm-identity (I7)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", ["alice@example.com", "cur-1", "CUR-001", "syn-0001", "", "member 1",
                                 "syn-001@example.com", "Jane Doe"])
async def test_non_opaque_member_id_rejected_before_adapter(adapter, bad):
    spy = SpyAdapter(adapter)
    with pytest.raises(ValueError):
        await MemoryLhm(spy).metric_state(bad, day(20))
    assert spy.scopes == []
    assert spy.notes == []


async def test_adapter_only_ever_receives_member_id(adapter):
    spy = SpyAdapter(adapter)
    lhm = MemoryLhm(spy)
    for mid in SAMPLE_MEMBERS + (UNKNOWN_MEMBER,):
        spy.scopes.clear()
        await lhm.metric_state(mid, day(41))
        assert spy.scopes, mid
        assert set(spy.scopes) == {mid}
    for bad in ("bob@example.org", "cur-01"):
        spy.scopes.clear()
        with pytest.raises(ValueError):
            await lhm.metric_state(bad, day(41))
        assert spy.scopes == []


async def test_spy_with_events_still_only_member_id(adapter):
    spy = SpyAdapter(adapter)
    log = EventLog()
    log.extend(action_stream("cur-009", day(30), "spy"))
    log.append(sig("spy-s", "cur-009", day(29), PRIMARY_NODE, 1.0))
    await MemoryLhm(spy, events=log).metric_state("cur-009", day(30))
    assert set(spy.scopes) == {"cur-009"}


# --------------------------------------------------------------------------- #
# functional/lhm-determinism
# --------------------------------------------------------------------------- #


async def test_deterministic_equal_inputs_equal_states(adapter):
    as_of = day(33)
    log = EventLog()
    log.extend(action_stream("syn-077", as_of, "det"))
    log.append(sig("det-s", "syn-077", day(32), CHECKIN, -0.25))
    lhm = MemoryLhm(adapter, events=log)
    a = await lhm.metric_state("syn-077", as_of)
    b = await lhm.metric_state("syn-077", as_of)
    c = await MemoryLhm(adapter, events=log).metric_state("syn-077", as_of)
    for x in (b, c):
        assert x.member_id == a.member_id and x.as_of == a.as_of
        assert x.ontology_version == a.ontology_version
        assert dict(x.nodes) == dict(a.nodes)
        assert x.signals == a.signals
        assert dict(x.alignment) == dict(a.alignment)
        assert dict(x.guard_versions) == dict(a.guard_versions)
    assert a == b


async def test_fresh_cohort_adapter_gives_equal_state(adapter):
    from daypart.simmember.cohort import generate
    from daypart.simmember.lhm_records import SimMemberLhmAdapter

    other = SimMemberLhmAdapter(generate(0))
    a = await state(adapter, "cur-011", day(41))
    b = await state(other, "cur-011", day(41))
    assert dict(a.nodes) == dict(b.nodes) and a.signals == b.signals
