import asyncio
import math
from datetime import datetime, timedelta

import pytest
from syntropy.lhm.metrics.claims import CLAIMS_GUARD_VERSION, VALENCE_GUARD_VERSION

from daypart.memory.events import ACTION_CODES, ActionEvent, EventLog, SignalEvent
from daypart.memory.lhm import (
    GUARD_VERSIONS,
    PRIMARY_NODE,
    SERIES_NODES,
    MemoryLhm,
    alignment_score,
)
from daypart.memory.ontology import CORE_CAVEATS, LOCAL_SYSTEM, Node, OntologyUpdate, apply_update, core
from daypart.simmember.cohort import generate
from daypart.simmember.lhm_records import SimMemberLhmAdapter
from daypart.simmember.trajectory import START_DATE

D4 = START_DATE + timedelta(days=4)
D41 = START_DATE + timedelta(days=41)


@pytest.fixture(scope="module")
def cohort0():
    return generate(0)


@pytest.fixture(scope="module")
def adapter(cohort0):
    return SimMemberLhmAdapter(cohort0)


class Spy:
    """Adapter wrapper recording every user_scope it receives."""

    def __init__(self, inner):
        self.inner, self.scopes = inner, []

    async def read_orders(self, s):
        self.scopes.append(s)
        return await self.inner.read_orders(s)

    async def read_regimen_starts(self, s):
        self.scopes.append(s)
        return await self.inner.read_regimen_starts(s)

    async def read_outcome_series(self, s):
        self.scopes.append(s)
        return await self.inner.read_outcome_series(s)

    def sentiment_fn(self, note):
        return self.inner.sentiment_fn(note)


def state(adapter, mid, as_of, **kw):
    return asyncio.run(MemoryLhm(adapter, **kw).metric_state(mid, as_of))


def test_constants():
    assert PRIMARY_NODE == "dp:adherence.protocol"
    assert GUARD_VERSIONS == {"claims": CLAIMS_GUARD_VERSION, "valence": VALENCE_GUARD_VERSION}
    assert set(SERIES_NODES.values()) <= set(core().nodes)


def test_alignment_score():
    o = core()
    e = ActionEvent(event_id="a", member_id="syn-001", occurred_at=START_DATE, code="act:reminder_sent")
    assert alignment_score(e, o) == 0.3
    assert alignment_score(e, o, ()) == 0.0
    assert alignment_score(e, o, (PRIMARY_NODE, "dp:affect.checkin_sentiment")) == pytest.approx(0.15)
    r = ActionEvent(event_id="b", member_id="syn-001", occurred_at=START_DATE, code="act:reply_sent")
    assert alignment_score(r, o) == 0.0


def test_readiness_5_vs_42_days(cohort0, adapter):
    mlhm = MemoryLhm(adapter)

    async def go():
        ready42 = 0
        for p in cohort0.profiles:
            s5 = await mlhm.metric_state(p.member_id, D4)
            n5 = s5.nodes[PRIMARY_NODE]
            assert not n5.ready and n5.value is None
            s42 = await mlhm.metric_state(p.member_id, D41)
            n42 = s42.nodes[PRIMARY_NODE]
            if n42.ready:
                ready42 += 1
                assert n42.value == n42.score.score and 0.0 <= n42.value <= 100.0
            for st in (*s5.nodes.values(), *s42.nodes.values()):
                if not st.ready:
                    assert st.value is None
            assert all(v is None for c, v in s42.surfaced().items() if not s42.nodes[c].ready)
        return ready42

    assert asyncio.run(go()) >= 0.95 * len(cohort0.profiles)


def test_state_shape_and_unmapped_never_ready(adapter):
    s = state(adapter, "syn-001", D41)
    o = core()
    assert set(s.nodes) == {n.code for n in o.metric_nodes()}
    assert s.ontology_version == o.version and s.guard_versions == GUARD_VERSIONS
    assert s.member_id == "syn-001" and s.as_of == D41
    for n in o.metric_nodes():
        st = s.nodes[n.code]
        assert st.lhm_domain == n.lhm_domain
        if n.lhm_domain is None:
            assert st.score is None and not st.ready and st.value is None
    assert PRIMARY_NODE in s.ready_codes() and list(s.ready_codes()) == sorted(s.ready_codes())
    assert len(s.signals) >= 1
    assert state(adapter, "syn-001", D41) == s  # deterministic


def test_unmapped_node_with_abundant_signals_not_ready(adapter):
    log = EventLog()
    log.extend(SignalEvent(event_id=f"hr{i}", member_id="syn-001", occurred_at=START_DATE + timedelta(days=i),
                           code="8867-4", value=60.0) for i in range(42))
    st = state(adapter, "syn-001", D41, events=log).nodes["8867-4"]
    assert st.score is None and not st.ready and st.value is None


def test_signal_event_adds_to_n_effective(adapter):
    base = state(adapter, "syn-002", D41).nodes[PRIMARY_NODE].score.n_effective
    log = EventLog()
    log.extend([
        SignalEvent(event_id="x1", member_id="syn-002", occurred_at=D41 - timedelta(hours=1), code=PRIMARY_NODE, value=1.0),
        SignalEvent(event_id="x2", member_id="syn-002", occurred_at=D41 - timedelta(days=2), code=PRIMARY_NODE, value=0.0),
        SignalEvent(event_id="x3", member_id="syn-002", occurred_at=D41 - timedelta(days=60), code=PRIMARY_NODE, value=1.0),
        SignalEvent(event_id="x4", member_id="syn-002", occurred_at=D41 + timedelta(days=1), code=PRIMARY_NODE, value=1.0),
    ])
    assert state(adapter, "syn-002", D41, events=log).nodes[PRIMARY_NODE].score.n_effective == base + 2


@pytest.mark.parametrize("mid", ["syn-003", "cur-007"])
@pytest.mark.parametrize("as_of", [D4, D41])
def test_d14_actions_do_not_change_nodes_or_signals(adapter, mid, as_of):
    plain = state(adapter, mid, as_of)
    log = EventLog()
    codes = sorted(ACTION_CODES)
    log.extend(ActionEvent(event_id=f"a{i}", member_id=mid, occurred_at=START_DATE + timedelta(hours=7 * i),
                           code=codes[i % len(codes)]) for i in range(200))
    with_actions = state(adapter, mid, as_of, events=log)
    assert with_actions.nodes == plain.nodes and with_actions.signals == plain.signals
    expected = {e.event_id for e in log.events(mid, as_of=as_of, kind="action")}
    assert set(with_actions.alignment) == expected and plain.alignment == {}
    assert with_actions.alignment["a0"] == alignment_score(log.events(mid)[0], core())


def test_g3_extension(adapter):
    upd = OntologyUpdate(proposer="t", rationale="g3", add_nodes=(
        Node(code="dp:behavior.walk", system=LOCAL_SYSTEM, display="Walk logged", kind="metric", parent="dp:behavior",
             lhm_domain="consistency", value_range=(0.0, 1.0), label="Walk logged", caveats=CORE_CAVEATS),
        Node(code="dp:behavior.stretch", system=LOCAL_SYSTEM, display="Stretch logged", kind="metric",
             parent="dp:behavior", value_range=(0.0, 1.0), label="Stretch logged", caveats=CORE_CAVEATS),
    ))
    ont = apply_update(core(), upd)
    log = EventLog(ont)
    for i in range(28):
        at = D41 - timedelta(days=i)
        for code in ("dp:behavior.walk", "dp:behavior.stretch"):
            log.append(SignalEvent(event_id=f"{code}{i}", member_id="syn-004", occurred_at=at, code=code,
                                   value=float(i % 2)))
    s = state(adapter, "syn-004", D41, ontology=ont, events=log)
    assert s.ontology_version == "core-v1+u1"
    assert s.nodes["dp:behavior.walk"].ready and s.nodes["dp:behavior.walk"].value is not None
    st = s.nodes["dp:behavior.stretch"]
    assert not st.ready and st.score is None and st.value is None


def test_unknown_member_and_early_as_of(adapter):
    s = state(adapter, "syn-999", D41)
    assert s.signals == () and s.ready_codes() == ()
    assert all(v is None for v in s.surfaced().values())
    early = state(adapter, "syn-001", START_DATE - timedelta(days=1))
    assert early.signals == () and early.ready_codes() == ()


def test_as_of_restricts_adapter_view(adapter):
    a = state(adapter, "syn-005", D4)
    b = state(adapter, "syn-005", D41)
    assert a.nodes[PRIMARY_NODE].score.n_effective <= 5
    assert a.signals != b.signals


@pytest.mark.parametrize("mid", ["alice@example.com", "cur-1", "syn-0001", ""])
def test_i7_rejects_non_opaque_ids_before_adapter(adapter, mid):
    spy = Spy(adapter)
    with pytest.raises(ValueError):
        state(spy, mid, D41)
    assert spy.scopes == []


def test_naive_as_of_rejected(adapter):
    with pytest.raises(ValueError):
        state(adapter, "syn-001", datetime(2026, 10, 1))


def test_adapter_only_sees_member_id(adapter):
    spy = Spy(adapter)
    state(spy, "cur-003", D41)
    assert spy.scopes and set(spy.scopes) == {"cur-003"}


def test_values_finite_when_ready(adapter):
    s = state(adapter, "cur-001", D41)
    for code in s.ready_codes():
        assert math.isfinite(s.nodes[code].value)
