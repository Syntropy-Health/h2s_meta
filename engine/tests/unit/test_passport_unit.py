import json
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

import pydantic
import pytest
from syntropy.lhm.contracts.viz import VizPayload, VizPoint
from syntropy.lhm.metrics.scoring import DomainScore

from daypart.memory.lhm import GUARD_VERSIONS, MemberMetricState, MemoryLhm, NodeState
from daypart.memory.ontology import CORE_CAVEATS, Node, core
from daypart.passport import (
    FALLBACK_TITLE,
    PASSPORT_TITLE,
    PENDING,
    WITHHELD_MESSAGE,
    Flag,
    Passport,
    PassportRenderer,
    flags_from_state,
    metric_key,
    passport_from_state,
)
from daypart.simmember.cohort import generate
from daypart.simmember.lhm_records import SimMemberLhmAdapter
from daypart.simmember.trajectory import START_DATE

AS_OF = datetime(2026, 10, 1, tzinfo=timezone.utc)
CORE4 = ["dp:adherence.protocol", "dp:affect.checkin_sentiment", "dp:nutrition.food_log", "dp:symptom.severity"]


def _payload(label="Protocol adherence (logged)", value=50.0):
    return VizPayload(metric_key="dp_adherence_protocol", label=label, series=(VizPoint(ts=AS_OF, value=value),),
                      confidence="low", caveats=CORE_CAVEATS, rx_input=False)


def _ready_kw(**over):
    kw = dict(code="dp:adherence.protocol", lhm_domain="consistency", title="Protocol adherence (logged)",
              status="ready", value=50.0, interval=(40.0, 60.0), caveats=CORE_CAVEATS, message=None,
              payload=_payload())
    kw.update(over)
    return kw


def _withheld_kw(**over):
    kw = dict(code="dp:adherence.protocol", lhm_domain="consistency", title="Protocol adherence (logged)",
              status="withheld", value=None, interval=None, caveats=(), message=WITHHELD_MESSAGE, payload=None)
    kw.update(over)
    return kw


def _state(nodes, ontology_version="core-v1"):
    return MemberMetricState(
        member_id="syn-001", as_of=AS_OF, ontology_version=ontology_version, nodes=MappingProxyType(nodes),
        signals=(), alignment=MappingProxyType({}), guard_versions=MappingProxyType(dict(GUARD_VERSIONS)),
    )


def _ns(code, domain, ready, value=55.5, interval=(45.0, 65.0)):
    if domain is None:
        return NodeState(code, None, None, False, None)
    score = DomainScore(domain=domain, score=value, interval=interval, n_effective=10)
    return NodeState(code, domain, score, ready, value if ready else None)


def _core_state(ready_codes=(), value=55.5):
    ont = core()
    nodes = {}
    for n in ont.metric_nodes():
        nodes[n.code] = _ns(n.code, n.lhm_domain, n.code in ready_codes, value)
    return _state(nodes)


# --------------------------------------------------------------------------- Flag model


def test_constants():
    assert (PASSPORT_TITLE, WITHHELD_MESSAGE, FALLBACK_TITLE, PENDING) == (
        "Pattern passport", "Not enough data yet.", "Personal pattern", ("R10", "R11"))


def test_flag_valid_and_roundtrip():
    for f in (Flag(**_ready_kw()), Flag(**_withheld_kw()), Flag(**_ready_kw(interval=None))):
        assert Flag.model_validate_json(f.model_dump_json()) == f
        assert f.pattern is None and f.citation is None and f.pending == PENDING


@pytest.mark.parametrize(
    "kw",
    [
        _ready_kw(value=None), _ready_kw(value=float("nan")), _ready_kw(value=float("inf")),
        _ready_kw(message=WITHHELD_MESSAGE), _ready_kw(caveats=()), _ready_kw(payload=None),
        _withheld_kw(value=1.0), _withheld_kw(interval=(0.0, 1.0)), _withheld_kw(payload=_payload()),
        _withheld_kw(caveats=CORE_CAVEATS), _withheld_kw(message=None), _withheld_kw(message="Soon."),
        _ready_kw(title="This will cure you"), _ready_kw(caveats=(*CORE_CAVEATS, "We recommend more.")),
        _withheld_kw(title="You should take more"), _ready_kw(pattern="x"), _ready_kw(citation="x"),
        _ready_kw(extra_field=1), _ready_kw(status="maybe"),
    ],
)
def test_flag_invalid(kw):
    with pytest.raises(pydantic.ValidationError):
        Flag(**kw)


def test_flag_frozen():
    f = Flag(**_withheld_kw())
    with pytest.raises(pydantic.ValidationError):
        f.title = "x"


def test_metric_key():
    assert metric_key("dp:adherence.protocol") == "dp_adherence_protocol"
    assert metric_key("72514-3") == "m_72514_3"
    assert metric_key("Dp:X") == "dp_x"


# --------------------------------------------------------------------------- flags_from_state


def test_all_withheld_core():
    flags = flags_from_state(_core_state())
    assert [f.code for f in flags] == CORE4
    for f in flags:
        assert f.status == "withheld" and f.value is None and f.interval is None and f.payload is None
        assert f.message == WITHHELD_MESSAGE and f.caveats == ()


def test_ready_card():
    flags = flags_from_state(_core_state(ready_codes={"dp:adherence.protocol"}, value=62.25))
    f = flags[0]
    node = core().node("dp:adherence.protocol")
    assert f.status == "ready" and f.value == 62.25 and f.interval == (45.0, 65.0)
    assert f.caveats == node.caveats and f.title == node.label and f.lhm_domain == "consistency"
    assert f.payload == VizPayload(contract="lhm-viz/v1", metric_key="dp_adherence_protocol", label=node.label,
                                   series=(VizPoint(ts=AS_OF, value=62.25),), regimen_start=None,
                                   confidence="low", caveats=node.caveats, rx_input=False)
    assert all(g.status == "withheld" for g in flags[1:])
    assert flags == flags_from_state(_core_state(ready_codes={"dp:adherence.protocol"}, value=62.25))


def test_unmapped_nodes_hidden_and_unknown_code():
    nodes = {
        "72514-3": _ns("72514-3", None, False),
        "dp:zzz.other": _ns("dp:zzz.other", "consistency", True),
    }
    flags = flags_from_state(_state(nodes))
    assert [f.code for f in flags] == ["dp:zzz.other"]
    assert flags[0].status == "withheld" and flags[0].title == FALLBACK_TITLE


def _ontology_with(label, caveats):
    node = Node(code="dp:behavior.extra", system="urn:daypart:metric", display="Extra", kind="metric",
                parent="dp:behavior", lhm_domain="consistency", value_range=(0.0, 1.0), label=label,
                caveats=caveats)
    # bypass the I9 gate: build the Ontology directly (the gate would reject unsafe text)
    from daypart.memory.ontology import Ontology

    base = core()
    return Ontology("core-v1+t", (*base.nodes.values(), node), base.edges)


def test_unsafe_title_falls_back():
    ont = _ontology_with("This will cure you", CORE_CAVEATS)
    flags = flags_from_state(_state({"dp:behavior.extra": _ns("dp:behavior.extra", "consistency", True)}), ont)
    assert flags[0].title == FALLBACK_TITLE
    assert flags[0].status == "ready"  # fallback title still passes the claim gate with safe caveats


@pytest.mark.parametrize("caveats", [(), ("Just numbers.",), ("We recommend this. n-of-1, uncertain.",)])
def test_bad_caveats_fail_closed(caveats):
    ont = _ontology_with("Extra metric", caveats)
    flags = flags_from_state(_state({"dp:behavior.extra": _ns("dp:behavior.extra", "consistency", True)}), ont)
    assert flags[0].status == "withheld" and flags[0].value is None and flags[0].payload is None


def test_nonfinite_ready_value_fail_closed():
    st = _state({"dp:adherence.protocol": _ns("dp:adherence.protocol", "consistency", True, value=float("nan"))})
    assert flags_from_state(st)[0].status == "withheld"


def test_passport_from_state():
    st = _core_state(ready_codes={"dp:symptom.severity"})
    p = passport_from_state(st)
    assert isinstance(p, Passport) and p.title == "Pattern passport"
    assert p.member_id == "syn-001" and p.as_of == AS_OF and p.ontology_version == "core-v1"
    assert dict(p.guard_versions) == dict(st.guard_versions)
    assert p.flags == flags_from_state(st)
    data = json.loads(p.model_dump_json())
    assert all(f["value"] is None for f in data["flags"] if f["status"] == "withheld")
    assert Passport.model_validate_json(p.model_dump_json()) == p


# --------------------------------------------------------------------------- renderer (real cohort, small)


@pytest.fixture(scope="module")
def renderer():
    return PassportRenderer(MemoryLhm(SimMemberLhmAdapter(generate(0))))


async def test_renderer_early_withheld_late_ready(renderer):
    ids = [p.member_id for p in generate(0).profiles][:12]
    ready = 0
    for mid in ids:
        early = await renderer.render(mid, START_DATE + timedelta(days=4))
        assert [f.code for f in early] == CORE4
        assert early[0].status == "withheld"
        late = await renderer.render(mid, START_DATE + timedelta(days=41))
        if late[0].status == "ready" and 0 <= late[0].value <= 100:
            ready += 1
    assert ready >= 11


async def test_renderer_unknown_and_invalid(renderer):
    flags = await renderer.render("syn-999", START_DATE + timedelta(days=41))
    assert [f.status for f in flags] == ["withheld"] * 4
    with pytest.raises(ValueError):
        await renderer.render("someone@example.com", START_DATE + timedelta(days=41))


async def test_renderer_passport_no_free_text(renderer):
    as_of = START_DATE + timedelta(days=41)
    p = await renderer.passport("syn-003", as_of)
    assert p.title == PASSPORT_TITLE and p.flags == await renderer.render("syn-003", as_of)
    dumped = p.model_dump_json()
    assert "note:" not in dumped
