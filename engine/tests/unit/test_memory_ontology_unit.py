import math

import pydantic
import pytest
from syntropy.lhm.metrics.claims import assert_claim_safe
from syntropy.lhm.metrics.scoring import V1_DOMAINS

from daypart.memory.ontology import (
    CORE_CAVEATS,
    CORE_VERSION,
    LOCAL_SYSTEM,
    LOINC_SYSTEM,
    Edge,
    Node,
    OntologyGateError,
    OntologyUpdate,
    apply_update,
    core,
)

LOINC = {"72514-3", "44261-6", "70274-6", "8867-4", "29463-7", "55423-8", "93832-4"}


def metric(code="dp:behavior.extra", parent="dp:behavior", domain=None, label="Extra marker", caveats=CORE_CAVEATS):
    return Node(code=code, system=LOCAL_SYSTEM, display="Extra", kind="metric", parent=parent,
                lhm_domain=domain, value_range=(0.0, 1.0), label=label, caveats=caveats)


def upd(**kw):
    return OntologyUpdate(proposer="tester", rationale="unit", **kw)


def test_core_shape():
    o = core()
    assert o.version == CORE_VERSION and o.history == ()
    assert o == core() and o.digest() == core().digest()
    assert len(o.nodes) == 18
    cats = [n for n in o.nodes.values() if n.kind == "category"]
    assert len(cats) == 7 and all(n.parent is None and n.system == LOCAL_SYSTEM for n in cats)
    ms = o.metric_nodes()
    assert [n.code for n in ms] == sorted(n.code for n in ms) and len(ms) == 11
    assert {n.code for n in ms if n.system == LOINC_SYSTEM} == LOINC
    assert all(o.node(c).lhm_domain is None for c in LOINC)
    mapped = [n.lhm_domain for n in ms if n.lhm_domain]
    assert sorted(mapped) == sorted(V1_DOMAINS)
    for n in ms:
        assert n.caveats == CORE_CAVEATS and n.label
        assert_claim_safe(n.label, n.caveats, "low", False)
        assert o.is_core(n.code)
    assert o.node("dp:symptom.severity").value_range == (1.0, 3.0)
    assert o.node("8867-4").value_range == (20.0, 250.0)
    assert o.node("72514-3").parent == "dp:symptom"
    with pytest.raises(KeyError):
        o.node("nope")
    with pytest.raises(TypeError):
        o.nodes["x"] = metric()  # type: ignore[index]
    assert not o.is_core("nope")


def test_core_edges():
    o = core()
    assert len(o.edges) == 10
    assert [(e.source, e.target) for e in o.edges] == sorted((e.source, e.target) for e in o.edges)
    assert o.weight("act:reminder_sent", "dp:adherence.protocol") == 0.3
    assert o.weight("72514-3", "dp:affect.checkin_sentiment") == -0.3
    assert o.weight("act:reply_sent", "dp:adherence.protocol") is None


@pytest.mark.parametrize(
    "kw",
    [
        dict(code="dp:x", system=LOINC_SYSTEM),
        dict(code="1234-5", system=LOCAL_SYSTEM),
        dict(code="dp:Bad", system=LOCAL_SYSTEM),
        dict(lhm_domain="mood"),
        dict(value_range=None),
        dict(value_range=(1.0, 1.0)),
        dict(value_range=(0.0, math.inf)),
        dict(parent=None),
        dict(display=""),
        dict(extra_field=1),
    ],
)
def test_node_validation(kw):
    base = dict(code="dp:behavior.extra", system=LOCAL_SYSTEM, display="Extra", kind="metric",
                parent="dp:behavior", value_range=(0.0, 1.0))
    with pytest.raises(pydantic.ValidationError):
        Node(**{**base, **kw})


def test_category_validation():
    with pytest.raises(pydantic.ValidationError):
        Node(code="dp:cat", system=LOCAL_SYSTEM, display="C", kind="category", parent=None, lhm_domain="consistency")
    with pytest.raises(pydantic.ValidationError):
        Node(code="dp:cat", system=LOCAL_SYSTEM, display="C", kind="category", parent=None, value_range=(0, 1))


@pytest.mark.parametrize("w", [math.nan, math.inf, 1.01, -1.5])
def test_edge_weight(w):
    with pytest.raises(pydantic.ValidationError):
        Edge(source="a", target="b", weight=w)


def test_edge_weight_required():
    with pytest.raises(pydantic.ValidationError):
        Edge(source="a", target="b")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "update",
    [
        lambda: upd(),
        lambda: upd(add_nodes=(metric(code="dp:adherence.protocol"),)),
        lambda: upd(add_nodes=(metric(), metric())),
        lambda: upd(add_nodes=(metric(parent="dp:nowhere"),)),
        lambda: upd(add_nodes=(metric(caveats=()),)),
        lambda: upd(add_nodes=(metric(label="Helps improve sleep"),)),
        lambda: upd(add_nodes=(Node(  # empty label → display is gated
            code="dp:behavior.extra", system=LOCAL_SYSTEM, display="Clinical marker", kind="metric",
            parent="dp:behavior", value_range=(0.0, 1.0), caveats=CORE_CAVEATS),)),
        lambda: upd(set_edges=(Edge(source="act:goal_set", target="dp:behavior", weight=0.1),)),
        lambda: upd(set_edges=(Edge(source="act:unknown", target="dp:adherence.protocol", weight=0.1),)),
        lambda: upd(set_edges=(Edge(source="dp:sleep", target="dp:adherence.protocol", weight=0.1),)),
        lambda: upd(set_edges=(Edge(source="act:goal_set", target="dp:adherence.protocol", weight=0.1),
                               Edge(source="act:goal_set", target="dp:adherence.protocol", weight=0.2))),
    ],
)
def test_gate_rejects(update):
    o = core()
    d = o.digest()
    with pytest.raises(OntologyGateError):
        apply_update(o, update())
    assert o.digest() == d and o == core()


def test_gate_error_is_value_error():
    assert issubclass(OntologyGateError, ValueError)


def test_update_empty_proposer():
    with pytest.raises(pydantic.ValidationError):
        OntologyUpdate(proposer="", rationale="x")


def test_apply_update_success():
    o = core()
    parent = Node(code="dp:behavior.sub", system=LOCAL_SYSTEM, display="Sub", kind="category", parent="dp:behavior")
    child = metric(code="dp:behavior.sub.x", parent="dp:behavior.sub", domain="consistency")
    u = upd(add_nodes=(parent, child), set_edges=(
        Edge(source="act:goal_set", target="dp:behavior.sub.x", weight=0.5),
        Edge(source="act:reminder_sent", target="dp:adherence.protocol", weight=-0.4),
    ))
    o1 = apply_update(o, u)
    assert o1.version == "core-v1+u1" and o.version == CORE_VERSION
    assert len(o1.history) == 1
    rec = o1.history[0]
    assert (rec.from_version, rec.to_version, rec.proposer, rec.gate) == (CORE_VERSION, "core-v1+u1", "tester", "passed")
    assert len(rec.update_digest) == 64
    assert not o1.is_core("dp:behavior.sub.x") and o1.is_core("dp:adherence.protocol")
    assert o1.weight("act:reminder_sent", "dp:adherence.protocol") == -0.4
    assert o.weight("act:reminder_sent", "dp:adherence.protocol") == 0.3
    assert o1.weight("act:goal_set", "dp:behavior.sub.x") == 0.5
    assert len(o1.edges) == 11
    for code, n in o.nodes.items():
        assert o1.node(code) == n
    assert apply_update(o, u) == o1 and apply_update(o, u).digest() == o1.digest()
    o2 = apply_update(o1, upd(add_nodes=(metric(code="dp:behavior.y"),)))
    assert o2.version == "core-v1+u2" and o2.history[0] == rec and o2.history[1].from_version == "core-v1+u1"
    assert o2.digest() != o1.digest() != o.digest()


def test_edge_may_reference_node_added_in_same_update():
    o1 = apply_update(core(), upd(add_nodes=(metric(),),
                                  set_edges=(Edge(source="dp:behavior.extra", target="dp:adherence.protocol", weight=0.1),)))
    assert o1.weight("dp:behavior.extra", "dp:adherence.protocol") == 0.1
