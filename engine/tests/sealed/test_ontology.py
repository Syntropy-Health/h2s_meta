"""Sealed challenge suite — slice D2: daypart.memory.ontology (D13, I9, G3).

Contract: docs/daypart/contracts/CONTRACT-slice-D-memory.md §D2, §D2.1, §D2.2.

Categories: functional/ontology-models, functional/ontology-core, functional/ontology-readonly,
functional/ontology-gate, functional/ontology-update, functional/ontology-determinism.

Ambiguities recorded for the principal (not invented behaviour; left untested):
- Exact `display` strings of core nodes (e.g. LOINC display names) are unspecified;
  only non-emptiness is asserted.
- `update_digest` is "sha256 hex of the update's canonical JSON" but the canonicalisation
  is not pinned, so it is not recomputed; only format (64 lowercase hex), equality for
  equal updates and inequality for different updates are asserted.
- An empty OntologyUpdate is listed as a GATE rejection (D2.2 #1), so the model itself
  is assumed constructible with no add_nodes/set_edges; rejection is asserted at
  apply_update as OntologyGateError.
- D2.2 rule 2 rejects an added code that "already exists in ont"; a code duplicated
  WITHIN one update's add_nodes is not covered by any listed rule, so it is untested.
- Mutating a frozen model may raise pydantic.ValidationError, TypeError or
  AttributeError; any is accepted.
"""

from __future__ import annotations

import math
import re

import pydantic
import pytest

from daypart.memory.events import ACTION_CODES
from daypart.memory.ontology import (
    CORE_CAVEATS,
    CORE_VERSION,
    LOCAL_SYSTEM,
    LOINC_SYSTEM,
    Edge,
    Node,
    Ontology,
    OntologyGateError,
    OntologyUpdate,
    UpdateRecord,
    apply_update,
    core,
)
from syntropy.lhm.metrics.claims import ClaimScopeError, assert_claim_safe
from syntropy.lhm.metrics.scoring import V1_DOMAINS
from tests.sealed.conftest_memory import CORE_CAVEATS_EXPECTED

HEX64 = re.compile(r"^[0-9a-f]{64}$")
L, X = "urn:daypart:metric", "http://loinc.org"

CORE_NODES = {
    # code: (system, kind, parent, lhm_domain, value_range)
    "dp:behavior": (L, "category", None, None, None),
    "dp:affect": (L, "category", None, None, None),
    "dp:symptom": (L, "category", None, None, None),
    "dp:nutrition": (L, "category", None, None, None),
    "dp:vital": (L, "category", None, None, None),
    "dp:activity": (L, "category", None, None, None),
    "dp:sleep": (L, "category", None, None, None),
    "dp:adherence.protocol": (L, "metric", "dp:behavior", "consistency", (0.0, 1.0)),
    "dp:affect.checkin_sentiment": (L, "metric", "dp:affect", "sentiment_trajectory", (-1.0, 1.0)),
    "dp:nutrition.food_log": (L, "metric", "dp:nutrition", "nutrition_regularity", (0.0, 1.0)),
    "dp:symptom.severity": (L, "metric", "dp:symptom", "symptom_load", (1.0, 3.0)),
    "72514-3": (X, "metric", "dp:symptom", None, (0.0, 10.0)),
    "44261-6": (X, "metric", "dp:affect", None, (0.0, 27.0)),
    "70274-6": (X, "metric", "dp:affect", None, (0.0, 21.0)),
    "8867-4": (X, "metric", "dp:vital", None, (20.0, 250.0)),
    "29463-7": (X, "metric", "dp:vital", None, (0.0, 500.0)),
    "55423-8": (X, "metric", "dp:activity", None, (0.0, 100000.0)),
    "93832-4": (X, "metric", "dp:sleep", None, (0.0, 24.0)),
}

CORE_EDGES = {
    ("act:reminder_sent", "dp:adherence.protocol"): 0.3,
    ("act:nudge_sent", "dp:adherence.protocol"): 0.2,
    ("act:goal_set", "dp:adherence.protocol"): 0.2,
    ("act:education_viewed", "dp:adherence.protocol"): 0.1,
    ("act:protocol_adjusted", "dp:adherence.protocol"): 0.1,
    ("act:reply_sent", "dp:affect.checkin_sentiment"): 0.1,
    ("93832-4", "dp:affect.checkin_sentiment"): 0.2,
    ("55423-8", "dp:affect.checkin_sentiment"): 0.1,
    ("72514-3", "dp:affect.checkin_sentiment"): -0.3,
    ("dp:symptom.severity", "dp:affect.checkin_sentiment"): -0.3,
}


def metric(code="dp:behavior.walk_logged", parent="dp:behavior", lhm_domain=None, value_range=(0.0, 1.0),
           label="Walk logged", caveats=CORE_CAVEATS_EXPECTED, display="Walk logged", system=L):
    return Node(code=code, system=system, display=display, kind="metric", parent=parent,
                lhm_domain=lhm_domain, value_range=value_range, label=label, caveats=caveats)


def category(code="dp:social", parent=None, display="Social"):
    return Node(code=code, system=L, display=display, kind="category", parent=parent)


def upd(add_nodes=(), set_edges=(), proposer="test-author", rationale="sealed test"):
    return OntologyUpdate(proposer=proposer, rationale=rationale, add_nodes=tuple(add_nodes),
                          set_edges=tuple(set_edges))


def snapshot(o: Ontology):
    return (o.version, o.digest(), dict(o.nodes), o.edges, o.history)


# --------------------------------------------------------------------------- #
# functional/ontology-models
# --------------------------------------------------------------------------- #


def test_constants():
    assert LOINC_SYSTEM == "http://loinc.org"
    assert LOCAL_SYSTEM == "urn:daypart:metric"
    assert CORE_VERSION == "core-v1"
    assert CORE_CAVEATS == CORE_CAVEATS_EXPECTED
    assert issubclass(OntologyGateError, ValueError)


@pytest.mark.parametrize("code", ["1-0", "8867-4", "1234567-8"])
def test_loinc_code_pattern_accepted(code):
    assert metric(code=code, system=X).code == code


@pytest.mark.parametrize("code", ["12345678-9", "8867-45", "8867", "-4", "abc-1", "dp:vital.hr", "8867_4", ""])
def test_loinc_code_pattern_rejected(code):
    with pytest.raises(pydantic.ValidationError):
        metric(code=code, system=X)


@pytest.mark.parametrize("code", ["dp:x", "dp:behavior.walk_logged", "dp:a.b.c_d"])
def test_local_code_pattern_accepted(code):
    assert metric(code=code).code == code


@pytest.mark.parametrize("code", ["dp:", "dp:Walk", "dp:walk.", "dp:.walk", "dp:walk1", "behavior", "8867-4",
                                  "dp:walk-logged", "DP:walk", "dp:walk..x"])
def test_local_code_pattern_rejected(code):
    with pytest.raises(pydantic.ValidationError):
        metric(code=code)


def test_unknown_system_rejected():
    with pytest.raises(pydantic.ValidationError):
        metric(system="http://snomed.info/sct")


@pytest.mark.parametrize("dom", list(V1_DOMAINS))
def test_lhm_domain_v1_accepted(dom):
    assert metric(lhm_domain=dom).lhm_domain == dom


@pytest.mark.parametrize("dom", ["sleep", "", "Consistency", "mood"])
def test_lhm_domain_outside_v1_rejected(dom):
    with pytest.raises(pydantic.ValidationError):
        metric(lhm_domain=dom)


def test_category_rejects_lhm_domain_or_value_range():
    with pytest.raises(pydantic.ValidationError):
        Node(code="dp:x", system=L, display="X", kind="category", parent=None, lhm_domain="consistency")
    with pytest.raises(pydantic.ValidationError):
        Node(code="dp:x", system=L, display="X", kind="category", parent=None, value_range=(0.0, 1.0))
    assert category().value_range is None and category().lhm_domain is None


@pytest.mark.parametrize("vr", [None, (1.0, 1.0), (2.0, 1.0), (0.0, math.inf), (-math.inf, 0.0),
                                (math.nan, 1.0), (0.0, math.nan)])
def test_metric_value_range_rules(vr):
    with pytest.raises(pydantic.ValidationError):
        metric(value_range=vr)


def test_metric_requires_parent():
    with pytest.raises(pydantic.ValidationError):
        metric(parent=None)


def test_display_non_empty():
    with pytest.raises(pydantic.ValidationError):
        metric(display="")


def test_node_extra_forbidden_and_frozen():
    with pytest.raises(pydantic.ValidationError):
        Node(code="dp:x", system=L, display="X", kind="category", parent=None, note="free text")
    n = metric()
    with pytest.raises((pydantic.ValidationError, TypeError, AttributeError)):
        n.parent = "dp:affect"  # type: ignore[misc]


def test_node_kind_literal():
    with pytest.raises(pydantic.ValidationError):
        Node(code="dp:x", system=L, display="X", kind="group", parent=None)


def test_node_defaults():
    n = Node(code="dp:x.y", system=L, display="X", kind="metric", parent="dp:behavior", value_range=(0.0, 1.0))
    assert n.lhm_domain is None and n.label == "" and n.caveats == ()


@pytest.mark.parametrize("w", [-1.0, -0.5, 0.0, 0.3, 1.0])
def test_edge_weight_accepted(w):
    assert Edge(source="act:goal_set", target="dp:adherence.protocol", weight=w).weight == w


@pytest.mark.parametrize("w", [math.nan, math.inf, -math.inf, 1.0001, -1.0001, 2.0])
def test_edge_weight_rejected(w):
    with pytest.raises(pydantic.ValidationError):
        Edge(source="act:goal_set", target="dp:adherence.protocol", weight=w)


def test_edge_weight_required_and_extra_forbidden():
    with pytest.raises(pydantic.ValidationError):
        Edge(source="act:goal_set", target="dp:adherence.protocol")  # type: ignore[call-arg]
    with pytest.raises(pydantic.ValidationError):
        Edge(source="act:goal_set", target="dp:adherence.protocol", weight=0.1, kind="parent")
    e = Edge(source="act:goal_set", target="dp:adherence.protocol", weight=0.1)
    with pytest.raises((pydantic.ValidationError, TypeError, AttributeError)):
        e.weight = 0.5  # type: ignore[misc]


def test_update_model_validation():
    with pytest.raises(pydantic.ValidationError):
        OntologyUpdate(proposer="", rationale="r", add_nodes=(metric(),))
    with pytest.raises(pydantic.ValidationError):
        OntologyUpdate(proposer="p", rationale="", add_nodes=(metric(),))
    with pytest.raises(pydantic.ValidationError):
        OntologyUpdate(proposer="p", rationale="r", add_nodes=(metric(),), remove_nodes=("dp:behavior",))
    u = OntologyUpdate(proposer="p", rationale="r")
    assert u.add_nodes == () and u.set_edges == ()


# --------------------------------------------------------------------------- #
# functional/ontology-core
# --------------------------------------------------------------------------- #


def test_core_version_and_history():
    o = core()
    assert isinstance(o, Ontology)
    assert o.version == CORE_VERSION == "core-v1"
    assert o.history == ()


def test_core_node_table_exact():
    o = core()
    assert set(o.nodes) == set(CORE_NODES)
    for code, (system, kind, parent, dom, vr) in CORE_NODES.items():
        n = o.node(code)
        assert o.nodes[code] == n
        assert n.code == code
        assert n.system == system, code
        assert n.kind == kind, code
        assert n.parent == parent, code
        assert n.lhm_domain == dom, code
        if vr is None:
            assert n.value_range is None, code
        else:
            assert tuple(float(x) for x in n.value_range) == vr, code
        assert n.display, code


def test_each_v1_domain_mapped_exactly_once_loinc_unmapped():
    o = core()
    mapped = [n.lhm_domain for n in o.nodes.values() if n.lhm_domain is not None]
    assert sorted(mapped) == sorted(V1_DOMAINS)
    for n in o.nodes.values():
        if n.system == LOINC_SYSTEM:
            assert n.lhm_domain is None


def test_core_metric_nodes_claim_safe():
    o = core()
    for n in o.metric_nodes():
        assert n.caveats == CORE_CAVEATS, n.code
        assert n.label, n.code
        assert_claim_safe(n.label, n.caveats, "low", False)


def test_core_categories_have_no_metric_payload():
    for n in core().nodes.values():
        if n.kind == "category":
            assert n.value_range is None and n.lhm_domain is None


def test_metric_nodes_sorted_and_complete():
    o = core()
    mn = o.metric_nodes()
    assert isinstance(mn, tuple)
    assert [n.code for n in mn] == sorted(c for c, v in CORE_NODES.items() if v[1] == "metric")
    assert all(n.kind == "metric" for n in mn)
    assert len(mn) == 11


def test_core_edges_exact_and_sorted():
    o = core()
    assert isinstance(o.edges, tuple)
    assert {(e.source, e.target): e.weight for e in o.edges} == pytest.approx(CORE_EDGES)
    assert len(o.edges) == len(CORE_EDGES)
    keys = [(e.source, e.target) for e in o.edges]
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)


def test_core_edge_endpoints_valid():
    o = core()
    for e in o.edges:
        assert o.node(e.target).kind == "metric"
        assert e.source in ACTION_CODES or o.node(e.source).kind == "metric"


def test_weight_lookup():
    o = core()
    for (s, t), w in CORE_EDGES.items():
        assert o.weight(s, t) == pytest.approx(w)
    assert o.weight("act:reply_sent", "dp:adherence.protocol") is None
    assert o.weight("dp:adherence.protocol", "act:reminder_sent") is None
    assert o.weight("nope", "nope") is None


def test_node_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        core().node("dp:nope")


def test_is_core():
    o = core()
    for code in CORE_NODES:
        assert o.is_core(code)
    assert not o.is_core("dp:nope")


def test_no_edges_encode_hierarchy():
    # Hierarchy lives on Node.parent; no edge targets a category.
    o = core()
    cats = {c for c, v in CORE_NODES.items() if v[1] == "category"}
    assert not any(e.source in cats or e.target in cats for e in o.edges)


# --------------------------------------------------------------------------- #
# functional/ontology-readonly
# --------------------------------------------------------------------------- #


def test_nodes_mapping_read_only():
    o = core()
    with pytest.raises(TypeError):
        o.nodes["dp:behavior"] = category(code="dp:behavior")  # type: ignore[index]
    with pytest.raises(TypeError):
        o.nodes["dp:new"] = category(code="dp:new")  # type: ignore[index]
    with pytest.raises(TypeError):
        del o.nodes["dp:behavior"]  # type: ignore[attr-defined]
    assert o.digest() == core().digest()


def test_edges_tuple_immutable():
    o = core()
    with pytest.raises((TypeError, AttributeError)):
        o.edges.append(Edge(source="act:goal_set", target="8867-4", weight=0.1))  # type: ignore[attr-defined]


def test_core_returns_equal_independent_values():
    a, b = core(), core()
    assert a.digest() == b.digest()
    assert HEX64.match(a.digest())
    assert dict(a.nodes) == dict(b.nodes)
    assert a.edges == b.edges


# --------------------------------------------------------------------------- #
# functional/ontology-gate  (D2.2 rejection cases)
# --------------------------------------------------------------------------- #


def assert_rejected(o: Ontology, u: OntologyUpdate):
    before = snapshot(o)
    with pytest.raises(OntologyGateError):
        apply_update(o, u)
    assert snapshot(o) == before


def test_gate_rejects_empty_update():
    assert_rejected(core(), upd())


@pytest.mark.parametrize("code", sorted(CORE_NODES))
def test_gate_rejects_replacing_core_node_identically(code):
    o = core()
    assert_rejected(o, upd(add_nodes=[o.node(code)]))


def test_gate_rejects_reparent_remap_or_replace_core_node():
    o = core()
    assert_rejected(o, upd(add_nodes=[metric(code="dp:adherence.protocol", parent="dp:affect",
                                             lhm_domain="consistency")]))
    assert_rejected(o, upd(add_nodes=[metric(code="dp:adherence.protocol", parent="dp:behavior",
                                             lhm_domain="sentiment_trajectory")]))
    assert_rejected(o, upd(add_nodes=[metric(code="93832-4", system=X, parent="dp:sleep",
                                             lhm_domain="consistency", value_range=(0.0, 24.0))]))
    assert_rejected(o, upd(add_nodes=[category(code="dp:sleep", display="Rest")]))
    assert o.node("dp:adherence.protocol").parent == "dp:behavior"


def test_gate_rejects_re_adding_extension_node():
    o1 = apply_update(core(), upd(add_nodes=[metric()]))
    assert_rejected(o1, upd(add_nodes=[metric(display="Walk logged v2")]))


def test_gate_rejects_unknown_parent():
    assert_rejected(core(), upd(add_nodes=[metric(parent="dp:nowhere")]))


def test_gate_parent_must_be_added_earlier_in_same_update():
    o = core()
    child = metric(code="dp:social.contact_logged", parent="dp:social")
    assert_rejected(o, upd(add_nodes=[child, category(code="dp:social")]))
    ok = apply_update(o, upd(add_nodes=[category(code="dp:social"), child]))
    assert ok.node("dp:social.contact_logged").parent == "dp:social"


def test_gate_rejects_metric_without_caveats():
    assert_rejected(core(), upd(add_nodes=[metric(caveats=())]))


@pytest.mark.parametrize("label", ["Improves adherence", "Helps treat fatigue", "We recommend walks",
                                   "Clinical walk score"])
def test_gate_rejects_unsafe_label(label):
    assert_rejected(core(), upd(add_nodes=[metric(label=label)]))


def test_gate_uses_display_when_label_empty():
    assert_rejected(core(), upd(add_nodes=[metric(label="", display="Improvement streak")]))
    ok = apply_update(core(), upd(add_nodes=[metric(label="", display="Walk logged")]))
    assert ok.node("dp:behavior.walk_logged").display == "Walk logged"


def test_gate_rejects_caveats_missing_triad_marker():
    # association marker only (no uncertainty marker)
    assert_rejected(core(), upd(add_nodes=[metric(caveats=("An association in your own data.",))]))
    # uncertainty marker only (no association marker)
    assert_rejected(core(), upd(add_nodes=[metric(caveats=("Small samples are uncertain.",))]))


def test_gate_does_not_require_caveats_on_categories():
    o = apply_update(core(), upd(add_nodes=[category(code="dp:social")]))
    assert o.node("dp:social").kind == "category"


def test_unsafe_labels_are_unsafe_per_lhm():
    # Fixture sanity: the labels used above are rejected by the pinned LHM guard itself.
    for label in ("Improves adherence", "Helps treat fatigue", "We recommend walks", "Clinical walk score",
                  "Improvement streak"):
        with pytest.raises(ClaimScopeError):
            assert_claim_safe(label, CORE_CAVEATS_EXPECTED, "low", False)
    assert_claim_safe("Walk logged", CORE_CAVEATS_EXPECTED, "low", False)


@pytest.mark.parametrize(
    "src,tgt",
    [
        ("act:goal_set", "dp:behavior"),           # target a category
        ("act:goal_set", "dp:nope"),               # target unknown
        ("act:goal_set", "act:reply_sent"),        # target an action code
        ("act:unknown", "dp:adherence.protocol"),  # source unknown action
        ("dp:affect", "dp:adherence.protocol"),    # source a category
        ("dp:nope", "dp:adherence.protocol"),      # source unknown node
        ("", "dp:adherence.protocol"),
    ],
)
def test_gate_rejects_bad_edge_endpoints(src, tgt):
    assert_rejected(core(), upd(set_edges=[Edge(source=src, target=tgt, weight=0.1)]))


def test_gate_edges_resolved_after_nodes_added():
    o = core()
    n = metric()
    # target is a node added in the same update → accepted
    o1 = apply_update(o, upd(add_nodes=[n], set_edges=[Edge(source="act:nudge_sent", target=n.code, weight=0.4)]))
    assert o1.weight("act:nudge_sent", n.code) == pytest.approx(0.4)
    # source is a node added in the same update → accepted
    o2 = apply_update(o, upd(add_nodes=[n], set_edges=[Edge(source=n.code, target="dp:adherence.protocol",
                                                            weight=-0.2)]))
    assert o2.weight(n.code, "dp:adherence.protocol") == pytest.approx(-0.2)
    # an added CATEGORY is still not a valid target
    assert_rejected(o, upd(add_nodes=[category(code="dp:social")],
                           set_edges=[Edge(source="act:goal_set", target="dp:social", weight=0.1)]))


def test_gate_rejects_duplicate_edge_pair_in_update():
    assert_rejected(core(), upd(set_edges=[
        Edge(source="act:goal_set", target="8867-4", weight=0.1),
        Edge(source="act:goal_set", target="8867-4", weight=0.2),
    ]))
    assert_rejected(core(), upd(set_edges=[
        Edge(source="act:reminder_sent", target="dp:adherence.protocol", weight=0.5),
        Edge(source="act:reminder_sent", target="dp:adherence.protocol", weight=0.5),
    ]))


def test_gate_rejection_atomic_mixed_update():
    # A valid node with an invalid edge: nothing is applied.
    o = core()
    assert_rejected(o, upd(add_nodes=[metric()], set_edges=[Edge(source="act:goal_set", target="dp:vital",
                                                                  weight=0.1)]))
    assert "dp:behavior.walk_logged" not in o.nodes


# --------------------------------------------------------------------------- #
# functional/ontology-update  (accepted updates)
# --------------------------------------------------------------------------- #


def test_accepted_update_versions_and_history():
    o0 = core()
    before = snapshot(o0)
    u1 = upd(add_nodes=[metric()], proposer="alice-agent", rationale="add walks")
    o1 = apply_update(o0, u1)
    assert snapshot(o0) == before  # original untouched
    assert o1 is not o0
    assert o1.version == "core-v1+u1"
    assert len(o1.history) == 1
    r = o1.history[0]
    assert isinstance(r, UpdateRecord)
    assert (r.from_version, r.to_version, r.proposer, r.rationale, r.gate) == (
        "core-v1", "core-v1+u1", "alice-agent", "add walks", "passed")
    assert HEX64.match(r.update_digest)

    u2 = upd(set_edges=[Edge(source="act:goal_set", target="dp:behavior.walk_logged", weight=0.25)],
             proposer="bob-agent", rationale="link goals")
    snap1 = snapshot(o1)
    o2 = apply_update(o1, u2)
    assert snapshot(o1) == snap1
    assert o2.version == "core-v1+u2"
    assert o2.history[0] == r
    assert len(o2.history) == 2
    r2 = o2.history[1]
    assert (r2.from_version, r2.to_version, r2.proposer, r2.rationale, r2.gate) == (
        "core-v1+u1", "core-v1+u2", "bob-agent", "link goals", "passed")
    assert r2.update_digest != r.update_digest
    assert len({o0.digest(), o1.digest(), o2.digest()}) == 3


def test_rejected_update_after_accepted_leaves_extended_unchanged():
    o1 = apply_update(core(), upd(add_nodes=[metric()]))
    assert_rejected(o1, upd(add_nodes=[metric(code="dp:x.y", parent="dp:nowhere")]))
    assert o1.version == "core-v1+u1"
    o2 = apply_update(o1, upd(add_nodes=[metric(code="dp:x.y")]))
    assert o2.version == "core-v1+u2"


def test_added_nodes_not_core_and_core_nodes_byte_identical():
    o0 = core()
    n = metric(lhm_domain="consistency")
    o1 = apply_update(o0, upd(add_nodes=[category(code="dp:social"), n]))
    assert not o1.is_core("dp:social") and not o1.is_core(n.code)
    assert o1.node(n.code) == n
    for code in CORE_NODES:
        assert o1.is_core(code)
        assert o1.node(code) == o0.node(code)
        assert o1.node(code).model_dump_json() == o0.node(code).model_dump_json()
    assert set(o1.nodes) == set(CORE_NODES) | {"dp:social", n.code}
    assert [m.code for m in o1.metric_nodes()] == sorted(
        [c for c, v in CORE_NODES.items() if v[1] == "metric"] + [n.code])


def test_reweight_existing_edge():
    o0 = core()
    o1 = apply_update(o0, upd(set_edges=[Edge(source="act:reminder_sent", target="dp:adherence.protocol",
                                               weight=-0.5)]))
    assert o1.weight("act:reminder_sent", "dp:adherence.protocol") == pytest.approx(-0.5)
    assert o0.weight("act:reminder_sent", "dp:adherence.protocol") == pytest.approx(0.3)
    assert len(o1.edges) == len(o0.edges)
    keys = [(e.source, e.target) for e in o1.edges]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    others = {k: w for k, w in CORE_EDGES.items() if k != ("act:reminder_sent", "dp:adherence.protocol")}
    for (s, t), w in others.items():
        assert o1.weight(s, t) == pytest.approx(w)
    assert o1.version == "core-v1+u1"


def test_add_new_edge_keeps_sorted_unique():
    o1 = apply_update(core(), upd(set_edges=[
        Edge(source="8867-4", target="dp:affect.checkin_sentiment", weight=-0.1),
        Edge(source="act:education_viewed", target="dp:affect.checkin_sentiment", weight=0.05),
    ]))
    assert len(o1.edges) == len(CORE_EDGES) + 2
    keys = [(e.source, e.target) for e in o1.edges]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    assert o1.weight("8867-4", "dp:affect.checkin_sentiment") == pytest.approx(-0.1)


def test_mixed_reweight_and_add_in_one_update():
    o1 = apply_update(core(), upd(set_edges=[
        Edge(source="act:nudge_sent", target="dp:adherence.protocol", weight=0.9),
        Edge(source="act:nudge_sent", target="dp:affect.checkin_sentiment", weight=0.1),
    ]))
    assert o1.weight("act:nudge_sent", "dp:adherence.protocol") == pytest.approx(0.9)
    assert o1.weight("act:nudge_sent", "dp:affect.checkin_sentiment") == pytest.approx(0.1)
    assert len(o1.edges) == len(CORE_EDGES) + 1


# --------------------------------------------------------------------------- #
# functional/ontology-determinism
# --------------------------------------------------------------------------- #


def test_same_update_twice_equal_results():
    u = upd(add_nodes=[category(code="dp:social"), metric(code="dp:social.contact_logged", parent="dp:social")],
            set_edges=[Edge(source="act:reply_sent", target="dp:social.contact_logged", weight=0.2)])
    a = apply_update(core(), u)
    b = apply_update(core(), u)
    c = apply_update(core(), OntologyUpdate.model_validate(u.model_dump()))
    assert a.digest() == b.digest() == c.digest()
    assert a.version == b.version
    assert a.history == b.history
    assert a.history[0].update_digest == c.history[0].update_digest
    assert dict(a.nodes) == dict(b.nodes) and a.edges == b.edges


def test_digest_depends_on_edges_and_nodes():
    base = core().digest()
    d_edge = apply_update(core(), upd(set_edges=[Edge(source="act:goal_set", target="8867-4", weight=0.1)])).digest()
    d_node = apply_update(core(), upd(add_nodes=[metric()])).digest()
    assert len({base, d_edge, d_node}) == 3
    assert all(HEX64.match(d) for d in (base, d_edge, d_node))


def test_update_digest_differs_for_different_updates():
    a = apply_update(core(), upd(add_nodes=[metric()]))
    b = apply_update(core(), upd(add_nodes=[metric()], rationale="different rationale"))
    assert a.history[0].update_digest != b.history[0].update_digest
