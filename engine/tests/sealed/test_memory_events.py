"""Sealed challenge suite — slice D1: daypart.memory.events (D14, I7).

Contract: docs/daypart/contracts/CONTRACT-slice-D-memory.md §D1 (and the §D2.1 core
registry for EventLog's default ontology / value_range checks).

Categories: functional/events-models, functional/events-parse, functional/eventlog-append,
functional/eventlog-query, integration/events-imports.

Ambiguities recorded for the principal (not invented behaviour; left untested):
- EventLog.extend when a middle event is rejected: the contract says "append each, in
  order", but does not say whether the earlier events stay stored. Only "raises
  ValueError" and "the rejected event is not stored" are asserted.
- Mutating a frozen model may raise pydantic.ValidationError, TypeError or
  AttributeError; any of these is accepted.
"""

from __future__ import annotations

import ast
import importlib.util
import math
from datetime import datetime, timedelta, timezone

import pydantic
import pytest

from daypart.memory import events as ev_mod
from daypart.memory.events import (
    ACTION_CODES,
    MEMBER_ID_RE,
    ActionEvent,
    Event,
    EventLog,
    SignalEvent,
    parse_event,
)
from tests.sealed.conftest_memory import day

UTC = timezone.utc
T0 = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)


def act(eid="a1", mid="syn-001", at=T0, code="act:nudge_sent"):
    return ActionEvent(event_id=eid, member_id=mid, occurred_at=at, code=code)


def sig(eid="s1", mid="syn-001", at=T0, code="dp:adherence.protocol", value=1.0):
    return SignalEvent(event_id=eid, member_id=mid, occurred_at=at, code=code, value=value)


# --------------------------------------------------------------------------- #
# functional/events-models
# --------------------------------------------------------------------------- #


def test_member_id_re_and_action_codes_exact():
    assert MEMBER_ID_RE == r"^(cur|syn)-\d{3}$"
    assert isinstance(ACTION_CODES, frozenset)
    assert ACTION_CODES == frozenset(
        {
            "act:nudge_sent",
            "act:reminder_sent",
            "act:goal_set",
            "act:protocol_adjusted",
            "act:education_viewed",
            "act:reply_sent",
        }
    )


@pytest.mark.parametrize("code", sorted(
    ["act:nudge_sent", "act:reminder_sent", "act:goal_set",
     "act:protocol_adjusted", "act:education_viewed", "act:reply_sent"]))
def test_action_event_accepts_every_action_code(code):
    e = act(code=code)
    assert e.kind == "action"
    assert e.code == code


@pytest.mark.parametrize("code", ["act:unknown", "nudge_sent", "", "dp:adherence.protocol", "ACT:NUDGE_SENT"])
def test_action_event_rejects_unknown_code(code):
    with pytest.raises(pydantic.ValidationError):
        act(code=code)


def test_signal_event_defaults_kind_and_code_not_checked_by_model():
    s = sig(code="not-an-ontology-node", value=3.5)
    assert s.kind == "signal"
    assert s.value == 3.5


def test_action_event_has_no_value_field():
    with pytest.raises(pydantic.ValidationError):
        ActionEvent(event_id="a", member_id="syn-001", occurred_at=T0, code="act:goal_set", value=1.0)


@pytest.mark.parametrize("extra", ["note", "text", "email", "free_text", "name"])
def test_action_event_rejects_extra_keys(extra):
    with pytest.raises(pydantic.ValidationError):
        ActionEvent(event_id="a", member_id="syn-001", occurred_at=T0, code="act:goal_set", **{extra: "hello"})


@pytest.mark.parametrize("extra", ["note", "text", "email", "free_text", "name"])
def test_signal_event_rejects_extra_keys(extra):
    with pytest.raises(pydantic.ValidationError):
        SignalEvent(event_id="s", member_id="syn-001", occurred_at=T0,
                    code="dp:adherence.protocol", value=1.0, **{extra: "hello"})


@pytest.mark.parametrize("cls", ["action", "signal"])
def test_events_have_no_free_text_fields(cls):
    model = ActionEvent if cls == "action" else SignalEvent
    allowed = {"kind", "event_id", "member_id", "occurred_at", "code"} | ({"value"} if cls == "signal" else set())
    assert set(model.model_fields) == allowed


@pytest.mark.parametrize("bad", ["alice@example.com", "cur-1", "cur-0001", "CUR-001", "xyz-001",
                                 "syn001", "", " syn-001", "syn-001 ", "syn-00a"])
def test_member_id_pattern_rejected(bad):
    with pytest.raises(pydantic.ValidationError):
        act(mid=bad)
    with pytest.raises(pydantic.ValidationError):
        sig(mid=bad)


@pytest.mark.parametrize("good", ["cur-001", "cur-025", "syn-001", "syn-175", "syn-999", "cur-000"])
def test_member_id_pattern_accepted(good):
    assert act(mid=good).member_id == good
    assert sig(mid=good).member_id == good


def test_event_id_must_be_non_empty():
    with pytest.raises(pydantic.ValidationError):
        act(eid="")
    with pytest.raises(pydantic.ValidationError):
        sig(eid="")


def test_naive_occurred_at_rejected():
    naive = datetime(2026, 9, 10, 12, 0)
    with pytest.raises(pydantic.ValidationError):
        act(at=naive)
    with pytest.raises(pydantic.ValidationError):
        sig(at=naive)


def test_non_utc_aware_occurred_at_accepted():
    plus5 = timezone(timedelta(hours=5))
    assert act(at=datetime(2026, 9, 10, 12, 0, tzinfo=plus5)).occurred_at.utcoffset() == timedelta(hours=5)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_signal_value_must_be_finite(bad):
    with pytest.raises(pydantic.ValidationError):
        sig(value=bad)


def test_events_are_frozen():
    a, s = act(), sig()
    with pytest.raises((pydantic.ValidationError, TypeError, AttributeError)):
        a.code = "act:goal_set"  # type: ignore[misc]
    with pytest.raises((pydantic.ValidationError, TypeError, AttributeError)):
        s.value = 0.0  # type: ignore[misc]


def test_events_are_hashable_value_objects():
    assert act() == act()
    assert hash(act()) == hash(act())
    assert sig() == sig()


# --------------------------------------------------------------------------- #
# functional/events-parse
# --------------------------------------------------------------------------- #


def test_parse_event_dispatches_on_kind():
    a = parse_event({"kind": "action", "event_id": "a", "member_id": "cur-002",
                     "occurred_at": "2026-09-10T12:00:00+00:00", "code": "act:reply_sent"})
    assert isinstance(a, ActionEvent)
    s = parse_event({"kind": "signal", "event_id": "s", "member_id": "cur-002",
                     "occurred_at": "2026-09-10T12:00:00+00:00", "code": "8867-4", "value": 72})
    assert isinstance(s, SignalEvent)
    assert s.value == 72.0


def test_parse_event_round_trips_model_dump():
    for e in (act(), sig()):
        assert parse_event(e.model_dump(mode="json")) == e
        assert parse_event(e.model_dump()) == e


@pytest.mark.parametrize("kind", [None, "observation", "", "ACTION"])
def test_parse_event_missing_or_unknown_kind(kind):
    obj = {"event_id": "a", "member_id": "cur-002", "occurred_at": "2026-09-10T12:00:00+00:00",
           "code": "act:reply_sent"}
    if kind is not None:
        obj["kind"] = kind
    with pytest.raises(pydantic.ValidationError):
        parse_event(obj)


def test_parse_event_action_with_value_rejected():
    with pytest.raises(pydantic.ValidationError):
        parse_event({"kind": "action", "event_id": "a", "member_id": "cur-002",
                     "occurred_at": "2026-09-10T12:00:00+00:00", "code": "act:reply_sent", "value": 1.0})


def test_parse_event_rejects_free_text_and_email_id():
    with pytest.raises(pydantic.ValidationError):
        parse_event({"kind": "signal", "event_id": "s", "member_id": "cur-002",
                     "occurred_at": "2026-09-10T12:00:00+00:00", "code": "8867-4", "value": 70,
                     "note": "felt dizzy"})
    with pytest.raises(pydantic.ValidationError):
        parse_event({"kind": "action", "event_id": "a", "member_id": "alice@example.com",
                     "occurred_at": "2026-09-10T12:00:00+00:00", "code": "act:reply_sent"})


def test_event_annotated_union_discriminates():
    ta = pydantic.TypeAdapter(Event)
    a = ta.validate_python({"kind": "action", "event_id": "a", "member_id": "syn-003",
                            "occurred_at": T0, "code": "act:goal_set"})
    s = ta.validate_python({"kind": "signal", "event_id": "s", "member_id": "syn-003",
                            "occurred_at": T0, "code": "8867-4", "value": 60.0})
    assert isinstance(a, ActionEvent) and isinstance(s, SignalEvent)
    with pytest.raises(pydantic.ValidationError):
        ta.validate_python({"kind": "other", "event_id": "s", "member_id": "syn-003", "occurred_at": T0})


# --------------------------------------------------------------------------- #
# functional/eventlog-append
# --------------------------------------------------------------------------- #


def test_eventlog_default_is_core_ontology_and_starts_empty():
    log = EventLog()
    assert len(log) == 0
    log.append(sig(code="dp:adherence.protocol", value=1.0))
    log.append(sig(eid="s2", code="93832-4", value=7.5))
    assert len(log) == 2


def test_eventlog_explicit_core_ontology():
    from daypart.memory.ontology import core

    log = EventLog(core())
    log.append(sig(code="55423-8", value=1000.0))
    assert len(log) == 1


def test_duplicate_event_id_rejected_and_nothing_stored():
    log = EventLog()
    log.append(act(eid="x"))
    with pytest.raises(ValueError):
        log.append(act(eid="x", at=T0 + timedelta(hours=1)))
    # duplicate across kinds and across members also rejected
    with pytest.raises(ValueError):
        log.append(sig(eid="x"))
    with pytest.raises(ValueError):
        log.append(act(eid="x", mid="cur-005"))
    assert len(log) == 1
    assert log.events("syn-001") == (act(eid="x"),)
    assert log.events("cur-005") == ()


def test_identical_event_appended_twice_rejected():
    log = EventLog()
    log.append(sig())
    with pytest.raises(ValueError):
        log.append(sig())
    assert len(log) == 1


@pytest.mark.parametrize("code", ["dp:unknown", "12345-6", "act:nudge_sent", "", "journal-metric"])
def test_signal_on_unknown_node_rejected(code):
    log = EventLog()
    with pytest.raises(ValueError):
        log.append(sig(code=code, value=0.5))
    assert len(log) == 0


@pytest.mark.parametrize("code", ["dp:behavior", "dp:affect", "dp:symptom", "dp:nutrition",
                                  "dp:vital", "dp:activity", "dp:sleep"])
def test_signal_on_category_node_rejected(code):
    log = EventLog()
    with pytest.raises(ValueError):
        log.append(sig(code=code, value=0.5))
    assert len(log) == 0


@pytest.mark.parametrize(
    "code,lo,hi",
    [
        ("dp:adherence.protocol", 0.0, 1.0),
        ("dp:affect.checkin_sentiment", -1.0, 1.0),
        ("dp:nutrition.food_log", 0.0, 1.0),
        ("dp:symptom.severity", 1.0, 3.0),
        ("72514-3", 0.0, 10.0),
        ("44261-6", 0.0, 27.0),
        ("70274-6", 0.0, 21.0),
        ("8867-4", 20.0, 250.0),
        ("29463-7", 0.0, 500.0),
        ("55423-8", 0.0, 100000.0),
        ("93832-4", 0.0, 24.0),
    ],
)
def test_signal_value_range_inclusive(code, lo, hi):
    log = EventLog()
    log.append(sig(eid=f"{code}-lo", code=code, value=lo))
    log.append(sig(eid=f"{code}-hi", code=code, value=hi))
    log.append(sig(eid=f"{code}-mid", code=code, value=(lo + hi) / 2))
    assert len(log) == 3
    span = hi - lo
    with pytest.raises(ValueError):
        log.append(sig(eid=f"{code}-below", code=code, value=lo - span * 1e-3))
    with pytest.raises(ValueError):
        log.append(sig(eid=f"{code}-above", code=code, value=hi + span * 1e-3))
    assert len(log) == 3


def test_action_events_need_no_ontology_node():
    log = EventLog()
    for i, code in enumerate(sorted(ACTION_CODES)):
        log.append(act(eid=f"a{i}", code=code))
    assert len(log) == len(ACTION_CODES)


def test_eventlog_over_extended_ontology_accepts_new_node():
    from daypart.memory.ontology import CORE_CAVEATS, LOCAL_SYSTEM, Node, OntologyUpdate, apply_update, core

    new = Node(code="dp:behavior.walk_logged", system=LOCAL_SYSTEM, display="Walk logged",
               kind="metric", parent="dp:behavior", lhm_domain=None, value_range=(0.0, 1.0),
               label="Walk logged", caveats=CORE_CAVEATS)
    ext = apply_update(core(), OntologyUpdate(proposer="p", rationale="r", add_nodes=(new,)))
    EventLog(ext).append(sig(code="dp:behavior.walk_logged", value=1.0))
    with pytest.raises(ValueError):
        EventLog().append(sig(code="dp:behavior.walk_logged", value=1.0))
    with pytest.raises(ValueError):
        EventLog(ext).append(sig(code="dp:behavior.walk_logged", value=1.5))


def test_extend_appends_in_order():
    log = EventLog()
    evs = [act(eid="a1"), sig(eid="s1"), act(eid="a2", mid="cur-003")]
    log.extend(evs)
    assert len(log) == 3
    log.extend(iter([sig(eid="s2", at=T0 - timedelta(days=1))]))
    assert len(log) == 4


def test_extend_rejects_invalid_event():
    log = EventLog()
    with pytest.raises(ValueError):
        log.extend([act(eid="a1"), sig(eid="bad", code="dp:behavior")])
    assert all(e.event_id != "bad" for e in log.events("syn-001"))
    with pytest.raises(ValueError):
        log.extend([act(eid="dup"), act(eid="dup")])


def test_eventlog_has_no_update_or_delete():
    log = EventLog()
    for name in ("delete", "remove", "update", "purge", "pop", "clear", "discard", "replace", "__delitem__",
                 "__setitem__"):
        assert not hasattr(log, name), name


# --------------------------------------------------------------------------- #
# functional/eventlog-query
# --------------------------------------------------------------------------- #


def _populated():
    log = EventLog()
    plus5 = timezone(timedelta(hours=5))
    evs = [
        sig(eid="s-late", at=day(10)),
        act(eid="b", at=day(5)),
        act(eid="a", at=day(5)),  # same instant as "b" → tie broken by event_id
        sig(eid="s-early", at=day(1), value=0.0),
        # 2026-09-04T03:00+05:00 == 2026-09-03T22:00Z → between day(2) and day(3) in instant order
        act(eid="tz", at=datetime(2026, 9, 4, 3, 0, tzinfo=plus5), code="act:reply_sent"),
        sig(eid="other-member", mid="cur-010", at=day(2), value=1.0),
        act(eid="c", at=day(20), code="act:goal_set"),
    ]
    for e in evs:
        log.append(e)
    return log


def test_events_sorted_by_time_then_id():
    log = _populated()
    got = log.events("syn-001")
    assert isinstance(got, tuple)
    assert [e.event_id for e in got] == ["s-early", "tz", "a", "b", "s-late", "c"]
    assert len(log) == 7


def test_events_filtered_per_member():
    log = _populated()
    assert [e.event_id for e in log.events("cur-010")] == ["other-member"]


def test_events_unknown_member_is_empty_tuple():
    log = _populated()
    assert log.events("syn-150") == ()
    assert EventLog().events("cur-001") == ()


def test_events_as_of_inclusive():
    log = _populated()
    assert [e.event_id for e in log.events("syn-001", as_of=day(5))] == ["s-early", "tz", "a", "b"]
    assert [e.event_id for e in log.events("syn-001", as_of=day(5) - timedelta(microseconds=1))] == [
        "s-early", "tz"]
    assert log.events("syn-001", as_of=day(0)) == ()
    assert len(log.events("syn-001", as_of=day(100))) == 6


def test_events_as_of_compares_instants_across_offsets():
    log = _populated()
    minus7 = timezone(timedelta(hours=-7))
    # day(5) expressed in -07:00
    as_of = day(5).astimezone(minus7)
    assert [e.event_id for e in log.events("syn-001", as_of=as_of)] == ["s-early", "tz", "a", "b"]


def test_events_kind_filter():
    log = _populated()
    assert [e.event_id for e in log.events("syn-001", kind="action")] == ["tz", "a", "b", "c"]
    assert [e.event_id for e in log.events("syn-001", kind="signal")] == ["s-early", "s-late"]
    assert all(isinstance(e, ActionEvent) for e in log.events("syn-001", kind="action"))
    assert all(isinstance(e, SignalEvent) for e in log.events("syn-001", kind="signal"))


def test_events_kind_and_as_of_combined():
    log = _populated()
    assert [e.event_id for e in log.events("syn-001", as_of=day(5), kind="action")] == ["tz", "a", "b"]
    assert [e.event_id for e in log.events("syn-001", as_of=day(5), kind="signal")] == ["s-early"]


def test_events_naive_as_of_rejected():
    log = _populated()
    with pytest.raises(ValueError):
        log.events("syn-001", as_of=datetime(2026, 9, 5))


def test_events_result_independent_of_append_order():
    a = _populated()
    b = EventLog()
    for e in reversed(a.events("syn-001")):
        b.append(e)
    for e in a.events("cur-010"):
        b.append(e)
    assert a.events("syn-001") == b.events("syn-001")
    assert a.events("cur-010") == b.events("cur-010")


# --------------------------------------------------------------------------- #
# integration/events-imports
# --------------------------------------------------------------------------- #


def _module_level_imports(modname: str) -> set[str]:
    spec = importlib.util.find_spec(modname)
    assert spec is not None and spec.origin
    with open(spec.origin, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    pkg = modname.rsplit(".", 1)[0]
    out: set[str] = set()
    # Unconditional top-level statements only: lazy imports inside functions and
    # `if TYPE_CHECKING:` blocks are permitted by the contract ("imported lazily").
    for node in tree.body:
        for st in [node]:
            if isinstance(st, ast.Import):
                out.update(a.name for a in st.names)
            elif isinstance(st, ast.ImportFrom):
                if st.level:
                    base = pkg if st.level == 1 else pkg.rsplit(".", st.level - 1)[0]
                    mod = f"{base}.{st.module}" if st.module else base
                    out.add(mod)
                    if not st.module:
                        out.update(f"{base}.{a.name}" for a in st.names)
                else:
                    out.add(st.module or "")
                    out.update(f"{st.module}.{a.name}" for a in st.names)
    return out


def test_events_has_no_import_time_dependency_on_ontology_or_lhm():
    assert ev_mod.__name__ == "daypart.memory.events"
    imports = _module_level_imports("daypart.memory.events")
    assert not any(i.startswith("daypart.memory.ontology") for i in imports), imports
    assert not any(i.startswith("daypart.memory.lhm") for i in imports), imports


def test_ontology_does_not_import_lhm_module():
    imports = _module_level_imports("daypart.memory.ontology")
    assert not any(i.startswith("daypart.memory.lhm") for i in imports), imports
