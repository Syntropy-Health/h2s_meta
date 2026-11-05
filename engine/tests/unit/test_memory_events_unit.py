import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pydantic
import pytest

from daypart.memory.events import ACTION_CODES, ActionEvent, EventLog, SignalEvent, parse_event

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def act(eid="a1", mid="syn-001", at=T0, code="act:reminder_sent"):
    return ActionEvent(event_id=eid, member_id=mid, occurred_at=at, code=code)


def sig(eid="s1", mid="syn-001", at=T0, code="dp:adherence.protocol", value=1.0):
    return SignalEvent(event_id=eid, member_id=mid, occurred_at=at, code=code, value=value)


def test_events_module_has_no_import_time_ontology_dependency():
    code = "import sys, daypart.memory.events; assert 'daypart.memory.ontology' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_action_codes():
    assert len(ACTION_CODES) == 6 and "act:reply_sent" in ACTION_CODES


@pytest.mark.parametrize(
    "kw",
    [
        {"value": 1.0},
        {"note": "hi"},
        {"email": "a@b.c"},
        {"code": "act:unknown"},
        {"member_id": "alice@example.com"},
        {"member_id": "cur-1"},
        {"occurred_at": datetime(2026, 9, 1)},
        {"event_id": ""},
    ],
)
def test_action_event_rejects(kw):
    base = dict(event_id="a1", member_id="syn-001", occurred_at=T0, code="act:goal_set")
    with pytest.raises(pydantic.ValidationError):
        ActionEvent(**{**base, **kw})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_signal_value_finite(value):
    with pytest.raises(pydantic.ValidationError):
        sig(value=value)


def test_signal_rejects_text_and_bad_id():
    with pytest.raises(pydantic.ValidationError):
        SignalEvent(event_id="s", member_id="syn-001", occurred_at=T0, code="8867-4", value=60, text="x")
    with pytest.raises(pydantic.ValidationError):
        sig(mid="alice@example.com")


def test_frozen():
    with pytest.raises(pydantic.ValidationError):
        act().code = "act:goal_set"  # type: ignore[misc]


def test_parse_event():
    a = parse_event({"kind": "action", "event_id": "a", "member_id": "cur-001", "occurred_at": T0, "code": "act:goal_set"})
    s = parse_event({"kind": "signal", "event_id": "s", "member_id": "cur-001", "occurred_at": T0.isoformat(),
                     "code": "8867-4", "value": 60})
    assert isinstance(a, ActionEvent) and isinstance(s, SignalEvent) and s.value == 60.0
    for bad in ({"event_id": "a", "member_id": "cur-001", "occurred_at": T0, "code": "act:goal_set"},
                {"kind": "other", "event_id": "a", "member_id": "cur-001", "occurred_at": T0, "code": "x"},
                {"kind": "action", "event_id": "a", "member_id": "cur-001", "occurred_at": T0,
                 "code": "act:goal_set", "value": 1.0}):
        with pytest.raises(pydantic.ValidationError):
            parse_event(bad)


def test_append_gates():
    log = EventLog()
    log.append(sig())
    for bad in (
        sig(),  # duplicate id
        sig("s2", code="dp:behavior"),  # category
        sig("s3", code="dp:unknown"),  # not in ontology
        sig("s4", value=1.5),  # out of range
        sig("s5", code="8867-4", value=19.9),
        act("s1"),  # duplicate id across kinds
    ):
        with pytest.raises(ValueError):
            log.append(bad)
    assert len(log) == 1
    log.append(sig("s6", code="8867-4", value=20.0))  # inclusive bound
    log.append(sig("s7", code="8867-4", value=250.0))
    assert len(log) == 3


def test_extend_stops_at_failure():
    log = EventLog()
    with pytest.raises(ValueError):
        log.extend([act("a1"), act("a1")])
    assert len(log) == 1


def test_events_query():
    log = EventLog()
    log.extend([
        act("b", at=T0 + timedelta(days=1)),
        sig("a", at=T0 + timedelta(days=1)),
        act("c", at=T0),
        act("x", mid="cur-002", at=T0),
    ])
    assert [e.event_id for e in log.events("syn-001")] == ["c", "a", "b"]
    assert [e.event_id for e in log.events("syn-001", as_of=T0)] == ["c"]
    assert [e.event_id for e in log.events("syn-001", kind="signal")] == ["a"]
    assert [e.event_id for e in log.events("syn-001", as_of=T0 + timedelta(days=1), kind="action")] == ["c", "b"]
    assert log.events("syn-099") == ()
    assert len(log) == 4
    with pytest.raises(ValueError):
        log.events("syn-001", as_of=datetime(2026, 9, 2))
