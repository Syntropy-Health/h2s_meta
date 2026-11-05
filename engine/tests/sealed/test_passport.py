"""Sealed challenge suite — E2 pattern passport (``daypart.passport``).

Written double-blind from docs/daypart/contracts/CONTRACT-slice-E-eval-passport.md
(E2, E2.1-E2.3). Reference states come from slice A/D public APIs, or are
hand-built ``MemberMetricState`` values over a test-only ontology.

Categories:
  functional/passport-flag      E2.1 Flag invariants, narration safety, JSON round-trip
  functional/passport-model     Passport model + module constants
  functional/passport-flags     E2.2 flags_from_state (ordering, withheld/ready, metric_key,
                                payload, title fallback, fail-closed, determinism)
  integration/passport-render   E2.3 PassportRenderer over the generate(0) cohort

Ambiguities resolved:
  * "Flag constructed with a string that fails assert_narration_safe → ValidationError"
    is checked on ``title`` and on caveats. ``message`` is pinned to
    ``WITHHELD_MESSAGE`` or ``None`` anyway.
  * The E2.3 "free text" check uses the adapter's series-point ``note`` values. For
    this cohort they are note-id tokens, and they must not appear in the passport JSON.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

import pytest
from pydantic import ValidationError

UTC = timezone.utc
CORE_FOUR = (
    "dp:adherence.protocol",
    "dp:affect.checkin_sentiment",
    "dp:nutrition.food_log",
    "dp:symptom.severity",
)
CAVEATS = (
    "Personal n-of-1 observation: an association in your own data, not causal.",
    "Scores carry an interval; small samples are uncertain.",
)
AS_OF = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def _p():
    import daypart.passport as p

    return p


def day(n: float) -> datetime:
    from daypart.simmember.trajectory import START_DATE

    return START_DATE + timedelta(days=n)


def _payload(key="dp_adherence_protocol", label="Protocol adherence (logged)", value=50.0, ts=AS_OF, caveats=CAVEATS):
    from syntropy.lhm.contracts.viz import VizPayload, VizPoint

    return VizPayload(
        contract="lhm-viz/v1",
        metric_key=key,
        label=label,
        series=(VizPoint(ts=ts, value=value),),
        regimen_start=None,
        confidence="low",
        caveats=caveats,
        rx_input=False,
    )


def _ready_kw(**over):
    kw = dict(
        code="dp:adherence.protocol",
        lhm_domain="consistency",
        title="Protocol adherence (logged)",
        status="ready",
        value=50.0,
        interval=(40.0, 60.0),
        caveats=CAVEATS,
        message=None,
        payload=_payload(),
    )
    kw.update(over)
    return kw


def _withheld_kw(**over):
    kw = dict(
        code="dp:adherence.protocol",
        lhm_domain="consistency",
        title="Protocol adherence (logged)",
        status="withheld",
        value=None,
        interval=None,
        caveats=(),
        message="Not enough data yet.",
        payload=None,
    )
    kw.update(over)
    return kw


# --------------------------------------------------------------------------- #
# Hand-built states over a test-only ontology
# --------------------------------------------------------------------------- #


def _node(code, system, display, parent, domain, label, caveats=CAVEATS):
    from daypart.memory.ontology import Node

    return Node(
        code=code,
        system=system,
        display=display,
        kind="metric",
        parent=parent,
        lhm_domain=domain,
        value_range=(0.0, 100.0),
        label=label,
        caveats=caveats,
    )


def _test_ontology():
    from daypart.memory.ontology import Node, Ontology

    L, D = "http://loinc.org", "urn:daypart:metric"
    cats = [
        Node(code="dp:symptom", system=D, display="Symptom", kind="category", parent=None),
        Node(code="dp:behavior", system=D, display="Behavior", kind="category", parent=None),
    ]
    metrics = [
        _node("72514-3", L, "Pain severity", "dp:symptom", "symptom_load", "Pain rating (0-10)"),
        _node("dp:behavior.zeta", D, "Zeta display", "dp:behavior", "consistency", "This cures your fatigue"),
        _node("dp:behavior.alpha", D, "Alpha display", "dp:behavior", "consistency", ""),
        _node(
            "dp:behavior.unsafe_caveat",
            D,
            "Unsafe caveat",
            "dp:behavior",
            "consistency",
            "Unsafe caveat card",
            caveats=(CAVEATS[0], "We recommend you log more often."),
        ),
        _node(
            "dp:behavior.no_triad",
            D,
            "No triad",
            "dp:behavior",
            "consistency",
            "No triad card",
            caveats=("Your logged data is shown here.",),
        ),
        _node("dp:behavior.unmapped", D, "Unmapped", "dp:behavior", None, "Unmapped card"),
    ]
    return Ontology("sealed-test-v1", cats + metrics, ())


def _score(domain, value, interval=(30.0, 55.0)):
    from syntropy.lhm.metrics.scoring import DomainScore

    return DomainScore(domain=domain, score=value, n_effective=20, interval=interval, method="sealed")


def _ns(code, domain, value, ready, interval=(30.0, 55.0)):
    from daypart.memory.lhm import NodeState

    if domain is None:
        return NodeState(code, None, score=None, ready=False, value=None)
    return NodeState(code, domain, score=_score(domain, value, interval), ready=ready, value=value if ready else None)


def _state(nodes, as_of=AS_OF, member="syn-001", version="sealed-test-v1"):
    from daypart.memory.lhm import GUARD_VERSIONS, MemberMetricState

    return MemberMetricState(
        member_id=member,
        as_of=as_of,
        ontology_version=version,
        nodes=MappingProxyType({n.code: n for n in nodes}),
        signals=(),
        alignment=MappingProxyType({}),
        guard_versions=MappingProxyType(dict(GUARD_VERSIONS)),
    )


def _all_ready_state():
    return _state(
        [
            _ns("72514-3", "symptom_load", 42.5, True),
            _ns("dp:behavior.zeta", "consistency", 61.0, True),
            _ns("dp:behavior.alpha", "consistency", 12.25, True, interval=None),
            _ns("dp:behavior.unsafe_caveat", "consistency", 70.0, True),
            _ns("dp:behavior.no_triad", "consistency", 80.0, True),
            _ns("dp:behavior.unmapped", None, 0.0, False),
        ]
    )


def _by_code(flags):
    return {f.code: f for f in flags}


# --------------------------------------------------------------------------- #
# Constants + models
# --------------------------------------------------------------------------- #


@pytest.mark.category("functional/passport-model")
class TestConstants:
    def test_values(self):
        p = _p()
        assert p.PASSPORT_TITLE == "Pattern passport"
        assert p.WITHHELD_MESSAGE == "Not enough data yet."
        assert p.FALLBACK_TITLE == "Personal pattern"
        assert tuple(p.PENDING) == ("R10", "R11")

    def test_member_facing_constants_are_narration_safe(self):
        from syntropy.lhm.metrics.claims import assert_narration_safe

        p = _p()
        for s in (p.PASSPORT_TITLE, p.WITHHELD_MESSAGE, p.FALLBACK_TITLE):
            assert_narration_safe(s)


@pytest.mark.category("functional/passport-flag")
class TestFlagValid:
    def test_ready(self):
        p = _p()
        f = p.Flag(**_ready_kw())
        assert f.status == "ready" and f.value == 50.0
        assert f.pattern is None and f.citation is None
        assert tuple(f.pending) == ("R10", "R11")
        assert f.caveats == CAVEATS

    def test_ready_interval_none_allowed(self):
        p = _p()
        assert p.Flag(**_ready_kw(interval=None)).interval is None

    def test_withheld(self):
        p = _p()
        f = p.Flag(**_withheld_kw())
        assert f.status == "withheld"
        assert f.value is None and f.interval is None and f.payload is None
        assert f.caveats == ()
        assert f.message == "Not enough data yet."
        assert tuple(f.pending) == ("R10", "R11")

    def test_frozen(self):
        p = _p()
        f = p.Flag(**_withheld_kw())
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            f.title = "x"  # type: ignore[misc]


@pytest.mark.category("functional/passport-flag")
class TestFlagInvariants:
    @pytest.mark.parametrize(
        "over",
        [
            {"value": None},
            {"value": float("nan")},
            {"value": float("inf")},
            {"value": float("-inf")},
            {"message": "Not enough data yet."},
            {"caveats": ()},
            {"payload": None},
        ],
        ids=["value-none", "value-nan", "value-inf", "value-neginf", "message", "no-caveats", "no-payload"],
    )
    def test_ready_violations(self, over):
        p = _p()
        with pytest.raises(ValidationError):
            p.Flag(**_ready_kw(**over))

    @pytest.mark.parametrize(
        "over",
        [
            {"value": 1.0},
            {"value": 0.0},
            {"interval": (0.0, 1.0)},
            {"payload": "PAYLOAD"},
            {"caveats": CAVEATS},
            {"message": None},
            {"message": "Some other text."},
        ],
        ids=["value", "value-zero", "interval", "payload", "caveats", "message-none", "message-other"],
    )
    def test_withheld_violations(self, over):
        p = _p()
        if over.get("payload") == "PAYLOAD":
            over = {"payload": _payload()}
        with pytest.raises(ValidationError):
            p.Flag(**_withheld_kw(**over))

    @pytest.mark.parametrize(
        "over",
        [{"pattern": "a pattern"}, {"citation": "a citation"}, {"pattern": {}}, {"surprise": 1}, {"status": "maybe"}],
        ids=["pattern", "citation", "pattern-dict", "extra", "status"],
    )
    def test_other_violations(self, over):
        p = _p()
        with pytest.raises(ValidationError):
            p.Flag(**_withheld_kw(**over))
        with pytest.raises(ValidationError):
            p.Flag(**_ready_kw(**over))

    @pytest.mark.parametrize(
        "title",
        ["This cures your fatigue", "We recommend you log more", "You should take 400 mg of magnesium."],
    )
    def test_unsafe_title(self, title):
        p = _p()
        with pytest.raises(ValidationError):
            p.Flag(**_withheld_kw(title=title))
        with pytest.raises(ValidationError):
            p.Flag(**_ready_kw(title=title))

    @pytest.mark.parametrize("bad", ["We recommend you log more often.", "This treatment cures anxiety."])
    def test_unsafe_caveat(self, bad):
        p = _p()
        caveats = (CAVEATS[0], bad)
        with pytest.raises(ValidationError):
            p.Flag(**_ready_kw(caveats=caveats, payload=_payload()))


@pytest.mark.category("functional/passport-flag")
class TestFlagJson:
    @pytest.mark.parametrize("kind", ["ready", "ready_no_interval", "withheld"])
    def test_round_trip(self, kind):
        p = _p()
        kw = {"ready": _ready_kw(), "ready_no_interval": _ready_kw(interval=None), "withheld": _withheld_kw()}[kind]
        f = p.Flag(**kw)
        back = p.Flag.model_validate_json(f.model_dump_json())
        assert back == f
        assert back.payload == f.payload

    def test_withheld_json_has_no_number(self):
        p = _p()
        data = json.loads(p.Flag(**_withheld_kw()).model_dump_json())
        assert data["value"] is None and data["interval"] is None and data["payload"] is None


@pytest.mark.category("functional/passport-model")
class TestPassportModel:
    def test_build_and_defaults(self):
        from daypart.memory.lhm import GUARD_VERSIONS

        p = _p()
        flags = (p.Flag(**_withheld_kw()),)
        pp = p.Passport(
            member_id="syn-001",
            as_of=AS_OF,
            ontology_version="core-v1",
            guard_versions=dict(GUARD_VERSIONS),
            flags=flags,
        )
        assert pp.title == "Pattern passport"
        assert pp.flags == flags

    @pytest.mark.parametrize("over", [{"title": "Other"}, {"surprise": True}])
    def test_invalid(self, over):
        p = _p()
        kw = dict(member_id="syn-001", as_of=AS_OF, ontology_version="v", guard_versions={}, flags=())
        kw.update(over)
        with pytest.raises(ValidationError):
            p.Passport(**kw)


# --------------------------------------------------------------------------- #
# E2.2 flags_from_state (hand-built states)
# --------------------------------------------------------------------------- #


@pytest.mark.category("functional/passport-flags")
class TestFlagsSynthetic:
    def test_mapped_only_sorted(self):
        p = _p()
        flags = p.flags_from_state(_all_ready_state(), _test_ontology())
        codes = [f.code for f in flags]
        assert codes == sorted(codes)
        assert set(codes) == {
            "72514-3",
            "dp:behavior.zeta",
            "dp:behavior.alpha",
            "dp:behavior.unsafe_caveat",
            "dp:behavior.no_triad",
        }
        assert "dp:behavior.unmapped" not in codes

    def test_loinc_metric_key(self):
        p = _p()
        f = _by_code(p.flags_from_state(_all_ready_state(), _test_ontology()))["72514-3"]
        assert f.status == "ready"
        assert f.payload.metric_key == "m_72514_3"
        assert f.lhm_domain == "symptom_load"
        assert f.title == "Pain rating (0-10)"
        assert f.value == 42.5
        assert f.interval == (30.0, 55.0)
        assert f.caveats == CAVEATS
        assert f.message is None
        assert f.payload == _payload(key="m_72514_3", label="Pain rating (0-10)", value=42.5)

    def test_payload_round_trips(self):
        from syntropy.lhm.contracts.viz import VizPayload

        p = _p()
        for f in p.flags_from_state(_all_ready_state(), _test_ontology()):
            if f.status == "ready":
                assert VizPayload.model_validate(f.payload.model_dump(mode="json")) == f.payload
                assert f.payload.contract == "lhm-viz/v1"
                assert f.payload.series[0].ts == AS_OF
                assert f.payload.series[0].value == f.value
                assert f.payload.regimen_start is None
                assert f.payload.confidence == "low"
                assert f.payload.rx_input is False
                assert tuple(f.payload.caveats) == f.caveats
                assert f.payload.label == f.title

    def test_title_fallback_on_unsafe_label(self):
        p = _p()
        f = _by_code(p.flags_from_state(_all_ready_state(), _test_ontology()))["dp:behavior.zeta"]
        assert f.title == "Personal pattern"
        assert f.status == "ready"
        assert f.payload.label == "Personal pattern"
        assert f.payload.metric_key == "dp_behavior_zeta"

    def test_title_uses_display_when_no_label(self):
        p = _p()
        f = _by_code(p.flags_from_state(_all_ready_state(), _test_ontology()))["dp:behavior.alpha"]
        assert f.title == "Alpha display"
        assert f.status == "ready"
        assert f.interval is None
        assert f.value == 12.25

    @pytest.mark.parametrize("code", ["dp:behavior.unsafe_caveat", "dp:behavior.no_triad"])
    def test_fail_closed(self, code):
        p = _p()
        f = _by_code(p.flags_from_state(_all_ready_state(), _test_ontology()))[code]
        assert f.status == "withheld"
        assert f.value is None and f.interval is None and f.payload is None
        assert f.caveats == ()
        assert f.message == "Not enough data yet."

    def test_not_ready_is_withheld(self):
        p = _p()
        state = _state([_ns("72514-3", "symptom_load", 42.5, False), _ns("dp:behavior.zeta", "consistency", 9.0, False)])
        flags = p.flags_from_state(state, _test_ontology())
        assert [f.code for f in flags] == ["72514-3", "dp:behavior.zeta"]
        for f in flags:
            assert f.status == "withheld"
            assert f.value is None and f.interval is None and f.payload is None
            assert f.message == "Not enough data yet."
        assert flags[1].title == "Personal pattern"

    def test_ready_value_is_exact(self):
        p = _p()
        v = 33.333333333333336
        state = _state([_ns("72514-3", "symptom_load", v, True)])
        (f,) = p.flags_from_state(state, _test_ontology())
        assert f.value == v

    def test_deterministic(self):
        p = _p()
        assert p.flags_from_state(_all_ready_state(), _test_ontology()) == p.flags_from_state(
            _all_ready_state(), _test_ontology()
        )

    def test_passport_from_state(self):
        p = _p()
        st = _all_ready_state()
        pp = p.passport_from_state(st, _test_ontology())
        assert isinstance(pp, p.Passport)
        assert pp.title == "Pattern passport"
        assert pp.member_id == "syn-001"
        assert pp.as_of == AS_OF
        assert pp.ontology_version == "sealed-test-v1"
        assert dict(pp.guard_versions) == dict(st.guard_versions)
        assert pp.flags == p.flags_from_state(st, _test_ontology())


# --------------------------------------------------------------------------- #
# E2.2 / E2.3 over the real cohort
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def adapter_m():
    from daypart.simmember.cohort import generate
    from daypart.simmember.lhm_records import SimMemberLhmAdapter

    cohort = generate(0)
    return cohort, SimMemberLhmAdapter(cohort)


@pytest.fixture
def memory(adapter_m):
    from daypart.memory.lhm import MemoryLhm

    return MemoryLhm(adapter_m[1])


def _member_ids(cohort):
    return [p.member_id for p in cohort.profiles]


@pytest.mark.category("functional/passport-flags")
class TestFlagsCore:
    async def test_four_core_nodes(self, memory):
        p = _p()
        st = await memory.metric_state("syn-001", day(41))
        flags = p.flags_from_state(st)
        assert tuple(f.code for f in flags) == CORE_FOUR
        assert flags == p.flags_from_state(st, None)

    async def test_ready_card_matches_state(self, memory):
        from daypart.memory.ontology import core

        p = _p()
        ont = core()
        st = await memory.metric_state("syn-001", day(41))
        for f in p.flags_from_state(st):
            ns = st.nodes[f.code]
            node = ont.node(f.code)
            assert f.lhm_domain == node.lhm_domain
            assert f.title == (node.label or node.display)
            if ns.ready:
                assert f.status == "ready"
                assert f.value == ns.value
                assert f.interval == (None if ns.score.interval is None else tuple(ns.score.interval))
                assert f.caveats == tuple(node.caveats)
                key = f.code.lower().replace(":", "_").replace(".", "_")
                assert f.payload == _payload(key=key, label=f.title, value=ns.value, ts=st.as_of, caveats=node.caveats)
            else:
                assert f.status == "withheld"
                assert f.value is None and f.payload is None and f.interval is None

    async def test_metric_keys_core(self, memory):
        p = _p()
        st = await memory.metric_state("syn-001", day(41))
        keys = {f.code: f.payload.metric_key for f in p.flags_from_state(st) if f.payload is not None}
        assert keys.get("dp:adherence.protocol") == "dp_adherence_protocol"
        expected = {
            "dp:adherence.protocol": "dp_adherence_protocol",
            "dp:affect.checkin_sentiment": "dp_affect_checkin_sentiment",
            "dp:nutrition.food_log": "dp_nutrition_food_log",
            "dp:symptom.severity": "dp_symptom_severity",
        }
        for code, key in keys.items():
            assert key == expected[code]

    async def test_deterministic_equal_states(self, memory, adapter_m):
        from daypart.memory.lhm import MemoryLhm

        p = _p()
        a = await memory.metric_state("syn-088", day(41))
        b = await MemoryLhm(adapter_m[1]).metric_state("syn-088", day(41))
        assert p.flags_from_state(a) == p.flags_from_state(b)


@pytest.mark.category("integration/passport-render")
class TestRenderer:
    async def test_day4_adherence_withheld_everyone(self, memory, adapter_m):
        p = _p()
        r = p.PassportRenderer(memory)
        for mid in _member_ids(adapter_m[0]):
            f = _by_code(await r.render(mid, day(4)))["dp:adherence.protocol"]
            assert f.status == "withheld", mid
            assert f.value is None and f.payload is None

    async def test_day41_adherence_ready_95pct(self, memory, adapter_m):
        p = _p()
        r = p.PassportRenderer(memory)
        mids = _member_ids(adapter_m[0])
        ready = 0
        for mid in mids:
            f = _by_code(await r.render(mid, day(41)))["dp:adherence.protocol"]
            if f.status == "ready":
                assert 0.0 <= f.value <= 100.0
                ready += 1
        assert ready / len(mids) >= 0.95, (ready, len(mids))

    @pytest.mark.parametrize("mid", ["cur-001", "syn-001", "syn-175"])
    async def test_render_equals_flags_from_state(self, memory, mid):
        from daypart.memory.ontology import core

        p = _p()
        for r in (p.PassportRenderer(memory), p.PassportRenderer(memory, core())):
            for n in (4, 20, 41):
                got = await r.render(mid, day(n))
                assert got == p.flags_from_state(await memory.metric_state(mid, day(n)), core())

    async def test_passport(self, memory):
        from daypart.memory.lhm import GUARD_VERSIONS

        p = _p()
        r = p.PassportRenderer(memory)
        pp = await r.passport("syn-001", day(41))
        assert isinstance(pp, p.Passport)
        assert pp.title == "Pattern passport"
        assert pp.member_id == "syn-001"
        assert pp.as_of == day(41)
        assert pp.ontology_version == "core-v1"
        assert dict(pp.guard_versions) == dict(GUARD_VERSIONS)
        assert pp.flags == await r.render("syn-001", day(41))
        assert p.Passport.model_validate_json(pp.model_dump_json()) == pp

    async def test_unknown_member_all_withheld(self, memory):
        p = _p()
        flags = await p.PassportRenderer(memory).render("syn-999", day(41))
        assert tuple(f.code for f in flags) == CORE_FOUR
        for f in flags:
            assert f.status == "withheld"
            assert f.value is None and f.interval is None and f.payload is None
            assert f.message == "Not enough data yet."

    @pytest.mark.parametrize("bad", ["someone@example.com", "", "Syn-001", "member 1"])
    async def test_invalid_member_id(self, memory, bad):
        p = _p()
        with pytest.raises(ValueError):
            await p.PassportRenderer(memory).render(bad, day(41))

    @pytest.mark.parametrize("n", [4, 41])
    async def test_json_no_number_for_withheld(self, memory, adapter_m, n):
        p = _p()
        r = p.PassportRenderer(memory)
        for mid in _member_ids(adapter_m[0])[:: 25]:
            data = json.loads((await r.passport(mid, day(n))).model_dump_json())
            assert data["title"] == "Pattern passport"
            for f in data["flags"]:
                if f["status"] == "withheld":
                    assert f["value"] is None and f["interval"] is None and f["payload"] is None
                else:
                    assert isinstance(f["value"], (int, float)) and math.isfinite(f["value"])

    async def test_no_member_notes_in_json(self, memory, adapter_m):
        p = _p()
        _, adapter = adapter_m
        r = p.PassportRenderer(memory)
        checked = 0
        for mid in ("cur-001", "cur-013", "syn-001", "syn-088", "syn-175"):
            notes = {
                pt.note
                for rec in await adapter.read_outcome_series(mid)
                for pt in rec.points
                if getattr(pt, "note", None)
            }
            text = (await r.passport(mid, day(41))).model_dump_json()
            for note in notes:
                checked += 1
                assert note not in text, (mid, note)
        assert checked > 0
