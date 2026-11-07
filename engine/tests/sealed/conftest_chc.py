"""Shared helpers for the Clinical Human Clone P1 sealed suite (HAC-38 / HAC-39).

Written double-blind from the approved A&D (HAC-33, sections "Components & interfaces",
"Data & state" 4.1-4.9, "Failure handling", "Per-requirement spec") as amended by CTO AM-1
(HAC-38, 2026-11-07), plus the locked plan's interface decisions G4, G8 and G9.

Imported explicitly by the ``test_chc_*.py`` files. Every implementation import happens inside
a function, so collection never depends on the implementation existing.

Observation points used by the suite (all black-box at the contract boundary):
- HTTP: ``daypart.demo.app.create_app()`` through ``fastapi.testclient.TestClient``.
- Writes: ``EventLog.append`` (the one shared write path, J1) is wrapped to count the events
  that are actually appended; nothing else is patched.
- Pending store: live ``CheckIn`` objects (A&D 4.1, ``daypart.checkin.models.CheckIn``) are
  counted through ``gc`` to prove a refused check-in leaves none behind.
"""

from __future__ import annotations

import gc
import json
from collections.abc import Iterator, Mapping
from importlib.resources import files
from typing import Any

import pytest

# AM-1 item 1: the only review label in this build.
REVIEWED_LABEL = "reviewed: board sign-off 2026-11-07"

# A&D 4.1: the closed flag vocabulary; every flag is blocking.
FLAG_KINDS: tuple[str, ...] = (
    "unit_mismatch",
    "unit_assumed",
    "ambiguous_date",
    "contradiction",
    "unknown_code",
    "out_of_plausible_range",
    "missing_value",
)

# A&D 4.9: the 12 R9 governance gates.
GATE_IDS: frozenset[str] = frozenset(
    {
        "lawful_basis",
        "consent",
        "access_control",
        "encryption",
        "retention_deletion",
        "audit",
        "incident_response",
        "processor_terms",
        "jurisdiction_review",
        "clinical_intended_use",
        "qualified_oversight",
        "regulatory_review",
    }
)

VERTICALS: frozenset[str] = frozenset({"vitals_body", "behaviour_function", "symptoms_wellbeing"})

MIB = 1024 * 1024
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# R5 inspect-view fields (A&D 4.2, ObservationRecord).
OBSERVATION_FIELDS: tuple[str, ...] = (
    "record_id",
    "event_id",
    "concept_id",
    "loinc_code",
    "loinc_version",
    "display",
    "value",
    "unit",
    "value_type",
    "observed_at",
    "provenance",
    "quality",
    "status",
    "reviewer_status",
    "allowed_use",
)
PROVENANCE_FIELDS: tuple[str, ...] = ("source_kind", "source_ref", "checkin_id", "method")


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #


def string_leaves(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            yield str(k)
            yield from string_leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from string_leaves(v)


def body(resp) -> Any:
    try:
        return resp.json()
    except Exception:  # noqa: BLE001 - a non-JSON body is reported by the caller's assert
        return resp.text


def has_error(resp, code: str) -> bool:
    """True iff the error code appears as a string value anywhere in the JSON body.

    The A&D names the error codes but not the envelope (``{"error": ...}`` or FastAPI's
    ``{"detail": ...}``), so any placement is accepted.
    """
    return any(s == code for s in string_leaves(body(resp)))


def find_key(obj: Any, key: str) -> Any:
    """First value stored under ``key`` anywhere in a JSON body (None if absent)."""
    if isinstance(obj, Mapping):
        if key in obj:
            return obj[key]
        for v in obj.values():
            hit = find_key(v, key)
            if hit is not None:
                return hit
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            hit = find_key(v, key)
            if hit is not None:
                return hit
    return None


def get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


# --------------------------------------------------------------------------- #
# App + observation fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def chc_app(monkeypatch):
    monkeypatch.delenv("DAYPART_REAL_INPUT", raising=False)
    from daypart.demo.app import create_app

    return create_app()


@pytest.fixture
def client(chc_app):
    from fastapi.testclient import TestClient

    with TestClient(chc_app) as c:
        yield c


class WriteLog:
    """Events successfully appended to any ``EventLog`` while the fixture is active."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    @property
    def signals(self) -> list[Any]:
        return [e for e in self.events if get(e, "kind") == "signal"]

    @property
    def actions(self) -> list[Any]:
        return [e for e in self.events if get(e, "kind") == "action"]

    def __len__(self) -> int:
        return len(self.events)

    def clear(self) -> None:
        self.events.clear()


@pytest.fixture
def writes(monkeypatch) -> WriteLog:
    from daypart.memory.events import EventLog

    log = WriteLog()
    original = EventLog.append

    def counting_append(self, event):
        original(self, event)
        log.events.append(event)

    monkeypatch.setattr(EventLog, "append", counting_append)
    return log


def live_checkins() -> int:
    """Live pending-store ``CheckIn`` objects (A&D 4.1)."""
    gc.collect()
    n = 0
    for obj in gc.get_objects():
        t = type(obj)
        if t.__name__ == "CheckIn" and t.__module__.startswith("daypart.checkin"):
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Journey helpers (A&D API table)
# --------------------------------------------------------------------------- #


def a_member_id(client) -> str:
    resp = client.get("/api/members")
    assert resp.status_code == 200, resp.text
    for m in resp.json():
        mid = m.get("member_id") or m.get("id")
        if isinstance(mid, str) and mid.startswith("syn-"):
            return mid
    raise AssertionError("no syn- member listed by /api/members")


def new_journal(client) -> dict:
    resp = client.post("/api/journals", json={"member_id": a_member_id(client)})
    assert resp.status_code == 201, f"POST /api/journals -> {resp.status_code} {resp.text}"
    j = resp.json()
    for key in ("journal_id", "member_id", "today"):
        assert key in j, f"journal response lacks {key!r}: {j}"
    return j


def post_text(client, journal_id: str, text: str):
    return client.post("/api/checkins", json={"journal_id": journal_id, "text": text})


def post_fixture(client, journal_id: str, fixture_id: str):
    return client.post("/api/checkins", json={"journal_id": journal_id, "fixture_id": fixture_id})


def post_upload(client, journal_id: str, data: bytes, filename: str, content_type: str):
    return client.post(
        "/api/checkins",
        data={"journal_id": journal_id},
        files={"file": (filename, data, content_type)},
    )


def checkin_ok(resp) -> dict:
    assert resp.status_code == 201, f"check-in -> {resp.status_code} {resp.text}"
    c = resp.json()
    assert c.get("status") == "pending_review", c
    assert isinstance(c.get("checkin_id"), str) and c["checkin_id"], c
    assert isinstance(c.get("proposals"), list), c
    return c


def review(client, checkin_id: str, decisions: list[dict]):
    return client.post(f"/api/checkins/{checkin_id}/review", json={"decisions": decisions})


def discard_all_but(proposals: list[dict], keep: dict[str, dict]) -> list[dict]:
    """One decision per proposal: ``keep[proposal_id]`` where given, else ``discard``."""
    out = []
    for p in proposals:
        pid = p["proposal_id"]
        out.append({"proposal_id": pid, **keep[pid]} if pid in keep else {"proposal_id": pid, "action": "discard"})
    return out


def by_concept(proposals: list[dict], code: str) -> list[dict]:
    return [p for p in proposals if p.get("concept_code") == code]


def find_observations(client, journal_id: str, limit: int = 40) -> list[dict]:
    """All inspectable records of a journal; ids are ``obs-<n>`` (A&D 4.2)."""
    out = []
    for n in range(limit):
        r = client.get(f"/api/journals/{journal_id}/observations/obs-{n}")
        if r.status_code == 200:
            out.append(r.json())
    return out


def passport_or_skip(client, journal_id: str) -> dict:
    """GET passport; skip (not pass) when the P2 route is not registered yet."""
    paths = client.get("/openapi.json").json().get("paths", {})
    if not any(p.startswith("/api/journals/") and p.endswith("/passport") and "get" in ops for p, ops in paths.items()):
        pytest.skip("GET /api/journals/{id}/passport not registered yet (P2, plan task 7)")
    resp = client.get(f"/api/journals/{journal_id}/passport")
    assert resp.status_code == 200, resp.text
    return resp.json()


def observed_cards(passport: dict, codes: set[str], concept_ids: set[str]) -> list[dict]:
    cards = passport.get("cards", [])
    return [
        c
        for c in cards
        if c.get("origin") == "observed"
        and (c.get("concept_id") in concept_ids or c.get("concept_id") in codes)
    ]


# --------------------------------------------------------------------------- #
# Contract data files
# --------------------------------------------------------------------------- #


def _load_json_resource(package: str, name: str) -> Any:
    return json.loads(files(package).joinpath(name).read_text(encoding="utf-8"))


def registry_file() -> Any:
    return _load_json_resource("daypart.memory", "registry_a4.json")


_REGISTRY_API_NAMES = ("concepts", "all_concepts", "load_registry", "load_concepts", "registry", "CONCEPTS", "load")


def _as_dict(obj: Any) -> dict:
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "__dataclass_fields__"):
        import dataclasses

        return dataclasses.asdict(obj)
    return dict(vars(obj))


def _concept_list(raw: Any) -> list[dict] | None:
    if hasattr(raw, "concepts") and not isinstance(raw, Mapping) and not callable(raw.concepts):
        raw = raw.concepts
    if isinstance(raw, Mapping):
        raw = raw.get("concepts", list(raw.values()) if all(isinstance(v, Mapping) for v in raw.values()) else None)
    if raw is None:
        return None
    items = [_as_dict(c) for c in raw]
    return items if items and all("code" in c for c in items) else None


def registry_concepts() -> list[dict]:
    """The A4 concepts (A&D 4.3), as dicts.

    Prefers the Python API of ``daypart.memory.registry`` (so model defaults are visible); falls
    back to ``memory/registry_a4.json`` (a list, or under a ``concepts`` key).
    """
    import daypart.memory.registry as reg

    for name in _REGISTRY_API_NAMES:
        attr = getattr(reg, name, None)
        if attr is None:
            continue
        try:
            raw = attr() if callable(attr) else attr
            found = _concept_list(raw)
        except Exception:  # noqa: BLE001 - try the next candidate, then the JSON file
            found = None
        if found:
            return found
    found = _concept_list(registry_file())
    assert found, "neither daypart.memory.registry nor registry_a4.json exposes a concept list"
    return found


def concepts_by_code() -> dict[str, dict]:
    return {c["code"]: c for c in registry_concepts()}


def sources() -> dict[str, dict]:
    """``evidence/sources_v1.json`` keyed by evidence_id; empty if the file does not exist yet."""
    try:
        raw = _load_json_resource("daypart.evidence", "sources_v1.json")
    except (FileNotFoundError, ModuleNotFoundError):
        return {}
    items = raw.get("sources", raw) if isinstance(raw, Mapping) else raw
    if isinstance(items, Mapping):  # keyed by evidence_id
        return {k: {"evidence_id": k, **v} for k, v in items.items()}
    return {s["evidence_id"]: s for s in items}


def plausible_bounds(concept: Mapping) -> tuple[float, float]:
    p = concept["plausible"]
    if isinstance(p, Mapping):
        lo = p.get("lo", p.get("low", p.get("min")))
        hi = p.get("hi", p.get("high", p.get("max")))
    else:
        lo, hi = p
    return float(lo), float(hi)


def valid_correction(proposal: Mapping, today: str, concepts: Mapping[str, dict]) -> dict:
    """A ``correct`` decision that satisfies every A&D validation rule for the concept."""
    concept = concepts[proposal["concept_code"]]
    lo, hi = plausible_bounds(concept)
    value = proposal.get("value")
    if value is None or not (lo <= value <= hi):
        value = (lo + hi) / 2
    if concept.get("value_type") in ("integer", "ordinal"):
        value = int(round(value))
        value = min(max(value, int(-(-lo // 1))), int(hi // 1))
    return {"action": "correct", "value": value, "unit": concept["unit"], "date": today}


def fixture_manifest() -> dict[str, dict]:
    """``demo/fixtures/manifest.json`` (plan G8): fixture_id -> {file, sha256, type, size}."""
    raw = _load_json_resource("daypart.demo", "fixtures/manifest.json")
    items = raw.get("fixtures", raw) if isinstance(raw, Mapping) else raw
    if isinstance(items, Mapping):
        return {k: dict(v) for k, v in items.items()}
    return {i["fixture_id"]: dict(i) for i in items}


def fixture_bytes(entry: Mapping) -> bytes:
    return files("daypart.demo").joinpath("fixtures").joinpath(entry["file"]).read_bytes()


def fixture_kind(fixture_id: str, entry: Mapping) -> str:
    """'image' or 'data' from the manifest type, falling back to the file suffix."""
    t = str(entry.get("type", "")).lower()
    f = str(entry.get("file", fixture_id)).lower()
    if any(x in t for x in ("png", "jpeg", "jpg", "image")) or f.endswith((".png", ".jpg", ".jpeg")):
        return "image"
    if any(x in t for x in ("csv", "json")) or f.endswith((".csv", ".json")):
        return "data"
    return "other"
