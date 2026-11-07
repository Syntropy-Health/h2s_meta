"""HAC-38 P1 task 2 — real-input gate is hard off (AD7, A&D §4.9, R9 T9.1–T9.3, T9.5)."""

import logging
import re
from importlib.resources import files
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from daypart.demo.app import create_app
from daypart.governance import real_input
from daypart.governance.real_input import (
    REAL_INPUT_BUILD_ALLOWED,
    Gate,
    load_checklist,
    request_enable,
    requested_from_env,
    status,
)

GATE_IDS = (
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
)
DOC = Path(__file__).resolve().parents[3] / "docs" / "daypart" / "real-input-pathway.md"
YAML_TEXT = files("daypart.governance").joinpath("real_input.yaml").read_text(encoding="utf-8")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("DAYPART_REAL_INPUT", raising=False)
    with TestClient(create_app(cohort_seed=0)) as c:
        yield c


def _closed_checklist() -> tuple[Gate, ...]:
    return tuple(
        g.model_copy(update={"status": "closed", "owner": "A. Synthetic", "evidence": "https://example.org/e"})
        for g in load_checklist()
    )


# --- checklist -----------------------------------------------------------------------------


def test_build_constant_is_false():
    assert REAL_INPUT_BUILD_ALLOWED is False


def test_checklist_has_the_12_ids_in_order_all_open_unassigned():
    gates = load_checklist()
    assert tuple(g.id for g in gates) == GATE_IDS
    assert all(g.status == "open" and g.owner == "unassigned" and g.evidence is None for g in gates)
    assert all(g.label.strip() for g in gates)


def test_gate_is_frozen_and_rejects_bad_status():
    g = load_checklist()[0]
    with pytest.raises(ValidationError):
        g.status = "closed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Gate(id="x", label="x", status="maybe", owner="unassigned", evidence=None)


def test_checklist_missing_id_raises(tmp_path):
    data = yaml.safe_load(YAML_TEXT)
    items = data["gates"] if isinstance(data, dict) else data
    del items[3]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_checklist(p)


def test_checklist_reordered_ids_raise(tmp_path):
    data = yaml.safe_load(YAML_TEXT)
    items = data["gates"] if isinstance(data, dict) else data
    items[0], items[1] = items[1], items[0]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_checklist(p)


# --- status / request_enable ---------------------------------------------------------------


def test_status_default_is_disabled_with_12_open_gates():
    s = status()
    assert s.enabled is False and s.build_allowed is False
    assert tuple(g.id for g in s.open_gates) == GATE_IDS
    assert s.requested is False


def test_request_enable_refuses_build_disallowed():
    r = request_enable()
    assert r.reason == "build_disallowed"
    assert tuple(g.id for g in r.open_gates) == GATE_IDS


def test_t9_3_all_closed_still_refused(monkeypatch):
    closed = _closed_checklist()
    s = status(closed)
    assert s.open_gates == () and s.enabled is False
    assert request_enable(closed).reason == "build_disallowed"
    # also when the module's loader is monkeypatched to return the closed checklist
    monkeypatch.setattr(real_input, "load_checklist", lambda path=None: closed)
    assert real_input.status().enabled is False
    assert real_input.request_enable().reason == "build_disallowed"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("on", True),
        ("true", True),
        ("TRUE", True),
        (" On ", True),
        ("0", False),
        ("", False),
        ("off", False),
        ("yes", False),
        ("false", False),
        (None, False),
    ],
)
def test_requested_from_env_truth_table(value, expected):
    env = {} if value is None else {"DAYPART_REAL_INPUT": value}
    assert requested_from_env(env) is expected


# --- /api/mode (T9.1, T9.2) ----------------------------------------------------------------


def test_get_mode_default(client):
    body = client.get("/api/mode").json()
    assert body["mode"] == "demo"
    ri = body["real_input"]
    assert ri["enabled"] is False and ri["requested"] is False
    assert [g["id"] for g in ri["open_gates"]] == list(GATE_IDS)
    assert all(set(g) == {"id", "label", "owner"} for g in ri["open_gates"])


@pytest.mark.parametrize("flag", [True, False])
def test_t9_1_post_mode_always_403(client, flag):
    # Choice: the endpoint refuses regardless of the requested value (A&D: "always").
    r = client.post("/api/mode", json={"real_input": flag})
    assert r.status_code == 403
    body = r.json()
    assert body["error"] == "real_input_refused"
    assert body["reason"] == "build_disallowed"
    assert "detail" not in body
    assert len(body["open_gates"]) == 12
    assert [g["id"] for g in body["open_gates"]] == list(GATE_IDS)


def test_post_mode_rejects_extra_fields(client):
    assert client.post("/api/mode", json={"real_input": True, "force": True}).status_code == 422


def test_t9_2_env_requested_starts_in_demo_and_logs_once(monkeypatch, caplog):
    monkeypatch.setenv("DAYPART_REAL_INPUT", "1")
    with caplog.at_level(logging.WARNING, logger="daypart.demo"):
        app = create_app(cohort_seed=0)
    lines = [r for r in caplog.records if r.name == "daypart.demo" and r.levelno >= logging.WARNING]
    assert len(lines) == 1
    msg = lines[0].getMessage()
    assert "real input requested but refused: build_disallowed" in msg
    assert all(gid in msg for gid in GATE_IDS)
    with TestClient(app) as c:
        body = c.get("/api/mode").json()
        assert body["mode"] == "demo"
        assert body["real_input"]["requested"] is True
        assert body["real_input"]["enabled"] is False
        assert c.get("/api/health").status_code == 200
        assert c.post("/api/mode", json={"real_input": True}).status_code == 403


def test_no_log_when_not_requested(monkeypatch, caplog):
    monkeypatch.delenv("DAYPART_REAL_INPUT", raising=False)
    with caplog.at_level(logging.WARNING, logger="daypart.demo"):
        create_app(cohort_seed=0)
    assert not [r for r in caplog.records if r.name == "daypart.demo"]


# --- T9.5: pathway doc matches YAML --------------------------------------------------------


def _doc_rows() -> list[dict]:
    rows = []
    header = None
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("|"):
            if header is not None and rows:
                break
            continue
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if header is None:
            if [c.lower() for c in cells] == ["id", "gate", "owner", "status"]:
                header = cells
            continue
        if all(re.fullmatch(r":?-+:?", c) for c in cells):
            continue
        rows.append(dict(zip(("id", "gate", "owner", "status"), cells, strict=True)))
    assert header is not None, "doc has no | id | gate | owner | status | table"
    return rows


def test_t9_5_doc_table_matches_yaml():
    rows = _doc_rows()
    gates = load_checklist()
    assert [(r["id"], r["owner"], r["status"]) for r in rows] == [(g.id, g.owner, g.status) for g in gates]
    assert [r["gate"] for r in rows] == [g.label for g in gates]
    assert all(r["owner"] == "unassigned" or re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", r["owner"]) for r in rows)


def test_doc_states_hard_off_and_enforcement():
    text = DOC.read_text(encoding="utf-8")
    for needle in ("REAL_INPUT_BUILD_ALLOWED", "/api/mode", "DAYPART_REAL_INPUT", "AD7"):
        assert needle in text
