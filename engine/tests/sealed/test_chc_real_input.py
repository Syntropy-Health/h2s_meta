"""Sealed: real-input gate (R9) — T9.1, T9.2, T9.3.

Spec: A&D HAC-33 AD7, API table (``GET/POST /api/mode``), 4.9 (``real_input.status()``,
``request_enable()``, ``REAL_INPUT_BUILD_ALLOWED``, env ``DAYPART_REAL_INPUT``), T9.1-T9.3.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from tests.sealed.conftest_chc import (
    GATE_IDS,
    MIB,
    PNG_MAGIC,
    body,
    chc_app,  # noqa: F401 - fixture
    client,  # noqa: F401 - fixture
    find_key,
    fixture_manifest,
    get,
    has_error,
    new_journal,
    post_fixture,
    post_upload,
)

pytestmark = pytest.mark.category("integration/chc-real-input")


def _gates(payload) -> list:
    gates = find_key(payload, "open_gates")
    assert isinstance(gates, list), f"no open_gates list in {payload}"
    return gates


# --------------------------------------------------------------------------- #
# T9.1  POST /api/mode {"real_input": true} -> 403 with 12 open gates
# --------------------------------------------------------------------------- #


class TestT91ModeEndpoint:
    def test_t9_1_get_mode_reports_demo_and_disabled(self, client):
        """API table: GET /api/mode -> mode demo, real_input.enabled false, 12 open gates."""
        r = client.get("/api/mode")
        assert r.status_code == 200, r.text
        m = r.json()
        assert m["mode"] == "demo"
        ri = m["real_input"]
        assert ri["enabled"] is False
        assert ri["requested"] is False
        gates = ri["open_gates"]
        assert len(gates) == 12
        assert {g["id"] for g in gates} == GATE_IDS
        for g in gates:
            assert isinstance(g.get("label"), str) and g["label"].strip(), g
            assert "owner" in g, g

    def test_t9_1_enable_is_refused_with_12_open_gates(self, client):
        """T9.1: POST /api/mode {"real_input": true} -> 403 real_input_refused, 12 open gates."""
        r = client.post("/api/mode", json={"real_input": True})
        assert r.status_code == 403, r.text
        assert has_error(r, "real_input_refused"), body(r)
        gates = _gates(body(r))
        assert len(gates) == 12
        ids = {g["id"] if isinstance(g, dict) else g for g in gates}
        assert ids == GATE_IDS
        assert has_error(r, "build_disallowed"), body(r)

    def test_t9_1_refusal_does_not_change_mode(self, client):
        """T9.1: after a refused enable, GET /api/mode still reports demo, enabled false."""
        client.post("/api/mode", json={"real_input": True})
        m = client.get("/api/mode").json()
        assert m["mode"] == "demo"
        assert m["real_input"]["enabled"] is False

    @pytest.mark.parametrize("payload", [{"real_input": True}, {"real_input": "true"}, {"real_input": 1}])
    def test_t9_1_never_2xx(self, client, payload):
        """API table: POST /api/mode never returns 2xx."""
        assert client.get("/api/mode").status_code == 200
        r = client.post("/api/mode", json=payload)
        assert not (200 <= r.status_code < 300), r.text


# --------------------------------------------------------------------------- #
# T9.2  env DAYPART_REAL_INPUT=1 -> app starts, requested=true, enabled=false, fixtures only
# --------------------------------------------------------------------------- #


@pytest.fixture(params=["1", "on", "true"])
def env_client(request, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DAYPART_REAL_INPUT", request.param)
    from daypart.demo.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


class TestT92EnvRequest:
    def test_t9_2_app_starts_requested_but_disabled(self, env_client):
        """T9.2: with DAYPART_REAL_INPUT set, create_app() starts; /api/mode -> requested=true,
        enabled=false, still demo mode."""
        r = env_client.get("/api/mode")
        assert r.status_code == 200, r.text
        m = r.json()
        assert m["mode"] == "demo"
        assert m["real_input"]["requested"] is True
        assert m["real_input"]["enabled"] is False
        assert len(m["real_input"]["open_gates"]) == 12

    def test_t9_2_still_refuses_enable(self, env_client):
        """T9.2: the env request does not make POST /api/mode succeed."""
        r = env_client.post("/api/mode", json={"real_input": True})
        assert r.status_code == 403, r.text
        assert has_error(r, "real_input_refused")

    def test_t9_2_checkins_accept_fixtures_only(self, env_client):
        """T9.2: check-in endpoints still accept the bundled fixtures, and refuse a non-demo upload."""
        j = new_journal(env_client)
        fid = next(iter(fixture_manifest()))
        ok = post_fixture(env_client, j["journal_id"], fid)
        assert ok.status_code == 201, ok.text
        foreign = PNG_MAGIC + b"\x00\x00\x00\rIHDR" + b"real-input-probe" * 8
        r = post_upload(env_client, j["journal_id"], foreign, "photo.png", "image/png")
        assert r.status_code == 422, r.text
        assert has_error(r, "not_a_demo_fixture"), body(r)
        assert len(foreign) < MIB

    def test_t9_2_unset_env_is_not_requested(self, client):
        """T9.2 (control): without the env, requested is false."""
        assert client.get("/api/mode").json()["real_input"]["requested"] is False


# --------------------------------------------------------------------------- #
# T9.3  every checklist item closed -> still enabled=false (build_disallowed)
# --------------------------------------------------------------------------- #


def _closed(item):
    update = {"status": "closed", "owner": "sealed-test-owner", "evidence": "https://example.org/evidence"}
    if isinstance(item, dict):
        return {**item, **update}
    if hasattr(item, "model_copy"):
        return item.model_copy(update=update)
    import dataclasses

    if dataclasses.is_dataclass(item):
        return dataclasses.replace(item, **update)
    raise TypeError(f"cannot close checklist item of type {type(item).__name__}")


def _is_item_seq(v) -> bool:
    return (
        isinstance(v, (list, tuple))
        and len(v) == 12
        and all(get(i, "id") in GATE_IDS and get(i, "status") is not None for i in v)
    )


@pytest.fixture
def all_closed(monkeypatch, tmp_path):
    """Monkeypatch the governance checklist so that all 12 items are closed.

    The A&D fixes the module (``governance/real_input.py``) and its YAML, not the loader name, so
    every injection point is patched: a module path to the YAML, any checklist-loading function,
    and any module-level sequence of the 12 items. Caches are cleared.
    """
    import daypart.governance.real_input as ri

    yaml_path = Path(ri.__file__).with_name("real_input.yaml")
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    closed_raw = copy.deepcopy(raw)
    items = closed_raw.get("items", closed_raw.get("gates", closed_raw)) if isinstance(closed_raw, dict) else closed_raw
    if isinstance(items, dict):  # keyed by id
        for v in items.values():
            v.update(status="closed", owner="sealed-test-owner", evidence="https://example.org/evidence")
    else:
        for v in items:
            v.update(status="closed", owner="sealed-test-owner", evidence="https://example.org/evidence")
    closed_yaml = tmp_path / "real_input.yaml"
    closed_yaml.write_text(yaml.safe_dump(closed_raw), encoding="utf-8")

    for name in dir(ri):
        if name.startswith("__"):
            continue
        v = getattr(ri, name)
        if isinstance(v, (str, Path)) and str(v).endswith("real_input.yaml"):
            monkeypatch.setattr(ri, name, type(v)(closed_yaml) if isinstance(v, Path) else str(closed_yaml))
        elif _is_item_seq(v):
            monkeypatch.setattr(ri, name, type(v)(_closed(i) for i in v))
        elif callable(v) and getattr(v, "__module__", None) == ri.__name__ and (
            "checklist" in name.lower() or name.lower().startswith("load")
        ):
            if hasattr(v, "cache_clear"):
                v.cache_clear()
            original = v

            def patched(*a, _orig=original, **k):
                out = _orig(*a, **k)
                if _is_item_seq(out):
                    return type(out)(_closed(i) for i in out)
                seq = get(out, "items")
                if _is_item_seq(seq):
                    return {**out, "items": [_closed(i) for i in seq]} if isinstance(out, dict) else out
                return out

            monkeypatch.setattr(ri, name, patched)
    for name in dir(ri):
        v = getattr(ri, name, None)
        if hasattr(v, "cache_clear"):
            v.cache_clear()

    st = ri.status()
    if get(st, "open_gates"):
        pytest.fail(
            "sealed harness could not inject a closed checklist into daypart.governance.real_input "
            "(status().open_gates still non-empty); report to the Test Author"
        )
    return ri


class TestT93BuildDisallowed:
    def test_t9_3_constant_is_false(self):
        """AD7 / 4.9: REAL_INPUT_BUILD_ALLOWED is False in this build."""
        import daypart.governance.real_input as ri

        assert ri.REAL_INPUT_BUILD_ALLOWED is False

    def test_t9_3_status_default(self):
        """4.9: status() -> enabled False, build_allowed False, 12 open gates."""
        import daypart.governance.real_input as ri

        st = ri.status()
        assert get(st, "enabled") is False
        assert get(st, "build_allowed") is False
        assert len(get(st, "open_gates")) == 12

    def test_t9_3_all_closed_still_disabled(self, all_closed):
        """T9.3: with every checklist item closed, status().enabled is still False."""
        st = all_closed.status()
        assert get(st, "enabled") is False
        assert get(st, "build_allowed") is False
        assert not get(st, "open_gates")

    def test_t9_3_all_closed_request_enable_refused(self, all_closed):
        """T9.3 / components table: request_enable() is always Refused (build_disallowed)."""
        r = all_closed.request_enable()
        text = repr(r) + " " + str(get(r, "reason", ""))
        assert "Refused" in type(r).__name__ or "refused" in text.lower(), r
        assert "build_disallowed" in text, r

    def test_t9_3_all_closed_api_refuses(self, all_closed, monkeypatch):
        """T9.3: with every item closed, POST /api/mode is still 403 build_disallowed and
        GET /api/mode still reports enabled=false."""
        from fastapi.testclient import TestClient

        monkeypatch.delenv("DAYPART_REAL_INPUT", raising=False)
        from daypart.demo.app import create_app

        with TestClient(create_app()) as c:
            r = c.post("/api/mode", json={"real_input": True})
            assert r.status_code == 403, r.text
            assert has_error(r, "real_input_refused")
            assert has_error(r, "build_disallowed")
            assert c.get("/api/mode").json()["real_input"]["enabled"] is False
