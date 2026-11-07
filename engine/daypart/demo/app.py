"""FastAPI playground (HAC-20). Offline, no secrets, synthetic members only (I1)."""

from __future__ import annotations

import json
import logging
import os
from importlib.resources import files
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException

from daypart.checkin import fixtures, service
from daypart.checkin.journal import Journal, JournalError, JournalStore
from daypart.demo.playground import (
    MAX_HORIZON,
    EpisodeDone,
    Playground,
    UnknownEpisode,
    UnknownMember,
)
from daypart.demo.safety import safe_text
from daypart.governance import real_input
from daypart.simmember.dynamics import INTENTS

log = logging.getLogger("daypart.demo")

IntentName = Annotated[str, Field(json_schema_extra={"enum": list(INTENTS)})]


class EpisodeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    member_id: str
    seed: int = 0
    horizon: int = Field(14, ge=1, le=MAX_HORIZON)


class StepIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: IntentName
    text: str | None = Field(None, max_length=2000)  # None -> the intent's boundary-safe default


def _gate_view(gates: tuple[real_input.Gate, ...]) -> list[dict]:
    return [{"id": g.id, "label": g.label, "owner": g.owner} for g in gates]


# G7: the whole request body (JSON or multipart) is capped at the upload limit plus a 16 KiB
# envelope, enforced on Content-Length and again while streaming, before anything is parsed.
BODY_ENVELOPE_BYTES = 16 * 1024
MAX_BODY_BYTES = fixtures.MAX_UPLOAD_BYTES + BODY_ENVELOPE_BYTES
# Member-facing strings: each passes the layer-0 narration guard (fail closed). Numbers never do.
_SAFE_KEYS = frozenset({"display", "snippet", "text", "hint", "message", "warning", "label"})


class _BodyTooLarge(Exception):
    pass


def _error(status: int, body: dict[str, Any]) -> JSONResponse:
    return JSONResponse(status_code=status, content=_safe(body))


def _safe(obj: Any, key: str | None = None) -> Any:
    if isinstance(obj, dict):
        return {k: _safe(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe(v, key) for v in obj]
    if isinstance(obj, str) and key in _SAFE_KEYS:
        return safe_text(obj)
    return obj


async def _read_capped(request: Request) -> bytes:
    """The request body, refusing (``_BodyTooLarge``) past :data:`MAX_BODY_BYTES` without reading on."""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > MAX_BODY_BYTES:
                raise _BodyTooLarge
        except ValueError:
            pass  # a malformed header is left to the stream counter
    buf = bytearray()
    async for chunk in request.stream():
        buf.extend(chunk)
        if len(buf) > MAX_BODY_BYTES:
            raise _BodyTooLarge
    return bytes(buf)


async def _json_object(request: Request) -> dict[str, Any] | None:
    try:
        data = json.loads(await _read_capped(request))
    except (ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


async def _multipart(request: Request, body: bytes) -> tuple[str | None, bytes | None]:
    """``(journal_id, file bytes)`` from an already capped and buffered multipart body."""
    sent = False

    async def replay() -> dict[str, Any]:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    buffered = Request(request.scope, receive=replay)
    form = await buffered.form(max_files=1, max_fields=4)
    try:
        jid = form.get("journal_id")
        upload = form.get("file")
        data = await upload.read() if isinstance(upload, UploadFile) else None
    finally:
        await form.close()
    return (jid if isinstance(jid, str) else None), data


def create_app(cohort_seed: int = 0, results_url: str | None = None) -> FastAPI:
    """``results_url`` (or env ``DAYPART_RESULTS_URL``): where ``/results`` redirects (the hosted dashboard)."""
    pg = Playground(cohort_seed)
    results_url = results_url or os.environ.get("DAYPART_RESULTS_URL") or None
    app = FastAPI(
        title="Daypart playground",
        description="Synthetic SimMembers only. Not a consumer UI. Not medical advice.",
        version="0.1.0",
    )
    app.state.playground = pg
    app.state.journals = journals = JournalStore(pg.cohort)
    fixtures.manifest()  # warm and verify the fixture allow-list at startup (fail loudly on drift)
    # AD7: real input is hard off. The env request is read once at start, logged and refused.
    real_input_requested = real_input.requested_from_env()
    if real_input_requested:
        refused = real_input.request_enable()
        log.warning(
            "real input requested but refused: %s; open gates: %s",
            refused.reason,
            ", ".join(g.id for g in refused.open_gates),
        )
    index_html = files("daypart.demo").joinpath("static/index.html").read_text(encoding="utf-8")

    def _404(what: str, key: str) -> HTTPException:
        return HTTPException(status_code=404, detail=f"unknown {what}: {key}")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return index_html

    @app.get("/results", include_in_schema=False)
    def results() -> RedirectResponse:
        if not results_url:
            raise HTTPException(status_code=404, detail="no hosted dashboard; run `daypart-demo dashboard`")
        return RedirectResponse(results_url, status_code=307)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "cohort_seed": pg.cohort_seed}

    @app.get("/api/mode")
    def mode() -> dict:
        s = real_input.status(requested=real_input_requested)
        return {
            "mode": "demo",
            "real_input": {"enabled": s.enabled, "requested": s.requested, "open_gates": _gate_view(s.open_gates)},
        }

    @app.post("/api/mode")
    def set_mode() -> JSONResponse:
        # Always refused, whatever the body (A&D API table: "always"). The body is never parsed, so a
        # malformed or empty request gets the same 403 refusal shape instead of a 422.
        refused = real_input.request_enable()
        return JSONResponse(
            status_code=403,
            content={
                "error": "real_input_refused",
                "open_gates": _gate_view(refused.open_gates),
                "reason": refused.reason,
            },
        )

    @app.get("/api/intents")
    def intents() -> list[dict]:
        return pg.intents()

    @app.get("/api/members")
    def members() -> list[dict]:
        return pg.members()

    @app.post("/api/episodes", status_code=201)
    async def start(body: EpisodeIn) -> dict:
        try:
            return await pg.start(body.member_id, seed=body.seed, horizon=body.horizon)
        except UnknownMember:
            raise _404("member", body.member_id) from None

    @app.get("/api/episodes/{episode_id}")
    def episode(episode_id: str) -> dict:
        try:
            return pg.episode(episode_id)
        except UnknownEpisode:
            raise _404("episode", episode_id) from None

    @app.post("/api/episodes/{episode_id}/step")
    async def step(episode_id: str, body: StepIn) -> dict:
        if body.intent not in INTENTS:
            raise HTTPException(status_code=422, detail=f"intent must be one of {list(INTENTS)}")
        try:
            return await pg.step(episode_id, body.intent, body.text)
        except UnknownEpisode:
            raise _404("episode", episode_id) from None
        except EpisodeDone:
            raise HTTPException(status_code=409, detail="episode is done; start a new one") from None

    @app.get("/api/members/{member_id}/passport")
    async def passport(member_id: str, day: Annotated[int, Query(ge=0, le=MAX_HORIZON)] = 41) -> dict:
        try:
            return await pg.passport(member_id, day)
        except UnknownMember:
            raise _404("member", member_id) from None

    # --- check-in journey (A&D API table). Errors are top-level {"error": code, ...} JSON. ---------

    def _journal(journal_id: object) -> Journal:
        return journals.get(journal_id)

    @app.post("/api/journals", status_code=201)
    async def create_journal(request: Request) -> Response:
        try:
            body = await _json_object(request)
        except _BodyTooLarge:
            return _error(413, {"error": "too_large"})
        if body is None:
            return _error(422, {"error": "invalid_request"})
        try:
            j = journals.create(body.get("member_id"))
        except JournalError as e:
            return _error(e.status, e.body)
        return JSONResponse(
            status_code=201,
            content={"journal_id": j.journal_id, "member_id": j.member.member_id, "today": j.today.isoformat()},
        )

    @app.get("/api/fixtures")
    def list_fixtures() -> list[dict]:
        return _safe(fixtures.fixtures_list())

    @app.get("/api/fixtures/{fixture_id}")
    def get_fixture(fixture_id: str) -> Response:
        try:
            data, media_type = fixtures.fixture_bytes(fixture_id)
        except fixtures.UnknownFixture:
            return _error(404, {"error": "unknown_fixture"})
        return Response(content=data, media_type=media_type)

    @app.post("/api/checkins", status_code=201)
    async def create_checkin(request: Request) -> Response:
        # Request bodies are never logged (J4).
        try:
            raw = await _read_capped(request)
        except _BodyTooLarge:
            return _error(413, {"error": "too_large", "message": service.UPLOAD_MESSAGES["too_large"]})
        ctype = request.headers.get("content-type", "").lower()
        text = fixture_id = upload = None
        if ctype.startswith("multipart/form-data"):
            try:
                journal_id, upload = await _multipart(request, raw)
            except (StarletteHTTPException, MultiPartException, ValueError):  # never echo the parser
                return _error(422, {"error": "invalid_request"})
        else:
            try:
                body = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                body = None
            if not isinstance(body, dict):
                return _error(422, {"error": "invalid_request"})
            journal_id, text, fixture_id = body.get("journal_id"), body.get("text"), body.get("fixture_id")
            if not (text is None or isinstance(text, str)) or not (fixture_id is None or isinstance(fixture_id, str)):
                return _error(422, {"error": "invalid_request"})
        try:
            j = _journal(journal_id)
        except JournalError as e:
            return _error(e.status, e.body)
        async with j.lock:
            try:
                out = service.create_checkin(
                    j, text=text, fixture_id=fixture_id, upload=upload, real_input_requested=real_input_requested
                )
            except service.CheckinError as e:
                return _error(e.status, e.body)
        return JSONResponse(status_code=200 if out["status"] == "handoff" else 201, content=_safe(out))

    @app.post("/api/checkins/{checkin_id}/review")
    async def review_checkin(checkin_id: str, request: Request) -> Response:
        # Check-in ids are scoped to their journal: the body must carry journal_id, and the check-in
        # is looked up only inside that journal (no global checkin_id -> journal_id map to keep in
        # sync with eviction). A check-in id from another journal is a 404.
        try:
            body = await _json_object(request)
        except _BodyTooLarge:
            return _error(413, {"error": "too_large"})
        if body is None:
            return _error(422, {"error": "invalid_request"})
        try:
            j = _journal(body.get("journal_id"))
        except JournalError as e:
            return _error(e.status, e.body)
        decisions = body.get("decisions")
        if not isinstance(decisions, list):
            return _error(422, {"error": "invalid_request"})
        try:
            out = await service.review(j, checkin_id, decisions)
        except service.CheckinError as e:
            return _error(e.status, e.body)
        return JSONResponse(status_code=200, content=_safe(out))

    @app.get("/api/journals/{journal_id}/observations/{record_id}")
    def get_observation(journal_id: str, record_id: str) -> Response:
        try:
            j = _journal(journal_id)
            return JSONResponse(status_code=200, content=_safe(service.observation(j, record_id)))
        except JournalError as e:
            return _error(e.status, e.body)
        except service.CheckinError as e:
            return _error(e.status, e.body)

    return app


__all__ = ["create_app"]
