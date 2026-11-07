"""FastAPI playground (HAC-20). Offline, no secrets, synthetic members only (I1)."""

from __future__ import annotations

import logging
import os
from importlib.resources import files
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from daypart.demo.playground import (
    MAX_HORIZON,
    EpisodeDone,
    Playground,
    UnknownEpisode,
    UnknownMember,
)
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


class ModeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    real_input: bool


def _gate_view(gates: tuple[real_input.Gate, ...]) -> list[dict]:
    return [{"id": g.id, "label": g.label, "owner": g.owner} for g in gates]


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
    def set_mode(body: ModeIn) -> JSONResponse:
        # Always refused, whatever the requested value (A&D API table: "always").
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

    return app


__all__ = ["create_app"]
