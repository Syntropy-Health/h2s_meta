"""E2 — the ``Flag`` card and ``Passport`` models (I6, D11)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from typing import Literal

import pydantic
from pydantic import ConfigDict, model_validator
from syntropy.lhm.contracts.viz import VizPayload
from syntropy.lhm.metrics.claims import assert_narration_safe

from daypart.passport.constants import PASSPORT_TITLE, PENDING, WITHHELD_MESSAGE


def _narration_safe(what: str, text: str) -> None:
    try:
        assert_narration_safe(text)
    except Exception as exc:  # ClaimScopeError (or anything unexpected): fail closed
        raise ValueError(f"{what} fails the narration guard (D11): {exc}") from None


class Flag(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    lhm_domain: str
    title: str
    status: Literal["ready", "withheld"]
    value: float | None
    interval: tuple[float, float] | None
    caveats: tuple[str, ...]
    message: str | None
    payload: VizPayload | None
    pattern: None = None  # placeholder, pending R11
    citation: None = None  # placeholder, pending R10
    pending: tuple[str, ...] = PENDING

    @model_validator(mode="after")
    def _check(self) -> Flag:
        if self.status == "ready":
            if self.value is None or not math.isfinite(self.value):
                raise ValueError("a ready flag requires a finite value")
            if self.message is not None:
                raise ValueError("a ready flag carries no message")
            if not self.caveats:
                raise ValueError("a ready flag requires caveats")
            if self.payload is None:
                raise ValueError("a ready flag requires a viz payload")
        else:
            if self.value is not None or self.interval is not None or self.payload is not None:
                raise ValueError("a withheld flag never carries a number (I6)")
            if self.caveats != ():
                raise ValueError("a withheld flag carries no caveats")
            if self.message != WITHHELD_MESSAGE:
                raise ValueError(f"a withheld flag's message must be {WITHHELD_MESSAGE!r}")
        _narration_safe("title", self.title)
        if self.message is not None:
            _narration_safe("message", self.message)
        for caveat in self.caveats:
            _narration_safe("caveat", caveat)
        return self


class Passport(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    title: Literal["Pattern passport"] = PASSPORT_TITLE
    member_id: str
    as_of: datetime
    ontology_version: str
    guard_versions: Mapping[str, str]
    flags: tuple[Flag, ...]


__all__ = ["Flag", "Passport"]
