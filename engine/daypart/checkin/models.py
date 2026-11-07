"""Proposal and check-in models (A&D §4.1).

A :class:`Proposal` is one extracted value awaiting review. Every flag is blocking: ``blocking`` is
True iff ``flags`` is non-empty, which the validator enforces (it is filled in when omitted and
rejected when it disagrees). A :class:`CheckIn` moves one way only, ``pending_review`` → ``saved``.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

import pydantic
from pydantic import ConfigDict, Field, model_validator

Flag = Literal[
    "unit_mismatch",
    "unit_assumed",
    "ambiguous_date",
    "contradiction",
    "unknown_code",
    "out_of_plausible_range",
    "missing_value",
]
FLAGS: tuple[str, ...] = (
    "unit_mismatch",
    "unit_assumed",
    "ambiguous_date",
    "contradiction",
    "unknown_code",
    "out_of_plausible_range",
    "missing_value",
)

METHOD_TEXT = "rules-v1"
METHOD_FIXTURE = "fixture-sidecar-v1"

SNIPPET_MAX = 80


class Proposal(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_id: str = Field(pattern=r"^p-[1-9][0-9]*$")
    concept_code: str | None
    display: str = Field(min_length=1)
    value: float | None
    unit: str | None
    date: dt.date | None
    source_kind: Literal["text", "file_fixture", "image_fixture"]
    source_ref: str = Field(min_length=1)
    snippet: str = Field(max_length=SNIPPET_MAX)
    method: str = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]
    flags: tuple[Flag, ...] = ()
    blocking: bool

    @model_validator(mode="before")
    @classmethod
    def _fill_blocking(cls, data: Any) -> Any:
        if isinstance(data, dict) and "blocking" not in data:
            data = {**data, "blocking": bool(data.get("flags") or ())}
        return data

    @model_validator(mode="after")
    def _check(self) -> Proposal:
        if self.blocking != bool(self.flags):
            raise ValueError("blocking must be True iff flags is non-empty")
        if len(set(self.flags)) != len(self.flags):
            raise ValueError("flags must not repeat")
        return self

    def with_changes(self, **fields: Any) -> Proposal:
        """A revalidated copy with ``fields`` replaced (``model_copy`` would skip the validators).

        When ``flags`` changes and ``blocking`` is not given, ``blocking`` is recomputed from the
        new flags; an explicit ``blocking`` that disagrees with the flags raises.
        """
        data = {**self.model_dump(), **fields}
        if "flags" in fields and "blocking" not in fields:
            del data["blocking"]
        return Proposal.model_validate(data)


class CheckIn(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    checkin_id: str = Field(min_length=1)
    journal_id: str = Field(min_length=1)
    created_at: dt.datetime
    status: Literal["pending_review", "saved"]
    proposals: tuple[Proposal, ...]

    def mark_saved(self) -> CheckIn:
        """A copy with ``status="saved"``; only ``pending_review`` → ``saved`` is allowed."""
        if self.status != "pending_review":
            raise ValueError(f"check-in {self.checkin_id!r} is already {self.status!r}")
        return CheckIn.model_validate({**self.model_dump(), "status": "saved"})


__all__ = ["FLAGS", "METHOD_FIXTURE", "METHOD_TEXT", "SNIPPET_MAX", "CheckIn", "Flag", "Proposal"]
