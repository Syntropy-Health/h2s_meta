"""The typed extractor boundary (addendum G21).

An extractor turns one check-in source into proposals. Every extractor — ``rules-v1`` text,
``fixture-sidecar-v1`` and any future model-backed one — returns :class:`Proposal` models, and the
service passes its output through :func:`validate_proposals` before anything reaches the journal.

Contract for a model-backed extractor (none exists in this build; no live model is called): ask the
model for structured output in the ``Proposal`` JSON schema (``Proposal.model_json_schema()``), then
``Proposal.model_validate`` each item. ``extra="forbid"`` refuses any field the schema lacks (a risk
score, an interpretation), so a model cannot widen what is shown. Flags still come from
``detect_flags``; a model never sets ``blocking``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any, Protocol

import pydantic

from daypart.checkin.models import Proposal


class ExtractorOutputError(ValueError):
    """Extractor output that is not a sequence of valid, uniquely-identified proposals.

    The message names the item index and failing field only, never an input value.
    """


class Extractor(Protocol):
    def __call__(self, source: Any, today: dt.date, /, *, checkin_id: str) -> Sequence[Proposal]: ...


def _check(i: int, item: object) -> Proposal:
    # A Proposal instance is dumped and revalidated, so ``model_construct`` cannot bypass the schema.
    data = item.model_dump(warnings=False) if isinstance(item, Proposal) else item
    try:
        return Proposal.model_validate(data)
    except pydantic.ValidationError as e:
        fields = sorted({".".join(str(p) for p in err["loc"]) or "<item>" for err in e.errors()})
        raise ExtractorOutputError(f"proposal {i}: invalid {', '.join(fields)}") from None


def validate_proposals(output: object) -> tuple[Proposal, ...]:
    """``output`` as a tuple of validated proposals, or :class:`ExtractorOutputError`."""
    if isinstance(output, (str, bytes, dict)) or not isinstance(output, Sequence):
        raise ExtractorOutputError("extractor output must be a sequence of proposals")
    props = tuple(_check(i, item) for i, item in enumerate(output))
    ids = [p.proposal_id for p in props]
    if len(set(ids)) != len(ids):
        raise ExtractorOutputError("proposal ids must be unique")
    return props


__all__ = ["Extractor", "ExtractorOutputError", "validate_proposals"]
