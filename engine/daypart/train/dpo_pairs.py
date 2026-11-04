"""F5 — lapse-recovery DPO preference pairs.

``chosen`` is the first principled sample (coach prompt + ``PRINCIPLES``) that
passes the membrane (layer 0 + DSL) and the lapse-recovery rubric;
``rejected`` is one unconstrained sample of the same context (no principles).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pydantic
from pydantic import ConfigDict

from daypart.membrane import dsl
from daypart.membrane.post import post
from daypart.simmember.cohort import CURATED_PROFILES, Cohort
from daypart.simmember.manifest import CuratedLeakError, canonical_json
from daypart.simmember.narrative import NarrativeCache, TextClient
from daypart.train.sft_dataset import (  # write_jsonl re-exported: pairs use the F4 writer
    COACH_SYSTEM_PROMPT,
    _passing_by_member_day,
    _record,
    user_turn,
    write_jsonl,
)

RUBRIC_VERSION = "lapse-recovery-v1"

SHAME_TERMS: tuple[str, ...] = (
    "failed",
    "failure",
    "lazy",
    "disappointed",
    "disappointing",
    "no excuse",
    "should have",
    "shame",
    "guilt",
)

PRINCIPLES = (
    "Boundary: stay within routines, reminders, planning and motivation. Questions about amounts, "
    "changes or symptoms go to the member's clinician. "
    "Lapse recovery: after a missed day, respond without blame or judgment. Name one small, concrete "
    "next step, keep it brief, and end with an open question that invites the member back in."
)

_STATES = ("engaged", "strained", "lapsed", "recovering")
_INVITE_STATES = frozenset({"strained", "lapsed", "recovering"})
_SHAME_RE = re.compile(
    r"\b(?:" + "|".join(r"\s+".join(map(re.escape, t.split())) for t in SHAME_TERMS) + r")\b"
)
_CURATED_IDS = frozenset(p.member_id for p in CURATED_PROFILES)


@dataclass(frozen=True)
class RubricVerdict:
    passed: bool
    failed: tuple[str, ...]


def rubric_check(member_state: str, text: str) -> RubricVerdict:
    if member_state not in _STATES:
        raise ValueError(f"unknown member_state {member_state!r}")
    failed: list[str] = []
    if not (1 <= len(text.strip()) <= 480):
        failed.append("length")
    if _SHAME_RE.search(text.casefold()):
        failed.append("shame")
    if member_state in _INVITE_STATES and "?" not in text:
        failed.append("invitation")
    return RubricVerdict(passed=not failed, failed=tuple(failed))


class DpoContext(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    member_id: str
    day: int
    member_state: Literal["engaged", "strained", "lapsed", "recovering"]
    user_turn: str


def make_contexts(
    cohort: Cohort, cache: NarrativeCache, *, per_member: int = 2, max_members: int | None = None
) -> list[DpoContext]:
    if per_member < 1:
        raise ValueError("per_member must be >= 1")
    if max_members is not None and max_members < 0:
        raise ValueError("max_members must be >= 0")
    training = cohort.training_profiles()
    passing = _passing_by_member_day(cache, {p.member_id for p in training})
    out: list[DpoContext] = []
    members = 0
    for profile in training:
        if max_members is not None and members >= max_members:
            break
        candidates = []
        for (mid, day, kind), row in passing.items():
            if mid != profile.member_id or kind != "journal" or row.text is None:
                continue
            record = _record(cohort, mid, day)
            if record is None:
                continue
            candidates.append((record.state == "engaged", day, record, row.text))
        if not candidates:
            continue
        candidates.sort(key=lambda c: (c[0], c[1]))
        for _, day, record, journal in candidates[:per_member]:
            out.append(
                DpoContext(
                    member_id=profile.member_id,
                    day=day,
                    member_state=record.state,
                    user_turn=user_turn(record, journal),
                )
            )
        members += 1
    return out


def _seed(seed: int, ctx: DpoContext, i: int | str) -> int:
    payload = ["dpo", int(seed), ctx.member_id, ctx.day, ctx.member_state, ctx.user_turn, i]
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _is_curated_id(member_id: str) -> bool:
    return member_id.startswith("cur-") or member_id in _CURATED_IDS


def build_pairs(
    contexts: Sequence[DpoContext], sampler: TextClient, *, n_candidates: int = 4, seed: int = 0
) -> list[dict]:
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")
    for ctx in contexts:
        if _is_curated_id(ctx.member_id) or "cur-" in ctx.user_turn:
            raise CuratedLeakError(f"curated member {ctx.member_id!r} in DPO contexts (I1)")

    principled_system = COACH_SYSTEM_PROMPT + "\n\n" + PRINCIPLES
    rows: list[dict] = []
    for ctx in contexts:
        principled = [
            {"role": "system", "content": principled_system},
            {"role": "user", "content": ctx.user_turn},
        ]
        chosen: str | None = None
        for i in range(n_candidates):
            text = sampler.complete(principled, seed=_seed(seed, ctx, i))
            if post(text).decision == "pass" and rubric_check(ctx.member_state, text).passed:
                chosen = text
                break
        if chosen is None:
            continue
        prompt = [
            {"role": "system", "content": COACH_SYSTEM_PROMPT},
            {"role": "user", "content": ctx.user_turn},
        ]
        rejected = sampler.complete([dict(m) for m in prompt], seed=_seed(seed, ctx, "rejected"))
        if rejected == chosen:
            continue
        rejected_dsl = isinstance(rejected, str) and dsl.check(rejected).passed
        rejected_rubric = isinstance(rejected, str) and rubric_check(ctx.member_state, rejected).passed
        rows.append(
            {
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "meta": {
                    "member_id": ctx.member_id,
                    "day": ctx.day,
                    "member_state": ctx.member_state,
                    "rubric_version": RUBRIC_VERSION,
                    "rejected_dsl_passed": bool(rejected_dsl),
                    "rejected_rubric_passed": bool(rejected_rubric),
                },
            }
        )
    return rows


__all__ = [
    "PRINCIPLES",
    "RUBRIC_VERSION",
    "SHAME_TERMS",
    "DpoContext",
    "RubricVerdict",
    "build_pairs",
    "make_contexts",
    "rubric_check",
    "write_jsonl",
]
