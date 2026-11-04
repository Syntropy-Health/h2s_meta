"""F4 — the SFT chat dataset: cached coach dialogue + refusal-with-redirection mix.

Dialogue rows come only from passing journal + dialogue cache rows of the
cohort's *training* (synthetic) members. Every assistant turn passes the
boundary DSL (candidates that do not are dropped, never edited), and the
output is guarded by ``assert_no_curated`` against the cohort manifest (I1).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from daypart.membrane import dsl
from daypart.simmember.cohort import Cohort
from daypart.simmember.manifest import assert_no_curated, canonical_json, write
from daypart.simmember.narrative import TEMPLATE_REFUSALS, NarrativeCache, NarrativeRow
from daypart.simmember.trajectory import DayRecord

COACH_SYSTEM_PROMPT = (
    "You are Daypart, a supportive adherence coach. You help members keep the daily routines "
    "they have agreed with their care team: reminders, timing, planning around busy days and "
    "getting back on track after a missed day. Keep replies short, warm and free of judgment. "
    "Questions about amounts, changes or symptoms are outside your role: you redirect them to "
    "the member's clinician."
)

#: Member requests a coach must refuse (each FAILS ``dsl.check`` as text).
REFUSAL_PROMPTS: tuple[str, ...] = (
    "Should I take 500 mg of magnesium before bed?",
    "Can I double my dose tonight since I missed this morning?",
    "Is it fine to take 2 extra tablets when I feel worse?",
    "Is this tiredness a symptom of diabetes?",
    "Should I stop taking my metformin for a week?",
    "Would 1000 mg of vitamin C be better for me?",
    "Can I skip my medication on weekends?",
    "Does this sound like a sign of depression to you?",
    "Should I increase my dosage if I feel tired?",
    "Is semaglutide something I should start on?",
)

#: Refusal-with-redirection replies (each passes ``post``; mention "clinician"; ask a question).
REFUSALS: tuple[str, ...] = TEMPLATE_REFUSALS


def _rng(seed: int, tag: str) -> np.random.Generator:
    digest = hashlib.sha256(canonical_json(["daypart.train", tag, int(seed)]).encode()).digest()
    return np.random.default_rng([int.from_bytes(digest[i : i + 4], "big") for i in range(0, 16, 4)])


def user_turn(record: DayRecord, journal: str) -> str:
    """Member-facing day context + the member's journal. Never carries a member id."""
    return f"Day {record.day} check-in. Current state: {record.state}.\nMy journal: {journal}"


def _passing_by_member_day(
    cache: NarrativeCache, member_ids: set[str]
) -> dict[tuple[str, int, str], NarrativeRow]:
    """First passing row per ``(member_id, day, kind)`` in file order, training members only."""
    out: dict[tuple[str, int, str], NarrativeRow] = {}
    for row in cache.rows():
        if row.member_id not in member_ids or row.decision != "pass" or row.text is None:
            continue
        out.setdefault((row.member_id, row.day, row.kind), row)
    return out


def _record(cohort: Cohort, member_id: str, day: int) -> DayRecord | None:
    traj = cohort.trajectories.get(member_id)
    if traj is None or not (0 <= day < len(traj.days)):
        return None
    return traj.days[day]


def _manifest(cohort: Cohort) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        return write(cohort, Path(tmp))


def build_sft(
    cohort: Cohort, cache: NarrativeCache, *, refusal_ratio: float = 0.2, seed: int = 0
) -> list[dict]:
    if not isinstance(refusal_ratio, (int, float)) or not (0.0 <= refusal_ratio <= 0.5):
        raise ValueError(f"refusal_ratio must be within [0, 0.5], got {refusal_ratio!r}")
    training = cohort.training_profiles()
    training_ids = {p.member_id for p in training}
    passing = _passing_by_member_day(cache, training_ids)

    dialogue_rows: list[dict] = []
    for profile in training:
        days = sorted({day for (mid, day, kind) in passing if mid == profile.member_id})
        for day in days:
            journal = passing.get((profile.member_id, day, "journal"))
            reply = passing.get((profile.member_id, day, "dialogue"))
            record = _record(cohort, profile.member_id, day)
            if journal is None or reply is None or record is None:
                continue
            assert reply.text is not None and journal.text is not None
            if not dsl.check(reply.text).passed:
                continue  # dropped, never edited
            dialogue_rows.append(
                {
                    "messages": [
                        {"role": "system", "content": COACH_SYSTEM_PROMPT},
                        {"role": "user", "content": user_turn(record, journal.text)},
                        {"role": "assistant", "content": reply.text},
                    ],
                    "meta": {
                        "member_id": profile.member_id,
                        "day": day,
                        "source": "dialogue",
                        "prompt_hash": reply.prompt_hash,
                    },
                }
            )

    n_dialogue = len(dialogue_rows)
    n_refusal = (
        0 if refusal_ratio == 0 else int(round(refusal_ratio * n_dialogue / (1 - refusal_ratio)))
    )
    rng = _rng(seed, "sft")
    refusal_rows: list[dict] = []
    safe_refusals = [r for r in REFUSALS if dsl.check(r).passed]
    if n_refusal and training and safe_refusals:
        for _ in range(n_refusal):
            member = training[int(rng.integers(len(training)))]
            prompt = REFUSAL_PROMPTS[int(rng.integers(len(REFUSAL_PROMPTS)))]
            reply = safe_refusals[int(rng.integers(len(safe_refusals)))]
            refusal_rows.append(
                {
                    "messages": [
                        {"role": "system", "content": COACH_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": reply},
                    ],
                    "meta": {"member_id": member.member_id, "day": 0, "source": "refusal", "prompt_hash": None},
                }
            )

    # Deterministic interleave: refusals at seeded positions; dialogue keeps cohort/day order.
    total = len(dialogue_rows) + len(refusal_rows)
    slots = set(int(i) for i in rng.choice(total, size=len(refusal_rows), replace=False)) if refusal_rows else set()
    d_iter, r_iter = iter(dialogue_rows), iter(refusal_rows)
    rows = [next(r_iter) if i in slots else next(d_iter) for i in range(total)]

    assert_no_curated(rows, _manifest(cohort))
    return rows


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | os.PathLike) -> str:
    """Write canonical-JSON lines (each ending ``\\n``); return the file's sha256 hex."""
    data = "".join(canonical_json(dict(r)) + "\n" for r in rows).encode("utf-8")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


__all__ = ["COACH_SYSTEM_PROMPT", "REFUSALS", "REFUSAL_PROMPTS", "build_sft", "user_turn", "write_jsonl"]
