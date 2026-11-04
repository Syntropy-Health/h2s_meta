"""E1.2 — the R2 boundary corpus loader and recall / false-positive metrics."""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Iterable, Mapping
from typing import Any

import pydantic

from daypart.eval.models import CorpusRow, _coerce_row


def _r6(x: float | None) -> float | None:
    return None if x is None else round(float(x), 6)


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else _r6(num / den)


def load_corpus(path: str | os.PathLike) -> tuple[CorpusRow, ...]:
    """Parse an E1.2 JSONL corpus; a malformed row raises ``ValueError`` naming its 1-based line."""
    rows: list[CorpusRow] = []
    with pathlib.Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"corpus line {lineno}: invalid JSON: {exc.msg}") from None
            if not isinstance(obj, dict):
                raise ValueError(f"corpus line {lineno}: expected a JSON object")
            for key in ("text", "expected"):
                if key not in obj:
                    raise ValueError(f"corpus line {lineno}: missing {key!r}")
            try:
                rows.append(CorpusRow.model_validate(obj))
            except pydantic.ValidationError as exc:
                errs = "; ".join(e["msg"] for e in exc.errors())
                raise ValueError(f"corpus line {lineno}: {errs}") from None
    return tuple(rows)


def r2_metrics(rows: Iterable[CorpusRow | Mapping[str, Any]]) -> dict:
    from daypart.membrane import dsl

    parsed = [_coerce_row(r) for r in rows]
    rule_ids = sorted(dsl.RULE_IDS)
    n_violation = n_compliant = missed = false_positives = 0
    labeled = dict.fromkeys(rule_ids, 0)
    caught = dict.fromkeys(rule_ids, 0)
    fired_on_compliant = dict.fromkeys(rule_ids, 0)

    for row in parsed:
        verdict = dsl.check(row.text, crisis_context=row.crisis_context)
        fired = {v.rule_id for v in verdict.violations}
        if row.expected == "violation":
            n_violation += 1
            if verdict.passed:
                missed += 1
            for rid in set(row.rule_ids):
                labeled[rid] += 1
                if rid in fired:
                    caught[rid] += 1
        else:
            n_compliant += 1
            if not verdict.passed:
                false_positives += 1
            for rid in fired:
                if rid in fired_on_compliant:
                    fired_on_compliant[rid] += 1

    return {
        "rule_version": dsl.RULE_VERSION,
        "n_rows": len(parsed),
        "n_violation": n_violation,
        "n_compliant": n_compliant,
        "recall": _ratio(n_violation - missed, n_violation),
        "fp_rate": _ratio(false_positives, n_compliant),
        "missed": missed,
        "false_positives": false_positives,
        "per_rule": {
            rid: {
                "n_labeled": labeled[rid],
                "recall": _ratio(caught[rid], labeled[rid]),
                "fired_on_compliant": fired_on_compliant[rid],
                "fp_rate": _ratio(fired_on_compliant[rid], n_compliant),
            }
            for rid in rule_ids
        },
    }


__all__ = ["load_corpus", "r2_metrics"]
