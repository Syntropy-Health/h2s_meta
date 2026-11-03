"""Sealed challenge suite — R2 acceptance over the seeded boundary corpus (B4).

Category: functional/membrane-r2
Spec: docs/daypart/contracts/CONTRACT-slice-B-membrane.md section B4.

Corpus: ``data/boundary_corpus_v1.jsonl`` — hand-written, labeled. Each
violation row carries the rule_id it must trigger (and ``crisis_context``);
compliant rows are realistic adherence-coaching copy (a few are crisis
responses that DO include a handoff marker, checked with crisis_context=True).

Acceptance: recall == 1.0 on violations (the labeled rule_id must be among
the fired ids) and false-positive rate < 0.02 on compliant rows.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from daypart.membrane.dsl import RULE_IDS, check

CORPUS = Path(__file__).parent / "data" / "boundary_corpus_v1.jsonl"

B2_RULES = frozenset(
    {
        "dosing.unit_quantity",
        "dosing.imperative",
        "diagnosis.claim",
        "treatment.claim",
        "substance.rx_named",
        "lhm.banned_lexicon",
        "lhm.imperative_dose",
        "crisis.missing_handoff",
    }
)


def _load() -> list[dict]:
    rows = []
    with CORPUS.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


ROWS = _load()
VIOLATIONS = [r for r in ROWS if r["label"] == "violation"]
COMPLIANT = [r for r in ROWS if r["label"] == "compliant"]


class TestCorpusShape:
    """functional/membrane-r2 :: corpus is well-formed (B4 minimums)."""

    def test_labels_and_ids(self):
        assert all(r["label"] in ("violation", "compliant") for r in ROWS)
        ids = [r["id"] for r in ROWS]
        assert len(ids) == len(set(ids))
        texts = [(r["text"], r["crisis_context"]) for r in ROWS]
        assert len(texts) == len(set(texts))

    def test_rule_labels_match_rule_ids(self):
        assert {r["rule_id"] for r in VIOLATIONS} == B2_RULES
        assert RULE_IDS == B2_RULES

    def test_at_least_five_per_rule(self):
        counts = Counter(r["rule_id"] for r in VIOLATIONS)
        for rule in B2_RULES:
            assert counts[rule] >= 5, (rule, counts[rule])

    def test_at_least_100_compliant(self):
        assert len(COMPLIANT) >= 100


def _fired(row: dict) -> set[str]:
    return {v.rule_id for v in check(row["text"], crisis_context=row["crisis_context"]).violations}


class TestR2Acceptance:
    """functional/membrane-r2 :: recall and false-positive rate."""

    def test_recall_is_total(self):
        per_rule_total: Counter[str] = Counter()
        per_rule_missed: dict[str, list[str]] = defaultdict(list)
        for row in VIOLATIONS:
            per_rule_total[row["rule_id"]] += 1
            if row["rule_id"] not in _fired(row):
                per_rule_missed[row["rule_id"]].append(row["id"])
        missed = sum(len(v) for v in per_rule_missed.values())
        recall = 1 - missed / len(VIOLATIONS)
        if missed:
            lines = [f"recall={recall:.4f} ({missed}/{len(VIOLATIONS)} missed)"]
            for rule in sorted(B2_RULES):
                m = per_rule_missed.get(rule, [])
                lines.append(
                    f"  {rule}: {per_rule_total[rule] - len(m)}/{per_rule_total[rule]} caught"
                    + (f"; missed ids={m}" if m else "")
                )
            pytest.fail("\n".join(lines))
        assert recall == 1.0

    def test_false_positive_rate_below_2pct(self):
        fp_rows: list[tuple[str, list[str]]] = []
        per_rule_fp: Counter[str] = Counter()
        for row in COMPLIANT:
            ids = _fired(row)
            if ids:
                fp_rows.append((row["id"], sorted(ids)))
                per_rule_fp.update(ids)
        rate = len(fp_rows) / len(COMPLIANT)
        if rate >= 0.02:
            lines = [f"fp_rate={rate:.4f} ({len(fp_rows)}/{len(COMPLIANT)})"]
            for rule in sorted(B2_RULES):
                lines.append(f"  {rule}: {per_rule_fp.get(rule, 0)} false positives")
            lines.append(f"  rows: {fp_rows}")
            pytest.fail("\n".join(lines))
        assert rate < 0.02

    @pytest.mark.parametrize("rule", sorted(B2_RULES))
    def test_per_rule_recall(self, rule):
        rows = [r for r in VIOLATIONS if r["rule_id"] == rule]
        missed = [r["id"] for r in rows if rule not in _fired(r)]
        assert not missed, f"{rule}: missed {missed}"

    def test_compliant_crisis_rows_never_flag_handoff(self):
        rows = [r for r in COMPLIANT if r["crisis_context"]]
        assert rows
        flagged = [r["id"] for r in rows if "crisis.missing_handoff" in _fired(r)]
        assert not flagged, flagged
