"""Re-encode the sealed slice-B corpus into the E1.2 eval format (HAC-4 / HAC-23).

Source: ``boundary_corpus_v1.jsonl`` (``id``, ``label``, ``rule_id``, ``crisis_context``, ``text``).
Target: ``r2_corpus_v1.jsonl`` (``text``, ``expected``, ``rule_ids``, ``crisis_context``), the
format in docs/daypart/contracts/CONTRACT-slice-E-eval-passport.md section E1.2.

Same rows, same order. Every row's label is checked against ``dsl.check``: a violation row's
labelled rule must fire, and a compliant row must pass. A mismatch aborts the conversion; labels
are never rewritten to match the checker.

Run from ``engine/``:  uv run python tests/sealed/data/convert_r2_corpus.py [--check]
``--check`` verifies the committed target is byte-identical to a fresh conversion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from daypart.membrane.dsl import RULE_IDS, check

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "boundary_corpus_v1.jsonl"
TARGET = HERE / "r2_corpus_v1.jsonl"


def convert_row(src: dict) -> dict:
    label, rule_id, crisis = src["label"], src["rule_id"], bool(src["crisis_context"])
    if label == "violation":
        if rule_id not in RULE_IDS:
            raise ValueError(f"{src['id']}: unknown rule id {rule_id!r}")
        rule_ids = [rule_id]
    elif label == "compliant":
        if rule_id is not None:
            raise ValueError(f"{src['id']}: compliant row carries rule id {rule_id!r}")
        rule_ids = []
    else:
        raise ValueError(f"{src['id']}: label {label!r}")

    fired = {v.rule_id for v in check(src["text"], crisis_context=crisis).violations}
    if label == "violation" and not set(rule_ids) <= fired:
        raise ValueError(f"{src['id']}: labelled {rule_ids} but dsl.check fired {sorted(fired)}")
    if label == "compliant" and fired:
        raise ValueError(f"{src['id']}: compliant but dsl.check fired {sorted(fired)}")

    return {"text": src["text"], "expected": label, "rule_ids": rule_ids, "crisis_context": crisis}


def render() -> str:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return "".join(json.dumps(convert_row(r), ensure_ascii=False) + "\n" for r in rows)


def main(argv: list[str]) -> int:
    text = render()
    if "--check" in argv:
        if TARGET.read_text(encoding="utf-8") != text:
            print(f"{TARGET.name} is stale; re-run without --check", file=sys.stderr)
            return 1
        return 0
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET.name}: {text.count(chr(10))} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
