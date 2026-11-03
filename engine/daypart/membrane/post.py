"""Layer 0 + post — the membrane's outbound gate (CONTRACT slice B, B3).

Every draft runs LHM's real ``assert_narration_safe`` (layer 0, fail-closed),
then the boundary DSL. On failure the caller-supplied ``regenerate`` is invoked
at most once; if the final draft still fails, the decision is ``escalate``. An
accepted draft is returned byte-identical — the membrane never edits text.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from syntropy.lhm.metrics.claims import (
    CLAIMS_GUARD_VERSION,
    VALENCE_GUARD_VERSION,
    assert_narration_safe,
)

from daypart.membrane import dsl


@dataclass(frozen=True)
class PostResult:
    decision: Literal["pass", "escalate"]
    text: str | None  # accepted draft verbatim when "pass"; None when "escalate"
    trace: dict  # decision-trace fragment (I4); never contains draft text


def _attempt(draft: Any, crisis_context: bool) -> dict[str, Any]:
    """Run layer 0 then the DSL on one draft. Fail-closed on anything unexpected."""
    entry: dict[str, Any] = {"layer0": "fail", "dsl_passed": None, "violations": []}
    if not isinstance(draft, str):
        return entry
    try:
        assert_narration_safe(draft)
    except Exception:  # ClaimScopeError or anything else: fail closed
        return entry
    entry["layer0"] = "pass"
    try:
        verdict = dsl.check(draft, crisis_context=crisis_context)
    except Exception:
        entry["dsl_passed"] = False
        return entry
    entry["dsl_passed"] = verdict.passed
    entry["violations"] = sorted({v.rule_id for v in verdict.violations})
    return entry


def _ok(entry: dict[str, Any]) -> bool:
    return entry["layer0"] == "pass" and entry["dsl_passed"] is True


def post(
    draft: str,
    *,
    crisis_context: bool = False,
    regenerate: Callable[[], str] | None = None,
) -> PostResult:
    attempt_log = [_attempt(draft, crisis_context)]
    final_draft: Any = draft
    regenerated = False

    if not _ok(attempt_log[0]) and regenerate is not None:
        regenerated = True
        try:
            final_draft = regenerate()
        except Exception:
            final_draft = None  # regenerate failure -> fail-closed attempt
        attempt_log.append(_attempt(final_draft, crisis_context))

    last = attempt_log[-1]
    passed = _ok(last)
    decision: Literal["pass", "escalate"] = "pass" if passed else "escalate"

    trace = {
        "layer0": {
            "guard_versions": {"claims": CLAIMS_GUARD_VERSION, "valence": VALENCE_GUARD_VERSION},
            "outcome": last["layer0"],
        },
        "dsl": {
            "rule_version": dsl.RULE_VERSION,
            "passed": last["dsl_passed"],
            "violations": list(last["violations"]),
        },
        "attempts": len(attempt_log),
        "regenerated": regenerated,
        "decision": decision,
        "attempt_log": [
            {"layer0": e["layer0"], "dsl_passed": e["dsl_passed"], "violations": list(e["violations"])}
            for e in attempt_log
        ],
    }
    return PostResult(decision=decision, text=final_draft if passed else None, trace=trace)


__all__ = ["PostResult", "post"]
