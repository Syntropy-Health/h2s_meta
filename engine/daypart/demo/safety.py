"""Layer-0 gate for every member-facing string the demo returns (D11, fail closed)."""

from __future__ import annotations

from syntropy.lhm.metrics.claims import assert_narration_safe

WITHHELD_DRAFT = "[draft withheld: blocked by the membrane]"
WITHHELD_TEXT = "[text withheld: failed the narration guard]"


def safe_text(text: object, fallback: str = WITHHELD_TEXT) -> str:
    """``text`` when it passes ``assert_narration_safe``, else ``fallback``. Never raises."""
    if not isinstance(text, str):
        return fallback
    try:
        assert_narration_safe(text)
    except Exception:  # ClaimScopeError or anything unexpected: fail closed
        return fallback
    return text


__all__ = ["WITHHELD_DRAFT", "WITHHELD_TEXT", "safe_text"]
