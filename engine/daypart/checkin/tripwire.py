"""D-A7 PII tripwire on free check-in text.

:func:`scan` returns the reason codes for anything that looks like personal data; a non-empty
result means the whole check-in is refused and nothing is stored. This is defence in depth for a
public demo, not a privacy control, and is never claimed as one.
"""

from __future__ import annotations

import re

REASONS: tuple[str, ...] = ("email", "phone", "national_id", "self_name", "date_of_birth")

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# ISO dates are masked before the phone scan so "2026-11-03 72.5 kg" never reads as a number run.
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_PHONE = re.compile(r"(?<![\w.])\+?\(?\d[\d \t\-.()]*\d(?![\w])")
_SSN = re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])")
_DIGIT_RUN = re.compile(r"(?<![\d.])(?<!\d\.)\d{9,}(?!\d|\.\d)")
_SELF_NAME = re.compile(r"\b(?:my\s+name\s+is|i\s+am\s+called|i'm\s+called|call\s+me)\s+[a-z][a-z'-]*", re.IGNORECASE)
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_DATE = (
    r"(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|"
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}\b(?:,?\s+\d{{4}})?|"
    rf"{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?|(?:19|20)\d{{2}}\b)"
)
_DOB = re.compile(
    rf"\b(?:born\s+(?:on|in)|dob|d\.o\.b\.?|date\s+of\s+birth|birthday)\b"
    rf"(?:\s*[:=-]?\s*(?:is|was|on|in)\b)*\s*[:=-]?\s*{_DATE}",
    re.IGNORECASE,
)


def _has_phone(text: str) -> bool:
    masked = _ISO_DATE.sub(" ", text)
    return any(sum(ch.isdigit() for ch in m.group()) >= 10 for m in _PHONE.finditer(masked))


def scan(text: str) -> tuple[str, ...]:
    """Reason codes (in :data:`REASONS` order) for personal-data-looking content in ``text``."""
    found = {
        "email": bool(_EMAIL.search(text)),
        "phone": _has_phone(text),
        "national_id": bool(_SSN.search(text) or _DIGIT_RUN.search(text)),
        "self_name": bool(_SELF_NAME.search(text)),
        "date_of_birth": bool(_DOB.search(text)),
    }
    return tuple(r for r in REASONS if found[r])


__all__ = ["REASONS", "scan"]
