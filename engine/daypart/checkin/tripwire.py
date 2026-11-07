"""D-A7 PII tripwire on free check-in text.

:func:`scan` returns the reason codes for anything that looks like personal data; a non-empty
result means the whole check-in is refused and nothing is stored. This is defence in depth for a
public demo, not a privacy control, and is never claimed as one.
"""

from __future__ import annotations

import re

from daypart.membrane import dsl

REASONS: tuple[str, ...] = ("email", "phone", "national_id", "self_name", "date_of_birth")

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# ISO dates are masked before the phone scan so "2026-11-03 72.5 kg" never reads as a number run.
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_PHONE = re.compile(r"(?<![\w.])\+?\(?\d[\d \t\-.()]*\d(?![\w])")
_SSN = re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])")
_DIGIT_RUN = re.compile(r"(?<![\d.])(?<!\d\.)\d{9,}(?!\d|\.\d)")
# Self-name checks. Explicit phrases fire on any following word and run on dsl.normalize() output
# (casefold, NFKC, confusable and apostrophe fold). "call me" and "I'm" are ambiguous ("call me
# back", "I'm tired"), so they run on the case-preserving fold and need capitalised words.
_SELF_NAME = re.compile(
    r"\b(?:my\s+name\s+is|my\s+name's|(?:my\s+)?name\s*[:=]|i\s+am\s+called|i'?m\s+called)\s*[a-z][a-z'-]*"
)
_NAME_WORD = r"[A-Z][a-z'-]*+"
_CALL_ME = re.compile(rf"\b(?i:call\s+me)\s+{_NAME_WORD}")
_I_AM = re.compile(rf"\b(?i:i'?m|i\s+am)\s+{_NAME_WORD}(?:\s+{_NAME_WORD}|\s*(?:[.!?]|$))")
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
    rf"\b(?:born|dob|d\.o\.b\.?|date\s+of\s+birth|birthday)\b"
    # Atomic and bounded: an unbounded ``(?:\s*…(?:is|was|on|in)\b)*`` backtracks exponentially.
    rf"(?>\s*+[:=-]?\s*+(?:is|was|on|in)\b){{0,3}}\s*+[:=-]?\s*{_DATE}",
    re.IGNORECASE,
)


def _fold_keep_case(text: str) -> str:
    """:func:`dsl.normalize` per character, keeping upper case (so "Аlex" with a Cyrillic А → "Alex")."""
    return "".join(n.upper() if ch.isupper() else n for ch in text for n in (dsl.normalize(ch),))


def _has_phone(text: str) -> bool:
    masked = _ISO_DATE.sub(" ", text)
    return any(sum(ch.isdigit() for ch in m.group()) >= 10 for m in _PHONE.finditer(masked))


def scan(text: str) -> tuple[str, ...]:
    """Reason codes (in :data:`REASONS` order) for personal-data-looking content in ``text``."""
    norm = dsl.normalize(text)
    cased = _fold_keep_case(text)
    found = {
        "email": bool(_EMAIL.search(text)),
        "phone": _has_phone(text),
        "national_id": bool(_SSN.search(text) or _DIGIT_RUN.search(text)),
        "self_name": bool(_SELF_NAME.search(norm) or _CALL_ME.search(cased) or _I_AM.search(cased)),
        "date_of_birth": bool(_DOB.search(norm)),
    }
    return tuple(r for r in REASONS if found[r])


__all__ = ["REASONS", "scan"]
