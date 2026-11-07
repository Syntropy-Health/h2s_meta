"""Crisis pre-check on member check-in text (A&D failure table; plan G13).

Free text is checked for crisis language **before** extraction; a hit means nothing is extracted
or stored and the member gets :data:`HANDOFF_TEXT`.

Why a lexicon of its own: the ``boundary-v1`` rule pack's ``crisis_markers`` are the hand-off
markers a *reply* must contain ("988", "emergency services", ...), not the language a *member*
uses when in crisis, so reusing them (the original G6 reading) would detect nothing useful. This
module keeps a whole-phrase lexicon of member crisis language and matches it on the DSL's
:func:`~daypart.membrane.dsl.normalize` output (casefold, NFKC, confusable and apostrophe fold),
so case, accents and curly apostrophes cannot hide a phrase.
"""

from __future__ import annotations

import re

from daypart.eval import POLICY_TEXT
from daypart.membrane import dsl

CRISIS_PHRASES: tuple[str, ...] = (
    "suicide",
    "suicidal",
    "kill myself",
    "killing myself",
    "end my life",
    "take my own life",
    "want to die",
    "better off dead",
    "no reason to live",
    "self-harm",
    "self harm",
    "self-harming",
    "self harming",
    "hurt myself",
    "harm myself",
    "harming myself",
    "cutting myself",
    "overdose",
    "overdosed",
    "can't go on",
    "cant go on",
    "cannot go on",
)

_CRISIS_RE = re.compile(
    r"(?<![\w'-])(?:" + "|".join(re.escape(p).replace(r"\ ", r"\s+") for p in CRISIS_PHRASES) + r")(?![\w'-])"
)

HANDOFF_TEXT = (
    POLICY_TEXT["clinician_handoff"]
    + " If you might be in danger or thinking about harming yourself, contact your local emergency services now."
)


def is_crisis(text: str) -> bool:
    """True when ``text`` contains a whole phrase of member crisis language."""
    return bool(_CRISIS_RE.search(dsl.normalize(text)))


__all__ = ["CRISIS_PHRASES", "HANDOFF_TEXT", "is_crisis"]
