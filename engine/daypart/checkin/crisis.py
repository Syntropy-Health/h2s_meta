"""Crisis pre-check on member check-in text (A&D failure table; plan G13).

Free text is checked for crisis language **before** extraction; a hit means nothing is extracted
or stored and the member gets :data:`HANDOFF_TEXT`.

Why a lexicon of its own: the ``boundary-v1`` rule pack's ``crisis_markers`` are the hand-off
markers a *reply* must contain ("988", "emergency services", ...), not the language a *member*
uses when in crisis, so reusing them (the original G6 reading) would detect nothing useful. This
module keeps a regex lexicon of member crisis language (small word families, see
:data:`CRISIS_PATTERNS`) and matches it, as whole words, on the DSL's
:func:`~daypart.membrane.dsl.normalize` output (casefold, NFKC, confusable and apostrophe fold),
with every Unicode dash folded to ``-``, so case, accents, curly apostrophes and dashes cannot
hide a phrase.
"""

from __future__ import annotations

import re

from daypart.eval import POLICY_TEXT
from daypart.membrane import dsl

#: Member crisis language: small regex families (name, pattern) matched over normalised text.
CRISIS_PATTERNS: tuple[tuple[str, str], ...] = (
    ("die", r"(?:want(?:ed|s)?\s+to|wanna|going\s+to|gonna)\s+die"),
    ("end", r"(?:end|ending|ended)\s+my\s+(?:own\s+)?life|end(?:ing|ed)?\s+it\s+all|take\s+my\s+own\s+life"),
    ("kill", r"kill(?:ed|ing|s)?\s+my\s*self"),
    ("not_live", r"(?:don'?t|do\s+not)\s+want\s+to\s+(?:live|be\s+alive)|no\s+reason\s+to\s+live"),
    ("dead", r"wish\s+i\s+(?:was|were)\s+dead|better\s+off\s+dead"),
    ("pills", r"(?:took|taken|take|taking)\s+too\s+many\s+(?:pills|tablets)|overdos(?:e|ed|es|ing)"),
    ("self_harm", r"self\s*-?\s*harm(?:ing|ed|s)?|(?:hurt|harm|harming|hurting|cutting)\s+my\s*self"),
    ("cant_go_on", r"(?:can'?t|cannot|can\s+not)\s+go\s+on"),
    ("suicide", r"suicid\w*"),
)

_CRISIS_RE = re.compile(r"(?<![\w'-])(?:" + "|".join(p for _, p in CRISIS_PATTERNS) + r")(?![\w'-])")
# Any Unicode dash or minus -> "-", so "self–harm" / "self—harm" match the self-harm family.
_DASH_FOLD = str.maketrans(dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d", "-"))

HANDOFF_TEXT = (
    POLICY_TEXT["clinician_handoff"]
    + " If you might be in danger or thinking about harming yourself, contact your local emergency services now."
)


def is_crisis(text: str) -> bool:
    """True when ``text`` matches a member-crisis family (whole words, after normalisation)."""
    return bool(_CRISIS_RE.search(dsl.normalize(text).translate(_DASH_FOLD)))


__all__ = ["CRISIS_PATTERNS", "HANDOFF_TEXT", "is_crisis"]
