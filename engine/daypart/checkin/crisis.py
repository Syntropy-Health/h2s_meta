"""Crisis pre-check on member check-in text (A&D failure table; plan G13).

Free text is checked for crisis language **before** extraction; a hit means nothing is extracted
or stored and the member gets :data:`HANDOFF_TEXT`.

Two vocabularies, either one is a hit (A&D failure row 1; plan G6, G13, G18):

* the ``boundary-v1`` rule pack's ``crisis_markers`` (:data:`CRISIS_MARKERS`: "988", "emergency
  services", "crisis line", ...), the vocabulary the A&D names. A numeric marker followed by a unit
  ("walked 988 steps") is a measurement, not a hand-off, and does not count;
* a regex lexicon of the language a *member* uses when in crisis (:data:`CRISIS_PATTERNS`, small
  word families), because the pack's markers are what a *reply* must contain and miss most member
  phrasing.

Both match as whole words on the DSL's :func:`~daypart.membrane.dsl.normalize` output (casefold,
NFKC, confusable and apostrophe fold), with every Unicode dash folded to ``-``, so case, accents,
curly apostrophes and dashes cannot hide a phrase.
"""

from __future__ import annotations

import re

from daypart.checkin.extract_text import unit_at
from daypart.eval import POLICY_TEXT
from daypart.membrane import dsl

#: The rule pack's crisis hand-off vocabulary, reused as-is (A&D failure row 1).
CRISIS_MARKERS: tuple[str, ...] = tuple(dsl.load_rules(dsl.RULE_VERSION).crisis_markers)
_MARKER_RES = tuple(
    re.compile(r"(?<![\w.,'-])" + r"\s+".join(map(re.escape, dsl.normalize(m).split())) + r"(?![\w'-]|[.,]\d)")
    for m in CRISIS_MARKERS
)

#: Member crisis language: small regex families (name, pattern) matched over normalised text.
CRISIS_PATTERNS: tuple[tuple[str, str], ...] = (
    ("die", r"(?:want(?:ed|s)?\s+to|wanna|going\s+to|gonna)\s+die"),
    ("end", r"(?:end|ending|ended)\s+my\s+(?:own\s+)?life|end(?:ing|ed)?\s+it\s+all"
            r"|(?:take|taking|took)\s+my\s+(?:own\s+)?life"),
    # "end it" only with intent and nothing after it but a time word ("end it tonight", not "end it early").
    ("end_it", r"(?:going\s+to|gonna|plan(?:ning)?\s+to|want\s+to)\s+end\s+it"
               r"(?=\s*(?:$|[.,!;?]|tonight|today|now|soon))"),
    ("kill", r"kill(?:ed|ing|s)?\s+my\s*self"),
    ("not_live", r"(?:don'?t|do\s+not)\s+want\s+to\s+(?:live|be\s+alive)|no\s+reason\s+to\s+live"),
    ("dead", r"wish\s+i\s+(?:was|were)\s+dead|better\s+off\s+dead"),
    ("pills", r"(?:took|taken|take|taking)\s+too\s+many\s+(?:pills|tablets)|overdos(?:e|ed|es|ing)"),
    ("self_harm", r"self\s*-?\s*harm(?:ing|ed|s)?|(?:hurt|harm|harming|hurting|cut|cutting)\s+my\s*self"),
    ("method", r"hang(?:ing)?\s+my\s*self|jump(?:ing)?\s+off\s+(?:a|the)\s+(?:bridge|building|roof|cliff|balcony)"),
    # Passive ideation, in the wording screening instruments use (wish to be dead / not wake up).
    ("no_wake", r"(?:go\s+to\s+sleep|fall\s+asleep)\s+and\s+(?:not|never)\s+wake\s+up"
                r"|(?:hope|wish)\s+i\s+(?:don'?t|do\s+not|never|won'?t)\s+wake\s+up"),
    ("not_here", r"(?:don'?t|do\s+not)\s+want\s+to\s+(?:be\s+here\s+any\s*more|exist)"
                 r"|better\s+off\s+(?:without\s+me|gone)|disappear\s+forever"),
    ("no_point", r"no\s+point\s+(?:in\s+)?living|(?:isn'?t|is\s+not|not)\s+worth\s+living|point\s+of\s+living"),
    ("unsafe", r"(?:i'?m|i\s+am)\s+not\s+safe|(?:can'?t|cannot|can\s+not)\s+keep\s+my\s*self\s+safe"),
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
    """True when ``text`` holds a pack crisis marker or a member-crisis family (whole words, normalised)."""
    norm = dsl.normalize(text).translate(_DASH_FOLD)
    if _CRISIS_RE.search(norm):
        return True
    for rx in _MARKER_RES:
        for m in rx.finditer(norm):
            if not (m.group().isdigit() and unit_at(norm, m.end()) is not None):
                return True
    return False


__all__ = ["CRISIS_MARKERS", "CRISIS_PATTERNS", "HANDOFF_TEXT", "is_crisis"]
