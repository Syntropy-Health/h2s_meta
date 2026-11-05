"""Membrane boundary DSL — the deterministic gate (CONTRACT slice B, B1/B2).

The rule pack (``rules/<version>.yaml``) is the versioned artifact; this module
only normalizes text, expands the pack's ``<<placeholder>>`` templates, compiles
every pattern once, and scans. Pure and deterministic — no LLM, no I/O after
the first (cached) pack load.

Normalization mirrors ``syntropy.lhm.metrics.claims._fold`` (casefold, NFKC,
Cf/Mn stripped, Cyrillic/Greek confusable fold before AND after NFKC) and the
negated-disclaimer excision mirrors ``claims._DISCLAIMER_RE``. Both are private
in LHM so they are re-implemented here faithfully; the banned lexicon itself is
imported from LHM at runtime (D10), never copied.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml
from syntropy.lhm.metrics.claims import BANNED_LEXICON

RULE_VERSION = "boundary-v1"

RULE_IDS: frozenset[str] = frozenset(
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


@dataclass(frozen=True)
class Violation:
    rule_id: str  # one of RULE_IDS
    match: str  # the matched substring (from the normalized text)


@dataclass(frozen=True)
class Verdict:
    passed: bool  # True iff violations == ()
    violations: tuple[Violation, ...]
    rule_version: str  # "boundary-v1"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    kind: str  # "pattern" | "absence"
    patterns: tuple[re.Pattern[str], ...]
    crisis_context_only: bool = False


@dataclass(frozen=True)
class RulePack:
    version: str
    rules: tuple[Rule, ...]
    number_words: tuple[str, ...]
    dose_units: tuple[str, ...]
    conditions: tuple[str, ...]
    rx_substances: tuple[str, ...]
    crisis_markers: tuple[str, ...]

    @property
    def rule_ids(self) -> frozenset[str]:
        return frozenset(r.rule_id for r in self.rules)


# --------------------------------------------------------------------------- #
# Normalization (mirror of syntropy.lhm.metrics.claims._fold)
# --------------------------------------------------------------------------- #

_CONFUSABLE_FOLD = str.maketrans(
    {
        # Cyrillic lowercase -> Latin lookalikes
        "а": "a",
        "в": "b",
        "е": "e",
        "ё": "e",
        "к": "k",
        "м": "m",
        "н": "h",
        "о": "o",
        "р": "p",
        "с": "c",
        "т": "t",
        "у": "y",
        "х": "x",
        "ѕ": "s",
        "і": "i",
        "ј": "j",
        "һ": "h",
        "ԛ": "q",
        "ԝ": "w",
        # Greek lowercase -> Latin lookalikes
        "α": "a",
        "ε": "e",
        "ι": "i",
        "κ": "k",
        "ν": "v",
        "ο": "o",
        "ρ": "p",
        "τ": "t",
        "υ": "u",
        "χ": "x",
        "ϲ": "c",
    }
)

# Typographic apostrophes -> ASCII so "you’re" / "it’s" match the pack's frames.
_APOSTROPHE_FOLD = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "＇": "'"})


def normalize(text: str) -> str:
    """Casefold + NFKC + strip Cf/Mn + Cyrillic/Greek confusable fold (LHM ``_fold``)."""
    prefolded = text.casefold().translate(_CONFUSABLE_FOLD)
    normalized = unicodedata.normalize("NFKC", prefolded)
    stripped = "".join(
        ch for ch in normalized if unicodedata.category(ch) not in ("Cf", "Mn")
    ).casefold()
    folded = stripped.translate(_CONFUSABLE_FOLD)
    # Hardening beyond LHM: NFKC re-composes a base letter with a following
    # combining mark ("m" + U+0301 -> "ḿ"), so the Mn strip above never sees
    # it. Decompose, drop marks, recompose — accents never hide a token.
    decomposed = unicodedata.normalize("NFKD", folded)
    unmarked = unicodedata.normalize(
        "NFC", "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    )
    return unmarked.translate(_CONFUSABLE_FOLD).translate(_APOSTROPHE_FOLD)


# Mirror of claims._DISCLAIMER_RE: both branches end AT the disclaimer noun/verb,
# so everything after it is still scanned.
_FDA_NEGATED_VERBS = r"(?:diagnose|treat|cure|prevent|mitigate)"
_DISCLAIMER_RE = re.compile(
    r"(?:"
    r"not\s+(?:intended|meant)\s+to\s+[^.;,:()—–-]*?"
    + _FDA_NEGATED_VERBS
    + r"(?:(?:\s*,\s*(?:or\s+|and\s+)?|\s+(?:or|and)\s+)"
    + _FDA_NEGATED_VERBS
    + r")*"
    r"|not\s+(?:a\s+|an\s+)?(?:medical\s+|clinical\s+)?"
    r"(?:recommendation|advice|diagnosis|prescription)"
    r")",
    re.IGNORECASE,
)


def strip_disclaimers(normalized: str) -> str:
    """Excise recognized negated disclaimers.

    The span is replaced by a sentence break (" . "), so no pattern can bridge
    the gap and fuse the words around the disclaimer into a new match ("this is
    [not intended to ... prevent] any disease"); the text after the disclaimer
    is still scanned as its own clause.
    """
    return _DISCLAIMER_RE.sub(" . ", normalized)


# --------------------------------------------------------------------------- #
# Rule-pack loading / compilation
# --------------------------------------------------------------------------- #

_VERSION_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_PLACEHOLDER_RE = re.compile(r"<<([A-Za-z_][A-Za-z0-9_]*)>>")
_MAX_EXPANSION_DEPTH = 10


def _alternation(phrases: list[str] | tuple[str, ...]) -> str:
    """Escaped alternation, longest first; internal spaces match any whitespace."""
    items = sorted({str(p).casefold() for p in phrases}, key=lambda s: (-len(s), s))
    return "|".join(r"\s+".join(re.escape(part) for part in p.split()) for p in items)


def _expand(template: str, vocab: dict[str, str], fragments: dict[str, str]) -> str:
    def sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in vocab:
            return vocab[name]
        if name in fragments:
            return fragments[name]
        raise ValueError(f"unknown rule-pack placeholder <<{name}>>")

    out = template
    for _ in range(_MAX_EXPANSION_DEPTH):
        new = _PLACEHOLDER_RE.sub(sub, out)
        if new == out:
            return out
        out = new
    raise ValueError("rule-pack placeholder expansion did not terminate")


def _read_pack_text(version: str) -> str:
    if not isinstance(version, str) or not _VERSION_RE.match(version):
        raise FileNotFoundError(f"no membrane rule pack for version {version!r}")
    res = resources.files("daypart.membrane").joinpath("rules", f"{version}.yaml")
    if not res.is_file():
        raise FileNotFoundError(f"no membrane rule pack for version {version!r}")
    return res.read_text(encoding="utf-8")


def _str_list(raw: Any, key: str) -> tuple[str, ...]:
    vals = raw.get(key) or []
    if not isinstance(vals, list) or not all(isinstance(v, (str, int)) for v in vals):
        raise ValueError(f"rule-pack vocab {key!r} must be a list of strings")
    return tuple(str(v) for v in vals)


@lru_cache(maxsize=None)
def load_rules(version: str = RULE_VERSION) -> RulePack:
    """Load + compile a rule pack shipped as package data. Unknown version -> FileNotFoundError."""
    data = yaml.safe_load(_read_pack_text(version))
    if not isinstance(data, dict) or data.get("version") != version:
        raise ValueError(f"rule pack {version!r} declares version {data.get('version')!r}")

    raw_vocab = data.get("vocab") or {}
    vocab_lists = {k: _str_list(raw_vocab, k) for k in raw_vocab}
    vocab = {k: _alternation(v) for k, v in vocab_lists.items()}
    # D10: the LHM banned lexicon is imported at runtime, never copied.
    vocab["lhm_banned_lexicon"] = _alternation(BANNED_LEXICON)
    fragments = {str(k): str(v) for k, v in (data.get("fragments") or {}).items()}

    rules: list[Rule] = []
    for spec in data.get("rules") or []:
        rid = spec["id"]
        kind = spec.get("kind", "pattern")
        sources = spec.get("markers") if kind == "absence" else spec.get("patterns")
        if not sources:
            raise ValueError(f"rule {rid!r} has no patterns")
        compiled = tuple(re.compile(_expand(str(p), vocab, fragments)) for p in sources)
        rules.append(
            Rule(
                rule_id=rid,
                kind=kind,
                patterns=compiled,
                crisis_context_only=bool(spec.get("crisis_context_only", False)),
            )
        )

    ids = [r.rule_id for r in rules]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate rule ids in rule pack")
    if version == RULE_VERSION and set(ids) != RULE_IDS:
        raise ValueError(f"rule pack ids {sorted(ids)} != contract RULE_IDS")

    return RulePack(
        version=version,
        rules=tuple(rules),
        number_words=vocab_lists.get("number_words", ()),
        dose_units=vocab_lists.get("dose_units", ()),
        conditions=vocab_lists.get("conditions", ()),
        rx_substances=vocab_lists.get("rx_substances", ()),
        crisis_markers=vocab_lists.get("crisis_markers", ()),
    )


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #


def scan(text: str, pack: RulePack, *, crisis_context: bool = False) -> Verdict:
    """Run ``pack`` over ``text`` (normalize -> excise disclaimers -> match)."""
    if not isinstance(text, str):
        raise TypeError(f"check() expects str, got {type(text).__name__}")
    normalized = normalize(text)
    scanned = strip_disclaimers(normalized)

    violations: list[Violation] = []
    seen: set[tuple[str, str]] = set()
    for rule in pack.rules:
        if rule.kind == "absence":
            if rule.crisis_context_only and not crisis_context:
                continue
            if not any(p.search(normalized) for p in rule.patterns):
                violations.append(Violation(rule.rule_id, ""))
            continue
        for pat in rule.patterns:
            for m in pat.finditer(scanned):
                key = (rule.rule_id, m.group(0))
                if key not in seen:
                    seen.add(key)
                    violations.append(Violation(rule.rule_id, m.group(0)))

    return Verdict(passed=not violations, violations=tuple(violations), rule_version=pack.version)


def check(text: str, *, crisis_context: bool = False) -> Verdict:
    """Deterministic boundary verdict for ``text`` under the current rule pack."""
    return scan(text, load_rules(RULE_VERSION), crisis_context=crisis_context)


__all__ = [
    "RULE_IDS",
    "RULE_VERSION",
    "Rule",
    "RulePack",
    "Verdict",
    "Violation",
    "check",
    "load_rules",
    "normalize",
    "strip_disclaimers",
]
