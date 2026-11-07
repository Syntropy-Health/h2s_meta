"""rules-v1: deterministic, offline extraction of proposals from check-in text (A&D §4.1, D-A2).

:func:`extract` turns free text into :class:`~daypart.checkin.models.Proposal` values that wait for
review; nothing here writes anywhere. :func:`detect_flags` re-runs flag detection over a whole
(possibly corrected) proposal set, which is what ``review()`` calls after applying corrections.

Rules, in order:

1. The text is lowercased (position-preserving). Registry keywords are matched as whole words,
   longest first, never overlapping; then symptom-like words that are not in the registry.
2. Dates: ISO ``yyyy-mm-dd``, ``Mon DD`` / ``DD Mon`` (optional year), ``today``, ``yesterday``,
   ``last week`` and weekday names. A date applies to its clause; a clause without one uses the
   only date in the text, else the nearest earlier clause's date, else ``today``.
3. Numbers (outside date spans), each with an optional unit token straight after it, and
   blood-pressure pairs ``A/B`` (2-3 digits each, ``B != 10``).
4. Each keyword group (same concept, same clause) takes the nearest unclaimed number in its clause
   within :data:`NEAR` characters on either side. A keyword that *is* the unit of a number
   ("8000 steps", "64 bpm") binds to that number first.
5. Confidence: ``high`` = keyword + number + allowed unit; ``medium`` = keyword + number with no
   unit (the concept default is used and ``unit_assumed`` is raised); ``low`` = anything else.
"""

from __future__ import annotations

import bisect
import datetime as dt
import re
from collections import Counter
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field

from daypart.checkin.models import FLAGS, METHOD_TEXT, SNIPPET_MAX, Proposal
from daypart.memory.registry import Concept, load_registry

NEAR = 30
SYSTOLIC, DIASTOLIC = "8480-6", "8462-4"
_BP_CODES = (SYSTOLIC, DIASTOLIC)

#: Symptom-like words with no registry concept: proposals with ``concept_code=None`` (unknown_code).
UNKNOWN_WORDS: tuple[str, ...] = (
    "headache", "headaches", "migraine", "nausea", "nauseous", "nauseated", "dizzy", "dizziness",
    "cough", "coughing", "fatigue", "fatigued", "tired", "exhausted", "anxious", "anxiety", "rash",
    "cramps", "cramping", "sore throat", "bloated", "bloating",
)

# Unit tokens straight after a number -> the UCUM code they stand for. Order matters (longest first).
_UNIT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"out\s+of\s+10(?!\d)", "{score}"),
    (r"/\s*10(?![\d.])", "{score}"),
    (r"(?:beats|breaths)\s*(?:per\s+minute|/\s*min)(?![a-z])", "/min"),
    (r"/\s*min(?:ute)?(?![a-z])", "/min"),
    (r"bpm(?![a-z])", "/min"),
    (r"breaths(?![a-z])", "/min"),
    (r"mm\s*hg(?![a-z])", "mm[Hg]"),
    (r"kg\s*/\s*m(?:2|²)", "kg/m2"),
    (r"(?:kilograms?|kilos?|kgs?)(?![a-z])", "kg"),
    (r"(?:pounds?|lbs?)(?![a-z])", "[lb_av]"),
    (r"(?:centimet(?:er|re)s?|cm)(?![a-z])", "cm"),
    (r"(?:inches|inch)(?![a-z])|in(?=\s*(?:$|[,;.!?]))", "[in_i]"),
    (r"(?:hours?|hrs?|h)(?![a-z])", "h"),
    (r"(?:minutes?|mins?|min)(?![a-z])", "min"),
    (r"steps?(?![a-z])", "{steps}"),
    (r"(?:percent(?![a-z])|%)", "%"),
    (r"(?:[°º]\s*f|fahrenheit|degf|f)(?![a-z])", "[degF]"),
    (r"(?:[°º]\s*c|celsius|degc|c)(?![a-z])", "Cel"),
)
_UNIT_RES = tuple((re.compile(r"\s*(?:" + p + ")"), u) for p, u in _UNIT_PATTERNS)
# A token that names the concept's own counting unit (e.g. "steps" for steps in 24 hours).
_UNIT_EQUIV: dict[str, dict[str, str]] = {"41950-7": {"{steps}": "{steps}/(24.h)"}}

_NUMBER = re.compile(r"(?<![\w.,/-])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\d])")
_BP_PAIR = re.compile(r"(?<![\w.,/-])([1-9]\d{1,2})\s*/\s*([1-9]\d{1,2})(?![\d.]|\s*/)")

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9,
    "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_DATE_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("iso", re.compile(r"(?<![\d-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d-])")),
    ("dmon", re.compile(rf"(?<![\w.])(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_RE})\.?(?![a-z])(?:,?\s+(\d{{4}})(?!\d))?")),
    ("mond", re.compile(rf"(?<![a-z])({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?![\d:])(?:,?\s+(\d{{4}})(?!\d))?")),
    ("today", re.compile(r"(?<![a-z])today(?![a-z])")),
    ("yesterday", re.compile(r"(?<![a-z])yesterday(?![a-z])")),
    ("last_week", re.compile(r"(?<![a-z])last\s+week(?![a-z])")),
    ("weekday", re.compile(rf"(?<![a-z])({'|'.join(_WEEKDAYS)})s?(?![a-z])")),
)
_CLAUSE_BREAK = re.compile(
    r"[;!?\n]|(?<!\d),|,(?!\d)|(?<!\d)\.|\.(?!\d)|(?<![a-z])(?:and|but|then|also|while|plus)(?![a-z])"
)


@dataclass
class _Num:
    start: int
    end: int  # end of the number (or pair)
    unit_end: int  # end of the unit token, == end when there is none
    value: float
    unit: str | None
    value2: float | None = None  # diastolic, for a blood-pressure pair
    claimed: bool = False

    @property
    def is_pair(self) -> bool:
        return self.value2 is not None


@dataclass
class _Hit:
    start: int
    end: int
    concept: Concept | None  # None = unknown symptom word
    word: str
    anchor: _Num | None = None


@dataclass
class _Group:
    concept: Concept
    clause: int
    hits: list[_Hit] = field(default_factory=list)


@dataclass
class _Draft:
    start: int
    end: int
    order: int
    concept: Concept | None
    display: str
    value: float | None
    unit: str | None
    date: dt.date
    confidence: str
    sticky: set[str]


def _lower(text: str) -> str:
    """Lowercase without changing the length, so spans index the original text."""
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text)


def _overlaps(a0: int, a1: int, spans: Iterable[tuple[int, int]]) -> bool:
    return any(a0 < b1 and b0 < a1 for b0, b1 in spans)


def _word_re(phrase: str) -> re.Pattern[str]:
    return re.compile(r"(?<![\w])" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?![\w])")


def _keyword_hits(low: str) -> list[_Hit]:
    reg = load_registry()
    pairs: list[tuple[str, Concept | None]] = []
    seen: set[str] = set()
    for c in reg.concepts():
        for kw in c.keywords:
            k = kw.lower()
            if k not in seen:
                seen.add(k)
                pairs.append((k, c))
    known = sorted(pairs, key=lambda p: (-len(p[0]), p[0]))
    unknown = sorted(((w, None) for w in UNKNOWN_WORDS if w not in seen), key=lambda p: (-len(p[0]), p[0]))
    hits: list[_Hit] = []
    for word, concept in (*known, *unknown):
        for m in _word_re(word).finditer(low):
            if not _overlaps(m.start(), m.end(), ((h.start, h.end) for h in hits)):
                hits.append(_Hit(m.start(), m.end(), concept, word))
    return sorted(hits, key=lambda h: h.start)


def _safe_date(y: int, m: int, d: int) -> dt.date | None:
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


def _date_tokens(low: str, today: dt.date, blocked: list[tuple[int, int]]) -> list[tuple[int, int, dt.date, bool]]:
    """``(start, end, date, ambiguous)`` per date mention, non-overlapping, in text order."""
    out: list[tuple[int, int, dt.date, bool]] = []
    for kind, rx in _DATE_RES:
        for m in rx.finditer(low):
            if _overlaps(m.start(), m.end(), blocked) or _overlaps(m.start(), m.end(), ((s, e) for s, e, _, _ in out)):
                continue
            resolved: dt.date | None
            ambiguous = False
            if kind == "iso":
                resolved = _safe_date(int(m[1]), int(m[2]), int(m[3]))
                ambiguous = resolved is not None and resolved > today
            elif kind in ("dmon", "mond"):
                day, mon = (m[1], m[2]) if kind == "dmon" else (m[2], m[1])
                month = _MONTHS[mon]
                if m[3]:
                    resolved = _safe_date(int(m[3]), month, int(day))
                    ambiguous = resolved is not None and resolved > today
                else:
                    resolved = _safe_date(today.year, month, int(day))
                    if resolved is not None and resolved > today:
                        ambiguous = True
                        resolved = _safe_date(today.year - 1, month, int(day)) or resolved
            elif kind == "today":
                resolved = today
            elif kind == "yesterday":
                resolved = today - dt.timedelta(days=1)
            elif kind == "last_week":
                resolved, ambiguous = today - dt.timedelta(days=7), True
            else:  # weekday: the most recent such day, still ambiguous
                back = (today.weekday() - _WEEKDAYS.index(m[1])) % 7
                resolved, ambiguous = today - dt.timedelta(days=back), True
            if resolved is None:  # "feb 30": a date-shaped mention that is not a date
                resolved, ambiguous = today, True
            out.append((m.start(), m.end(), resolved, ambiguous))
    return sorted(out)


def _numbers(low: str, masked: list[tuple[int, int]]) -> list[_Num]:
    nums: list[_Num] = []

    def unit_after(pos: int) -> tuple[str | None, int]:
        for rx, ucum in _UNIT_RES:
            m = rx.match(low, pos)
            if m:
                return ucum, m.end()
        return None, pos

    taken: list[tuple[int, int]] = list(masked)
    for m in _BP_PAIR.finditer(low):
        if m[2] == "10" or _overlaps(m.start(), m.end(), taken):
            continue
        unit, uend = unit_after(m.end())
        unit = unit if unit == "mm[Hg]" else None
        uend = uend if unit else m.end()
        nums.append(_Num(m.start(), m.end(), uend, float(m[1]), unit, value2=float(m[2])))
        taken.append((m.start(), uend))
    for m in _NUMBER.finditer(low):
        if _overlaps(m.start(), m.end(), taken):
            continue
        unit, uend = unit_after(m.end())
        nums.append(_Num(m.start(), m.end(), uend, float(m[1].replace(",", "") + (m[2] or "")), unit))
        taken.append((m.start(), uend))
    return sorted(nums, key=lambda n: n.start)


def _allowed(concept: Concept) -> dict[str, float]:
    """Unit -> factor to the canonical unit, for every unit the concept accepts."""
    return {concept.unit: 1.0, **dict(concept.alt_units)}


def _norm_unit(concept: Concept, unit: str | None) -> str | None:
    if unit is None:
        return None
    return _UNIT_EQUIV.get(concept.code, {}).get(unit, unit)


def _clauses(low: str) -> list[int]:
    return [m.start() for m in _CLAUSE_BREAK.finditer(low)]


def _gap(hit: _Hit, num: _Num) -> int:
    if num.start >= hit.end:
        return num.start - hit.end
    if num.unit_end <= hit.start:
        return hit.start - num.unit_end
    return 0


def extract(text: str, today: dt.date, *, checkin_id: str = "c-0") -> tuple[Proposal, ...]:
    """rules-v1 proposals for ``text`` (in text order, ids ``p-1``, ``p-2``, ...), flags already set."""
    low = _lower(text)
    breaks = _clauses(low)

    def clause(pos: int) -> int:
        return bisect.bisect_right(breaks, pos)

    hits = _keyword_hits(low)
    dates = _date_tokens(low, today, [(h.start, h.end) for h in hits])
    nums = _numbers(low, [(s, e) for s, e, _, _ in dates])

    clause_date: dict[int, tuple[dt.date, bool]] = {}
    for s, _e, d, amb in dates:
        clause_date.setdefault(clause(s), (d, amb))
    single = (dates[0][2], dates[0][3]) if len({(d, a) for _, _, d, a in dates}) == 1 else None

    def date_at(pos: int) -> tuple[dt.date, bool]:
        c = clause(pos)
        if c in clause_date:
            return clause_date[c]
        if single is not None:
            return single
        earlier = [k for k in clause_date if k < c]
        return clause_date[max(earlier)] if earlier else (today, False)

    # A keyword sitting on a number's unit token ("8000 steps", "64 bpm", "6 hr") binds to that
    # number when the unit fits its concept, and is not a keyword at all otherwise.
    kept: list[_Hit] = []
    for h in hits:
        on_unit = next((n for n in nums if n.end < n.unit_end and h.start < n.unit_end and n.end < h.end), None)
        if on_unit is not None and h.concept is not None:
            if _norm_unit(h.concept, on_unit.unit) in _allowed(h.concept) and not on_unit.is_pair:
                h.anchor = on_unit
            else:
                continue
        kept.append(h)

    groups: dict[tuple[str, int], _Group] = {}
    unknown: list[_Hit] = []
    for h in kept:
        if h.concept is None:
            unknown.append(h)
            continue
        groups.setdefault((h.concept.code, clause(h.start)), _Group(h.concept, clause(h.start))).hits.append(h)

    # Assign numbers: anchors first, then the smallest gap (ties: the number after the keyword).
    candidates: list[tuple[int, int, int, int, _Group, _Num]] = []
    for gi, g in enumerate(groups.values()):
        for h in g.hits:
            for ni, n in enumerate(nums):
                if clause(n.start) != g.clause or (n.is_pair and g.concept.code not in _BP_CODES):
                    continue
                if h.anchor is n:
                    candidates.append((-1, 0, gi, ni, g, n))
                    continue
                gap = _gap(h, n)
                if gap <= NEAR:
                    candidates.append((gap, 0 if n.start >= h.end else 1, gi, ni, g, n))
    chosen: dict[int, _Num] = {}
    for _gap_, _side, gi, _ni, _g, n in sorted(candidates, key=lambda c: c[:4]):
        if gi in chosen or n.claimed:
            continue
        chosen[gi] = n
        n.claimed = True

    reg = load_registry()
    drafts: list[_Draft] = []
    numbered_clauses = {g.clause for gi, g in enumerate(groups.values()) if gi in chosen}
    for gi, g in enumerate(groups.values()):
        start, end = min(h.start for h in g.hits), max(h.end for h in g.hits)
        n = chosen.get(gi)
        if n is None:
            if g.clause in numbered_clauses:
                continue  # a bare keyword beside a recorded value ("walked 8000 steps today")
            d, amb = date_at(start)
            drafts.append(_Draft(start, end, 0, g.concept, g.concept.display, None, None, d, "low",
                                 {"ambiguous_date"} if amb else set()))
            continue
        start, end = min(start, n.start), max(end, n.unit_end)
        d, amb = date_at(start)
        if n.is_pair:
            for order, code, value in ((0, SYSTOLIC, n.value), (1, DIASTOLIC, n.value2)):
                c = reg.by_code(code)
                assert c is not None
                drafts.append(_measure(start, end, order, c, value, n.unit, d, amb, keyword=True))
        else:
            drafts.append(_measure(start, end, 0, g.concept, n.value, n.unit, d, amb, keyword=True))

    for n in nums:  # a bare "120/80" with no keyword is still a blood pressure (low confidence)
        if n.is_pair and not n.claimed:
            d, amb = date_at(n.start)
            for order, code, value in ((0, SYSTOLIC, n.value), (1, DIASTOLIC, n.value2)):
                c = reg.by_code(code)
                assert c is not None
                drafts.append(_measure(n.start, n.unit_end, order, c, value, n.unit, d, amb, keyword=False))

    for h in unknown:
        d, amb = date_at(h.start)
        drafts.append(_Draft(h.start, h.end, 0, None, h.word, None, None, d, "low",
                             {"ambiguous_date"} if amb else set()))

    drafts.sort(key=lambda dr: (dr.start, dr.order, dr.end))
    proposals = []
    for i, dr in enumerate(drafts, start=1):
        proposals.append(
            Proposal(
                proposal_id=f"p-{i}",
                concept_code=dr.concept.code if dr.concept else None,
                display=dr.display,
                value=dr.value,
                unit=dr.unit,
                date=dr.date,
                source_kind="text",
                source_ref=f"{checkin_id}:{dr.start}-{dr.end}",
                snippet=text[dr.start:dr.end][:SNIPPET_MAX],
                method=METHOD_TEXT,
                confidence=dr.confidence,  # type: ignore[arg-type]
                flags=tuple(sorted(dr.sticky, key=FLAGS.index)),  # type: ignore[arg-type]
            )
        )
    return _detect(proposals, sticky={p.proposal_id: set(p.flags) for p in proposals})


def _measure(start: int, end: int, order: int, concept: Concept, value: float | None, unit_token: str | None,
             d: dt.date, ambiguous: bool, *, keyword: bool) -> _Draft:
    sticky = {"ambiguous_date"} if ambiguous else set()
    unit = _norm_unit(concept, unit_token)
    if unit is None:
        unit, confidence = concept.unit, "medium" if keyword else "low"
        sticky.add("unit_assumed")
    elif unit in _allowed(concept):
        confidence = "high" if keyword else "low"
    else:
        confidence = "low"
    return _Draft(start, end, order, concept, concept.display, value, unit, d, confidence, sticky)


_STICKY = frozenset({"unit_assumed", "ambiguous_date"})


def _detect(proposals: Iterable[Proposal], sticky: dict[str, set[str]]) -> tuple[Proposal, ...]:
    reg = load_registry()
    props = tuple(proposals)
    flags: list[set[str]] = []
    for p in props:
        f = set(sticky.get(p.proposal_id, set())) & _STICKY
        concept = reg.by_code(p.concept_code) if p.concept_code is not None else None
        if p.value is None:
            f.add("missing_value")
        if concept is None:
            f.add("unknown_code")
        else:
            allowed = _allowed(concept)
            if p.unit is None:
                if p.value is not None:
                    f.add("unit_assumed")
            elif p.unit not in allowed:
                f.add("unit_mismatch")
            elif p.value is not None:
                lo, hi = concept.plausible
                if not lo <= p.value * allowed[p.unit] <= hi:
                    f.add("out_of_plausible_range")
        if p.date is None:
            f.add("ambiguous_date")
        flags.append(f)

    counts = Counter(p.concept_code for p in props if p.concept_code is not None)
    for p, f in zip(props, flags, strict=True):
        if p.concept_code is not None and counts[p.concept_code] > 1:
            f.add("contradiction")
    for i, s in enumerate(props):
        if s.concept_code != SYSTOLIC or s.value is None:
            continue
        for j, dia in enumerate(props):
            if dia.concept_code == DIASTOLIC and dia.value is not None and s.value <= dia.value:
                flags[i].add("contradiction")
                flags[j].add("contradiction")

    out = []
    for p, f in zip(props, flags, strict=True):
        ordered = tuple(sorted(f, key=FLAGS.index))
        out.append(Proposal.model_validate({**p.model_dump(), "flags": ordered, "blocking": bool(ordered)}))
    return tuple(out)


def detect_flags(proposals: Iterable[Proposal], corrected: Collection[str] = frozenset()) -> tuple[Proposal, ...]:
    """Re-run every flag over a whole (possibly corrected) proposal set, from scratch.

    ``proposals``: the check-in's proposals after corrections were applied (order is kept).
    ``corrected``: ids of proposals whose unit or date was *explicitly* set by a correction.

    Re-derived from the registry each time: ``unit_mismatch`` (unit neither canonical nor an alt
    unit), ``out_of_plausible_range`` (value, converted to the canonical unit, outside
    ``plausible``), ``missing_value`` (value None), ``unknown_code`` (no registry concept) and the
    set-level ``contradiction`` (two proposals for one concept, or systolic ≤ diastolic).
    ``unit_assumed`` and ``ambiguous_date`` cannot be re-derived from the values, so they are kept
    from the incoming flags unless the id is in ``corrected``; a missing unit (with a value) or a
    missing date raises them again regardless. Returns new proposals with ``blocking`` recomputed.
    """
    props = tuple(proposals)
    done = set(corrected)
    return _detect(props, sticky={p.proposal_id: set() if p.proposal_id in done else set(p.flags) for p in props})


__all__ = ["NEAR", "UNKNOWN_WORDS", "detect_flags", "extract"]
