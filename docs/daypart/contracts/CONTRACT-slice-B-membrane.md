# CONTRACT slice B — `daypart.membrane` boundary DSL + layer 0 + `post`

Plan task 1.2 (HAC-2). Sources: PVR R2; A&D R2, D10, request-path step 5, I4; plan grill G4, G5. This file is the only surface shared between the test-author and the implementer.

Package root: `engine/daypart/membrane/`. Pinned LHM is importable as `syntropy.lhm`.

## B1. Rule pack

`engine/daypart/membrane/rules/boundary-v1.yaml` — versioned YAML (`version: boundary-v1`), shipped as package data. The DSL is the authoritative, deterministic gate; no LLM call anywhere in this slice.

```python
# daypart.membrane.dsl
RULE_VERSION = "boundary-v1"

@dataclass(frozen=True)
class Violation:
    rule_id: str      # one of RULE_IDS
    match: str        # the matched substring (from the normalized text)

@dataclass(frozen=True)
class Verdict:
    passed: bool                        # True iff violations == ()
    violations: tuple[Violation, ...]
    rule_version: str                   # "boundary-v1"

RULE_IDS: frozenset[str]                # exactly the ids in B2
def load_rules(version: str = "boundary-v1") -> RulePack     # unknown version → FileNotFoundError
def check(text: str, *, crisis_context: bool = False) -> Verdict
```

`check` is pure and deterministic. Before matching, text is normalized: casefold, Unicode NFKC, zero-width / format characters (category Cf) and combining marks removed, and Cyrillic/Greek lowercase homoglyphs folded to Latin (same fold as `syntropy.lhm.metrics.claims`). Recognized **negated disclaimers** are excised before scanning (same frames LHM excises): "not intended/meant to diagnose, treat, cure, or prevent …" and "not (a/an) (medical/clinical) recommendation/advice/diagnosis/prescription". Everything after the disclaimer noun is still scanned.

## B2. Rules (exhaustive for boundary-v1)

"Number" below = digits with optional `,` thousands and `.` decimals (`400`, `1,000`, `2.5`), or an English number word (`one`…`twenty`, `thirty`…`ninety`, `hundred`, `thousand`, `half`, `a couple of`). Unit may be attached (`400mg`) or space-separated.

| rule_id | Fires when |
|---|---|
| `dosing.unit_quantity` | Any number followed by a dose unit: `mg`, `mcg`, `µg`, `ug`, `g`, `iu`, `ml`, `milligram(s)`, `microgram(s)`, `gram(s)`, `international unit(s)`, `milliliter(s)`/`millilitre(s)`, `unit(s)`, or a dose form `tablet(s)`, `capsule(s)`, `pill(s)`, `softgel(s)`, `drop(s)`, `scoop(s)`. Fires on ANY such quantity, including past tense ("you logged 500 mg") — G5, stricter than LHM. |
| `dosing.imperative` | A dose-change directive with no quantity: `(take|increase|decrease|double|halve|reduce|raise|lower|skip|stop|start|adjust|change) (your|the|a|my)? (dose|dosage|medication|meds|prescription|pills?|supplements?)`, or `stop taking`, `start taking`, `double up`, `take (an )?extra`, `take more`, `take less`. |
| `diagnosis.claim` | Diagnostic framing: `you (probably |likely |might |may |could )?(have|are suffering from|are developing)` followed within 4 words by a condition noun from the rule pack's `conditions` list; `(this|that|it) (is|sounds like|looks like|could be|indicates|means) (a |an )?(sign|symptom)s? of`; `you are (pre)?diabetic|hypertensive|depressed|anemic|deficient`; any word with stem `diagnos`. The `conditions` list contains at least: diabetes, prediabetes, hypertension, depression, anxiety disorder, anemia, deficiency, insomnia, sleep apnea, hypothyroidism, infection, disorder, disease, syndrome, cancer, arrhythmia. |
| `treatment.claim` | Treatment/efficacy framing: `(will|can|could|should|to) (fix|heal|cure|reverse|treat)`; `(is|are) (a |an )?(treatment|cure|remedy|therapy) for`; `(treats|cures|heals|reverses)`. |
| `substance.rx_named` | Any mention of a substance on the rule pack's `rx_substances` list, which contains at least: metformin, rapamycin, sirolimus, semaglutide, ozempic, wegovy, tirzepatide, mounjaro, insulin, statin(s), atorvastatin, rosuvastatin, warfarin, testosterone, hgh, sertraline, fluoxetine, adderall, modafinil, levothyroxine, prednisone, antibiotic(s). Word-boundary match. |
| `lhm.banned_lexicon` | Any word with a stem in LHM `BANNED_LEXICON` (`improve`, `treat`, `cure`, `prevent`, `reverse`, `efficacy`, `clinical`) — D10 imports this list from `syntropy.lhm` at runtime (not copied). The advice lexicon (`recommend`, `you should`, …) is NOT part of the DSL (D10/HQ2). |
| `lhm.imperative_dose` | LHM's imperative-dose structure: `take/start/stop` (any inflection) followed within 80 chars (no sentence break) by number + `mg|mcg|g|iu|ml`. |
| `crisis.missing_handoff` | Only when `crisis_context=True`: the text contains none of the handoff markers `988`, `emergency services`, `crisis line`, `crisis text line`, `call 911`, `emergency number`. Never fires when `crisis_context=False`. |

A single text may produce several violations; each fired `rule_id` appears at least once. `check("")` passes (except `crisis_context=True` → `crisis.missing_handoff`).

**D10 pair (must hold):** `check("Try setting your evening reminder 15 minutes earlier tonight.").passed is True`; `check("You should take 400 mg of magnesium.").passed is False` with `dosing.unit_quantity` among the violations.

Compliant adherence copy that must pass (examples of the compliant class): "Nice work logging three days in a row — want to keep the streak going?", "What got in the way yesterday? We can plan around it together.", "Would a reminder at 8pm fit your evening better?", "Your clinician is the right person for questions about your protocol.", "It's okay to miss a day. Want to pick it back up tomorrow morning?". Note "15 minutes", "three days", "8pm" are not doses.

## B3. Layer 0 + post — `daypart.membrane.post`

```python
@dataclass(frozen=True)
class PostResult:
    decision: Literal["pass", "escalate"]
    text: str | None          # the accepted draft verbatim when "pass"; None when "escalate"
    trace: dict               # decision-trace fragment (I4), see below

def post(draft: str, *, crisis_context: bool = False,
         regenerate: Callable[[], str] | None = None) -> PostResult
```
Algorithm (request-path step 5, G4):
1. **Layer 0:** `syntropy.lhm.metrics.claims.assert_narration_safe(draft)` (the real LHM function, called on every draft, fail-closed).
2. If layer 0 passed: `dsl.check(draft, crisis_context=...)`.
3. If either failed and `regenerate` is given and no regeneration has happened yet: call `regenerate()` **exactly once**, and run steps 1–2 on the new draft.
4. If the (final) draft passed both → `decision="pass"`, `text` = that draft **byte-identical** (never edited). Otherwise → `decision="escalate"`, `text=None`. `regenerate` is never called more than once; never called when the first draft passes.
5. Exceptions from `regenerate()` → escalate (do not propagate).

Trace fragment (JSON-serializable dict):
```python
{
  "layer0": {"guard_versions": {"claims": CLAIMS_GUARD_VERSION, "valence": VALENCE_GUARD_VERSION},  # imported from syntropy.lhm.metrics.claims ("claims-v3", "valence-v2")
             "outcome": "pass" | "fail" },          # of the final attempt
  "dsl": {"rule_version": "boundary-v1",
          "passed": bool | None,                     # None if DSL did not run on the final attempt (layer 0 failed)
          "violations": [rule_id, ...]},             # sorted unique; [] if passed or not run
  "attempts": 1 | 2,
  "regenerated": bool,
  "decision": "pass" | "escalate",
  "attempt_log": [ {"layer0": "pass"|"fail", "dsl_passed": bool|None, "violations": [...]} , ... ]   # one per attempt, in order
}
```
The trace never contains the draft text.

## B4. R2 acceptance (sealed)

On a seeded corpus of labeled responses — violations covering every rule in B2 (≥ 5 per rule, including obfuscations: attached units, number words, uppercase, zero-width chars, homoglyphs) and ≥ 100 compliant adherence responses in the style above — `check` reaches **100 % recall** on violations and **< 2 % false positives** on compliant responses.
