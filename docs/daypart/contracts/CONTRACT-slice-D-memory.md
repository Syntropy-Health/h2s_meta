# CONTRACT slice D — `daypart.memory` events, metric ontology, LHM metric state

Plan tasks 3.1–3.3 (HAC-3, child of HAC-2). Sources: A&D D13, D14, I6, I7, I9; plan grill G3. This file is the only surface shared between the test-author and the implementer.

Package root: `engine/daypart/memory/`. Python ≥3.12. Deps: numpy, pydantic v2, pinned `syntropy.lhm` (with `[lhm-scoring]`), `daypart.simmember` (slice A).
Determinism: no wall-clock, no randomness. Every function here is a pure function of its inputs.
LHM names used below: `syntropy.lhm.metrics.scoring` (`V1_DOMAINS`, `DomainScore`, `score_domain`, `is_ready`, `WINDOW_DAYS`), `syntropy.lhm.metrics.claims` (`assert_claim_safe`, `assert_valence_safe`, `ScoreRendering`, `ClaimScopeError`, `CLAIMS_GUARD_VERSION`, `VALENCE_GUARD_VERSION`), `syntropy.lhm.tools.pipeline` (`LhmAdapter`, `CandidateSignal`, `fetch`, `analyze`).

Module import order (no cycles): `events` ← `ontology` ← `lhm`. `events` imports nothing from the other two.

## D1. Events (D14) — `daypart.memory.events`

```python
MEMBER_ID_RE = r"^(cur|syn)-\d{3}$"

ACTION_CODES: frozenset[str] = frozenset({
    "act:nudge_sent", "act:reminder_sent", "act:goal_set",
    "act:protocol_adjusted", "act:education_viewed", "act:reply_sent",
})

class ActionEvent(pydantic.BaseModel, frozen=True):     # extra="forbid"
    kind: Literal["action"] = "action"
    event_id: str            # non-empty
    member_id: str           # matches MEMBER_ID_RE
    occurred_at: datetime    # tz-aware (naive → ValidationError)
    code: str                # must be in ACTION_CODES

class SignalEvent(pydantic.BaseModel, frozen=True):     # extra="forbid"
    kind: Literal["signal"] = "signal"
    event_id: str            # non-empty
    member_id: str           # matches MEMBER_ID_RE
    occurred_at: datetime    # tz-aware
    code: str                # an ontology metric-node code (checked by EventLog, not the model)
    value: float             # finite (nan/inf → ValidationError)

Event = Annotated[ActionEvent | SignalEvent, pydantic.Field(discriminator="kind")]
def parse_event(obj: Mapping[str, Any]) -> ActionEvent | SignalEvent
```
- An action is something done (by the engine or the member). It carries **no value field** — supplying `value` to an `ActionEvent` is a `ValidationError` (extra="forbid"). A signal is an observation of one metric node.
- Neither event type has any free-text field (I7). Any extra key (e.g. `note`, `text`, `email`) → `ValidationError`.
- `parse_event`: dispatches on `kind`; missing or unknown `kind` → `pydantic.ValidationError`.
- `member_id` not matching `MEMBER_ID_RE` (e.g. `"alice@example.com"`, `"cur-1"`) → `ValidationError`.

```python
class EventLog:
    def __init__(self, ontology: "Ontology | None" = None) -> None   # None → ontology.core()
    def append(self, event: ActionEvent | SignalEvent) -> None
    def extend(self, events: Iterable[ActionEvent | SignalEvent]) -> None   # append each, in order; stops at the first rejected event (earlier ones stay stored)
    def events(self, member_id: str, *, as_of: datetime | None = None,
               kind: Literal["action", "signal"] | None = None) -> tuple[ActionEvent | SignalEvent, ...]
    def __len__(self) -> int        # total events across all members
```
- `append` raises `ValueError` (and stores nothing) when: the `event_id` already exists in the log; a `SignalEvent.code` is not a node of the log's ontology, or names a `kind="category"` node; a `SignalEvent.value` lies outside that node's `value_range` (inclusive bounds).
- The `ontology` module is imported lazily inside `EventLog` (so `events` has no import-time dependency on it).
- `events(...)` returns the member's events sorted by `(occurred_at, event_id)`, filtered to `occurred_at <= as_of` when given (inclusive), and to `kind` when given. Unknown member → `()`. Naive `as_of` → `ValueError`.
- Append-only: there is no update or delete method in this slice (purge arrives with 3.4).

## D2. Metric ontology (D13, I9, G3) — `daypart.memory.ontology`

```python
LOINC_SYSTEM = "http://loinc.org"
LOCAL_SYSTEM = "urn:daypart:metric"
CORE_VERSION = "core-v1"
CORE_CAVEATS: tuple[str, str] = (
    "Personal n-of-1 observation: an association in your own data, not causal.",
    "Scores carry an interval; small samples are uncertain.",
)

class Node(pydantic.BaseModel, frozen=True):    # extra="forbid"
    code: str
    system: Literal["http://loinc.org", "urn:daypart:metric"]
    display: str                     # non-empty
    kind: Literal["category", "metric"]
    parent: str | None               # code of the parent node; None only for categories at the root
    lhm_domain: str | None = None    # None or one of syntropy.lhm V1_DOMAINS
    value_range: tuple[float, float] | None = None   # metric: required, finite, lo < hi; category: None
    label: str = ""                  # metric: claim-gated display label (defaults to display if empty)
    caveats: tuple[str, ...] = ()    # metric: claim-gated caveats
```
Model validation (`ValidationError`):
- `system == LOINC_SYSTEM` → `code` matches `^\d{1,7}-\d$`. `system == LOCAL_SYSTEM` → `code` matches `^dp:[a-z_]+(\.[a-z_]+)*$`.
- `lhm_domain` not None and not in `V1_DOMAINS`.
- `kind="category"` with a non-None `lhm_domain` or `value_range`.
- `kind="metric"` with `value_range` None, non-finite, or `lo >= hi`; or `parent` None.

```python
class Edge(pydantic.BaseModel, frozen=True):    # extra="forbid"
    source: str          # a metric-node code or an ACTION_CODES code
    target: str          # a metric-node code
    weight: float        # REQUIRED (no default), finite, in [-1.0, 1.0]
```
Edges are the explicit linkage weights (clinical sensitivity, D14/I9). Hierarchy lives on `Node.parent`, not on edges. `Edge` missing `weight`, or weight nan/inf/outside [-1, 1] → `ValidationError`.

```python
class OntologyGateError(ValueError): ...

class UpdateRecord(pydantic.BaseModel, frozen=True):
    from_version: str
    to_version: str
    proposer: str
    rationale: str
    update_digest: str      # sha256 hex of the update's canonical JSON
    gate: Literal["passed"]

class OntologyUpdate(pydantic.BaseModel, frozen=True):   # extra="forbid"
    proposer: str            # non-empty
    rationale: str           # non-empty
    add_nodes: tuple[Node, ...] = ()
    set_edges: tuple[Edge, ...] = ()     # add a new edge or re-weight an existing (source, target)

class Ontology:              # immutable value object
    version: str
    nodes: Mapping[str, Node]            # read-only mapping (item assignment → TypeError)
    edges: tuple[Edge, ...]              # sorted by (source, target); one edge per (source, target)
    history: tuple[UpdateRecord, ...]    # () for core
    def node(self, code: str) -> Node                    # unknown → KeyError
    def is_core(self, code: str) -> bool                 # True iff the node is in the core registry
    def metric_nodes(self) -> tuple[Node, ...]           # kind == "metric", sorted by code
    def weight(self, source: str, target: str) -> float | None   # None when no such edge
    def digest(self) -> str                              # sha256 hex of canonical JSON of (version, nodes, edges)

def core() -> Ontology
def apply_update(ontology: Ontology, update: OntologyUpdate) -> Ontology
```

### D2.1 Core registry (`core()`, version `CORE_VERSION`)

`core()` returns equal ontologies on every call (same `digest()`), `history == ()`. Exactly these nodes:

| code | system | kind | parent | lhm_domain | value_range |
|---|---|---|---|---|---|
| `dp:behavior` | local | category | — | — | — |
| `dp:affect` | local | category | — | — | — |
| `dp:symptom` | local | category | — | — | — |
| `dp:nutrition` | local | category | — | — | — |
| `dp:vital` | local | category | — | — | — |
| `dp:activity` | local | category | — | — | — |
| `dp:sleep` | local | category | — | — | — |
| `dp:adherence.protocol` | local | metric | `dp:behavior` | `consistency` | (0, 1) |
| `dp:affect.checkin_sentiment` | local | metric | `dp:affect` | `sentiment_trajectory` | (-1, 1) |
| `dp:nutrition.food_log` | local | metric | `dp:nutrition` | `nutrition_regularity` | (0, 1) |
| `dp:symptom.severity` | local | metric | `dp:symptom` | `symptom_load` | (1, 3) |
| `72514-3` (pain severity 0–10 NRS) | LOINC | metric | `dp:symptom` | — | (0, 10) |
| `44261-6` (PHQ-9 total score) | LOINC | metric | `dp:affect` | — | (0, 27) |
| `70274-6` (GAD-7 total score) | LOINC | metric | `dp:affect` | — | (0, 21) |
| `8867-4` (heart rate) | LOINC | metric | `dp:vital` | — | (20, 250) |
| `29463-7` (body weight) | LOINC | metric | `dp:vital` | — | (0, 500) |
| `55423-8` (step count) | LOINC | metric | `dp:activity` | — | (0, 100000) |
| `93832-4` (sleep duration, h) | LOINC | metric | `dp:sleep` | — | (0, 24) |

- Each of the four LHM `V1_DOMAINS` is mapped by exactly one core node (D13: the upstream domains map onto seed nodes rather than being the schema). The seven LOINC nodes are unmapped (`lhm_domain=None`).
- Every core metric node has `caveats == CORE_CAVEATS`, a non-empty `label`, and `assert_claim_safe(label, caveats, "low", False)` passes for it.
- Core edges (exact; all explicit weights):

| source | target | weight |
|---|---|---|
| `act:reminder_sent` | `dp:adherence.protocol` | 0.3 |
| `act:nudge_sent` | `dp:adherence.protocol` | 0.2 |
| `act:goal_set` | `dp:adherence.protocol` | 0.2 |
| `act:education_viewed` | `dp:adherence.protocol` | 0.1 |
| `act:protocol_adjusted` | `dp:adherence.protocol` | 0.1 |
| `act:reply_sent` | `dp:affect.checkin_sentiment` | 0.1 |
| `93832-4` | `dp:affect.checkin_sentiment` | 0.2 |
| `55423-8` | `dp:affect.checkin_sentiment` | 0.1 |
| `72514-3` | `dp:affect.checkin_sentiment` | -0.3 |
| `dp:symptom.severity` | `dp:affect.checkin_sentiment` | -0.3 |

### D2.2 Gated updates (I9)

`apply_update(ont, upd)` never mutates `ont`; it returns a NEW `Ontology` or raises `OntologyGateError` (and `ont` is unchanged, including `digest()`). The gate rejects when:
1. the update is empty (no `add_nodes` and no `set_edges`);
2. an added node's `code` already exists in `ont` or repeats within the same update (this is how the frozen core is protected: no core node can be replaced, re-parented, re-mapped or removed — there is no remove operation);
3. an added node's `parent` is neither in `ont` nor added earlier in the same update;
4. an added metric node fails `assert_claim_safe(label or display, caveats, "low", False)` (e.g. no caveats, or a label containing "improve");
5. an edge's `target` is not a metric node, or its `source` is neither a metric node nor in `ACTION_CODES` (resolved after the update's nodes are added);
6. the same `(source, target)` appears twice in `set_edges`.

On success: `version` = `f"{CORE_VERSION}+u{n}"` where `n = len(history)` of the result (so core → `core-v1+u1` → `core-v1+u2`); `history` = previous history + one `UpdateRecord(from_version=ont.version, to_version=<new>, proposer, rationale, update_digest, gate="passed")`; added nodes are not core (`is_core` False); a `set_edges` entry on an existing `(source, target)` replaces its weight, otherwise adds the edge. Core nodes remain byte-identical (`node(code)` equal before and after). Applying the same update to the same ontology twice yields equal results (same `digest()`).

## D3. LHM metric state (I6, I7, G3) — `daypart.memory.lhm`

```python
PRIMARY_NODE = "dp:adherence.protocol"
SERIES_NODES: dict[str, str] = {                # SeriesRecord.metric_display -> node code
    "Protocol adherence (logged)": "dp:adherence.protocol",
    "Check-in sentiment": "dp:affect.checkin_sentiment",
}
GUARD_VERSIONS: dict[str, str] = {"claims": CLAIMS_GUARD_VERSION, "valence": VALENCE_GUARD_VERSION}

@dataclass(frozen=True)
class NodeState:
    code: str
    lhm_domain: str | None
    score: DomainScore | None    # None iff lhm_domain is None
    ready: bool
    value: float | None          # == score.score iff ready, else None (I6: no number unless ready)

@dataclass(frozen=True)
class MemberMetricState:
    member_id: str
    as_of: datetime
    ontology_version: str
    nodes: Mapping[str, NodeState]            # one entry per ontology metric node, keyed by code
    signals: tuple[CandidateSignal, ...]
    alignment: Mapping[str, float]            # action event_id -> alignment score, actions with occurred_at <= as_of
    guard_versions: Mapping[str, str]         # == GUARD_VERSIONS
    def ready_codes(self) -> tuple[str, ...]  # sorted codes with ready=True
    def surfaced(self) -> dict[str, float | None]   # code -> value (None when not ready)

def alignment_score(event: ActionEvent, ontology: Ontology,
                    goal_nodes: Sequence[str] = (PRIMARY_NODE,)) -> float

class MemoryLhm:
    def __init__(self, adapter: LhmAdapter, *, ontology: Ontology | None = None,
                 events: EventLog | None = None,
                 goal_nodes: Sequence[str] = (PRIMARY_NODE,)) -> None
    async def metric_state(self, member_id: str, as_of: datetime) -> MemberMetricState
```
`adapter` is any `LhmAdapter` that also has `sentiment_fn(note: str) -> float` (e.g. `daypart.simmember.lhm_records.SimMemberLhmAdapter`). `ontology=None` → `core()`. `events=None` → an empty `EventLog` over the same ontology.

`alignment_score`: mean over `goal_nodes` of `ontology.weight(event.code, g)` (a missing edge counts 0.0); empty `goal_nodes` → 0.0. E.g. `act:reminder_sent` against the default goal → 0.3.

`metric_state(member_id, as_of)`:
- `member_id` not matching `MEMBER_ID_RE` → `ValueError` (I7: only opaque ids; e.g. an email never reaches the adapter). Naive `as_of` → `ValueError`.
- Samples per node, all restricted to timestamps `<= as_of`:
  - adapter series whose `metric_display` is in `SERIES_NODES` → that node; a point with `value` not None contributes `(observed_at, value)`; a point with `value is None` and a `note` contributes `(observed_at, adapter.sentiment_fn(note))`;
  - plus every `SignalEvent` of the member in `events` whose `code` is that node → `(occurred_at, value)`;
  - **`ActionEvent`s contribute no samples to any node (D14).**
- Per metric node:
  - `lhm_domain is None` → `NodeState(code, None, score=None, ready=False, value=None)` (G3: an unmapped node is never ready, even with abundant samples).
  - otherwise `score = score_domain(lhm_domain, samples, as_of=as_of)` (LHM default `WINDOW_DAYS`); claim gate = `assert_claim_safe(node.label or node.display, node.caveats, "low", False)` did not raise; valence gate = `assert_valence_safe(ScoreRendering(domain=lhm_domain, score=score.score))` did not raise; `ready = is_ready(score, lhm_domain, claim_gate_passed=<claim gate>, valence_gate_passed=<valence gate>)`; `value = score.score if ready else None`.
- `signals = analyze(fetch(<adapter view restricted to records with timestamps <= as_of>, member_id))` — order dates, regimen-start dates and series points after `as_of` are dropped before `fetch`. Unknown member or `as_of` before any record → `()`.
- `alignment` = `{e.event_id: alignment_score(e, ontology, goal_nodes)}` for the member's `ActionEvent`s with `occurred_at <= as_of`.
- `ontology_version == ontology.version`; `guard_versions == GUARD_VERSIONS`.
- Unknown member (valid id, no records) → every node not ready, every `value` None, `signals == ()`, no exception.
- Deterministic: two calls with equal inputs return equal states.

### D3.1 Behavioural guarantees (sealed)
- **D14 sealed property:** for any member and `as_of`, adding any stream of `ActionEvent`s (any codes, any times) to `events` leaves `nodes` and `signals` exactly equal to the state without them; only `alignment` changes.
- A `SignalEvent` on a mapped node changes that node's `score.n_effective` by the number of in-window signals.
- **I6 / fail-closed:** a SimMember cohort member (`generate(0)`) at `as_of = START_DATE + 4 days` (a 5-day member) has `PRIMARY_NODE` not ready and `value is None`; at `as_of = START_DATE + 41 days` (end of a 42-day trajectory) `PRIMARY_NODE` is ready and `value == score.score` in [0, 100], for at least 95% of the 200 members. Every node with `ready=False` has `value None`; `surfaced()` never carries a number for a not-ready node.
- **G3 via extension:** after `apply_update` adds a metric node mapped to `consistency` and a sibling metric node with `lhm_domain=None` (both value_range (0, 1)), feeding both 28 daily in-window `SignalEvent`s of 1.0/0.0 makes the mapped node ready and leaves the unmapped node not ready with `score None`. `MemberMetricState.ontology_version` reports the extended version.
- No identity or free text reaches the adapter: the only `user_scope` the adapter ever receives is the `member_id` argument.
