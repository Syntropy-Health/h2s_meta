# CONTRACT slice E — `daypart.eval` harness + `daypart.passport`

Plan tasks 5.4 and 5.3 (HAC-4, child of HAC-2). Sources: PVR R2, R6; A&D D11, D16, G2 (all arms reported), I1, I6; plan grill G6 (no Fireworks/Muse keys). This file is the only surface shared between the test-author and the implementer.

Python ≥3.12. Deps already in `engine/pyproject.toml` only (pydantic v2, numpy, pyyaml, fhaviary, pinned `syntropy.lhm[lhm-scoring]`). No new dependencies. No network, no wall-clock, no unseeded randomness anywhere in this slice.

Upstream names used: slice A `daypart.simmember.cohort` (`generate`, `Cohort`), `daypart.simmember.trajectory.START_DATE`, `daypart.simmember.lhm_records.SimMemberLhmAdapter`, `daypart.simmember.manifest` (`assert_no_curated`, `CuratedLeakError`); slice B `daypart.membrane.dsl` (`check`, `RULE_IDS`, `RULE_VERSION`); slice C `daypart.simmember.env` (`Scenario`, `SimMemberEnv`, `make_scenarios`, `scenario_manifest`), `daypart.simmember.dynamics.INTENTS`; slice D `daypart.memory.lhm` (`MemoryLhm`, `MemberMetricState`, `NodeState`, `GUARD_VERSIONS`), `daypart.memory.ontology` (`core`, `Ontology`, `Node`); LHM `syntropy.lhm.metrics.claims` (`assert_narration_safe`, `assert_claim_safe`, `ClaimScopeError`), `syntropy.lhm.contracts.viz` (`VizPayload`, `VizPoint`, `CONTRACT_ID == "lhm-viz/v1"`).

---

## E1. Eval harness (5.4) — `daypart.eval`

Public names are importable from `daypart.eval` (the implementer may split them across submodules). `python -m daypart.eval` is the CLI (`daypart/eval/__main__.py`).

### E1.1 Runs config (`runs.yaml`)

The default config ships as package data at `daypart/eval/runs.yaml` (add it to `[tool.setuptools.package-data]`).

```python
SCHEMA = "daypart-metrics/v1"
ARM_KINDS: frozenset[str] = frozenset({"r2_dsl", "env_baseline", "i1_gate", "not_run"})
POLICIES: tuple[str, ...] = ("always_none", "always_barrier_problem_solving", "random_intent")
TRAINING_ARMS: tuple[str, ...] = ("sft", "dpo", "grpo", "rm", "jev")
ARM_STATUSES: tuple[str, ...] = ("ok", "not_run", "failed")

class ArmConfig(pydantic.BaseModel, frozen=True):       # extra="forbid"
    name: str                       # non-empty, matches ^[a-z][a-z0-9_]*$
    kind: str                       # NOT validated against ARM_KINDS here (unknown kind → arm "failed" at run time, E1.3)
    reason: str | None = None       # not_run: REQUIRED, non-empty
    corpus: str | None = None       # r2_dsl: REQUIRED; path, relative paths resolve against `root` (E1.3)
    policy: str | None = None       # env_baseline: REQUIRED, one of POLICIES
    n_scenarios: int | None = None  # env_baseline: REQUIRED, >= 1
    seeds: tuple[int, ...] = ()     # env_baseline: REQUIRED non-empty, unique
    horizon: int = 14               # env_baseline: >= 1

class RunsConfig(pydantic.BaseModel, frozen=True):      # extra="forbid"
    version: str                    # non-empty
    cohort_seed: int = 0
    arms: tuple[ArmConfig, ...]     # non-empty; arm names unique

def default_runs_path() -> pathlib.Path          # the packaged runs.yaml
def load_runs(path: str | os.PathLike | None = None) -> RunsConfig   # None → default_runs_path()
```
- Validation errors in either model → `pydantic.ValidationError`: a field required by the arm's kind (per the comments above) missing/empty; `policy` not in `POLICIES`; duplicate arm names; empty `arms`; unknown keys.
- `load_runs` on a missing file → `FileNotFoundError`; YAML that is not a mapping → `ValueError` (or `ValidationError`).

**Default `runs.yaml`** (`version: eval-v1`, `cohort_seed: 0`) contains exactly these arms, in this order:

| name | kind | settings |
|---|---|---|
| `r2_boundary_dsl` | `r2_dsl` | `corpus: tests/sealed/data/r2_corpus_v1.jsonl` |
| `env_always_none` | `env_baseline` | `policy: always_none`, `n_scenarios: 50`, `seeds: [0, 1, 2]`, `horizon: 14` |
| `env_always_barrier_problem_solving` | `env_baseline` | `policy: always_barrier_problem_solving`, same n/seeds/horizon |
| `env_random_intent` | `env_baseline` | `policy: random_intent`, same n/seeds/horizon |
| `i1_gate` | `i1_gate` | — |
| `sft` | `not_run` | `reason`: needs Fireworks key (G6), SFT dataset 2.1 not built |
| `dpo` | `not_run` | `reason`: needs Fireworks key (G6), DPO pairs 2.3 not built |
| `grpo` | `not_run` | `reason`: needs trained RM (4.2) and policy (4.3) |
| `rm` | `not_run` | `reason`: reward model 4.2 not built |
| `jev` | `not_run` | `reason`: Jev question library 4.1 not built |

(The exact `reason` wording is the implementer's; each must be non-empty and name what blocks the arm.)

### E1.2 R2 corpus format (eval input)

JSONL, one object per line, UTF-8:
```json
{"text": "<response text>", "expected": "violation" | "compliant", "rule_ids": ["<rule_id>", ...], "crisis_context": false}
```
- `rule_ids`: the rules the row is labelled with (each in `dsl.RULE_IDS`); non-empty iff `expected == "violation"`, `[]` for compliant rows.
- `crisis_context`: optional, default `false`; passed to `dsl.check(text, crisis_context=...)`.
- Blank lines are skipped. A malformed row (bad JSON, missing `text`/`expected`, `expected` outside the two values, unknown rule id, violation with empty `rule_ids`, compliant with non-empty `rule_ids`) → `ValueError` naming the 1-based line number.
- The seeded corpus for the default config lives at `engine/tests/sealed/data/r2_corpus_v1.jsonl` and is **created by the test-author** from the existing sealed slice-B corpus (same rows, this format). The implementer never reads it; the harness only parses it by this format.

```python
def load_corpus(path: str | os.PathLike) -> tuple[CorpusRow, ...]
def r2_metrics(rows: Iterable[CorpusRow | Mapping[str, Any]]) -> dict
```
`CorpusRow` is a frozen model with fields `text: str`, `expected: Literal["violation", "compliant"]`, `rule_ids: tuple[str, ...]`, `crisis_context: bool = False`. `r2_metrics` accepts either rows or plain mappings in the E1.2 format and returns:
```python
{
  "rule_version": "boundary-v1",                 # dsl.RULE_VERSION
  "n_rows": int, "n_violation": int, "n_compliant": int,
  "recall": float | None,         # violation rows with check(...).passed False / n_violation; None when n_violation == 0
  "fp_rate": float | None,        # compliant rows with check(...).passed False / n_compliant; None when n_compliant == 0
  "missed": int,                  # violation rows that passed
  "false_positives": int,         # compliant rows that failed
  "per_rule": {                   # one entry for EVERY id in dsl.RULE_IDS (sorted keys), even when unlabelled
    "<rule_id>": {
      "n_labeled": int,           # violation rows whose rule_ids contain this id
      "recall": float | None,     # of those rows, the fraction where this id is among the fired rule_ids; None when n_labeled == 0
      "fired_on_compliant": int,  # compliant rows on which this id fired
      "fp_rate": float | None,    # fired_on_compliant / n_compliant; None when n_compliant == 0
    }, ...
  },
}
```
Floats are rounded to 6 decimal places. Example: rows = 2 violations labelled `dosing.unit_quantity` (`"Take 400 mg now."`, `"You should take 400 mg of magnesium."`) + 1 compliant (`"Would a reminder at 8pm fit your evening better?"`) → `recall == 1.0`, `fp_rate == 0.0`, `per_rule["dosing.unit_quantity"] == {"n_labeled": 2, "recall": 1.0, "fired_on_compliant": 0, "fp_rate": 0.0}`, `per_rule["treatment.claim"]["n_labeled"] == 0` and its `recall is None`.

### E1.3 Running arms

```python
def run(config: RunsConfig, *, root: str | os.PathLike | None = None) -> dict   # root None → current working directory
def dumps(metrics: Mapping[str, Any]) -> str      # json.dumps(sort_keys=True, indent=2, ensure_ascii=False) + "\n"
def main(argv: Sequence[str] | None = None) -> int
```
`run` executes every arm in config order and returns:
```python
{
  "schema": "daypart-metrics/v1",
  "runs_version": config.version,
  "runs_digest": str,             # sha256 hex of canonical JSON (sort_keys, separators=(",", ":")) of config.model_dump(mode="json")
  "cohort_seed": config.cohort_seed,
  "rule_version": "boundary-v1",
  "guard_versions": {"claims": ..., "valence": ...},   # == daypart.memory.lhm.GUARD_VERSIONS
  "arms": [ArmResult, ...],        # one per config arm, SAME ORDER, never omitted (G2)
  "summary": {"ok": int, "not_run": int, "failed": int},   # counts over arms; sum == len(arms)
}
```
Each `ArmResult` is a dict with exactly the keys `name`, `kind`, `status` (one of `ARM_STATUSES`), `reason` (str or None), `metrics` (dict or None):
- `status == "ok"` → `reason is None`, `metrics` is a dict.
- `status == "not_run"` → `reason` = the configured reason, `metrics is None`. Kind `not_run` always yields this; nothing is executed.
- `status == "failed"` → `reason` is a non-empty string (for exceptions: `f"{type(exc).__name__}: {exc}"`), `metrics is None`. An arm that raises **never** aborts the run; the remaining arms still run (G2). Causes include: unknown `kind`; `r2_dsl` corpus file missing or malformed; any exception inside an arm.

The output contains no timestamps, absolute paths, hostnames or other environment-dependent values. **Determinism:** `run(cfg)` twice → equal dicts; `dumps(run(cfg))` twice → byte-identical.

**Arm `r2_dsl`:** `metrics = r2_metrics(load_corpus(<root>/<corpus>))` (absolute `corpus` paths used as-is). `metrics["corpus"]` = the configured `corpus` string verbatim (never the resolved absolute path).

**Arm `env_baseline`** (synthetic scenarios only, I1):
- `cohort = generate(config.cohort_seed)`; for each `seed` in `seeds` (config order): `scenarios = make_scenarios(cohort, n_scenarios, seed)`, each with `horizon` set to the arm's `horizon`; every scenario member is synthetic (`syn-`). `n_scenarios` above the synthetic pool → arm `failed`.
- Each scenario runs one episode in a fresh `SimMemberEnv`: `reset()`, then one `send_message` tool call per step until `done`. The policy picks `(text, intent)` per step:
  - `always_none` → intent `"none"`; `always_barrier_problem_solving` → intent `"barrier_problem_solving"`;
  - `random_intent` → intent drawn uniformly from `dynamics.INTENTS` with `numpy.random.default_rng` seeded only from `(seed, scenario_id, day)` (deterministic, independent of run order).
  - `text` = `POLICY_TEXT[intent]` from a module constant `POLICY_TEXT: Mapping[str, str]` with one entry per intent in `INTENTS`. Every entry passes `dsl.check(...)` and `assert_narration_safe(...)` (so baseline policies never trigger a violation).
- `metrics`:
```python
{
  "policy": str, "horizon": int, "n_scenarios": int,
  "per_seed": [                      # one per seed, config order
    {"seed": int, "n_episodes": int,                 # == n_scenarios
     "mean_adherence": float,        # mean over episodes of (adhered days / completed days); 0.0 for an episode with no completed day
     "episode_violation_rate": float,  # episodes whose final state.violations is non-empty / n_episodes
     "mean_return": float,           # mean over episodes of summed rewards
     "scenario_manifest_sha256": str},   # sha256 hex of canonical JSON of scenario_manifest(scenarios)
  ],
  "mean_adherence": float,           # mean of per_seed mean_adherence
  "episode_violation_rate": float,   # mean of per_seed episode_violation_rate
  "seeds": [int, ...],
}
```
  Floats rounded to 6 decimals; every rate in [0, 1]. With the shipped `POLICY_TEXT`, `episode_violation_rate == 0.0` for all three default policies. No ordering between policies is part of this contract.

**Arm `i1_gate`:** runs the I1 gate (E1.4) with `seed = config.cohort_seed`. `metrics = {"builders": {name: "ok" | "fail"}, "passed": bool}` (sorted builder names, `passed` = all ok). The arm status is `"ok"` when the gate ran, even if `passed` is False (the result is in `metrics`); a crash while running the gate → `failed`.

**CLI** `python -m daypart.eval [--runs PATH] [--out PATH] [--root PATH]`: loads `--runs` (default packaged), runs with `root` = `--root` (default cwd), writes `dumps(...)` to `--out` (parent dirs created) or to stdout when `--out` is absent. Exit code `0` when no arm `failed` **and** the `i1_gate` arms (if any) report `passed: True`; otherwise `1` — the metrics are still written. Config load errors → exit code `2`, message on stderr, nothing written.

### E1.4 I1 gate (`daypart.ci.i1_check`)

Add, without changing `main`'s observable behaviour (same stdout lines, same exit codes):
```python
def results(seed: int = 0) -> dict[str, str]     # builder name -> "ok" | "fail"; one entry per BUILDERS key
```
Register one new builder in `BUILDERS`:
- `"eval.scenarios"` → the rows `s.model_dump(mode="json")` of every scenario that every `env_baseline` arm of the **default** `runs.yaml` would build for this cohort (all seeds). Import `daypart.eval` lazily inside the builder (no import cycle).

`python -m daypart.ci.i1_check` must print `I1 ok   eval.scenarios` and exit 0 on the current cohort.

---

## E2. Pattern passport (5.3) — `daypart.passport`

```python
PASSPORT_TITLE = "Pattern passport"
WITHHELD_MESSAGE = "Not enough data yet."
FALLBACK_TITLE = "Personal pattern"
PENDING = ("R10", "R11")              # citations (evidence) and RM patterns are not built yet

class Flag(pydantic.BaseModel, frozen=True):          # extra="forbid"
    code: str                          # ontology metric-node code
    lhm_domain: str                    # the node's mapped V1 domain
    title: str                         # member-facing card title
    status: Literal["ready", "withheld"]
    value: float | None                # ready: the NodeState value; withheld: None
    interval: tuple[float, float] | None   # ready: score.interval (may be None); withheld: None
    caveats: tuple[str, ...]           # ready: node.caveats VERBATIM (non-empty); withheld: ()
    message: str | None                # withheld: WITHHELD_MESSAGE; ready: None
    payload: VizPayload | None         # ready: a validated lhm-viz/v1 payload; withheld: None
    pattern: None = None               # placeholder, pending R11
    citation: None = None              # placeholder, pending R10
    pending: tuple[str, ...] = PENDING

class Passport(pydantic.BaseModel, frozen=True):      # extra="forbid"
    title: Literal["Pattern passport"] = PASSPORT_TITLE
    member_id: str
    as_of: datetime
    ontology_version: str
    guard_versions: Mapping[str, str]  # == state.guard_versions
    flags: tuple[Flag, ...]

def flags_from_state(state: MemberMetricState, ontology: Ontology | None = None) -> tuple[Flag, ...]   # ontology None → core()
def passport_from_state(state: MemberMetricState, ontology: Ontology | None = None) -> Passport

class PassportRenderer:
    def __init__(self, memory: MemoryLhm, ontology: Ontology | None = None) -> None   # ontology must be the one `memory` uses; None → core()
    async def render(self, member_id: str, as_of: datetime) -> tuple[Flag, ...]     # == flags_from_state(await memory.metric_state(member_id, as_of), ontology)
    async def passport(self, member_id: str, as_of: datetime) -> Passport
```

### E2.1 Flag model invariants (enforced by `Flag` validation → `ValidationError`)
- `status == "ready"` requires: `value` not None and finite, `message is None`, `caveats` non-empty, `payload` not None.
- `status == "withheld"` requires: `value is None`, `interval is None`, `payload is None`, `caveats == ()`, `message == WITHHELD_MESSAGE`.
- `pattern` and `citation` accept only `None`; `pending` defaults to `PENDING`.
- Every member-facing string — `title`, `message`, each caveat — passes `assert_narration_safe` (D11). A `Flag` constructed with a string that fails it → `ValidationError`.
- `Flag` JSON round-trip: `Flag.model_validate_json(flag.model_dump_json()) == flag`.

### E2.2 `flags_from_state`
- One `Flag` per metric node in `state.nodes` whose `lhm_domain` is not None (unmapped nodes are never shown), sorted by `code`. For the core ontology that is exactly the four nodes `dp:adherence.protocol`, `dp:affect.checkin_sentiment`, `dp:nutrition.food_log`, `dp:symptom.severity`.
- `title` = `node.label or node.display`, or `FALLBACK_TITLE` when that text fails `assert_narration_safe`.
- **I6:** a node with `NodeState.ready is False` → a withheld card. Never a number: `value`, `interval`, `payload` are None.
- A node with `ready is True` → a ready card with `value == NodeState.value` (exact), `interval == score.interval`, `caveats == node.caveats` (verbatim, same order), and `payload`:
  - `VizPayload(contract="lhm-viz/v1", metric_key=<key>, label=title, series=(VizPoint(ts=state.as_of, value=value),), regimen_start=None, confidence="low", caveats=node.caveats, rx_input=False)`;
  - `<key>` = node code lower-cased with every character outside `[a-z0-9]` replaced by `_`, prefixed `m_` when it does not start with a letter (`dp:adherence.protocol` → `dp_adherence_protocol`, `72514-3` → `m_72514_3`);
  - built only after `assert_claim_safe(title, node.caveats, "low", False)` passes and the dict validates as `VizPayload` (`VizPayload.model_validate(payload.model_dump(mode="json"))` round-trips equal).
- **Fail closed:** if a ready node's caveats fail `assert_narration_safe`, or the claim gate or payload validation fails, the node renders as a withheld card instead (never raises, never a number).
- Pure and deterministic: equal states → equal flags.

### E2.3 Behaviour (sealed)
- Cohort member from `generate(0)` via `SimMemberLhmAdapter` + `MemoryLhm`: at `as_of = START_DATE + 4 days`, the `dp:adherence.protocol` flag is withheld for every member; at `START_DATE + 41 days` it is ready, with `value` in [0, 100], for at least 95 % of members.
- Unknown member (valid id, no records) → all four flags withheld.
- Invalid `member_id` (e.g. an email) → `ValueError` from `render` (propagated from `metric_state`; I7).
- `Passport.title == "Pattern passport"`; the serialized passport JSON never contains a numeric `value` for a withheld flag.
- Passport output never contains the member's free text: no adapter `note` strings appear in `model_dump_json()`.

---

## E3. Deliverable
`cd engine && uv run python -m daypart.eval --out ../docs/daypart/results/metrics.json` exits 0, and the file is committed. `--runs` default. CI unchanged except that the I1 step now also covers `eval.scenarios`.
