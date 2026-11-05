# CONTRACT slice A — `daypart.simmember` cohort, trajectories, LHM records, I1 manifest

Plan task 1.1 (HAC-2). Sources: PVR R1; A&D R1, D12, I1, I7. This file is the only surface shared between the test-author and the implementer.

Package root: `engine/daypart/simmember/`. Python ≥3.12. Deps available: numpy, pydantic v2, pinned `syntropy.lhm`.
Determinism: all randomness comes from `numpy.random.default_rng`, seeded only from the `seed` argument (and per-member derivations of it). No wall-clock, no `random`, no hash-seed dependence.

## A1. Latents — `daypart.simmember.latents`

```python
AXES: dict[str, tuple[str, ...]] = {
    "motivation":         ("low", "moderate", "high"),
    "barrier_type":       ("time", "cost", "forgetfulness", "side_effect_concern", "social", "low_energy"),
    "escalation_trigger": ("none", "symptom_flare", "mood_drop", "missed_streak", "conflicting_info"),
    "age_band":           ("18-29", "30-44", "45-59", "60-74", "75+"),
    "literacy":           ("low", "medium", "high"),
    "engagement_pattern": ("steady", "weekend_dip", "burst", "fading", "erratic"),
}
class Latents(pydantic.BaseModel, frozen=True):   # extra="forbid"
    motivation: str; barrier_type: str; escalation_trigger: str
    age_band: str; literacy: str; engagement_pattern: str
    affect_baseline: float      # in [-1.0, 1.0], rounded to 3 decimals
```
Validation: each categorical field must be a level of its axis (else `pydantic.ValidationError`); `affect_baseline` outside [-1, 1] → `ValidationError`.

`MIN_PER_LEVEL = 2` — in a generated 200-member cohort every level of every axis occurs ≥ 2 times.

## A2. Profiles & cohort — `daypart.simmember.cohort`

```python
class Profile(pydantic.BaseModel, frozen=True):   # extra="forbid"
    member_id: str        # "cur-001".."cur-025" or "syn-001".."syn-175" (3-digit, zero-padded)
    cohort: Literal["curated", "synthetic"]   # must agree with the id prefix
    latents: Latents

CURATED_PROFILES: tuple[Profile, ...]   # fixed table, exactly 25, ids cur-001..cur-025, independent of any seed
```
- Profiles carry no names, emails, free-text bios, or any identity field (I7).
- `cohort` disagreeing with the prefix, or an id not matching `^(cur|syn)-\d{3}$` → `ValidationError`.
- **Curated coverage (G8):** across the 25 curated profiles every level of every categorical axis appears ≥ 1 time.
- **Disjointness:** no synthetic profile's `latents` equals any curated profile's `latents` (generator resamples on collision).

```python
@dataclass(frozen=True)
class Cohort:
    seed: int
    profiles: tuple[Profile, ...]           # 25 curated first (in id order), then synthetic in id order
    trajectories: dict[str, Trajectory]     # member_id -> Trajectory, one per profile
    def training_profiles(self) -> tuple[Profile, ...]   # synthetic only
    def heldout_profiles(self) -> tuple[Profile, ...]    # curated only

def generate(seed: int, *, n_synthetic: int = 175, days: int = 42) -> Cohort
```
`generate(s)` called twice returns equal cohorts (profiles and trajectories equal, and `write` output byte-identical). Different seeds → different synthetic latents; curated profiles identical across seeds.

## A3. Trajectories — `daypart.simmember.trajectory`

```python
STATES = ("engaged", "strained", "lapsed", "recovering")
START_DATE = datetime(2026, 9, 1, tzinfo=timezone.utc)

class DayRecord(pydantic.BaseModel, frozen=True):    # extra="forbid"
    day: int                    # 0..days-1, contiguous
    date: datetime              # tz-aware == START_DATE + day days
    state: Literal["engaged", "strained", "lapsed", "recovering"]
    adhered: bool               # protocol logged that day
    checkin: bool               # member checked in that day
    sentiment: float | None     # in [-1, 1] iff checkin, else None
    note_id: str | None         # iff checkin: f"note:{member_id}:d{day}", else None

class Trajectory(pydantic.BaseModel, frozen=True):
    member_id: str
    days: tuple[DayRecord, ...]
```
- Day 0 state is `"engaged"`. Transitions are a latent-driven Markov chain over `STATES`: lower motivation / `fading`/`erratic` patterns make `strained`/`lapsed` more likely; `lapsed` can only move to `lapsed` or `recovering`; `recovering` moves to `engaged`, `recovering` or `lapsed`. (Exact probabilities are implementation-defined, but:)
- Across the generated 200-member cohort (seed 0, 42 days) every state in `STATES` occurs at least once, and at least one member has a `lapsed` day followed later by a `recovering` day.
- Adherence rate over `engaged` days > adherence rate over `lapsed` days (cohort-wide, seed 0).
- `sentiment` is a deterministic function of the member's `affect_baseline`, the day's `state`, and the seeded rng (D12); always within [-1, 1].
- Trajectories carry no free text. `note_id` is an opaque token, never prose.

## A4. I1 manifest & leak guard — `daypart.simmember.manifest`

```python
class CuratedLeakError(Exception): ...

def write(cohort: Cohort, out_dir: Path) -> dict
```
Writes three files and returns the manifest dict (identical to `manifest.json` content):
- `profiles.jsonl` — one `Profile.model_dump(mode="json")` per line, cohort order.
- `trajectories.jsonl` — one `Trajectory.model_dump(mode="json")` per line, same order.
- `manifest.json` — `{"generator_version": str, "seed": int, "counts": {"curated": 25, "synthetic": int}, "curated_ids": [...25 sorted ids], "curated_latent_hashes": {member_id: sha256hex}, "files": {"profiles.jsonl": sha256hex, "trajectories.jsonl": sha256hex}}`.
JSON is written with `sort_keys=True, separators=(",", ":")`; lines end with `\n`. `latent_hash(latents) -> str` = sha256 hex of that canonical JSON of `latents.model_dump(mode="json")`, exported from this module.

```python
def assert_no_curated(records: Iterable[Mapping[str, Any]], manifest: Mapping[str, Any]) -> None
```
Raises `CuratedLeakError` if ANY record (a JSON-able mapping, arbitrarily nested — e.g. a chat-JSONL row) contains:
1. a string anywhere matching `\bcur-\d{3}\b`, or any id in `manifest["curated_ids"]`; or
2. a mapping anywhere under key `"latents"` whose `latent_hash` equals one of `manifest["curated_latent_hashes"]` values (catches a relabeled curated profile).
Otherwise returns `None`. Synthetic-only records pass. An empty iterable passes.

```python
def verify_manifest(out_dir: Path) -> dict
```
Re-hashes the files; raises `CuratedLeakError` if `profiles.jsonl` file hash mismatches the manifest, or `ValueError` on any other hash mismatch; returns the manifest.

## A5. LHM records (I7, D12) — `daypart.simmember.lhm_records`

```python
NON_RX_SOURCES = frozenset({"intake", "shopify", "stripe"})
def is_rx_for_source(source: str) -> bool   # "prescription" → True; NON_RX_SOURCES → False; anything else (unknown/empty) → True (fail-closed)

class SimMemberLhmAdapter:                  # satisfies syntropy.lhm.tools.pipeline.LhmAdapter
    def __init__(self, cohort: Cohort) -> None
    async def read_orders(self, user_scope: str) -> Sequence[OrderRecord]
    async def read_regimen_starts(self, user_scope: str) -> Sequence[RegimenStartRecord]
    async def read_outcome_series(self, user_scope: str) -> Sequence[SeriesRecord]
    def sentiment_fn(self, note: str) -> float
```
(`OrderRecord`, `RegimenStartRecord`, `SeriesRecord`, `SeriesPoint` from `syntropy.lhm.models.mapping`.)
- `user_scope` is the `member_id`; unknown scope → empty sequences. `subject_ref` on every record == the `member_id` (opaque; never an email, name, or other identity).
- Orders: at least one per member, `source="intake"`, `is_rx = is_rx_for_source(source)` — therefore `False` for the whole synthetic cohort (no Rx orders). `item_display` is a generic claim-safe label (e.g. "Daily protocol kit"), identical for all members.
- Regimen start: exactly one per member, `started_at == START_DATE`, `dose_value=None`, `dose_unit=""`.
- Series: one numeric `SeriesRecord` (`code="journal-metric"`, `metric_display="Protocol adherence (logged)"`, unit `""`) with one point per day, `value` 1.0 if adhered else 0.0; and one check-in `SeriesRecord` whose points are exactly the check-in days with `note=note_id`, `value=None`.
- Every datetime is tz-aware.
- **No free text reaches LHM:** every `SeriesPoint.note` is a `note_id` token matching `^note:(cur|syn)-\d{3}:d\d+$`.
- `sentiment_fn(note_id)` returns that day's `sentiment` float (deterministic; D12). Unknown token → `KeyError`.
- Running `syntropy.lhm.models.mapping.map_order / map_regimen_start / map_series(rec, sentiment_fn=adapter.sentiment_fn)` on every record of every member raises nothing; for each check-in series the number of sentiment Observations equals the number of check-in days.

## A6. CLI

`python -m daypart.simmember --seed 0 --out <dir>` → calls `write(generate(seed), out)`; exit 0.
