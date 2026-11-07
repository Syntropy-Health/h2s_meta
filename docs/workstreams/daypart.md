# daypart — dev doc

## FHDIC P1 · Data core for the Clinical Human Clone (HAC-38)

**Overview.** P1 builds the backend for the check-in → review loop of A&D HAC-33
(`workstreams/daypart/AND.md`). Nothing the person enters reaches memory until they confirm it.
Real-input mode exists only as a gate that always refuses. All inputs are synthetic.

```mermaid
flowchart LR
  T[text ≤500] --> CR{crisis pre-check} -->|crisis| H[200 handoff, nothing stored]
  CR --> PII{PII tripwire} -->|hit| R422[422 looks_like_personal_data]
  PII --> X[rules-v1 extractor]
  F[demo fixture / upload ≤1 MiB] --> S{magic bytes · sha256 allow-list} --> SC[sidecar extraction]
  X --> P[(pending CheckIn<br/>proposals + blocking flags)]
  SC --> P
  P -->|review: all-or-nothing| V{corrections re-validated<br/>flags re-run}
  V -->|confirm/correct| E[(EventLog core-v1+u1)]
  V -->|save_unmapped| U[(provenance only)]
  V -->|discard| D[gone]
  G{{real-input gate<br/>REAL_INPUT_BUILD_ALLOWED=False}} -.-> T
```

**What changed.**
- `memory/registry.py` + `registry_a4.json` + `loinc_verify_v1.csv`: 18 LOINC 2.83 codes and 2 local
  ones in 3 verticals. `slice_ontology()` = `apply_update(core(), …)`, version `core-v1+u1`, adding
  11 nodes; `core()` is unchanged. The LOINC notice is in ATTRIBUTION.
- `governance/real_input.py` + YAML (12 gates, all open and unassigned), `/api/mode` (always 403),
  and `docs/daypart/real-input-pathway.md`.
- `checkin/`: proposal models, the rules-v1 text extractor with 7 blocking flags, the PII tripwire,
  the crisis pre-check, fixture intake (4 synthetic fixtures + sidecars), per-visitor journals
  (LRU 256, 60 min idle), and the review service (atomic, per-journal lock).
- Routes: `POST /api/journals`, `GET /api/fixtures[/{id}]`, `POST /api/checkins` (JSON or
  multipart, with a stream cap), `POST /api/checkins/{id}/review`,
  `GET /api/journals/{j}/observations/{r}`. Modal `playground` runs with `max_containers=1`.

**Key decisions** (plan G1–G12 and addendum G13).
- 69725-0 failed LOINC verification (a single GAD item, not the GAD-2 total) and was dropped (D-A5).
- The crisis pre-check owns its member-crisis lexicon, because boundary-v1 only carries reply handoff markers (G13).
- An event value is stored in the canonical unit. The observation record keeps the value as recorded (e.g. 160 [lb_av]).
- The review body may omit `journal_id`: check-ins map to their journal at the store level.

**Run it**, from `engine/`: `uv run daypart-demo` (the FastAPI app), then use the routes above.
Tests: `env -u MUSE_API_KEY uv run pytest -q`.

## Slice G · 2.4 boundary eval (HAC-26)

**Overview.** Plan task 2.4 compares the 4-bit Muse Glimmer 30B base, base+SFT and base+DPO on held-out prompts. It
checks two things: boundary compliance (`membrane.dsl.check`) and the `lapse-recovery-v1` rubric. The contract is
`docs/daypart/contracts/CONTRACT-slice-G-bench.md`.

```mermaid
flowchart LR
  S[eval.scenarios<br/>env_baseline arms] --> P[heldout_prompts<br/>I1: syn- only, unique user turns,<br/>no training user turn]
  P --> G[Modal boundary_generate<br/>greedy, seed 0, one H100]
  G --> R[score_response · aggregate · verdicts<br/>R3 margin +0.05 pre-registered]
  R --> B[(boundary_eval.json)]
  T[(training.json)] --> H[harness arms base/sft/dpo]
  B --> H --> M[(metrics.json)]
  B --> D[dashboard training panel + boundary table]
  T --> D
```

**What changed.**
- `daypart.eval.boundary` (pure): builds the held-out prompts (registered as `eval.boundary_prompts` in I1 `BUILDERS`),
  scores and aggregates responses, computes the R3/R4 verdicts, builds and loads the report, and estimates cost.
- `daypart.train.results` (pure): the training summary and the loss curve from `training.json`.
- Harness arm kind `boundary_eval`: arms `base`, `sft` and `dpo` read the committed results. A missing file gives
  `not_run` and names it. An invalid file gives `failed`. A failing R3/R4 is reported as `passed: false` and the status
  stays `ok`. The Fireworks reasons are gone from `runs.yaml`.
- Dashboard: a training panel (steps, time, peak GPU, loss chart) and a base/SFT/DPO boundary table with R3/R4.
- `modal_app.py::boundary_eval` is the Modal entrypoint. It is untested glue.

**Key decisions.**
- `R3_MARGIN = 0.05` was fixed before any eval output existed.
- Held-out user turns are deduplicated before truncation (QG F1). Greedy decoding would otherwise double-count
  repeated prompts. The default scenario pool yields 141 unique prompts.
- A unit test fails CI when the committed `metrics.json` drifts from the harness (QG F2). The entrypoint regenerates it.

**Run it (CTO, Modal tokens), from `engine/`.**

    uv run --with modal modal run daypart/train/modal_app.py::boundary_eval --out ../docs/daypart/results/boundary_eval.json

Add `--dry-run` to print the estimate only: 141 prompts × 3 models, about 9 min on one H100, about $0.59. Commit
`boundary_eval.json` and the regenerated `metrics.json` together.

**Known issues / residual risk.**
- The Test Author still has to update 4 sealed `test_eval.py` tests for the slice-G `runs.yaml`. 3 are
  `PENDING_SLICE_G`, and `TestDefaultConfig::test_arms_and_summary` is in the R2 list. They stay deselected in CI.
- The R3/R4 results are unknown until the CTO runs the command above.
- The held-out prompts cover only the `strained` and `lapsed` states. The prompt text is held out, but the members
  are not: they also appear in training.
- Rollback: revert the phase commits. The arms then fall back to `not_run`, and no data is lost.
