# Sealed suite manifest — Daypart slices A + C

| Category | Spec section |
|---|---|
| functional/schema | Slice A, A1 Latents; A2 Profile |
| functional/curated | Slice A, A2 CURATED_PROFILES / G8 coverage |
| functional/cohort | Slice A, A2 Cohort / generate / determinism |
| functional/trajectory | Slice A, A3 DayRecord / Trajectory structure |
| functional/trajectory-dynamics | Slice A, A3 cohort-wide state properties |
| integration/manifest-write | Slice A, A4 write |
| functional/latent-hash | Slice A, A4 latent_hash |
| functional/leak-guard | Slice A, A4 assert_no_curated |
| integration/manifest-verify | Slice A, A4 verify_manifest |
| functional/rx-table | Slice A, A5 is_rx_for_source |
| functional/adapter-shape | Slice A, A5 SimMemberLhmAdapter interface |
| integration/lhm-records | Slice A, A5 record contents |
| functional/sentiment | Slice A, A5 sentiment_fn (D12) |
| integration/lhm-mapping | Slice A, A5 LHM map_* over cohort |
| integration/cli | Slice A, A6 CLI + determinism |
| functional/dynamics | Slice C, C1 transition |
| functional/scenarios | Slice C, C2 Scenario / make_scenarios / scenario_manifest / I1 |
| integration/env-reset | Slice C, C3 reset |
| integration/env-step | Slice C, C3 step semantics |
| integration/env-determinism | Slice C, C3 determinism / export_frame |


---

# Sealed suite manifest — Daypart slice B (membrane)

Spec: `docs/daypart/contracts/CONTRACT-slice-B-membrane.md`

| Category | File | Spec section |
|---|---|---|
| functional/membrane-dsl | test_membrane_dsl.py | B1 (rule pack, module surface), B2 (rules, normalization, disclaimer excision, D10 pair, compliant examples) |
| functional/membrane-r2 | test_membrane_r2.py + data/boundary_corpus_v1.jsonl | B4 (recall, false-positive rate) |
| integration/membrane-post | test_membrane_post.py | B3 (layer 0, post algorithm, trace fragment) |

## Slice D — `daypart.memory` (HAC-3, plan tasks 3.1–3.3)

Contract: `docs/daypart/contracts/CONTRACT-slice-D-memory.md`. Shared fixtures: `conftest_memory.py` (imported explicitly; `generate(0)` built once).

| file | contract section | invariants | categories |
|---|---|---|---|
| `test_memory_events.py` | D1 (+ D2.1 value ranges) | D14, I7 | `functional/events-models`, `functional/events-parse`, `functional/eventlog-append`, `functional/eventlog-query`, `integration/events-imports` |
| `test_ontology.py` | D2, D2.1, D2.2 | D13, I9, G3 | `functional/ontology-models`, `functional/ontology-core`, `functional/ontology-readonly`, `functional/ontology-gate`, `functional/ontology-update`, `functional/ontology-determinism` |
| `test_memory_lhm.py` | D3, D3.1 | I6, I7, G3, D14 | `functional/lhm-alignment`, `functional/lhm-state-shape`, `functional/lhm-scoring`, `functional/lhm-readiness`, `functional/lhm-d14`, `functional/lhm-g3`, `functional/lhm-identity`, `integration/lhm-signals`, `functional/lhm-determinism` |

## Slice E1 — `daypart.eval` R2 corpus (HAC-4, HAC-23)

Contract: `docs/daypart/contracts/CONTRACT-slice-E-eval-passport.md` section E1.2.

| file | source | rows | check |
|---|---|---|---|
| `data/r2_corpus_v1.jsonl` | `data/boundary_corpus_v1.jsonl` (slice B), same rows and order, re-encoded as `text`, `expected`, `rule_ids`, `crisis_context` | 202 (74 violation, 128 compliant; 8 rules × ≥8 rows; 14 `crisis_context`) | every label confirmed against `dsl.check`: the labelled rule fires on each violation row, and nothing fires on a compliant row |
| `data/convert_r2_corpus.py` | converter; `uv run python tests/sealed/data/convert_r2_corpus.py --check` from `engine/` confirms the committed corpus is current | | aborts on any label mismatch and never relabels |

Read by `test_eval.py` (`TestSeededCorpus`, `TestR2Metrics::test_seeded_corpus_meets_r2`, `TestDefaultConfig`) and by the default `runs.yaml` `r2_boundary_dsl` arm.
