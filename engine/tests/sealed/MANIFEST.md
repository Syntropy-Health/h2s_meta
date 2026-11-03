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
