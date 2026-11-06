# Build plan (locked): Clinical Human Clone & Health Clone Passport (FHDIC)

Issue: HAC-38. Spec: `workstreams/daypart/AND.md` (A&D rev 1, HAC-33), board-approved on HAC-36
(2026-11-06) with D-A9, D-A2, D-A3 and D-A1 as written. Locked through `/grill-me` in autonomous
mode on 2026-11-06. The CTO is the grill partner, so every call below is the agent's, recorded with
its `By`. Nothing in this plan changes the A&D's architecture or scope.

Branch: `daypart/hac-fhdic-build` from `daypart/hac33-design` @ efb3f65, so the design docs land
with P1. All four phases use this one branch. Each `/phase-complete` hands it to the CTO, who lands
it on `daypart/build` and deploys staging. Each iteration opens with `/iteration-start`, which merges
`origin/daypart/build`.

Paths are relative to `engine/` unless they start with `docs/`, `workstreams/`, `.github/` or
`ATTRIBUTION.md`. Each task is one iteration: failing unit test → minimal implementation → targeted
tests → `/iteration-complete`.

## Grill record (decisions that make the plan executable)

| # | Question | Decision | By |
|---|---|---|---|
| G1 | The A&D says "branch per phase off `daypart/build`"; HAC-38 says one branch from efb3f65. | One branch, `daypart/hac-fhdic-build`, for P1–P4. The issue is the later instruction. | Agent · delegated (HAC-38 description) |
| G2 | Who writes the `[sealed]` tests? | The Test Author seals them per phase from the A&D. The CTO commissions this; a worktree agent may not spawn it. Every T-id the plan cites also gets **my own** unit test in `tests/unit/` as the failing-test target. The sealed suite then runs as the referee in the phase gate. | Agent · assumed |
| G3 | How is a LOINC code "verified at build" (D-A5) without a licensed download? | Through the public HL7 terminology server `$lookup` (tx.fhir.org, R4), which returns the version, display and inactive flag. Run on 2026-11-06: **LOINC 2.83**. 18 of 19 candidates verify. **69725-0 fails**: it is the single GAD item "Feeling nervous, anxious or on edge in last 2 weeks", not the GAD-2 total. Under D-A5 it is dropped, not replaced. The CSV stores code, display, version and the lookup date. | Agent · assumed (D-A5 rule applied) |
| G4 | The slice count changes after G3. | 18 LOINC + 2 local. 7 codes are already in `core()`, so `add_nodes` holds **11** codes, not 12. Symptoms and wellbeing keeps 4 LOINC + 1 local (≥4). The LOINC count of 18 is inside 15–25. T5.1 reads "exactly the codes not in `core()`". | Agent · assumed |
| G5 | Where do the new LOINC nodes sit, and with what `value_range`? | Parents are existing core categories: vitals and body under `dp:vital`, steps and exercise under `dp:activity`, PHQ-2 under `dp:affect`. `value_range` = the registry `plausible` bounds (the EventLog validation range, never rendered). `lhm_domain=None` and `CORE_CAVEATS`, so G3 holds. | Agent · assumed |
| G6 | Crisis vocabulary for `checkin/crisis.py` | Reuse `crisis_markers` from the loaded `boundary-v1` rule pack (`dsl`) and match with the DSL's normaliser. No new lexicon. | Agent · assumed |
| G7 | Multipart upload and the 413 "stream limit" | Add `python-multipart` to the `demo` group. The handler reads `request.stream()` up to 1 MiB plus a 16 KiB envelope and returns 413 beyond it. The file part itself is also capped at 1 MiB, so 1 MiB + 1 byte → 413. Magic bytes are checked before the allow-list. | Agent · assumed |
| G8 | Fixture format, with no image library in the deps | 4 synthetic fixtures under `daypart/demo/fixtures/`: `scale-photo.png` and `bp-cuff.jpg` (synthetic images, generated once by a committed script with no real people or data), `wearable-export.csv` and `symptom-diary.json`. Each has a `<name>.extract.json` sidecar. `fixtures/manifest.json` holds `fixture_id → {file, sha256, type, size}`. Allow-listing is by sha256. | Agent · assumed |
| G9 | Journal `today` and the time zone | `today = START_DATE + 28 days`. Event times are 12:00 UTC on the resolved date. The correction date window is [today−28, today]. | Agent · assumed (A&D §4.2) |
| G10 | Where the UI test toolchain lives | A new `ui` dependency group (playwright) that is not in `default-groups`, so the `test` job is unchanged. The `demo-ui` job runs `uv sync --group ui` + `playwright install --with-deps chromium`, and `deploy-demo` gains `needs: demo-ui`. The vendored `axe.min.js` lives in `tests/ui/vendor/`, not `static/`, so it is never served. | Agent · assumed (AD9) |
| G11 | Phase approval. `tools/autonomy gate phase` prints `block`. | At each `/phase-complete` the QGR, Reviewer pass and CI go to the CTO, who lands per HAC-38. The board already approved the build scope on HAC-36. The phase boundary is recorded through scribe-writer; it does not wait on a new board question. | Agent · delegated (HAC-38 step 3) |
| G12 | The UIUX visual spec | HAC-35 `visual-spec` is done. P3 task 11 builds to it. No new UIUX child unless the spec lacks a state the A&D requires; a gap becomes a child issue to UIUX. | Agent · delegated (HAC-38 holds) |

## P1: Data core (R5, R9, R2–R4 backend)

1. **Concept registry.** Files: `daypart/memory/registry.py`, `daypart/memory/registry_a4.json`,
   `daypart/memory/loinc_verify_v1.csv`, `ATTRIBUTION.md` (LOINC notice).
   Failing test: `tests/unit/test_registry_unit.py` (T5.1, T5.2, T5.6).
   Done when: `slice_ontology().version == "core-v1+u1"`, `core()` digest unchanged, 1 history
   record, `add_nodes` = the 11 non-core codes, every concept has the T5.2 fields, every LOINC row
   matches the CSV, and the full `uv run pytest -q` is green.
2. **Real-input gate.** Files: `daypart/governance/{__init__,real_input}.py`,
   `daypart/governance/real_input.yaml`, `docs/daypart/real-input-pathway.md`, `/api/mode` in
   `daypart/demo/app.py`, and `create_app` reading `DAYPART_REAL_INPUT`.
   Failing test: `tests/unit/test_real_input_unit.py` (T9.1–T9.3, T9.5).
   Done when: POST /api/mode → 403 with 12 open gates; with the env set, `requested=true`,
   `enabled=false`; with every item closed it is still refused (`build_disallowed`); the doc table
   equals the YAML.
3. **Text check-in core.** Files: `daypart/checkin/{__init__,models,tripwire,crisis,extract_text}.py`.
   Failing test: `tests/unit/test_checkin_extract_unit.py` (T2.2 extraction, T2.3, T2.4 detection,
   T4.1, and T4.2 for every flag kind reachable from text).
   Done when: "slept 6 hours and walked 8000 steps" → 93832-4 = 6 h and 55423-8 = 8000; each of the
   7 flags is produced by a named input and is blocking; the tripwire catches e-mail, phone,
   ID-like numbers, "my name is" and DOB.
4. **Fixture intake.** Files: `daypart/checkin/fixtures.py`, `daypart/demo/fixtures/*`,
   `tools/make_fixtures.py` (the one-off generator, committed for reproducibility).
   Failing test: `tests/unit/test_checkin_fixtures_unit.py` (T3.1, T3.2 at the intake level).
   Done when: every manifest fixture → proposals with `method="fixture-sidecar-v1"`; an unknown
   PNG → `not_a_demo_fixture`; a PDF → `unsupported_type`; >1 MiB → `too_large`.
5. **Journal, service and routes.** Files: `daypart/checkin/{journal,service}.py`, routes in
   `daypart/demo/app.py` (`/api/journals`, `/api/checkins`, `/review`, `/observations/{id}`),
   `pyproject.toml` + `uv.lock` (`python-multipart`), and `deploy/modal_demo.py`
   (`max_containers=1`).
   Failing test: `tests/unit/test_checkin_service_unit.py` (T2.2 end to end, T2.3, T2.4, T3.2 via
   HTTP, T4.2–T4.6, T5.3, T5.5 for P1 endpoints, T9.2 fixtures-only).
   Done when: nothing reaches `EventLog` before review; review is all-or-nothing; a concurrent
   double review gives one 200 and one 409; unmapped values are in provenance only; eviction at
   60 min or 256 journals; `plausible` bounds appear in no response.

Phase gate P1: `/phase-complete`, then hand off to the CTO with the QGR, sealed P1 suite, CI and
Reviewer.

## P2: Passport, next step and lab (R6–R8)

6. **Evidence registry.** Files: `daypart/evidence/registry.py`, `daypart/evidence/sources_v1.json`.
   Failing test: `tests/unit/test_evidence_unit.py` (T6.3 data side, the no-range-in-`supports` rule).
   Done when: every source has `vetting="agent draft, not clinically reviewed"`, a public https URL
   and no digit+unit in `supports`, and at least one concept has no source (it exercises suppression).
7. **Passport view.** Files: `daypart/passport/view.py`, routes `/api/journals/{id}/passport` and
   `/api/inventory`.
   Failing test: `tests/unit/test_passport_view_unit.py` (T6.1–T6.6, T5.4).
   Done when: every card has 5 thread steps with non-null links; withheld cards carry no number;
   observation value = the recorded value byte for byte; >14 days old → withheld; existing sealed
   passport tests stay green.
8. **Next step.** Files: `daypart/checkin/next_step.py`, routes GET/POST `/api/journals/{id}/next-step`.
   Failing test: `tests/unit/test_next_step_unit.py` (T7.1–T7.4).
   Done when: rules apply in the §4.7 order; a monkeypatched dosing text → `replaced=true`; "choose"
   writes 1 `act:goal_set`; 409 before any save.
9. **Lab rows.** Files: `daypart/demo/lab.py`, route `/api/lab`.
   Failing test: `tests/unit/test_lab_unit.py` (T8.1–T8.3).
   Done when: base/sft/dpo plus 3 baselines match the §4.8 formulas over `docs/daypart/results/`;
   grpo is "proposed, not built" with no numbers; a missing seed → null; an adapter mismatch →
   "provenance mismatch".

Phase gate P2: as P1.

## P3: UI and accessibility (R1, R10)

10. **UIUX spec intake.** Read HAC-35 `visual-spec`; list any A&D state it lacks. Done when the gap
    list is empty or a UIUX child issue is open for each gap.
11. **UI rebuild.** Files: `daypart/demo/static/index.html` (hash routes, banner, research view,
    real-input control, LOINC footer), and the FastAPI title/description in `app.py`.
    Failing test: `tests/unit/test_demo_ui_html_unit.py` (T1.1, T1.3, T1.5 and the static parts of
    T2.1, T3.3, T8.4, T9.4).
12. **UI CI.** Files: `tests/ui/test_journey_a11y.py`, `tests/ui/vendor/axe.min.js`, the `ui`
    group in `pyproject.toml`/`uv.lock`, and the `demo-ui` job in `.github/workflows/engine.yml`
    gating `deploy-demo`.
    Failing test: the UI suite itself (T1.2, T2.1, T3.3, T8.4, T9.4, T10.1–T10.4).
    Done when it is green locally under headless chromium and in CI.
13. **Dashboard rename.** File: `daypart/demo/dashboard.py`. Failing test:
    `tests/unit/test_dashboard_strings_unit.py` (T1.4).

Phase gate P3: as P1, plus the Reviewer's `review-ux` pass.

## P4: Submission pack (R11)

14. **Pack.** Files: `docs/daypart/submission/{README,architecture,evidence-ledger,outcomes,assumptions-limits,organiser-rules,real-input-pathway}.md`,
    `tools/render_outcomes.py`. Failing test: `tests/unit/test_submission_pack_unit.py`
    (T11.1, T11.2, and the T5.6 README notice). Done when all 7 files exist, every ledger row
    resolves or is tagged `ASSUMPTION`, and `outcomes.md` equals the regenerated table.

Phase gate P4: `/phase-complete`, then `/plan-complete`, and hand off to the CTO for the L2 delivery
review against SC1–SC9.

## Holds (all phases)

Synthetic data only. No keys and no internal hostnames: the repo is public, and the one demo URL
already in `engine.yml` is the public Modal URL. The real-input gate stays hard off
(`REAL_INPUT_BUILD_ALLOWED = False`). Never read `tests/sealed` content beyond failure output.

## Human input

- 2026-11-06 (board, HAC-36): blueprint, D-A9, D-A2, D-A3 and D-A1 approved as written.
- 2026-11-06 (CTO, HAC-38 description): lock autonomously, record each decision, do not ask the
  board; branch from efb3f65; hand each phase to the CTO and never land it yourself.
- G1–G12 above: Agent decisions (assumed or delegated, as marked). No human answer was sought.
