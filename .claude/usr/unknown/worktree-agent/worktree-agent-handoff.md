---
type: session
trigger: iteration-complete
date: 2026-11-07
issue: HAC-38
branch: daypart/hac-fhdic-build
---

# Handoff — HAC-38 FHDIC build, after P2.1 (evidence registry)

## State
- P2.1 (plan task 6) boundary 24831e4, QGR `…iteration-complete-20261107-0254-8430ac7.md`, pushed.
  `evidence/registry.py`: `Source` (+ `stated_ranges`), `StatedRange` (statement = the page's own sentence; every
  bound appears in it, low before high), `EvidenceRegistry` (slotted; `sources()`, `by_id`, `for_concept(code)`
  accepts concept_id or code, raises on a code without a concept registry), `check_reference_ranges`,
  `load_evidence()` (cached; refuses unsourced or dropped ranges). URL: public https name only (no IP, no
  credentials, no port, no private suffix, no nip.io). Copy rule (no clinical/expert review, no "vetted") on
  title, publisher, supports, population, statement.
- `sources_v1.json`: MedlinePlus vital signs (pulse 60–100, BP 90/60–120/80 split by position, breathing 12–18,
  temperature 36.5–37.3 Cel), NHLBI BMI (18.5–24.9), NHLBI sleep (7–9 h), NHLBI high BP and WHO activity (no
  ranges). 7 concepts carry `reference_range`; the golden set is pinned in `test_evidence_unit.py`. AHA/CDC 403
  automated fetches, so they are not cited. Addendum G22. A&D §4.6 + Components table updated.
- Tests: unit 2425 passed; sealed P1 (HAC-39) 1500 passed, 3 skipped; pyright 0.
- Earlier: P1.7 boundary a46f759, P1.6 6d3413a, P1 phase commit 896eff7. The CTO lands the branch head
  (no squash; merge --no-ff). Reviewer HAC-40 still open. Records HAC-41/42.

## Next
P2 task 7 passport view (`passport/view.py`, `/api/journals/{id}/passport`, `/api/inventory`; T6.1–T6.6, T5.4).
- Read ranges ONLY through `load_evidence()` (A&D §4.6); render range beside the value with population + source
  link, or "no reference range"; never high/low. ESCAPE the url and population when rendering (QG P2.1 B8).
- Explanation suppression (§4.5): `for_concept(cid)` non-empty + template passes safe_text and dsl.check + ready.
Then 8 next step, 9 lab rows. Carried from P1.5: saved order / last_saved on the journal (V12), one Journal method
per J1 writer (V13), event_id→record_id index (V15). Observation cards read the as-recorded value/unit (G15).

## Working notes
- Clone: `gh repo clone Syntropy-Health/h2s_meta` into the run scratch dir (the workspace has no checkout).
- Tests: `cd engine && env -u MUSE_API_KEY -u META_MODEL_API_KEY PATH=$HOME/.local/bin:$PATH uv run pytest -q`.
- Mutation proofs: set PYTHONDONTWRITEBYTECODE=1 and clear __pycache__; a same-size edit inside one second
  reuses a stale .pyc and gives a false red/green.
- Lint my files: `$HOME/.local/bin/uvx ruff check <files>` (CI runs no ruff; repo-wide ruff has old findings).
- Plan file is signed (do not edit); decisions go in ADDENDUM-fhdic.md. No squash at phase-complete.
- Briefs: write `hacp-brief-<slug>` on HAC-38 FIRST, then create the `publish:` issue.
