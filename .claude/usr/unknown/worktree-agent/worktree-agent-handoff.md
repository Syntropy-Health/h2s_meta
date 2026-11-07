---
type: session
trigger: phase-complete
date: 2026-11-07
issue: HAC-38
branch: daypart/hac-fhdic-build
---

# Handoff — HAC-38 FHDIC build, after P1

## State
- Branch `daypart/hac-fhdic-build` (repo Syntropy-Health/h2s_meta), pushed. P1 phase commit 896eff7 (QGR
  `…phase-complete-20261107-0151-23af1bb.md`); CI green. b3024a3 adds phase-state + brief.
- Plan: `workstreams/daypart/plan/BUILD-PLAN-fhdic.md` (signed; do NOT edit — plan-gate hash) +
  `ADDENDUM-fhdic.md` (G13–G16; add new agent decisions here).
- Issues: Reviewer pass HAC-40 (open). Records HAC-41 / HAC-42 (scribe-writer). CTO asked twice to
  commission the Test Author for the sealed P1 suite.

## Next
P2 on the same branch. Begin with `/iteration-start` (merge origin/daypart/build if the CTO has landed P1):
- 6 evidence registry (`evidence/registry.py` + `sources_v1.json`, T6.3 data side).
- 7 passport view (`passport/view.py`, `/api/journals/{id}/passport`, `/api/inventory`; T6.1–T6.6, T5.4).
- 8 next step (`checkin/next_step.py`, GET/POST next-step; T7.1–T7.4).
- 9 lab rows (`demo/lab.py`, `/api/lab`; T8.1–T8.3).
Items carried from the P1.5 gate into P2: a saved order / `last_saved` checkin on the journal (V12), one Journal
method per J1 writer so `act:goal_set` does not rebuild event ids by hand (V13), and an event_id→record_id index
(V15). Observation cards read the as-recorded value and unit from provenance (G15).

## Working notes
- Tests: `cd engine && env -u MUSE_API_KEY -u META_MODEL_API_KEY PATH=$HOME/.local/bin:$PATH uv run pytest -q` (~22 s).
- Lint from `engine/`: `uvx ruff check <files>` (subagents have missed I001; re-check before signing).
- QG pattern per iteration: 4 reviewers + scorer → red test + per-finding commit → receipt-sign → `--boundary iteration`.
- No squash at phase-complete (pushed history; CTO merges --no-ff). Phase approval is delegated to the CTO (G11).
- Briefs: write `hacp-brief-<slug>` on HAC-38 FIRST, then create the `publish:` issue.
