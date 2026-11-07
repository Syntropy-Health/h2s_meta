# Build plan addendum (agent decisions after lock)

Issue: HAC-38. Companion to `BUILD-PLAN-fhdic.md`, which is locked and hash-signed and is not
edited. Rows here continue its grill record; where a row supersedes an earlier decision it says so.

| # | Question | Decision | By | Risk |
|---|---|---|---|---|
| G13 | Crisis pre-check vocabulary | `boundary-v1` carries only reply hand-off markers (`crisis_markers`), not member crisis language, so `checkin/crisis.py` owns a member-crisis regex lexicon over `dsl.normalize()`; supersedes G6 and A&D Failure-handling row 1 wording "reusing the boundary-v1 crisis vocabulary". | Agent · assumed | Sealed T2.4 rows may phrase crisis differently; the lexicon is broad on purpose. |
