---
type: plugin-feedback
target: airdlc plugin
plugin_version: 0.90.1
reporter: h2s_meta/unknown/worktree-agent
date: 2026-11-07
scope: plugin / operating-system behavior — NOT repo/app work
---

docs-ticket-align only greps context.json for "linear_issue|issue|ticket" keys. Under a Paperclip tracker, phase-state set-ws --paperclip-phase build=HAC-38 records the pointer under paperclip.phases, which the check never reads. So every Paperclip workstream gets "TICKET MISSING / ALIGNED 0" at each phase boundary. Seen on h2s_meta daypart P1 (HAC-38), 2026-11-07. Fix: also read paperclip.parent / paperclip.phases.*.

## 🟠 1. docs-ticket-align ignores Paperclip ticket pointers
**File:** <tool/hook/skill path, with line if known>
**Symptom:** <what the operator observed>
**Root cause:** <the underlying why — proven, not guessed>
**Fix:** <the smallest concrete change>
**Effect:** <what improves once applied>
**Status:** open (filed against airdlc 0.90.1)
