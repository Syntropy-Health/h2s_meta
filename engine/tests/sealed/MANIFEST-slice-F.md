# Sealed suite — slice F (narrative cache, SFT/DPO datasets, Modal job specs)

Contract: `docs/daypart/contracts/CONTRACT-slice-F-train.md`. Written double-blind
(no implementation read). Helper module: `conftest_train.py` (fakes only; imported
explicitly, not a pytest conftest).

| File | Category | Contract section |
|---|---|---|
| `test_train_narrative.py` | functional/narrative-client | F1 — MuseClient credentials + request, TemplateClient |
| | functional/narrative-prompt | F2 — build_prompt, prompt_hash, kinds |
| | functional/narrative-cache | F3 — NarrativeRow, NarrativeCache |
| | functional/narrative-realize | F3 — realize |
| | integration/narrative-i1 | F2/F3 — I1 refusals, member mismatch |
| `test_train_sft_dataset.py` | functional/sft-rows | F4 — row shape, cache selection |
| | functional/sft-filter | F4 — DSL filter |
| | functional/sft-refusal | F4 — constants, refusal mix |
| | functional/sft-jsonl | F4 — write_jsonl |
| | integration/sft-i1 | F4/F5 — I1 guard, BUILDERS, `i1_check.main` |
| `test_train_dpo_pairs.py` | functional/dpo-rubric | F5 — rubric_check, constants |
| | functional/dpo-contexts | F5 — DpoContext, make_contexts |
| | functional/dpo-pairs | F5 — build_pairs |
| | integration/dpo-i1 | F5 — curated exclusion end to end |
| `test_train_modal_jobs.py` | functional/jobspec | F6 — LoraConfig, JobSpec |
| | functional/jobspec-builders | F6 — sft_spec, dpo_spec, plan_pipeline |
| | functional/job-runner | F6 — PollResult, JobResult, JobRunner |
| | integration/no-modal-import | header + F6 — no `modal` import, no new dependency |
