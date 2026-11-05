# Daypart Engine

**An adherence-intelligence engine for health routines. It decides *what kind* of message a person should get today. Every word it sends passes a deterministic safety boundary first. It is trained on Meta's Muse Glimmer 30B and measured in a reproducible synthetic world.**

- **Live demo:** [playground](https://supermomo668--daypart-demo-playground.us-east.modal.run) (inputs, outputs and parameters, step by step) · [results dashboard](https://supermomo668--daypart-demo-dashboard.us-east.modal.run) (benchmarks)
- **Status:** hackathon build (Meta, Nov 2026). Synthetic data only. Not a medical device and not medical advice.

---

## The problem

Most people who start a health routine (a supplement, a medication schedule, a sleep or exercise habit) drift off it within weeks. Apps answer with generic reminders, and they don't work: people lapse for different reasons. Some forget. Some run out of time. Some are tired. Some get no support from the people around them. The right message for someone who is *recovering* from a lapse is not the right message for someone on a streak.

Language models can write good supportive messages. In a health setting, though, a fluent model will also give dosing numbers, promise outcomes or play clinician unless something stops it. Prompting is not a guarantee.

## The solution

Daypart splits the decision into two parts:

1. **What to say (the policy).** Each day, for each member, pick one of seven *intents*: `none`, `reminder`, `encouragement`, `barrier_problem_solving`, `celebrate_streak`, `check_in_question`, `clinician_handoff`. The pick is conditioned on the member's state (`on_track` → `strained` → `lapsed` → `recovering`) and on their latent profile (motivation, barrier type, health literacy, engagement pattern).
2. **How to say it (the language model), behind a membrane.** A fine-tuned Muse Glimmer writes the message for the chosen intent. Every draft then passes a **deterministic boundary DSL** (`boundary-v1`, no LLM in the loop). Any dose quantity, outcome or efficacy claim, diagnosis, or advice outside the lane fails, and the message is replaced by a safe one. In the RL environment a violation ends the episode with reward −1.

You can measure this because it runs inside **SimMember**, a seeded synthetic cohort. 200 members have latent-conditioned day-by-day adherence dynamics. 25 curated profiles are held out of all training (invariant I1, enforced in CI).

## Architecture

```
SimMember cohort (seeded, synthetic)        daypart.simmember
  latents → daily state chain → LHM health records → I1 manifest
        │
        ▼
SimMemberEnv  (FutureHouse aviary)          daypart.simmember.env
  send_message(text, intent) → next state, adherence reward; DSL violation = episode fail
        │
        ├── policy: choose intent ───────── baselines today; GRPO / reward model planned
        │
        ├── language: Muse Glimmer 30B      daypart.train (SFT → DPO, 4-bit QLoRA on Modal H100)
        │
        ▼
Membrane  boundary-v1 DSL + layer-0 LHM narration guard   daypart.membrane
        │
        ▼
Memory    typed action/signal events, LOINC-seeded ontology, LHM metric state   daypart.memory
        │
        ▼
Pattern passport   one card per metric; number shown only when ready + claim-safe   daypart.passport
        │
        ▼
Eval harness  runs.yaml → metrics.json (every arm reported, including failed / not run)   daypart.eval
        │
        ▼
Demo   FastAPI playground + Streamlit dashboard, hosted on Modal   daypart.demo, engine/deploy
```

| Package | What it does |
|---|---|
| `simmember` | Synthetic cohort, latents, trajectories, LHM-shaped records, narrative client (Muse or offline template) |
| `membrane` | `boundary-v1` rule pack and `check()`; `post()` runs the full layer-0 guard on every draft |
| `train` | SFT chat dataset, DPO pairs (`lapse-recovery-v1` rubric), job specs, Modal app (Muse Glimmer QLoRA) |
| `memory` | Event typing, ontology with gated versioned updates, `metric_state(member, as_of)` |
| `passport` | Member-facing pattern cards; withheld card when a metric is not ready (fail closed) |
| `eval` | Deterministic harness: boundary corpus, env baselines, I1 gate, training arms |
| `demo` | Playground API + static UI, results dashboard, `daypart-demo` CLI |

It is built on Syntropy Health's pre-existing open-source **LHM** SDK (claim and narration guards, scoring, readiness), which is pinned, not modified. The boundary between old and new work is a git line; see [ATTRIBUTION.md](ATTRIBUTION.md).

Every component was built by sealed TDD. A test author writes the tests from a contract (`docs/daypart/contracts/`) without seeing the implementation. The implementer never sees those tests. The suite is 1,761 tests.

## Run it

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Everything below runs offline: no keys, no GPU.

```bash
cd engine
uv sync --locked
uv run pytest -q                                  # full suite
uv run python -m daypart.eval --out metrics.json  # benchmark harness → metrics.json
uv run daypart-demo serve                         # playground on http://127.0.0.1:8000
uv run daypart-demo dashboard --metrics metrics.json   # results dashboard (Streamlit)
```

Training (needs Modal credentials and Hugging Face access to `meta-models/Muse-Glimmer-30B`):

```bash
cd engine
uv run python -m daypart.train.run --offline --out /tmp/daypart-data      # build SFT + DPO datasets
uv run --with modal modal run daypart/train/modal_app.py::smoke --data-dir /tmp/daypart-data
```

Deploy the demo (CI does this on every green push to `daypart/build` or `main`):

```bash
cd engine && uv run --with modal modal deploy deploy/modal_demo.py
```

## Results

All numbers come from `python -m daypart.eval` and the Modal run record in [`docs/daypart/results/`](docs/daypart/results/). The dashboard shows the same data live.

**Policy baselines in SimMemberEnv** (50 scenarios × 3 seeds, 14-day horizon):

| Policy | Mean adherence | Boundary violations |
|---|---|---|
| `always_none` (no messaging) | 0.637 | 0 |
| `random_intent` | 0.719 | 0 |
| `always_barrier_problem_solving` | **0.833** | 0 |

Choosing the intent matters: problem-solving the barrier beats no contact by +0.20 adherence and beats a random intent by +0.11. That gap is the headroom a learned policy competes for.

**Fine-tuning Muse Glimmer 30B on Modal** (4-bit nf4 QLoRA, LoRA r8 on the language tower only: 14.7M trainable of 29.8B parameters, 0.049%; one H100):

| Stage | Data | Steps | Wall time | Peak GPU | Result |
|---|---|---|---|---|---|
| SFT | 4,072 boundary-clean chat rows, 2 epochs | 1,018 | 12.3 min | 19.9 GB | train loss 0.084, token accuracy 0.98 |
| DPO | 308 lapse-recovery preference pairs, 2 epochs | 78 | 2.1 min | 21.2 GB | train loss 0.59, chosen-vs-rejected reward margin 0.96 |

Total GPU cost was about $1.

**Boundary DSL (R2):** 100% recall and 0 false positives across all 8 rules on the sealed 202-row corpus (74 violations, 128 compliant), written by an independent test author.

**Base vs SFT vs DPO on held-out prompts** (R3/R4; 141 held-out prompts from synthetic members never used in training: 90 boundary probes that invite dosing, diagnosis or outcome claims, and 51 check-ins; greedy decoding; [`boundary_eval.json`](docs/daypart/results/boundary_eval.json)):

| Model | Boundary pass rate | On probes | Lapse-recovery rubric |
|---|---|---|---|
| Muse Glimmer 30B base | 0.702 | 0.533 | 0.000 |
| + SFT adapter | **1.000** | **1.000** | **0.936** |
| + DPO adapter | (1.000, vacuous) | (1.000, vacuous) | 0.000 |

- **R3 passes.** SFT lifts boundary compliance from 0.702 to 1.000 (+0.30 against a pre-registered 0.05 margin) and does not lose rubric quality. The base model, asked the probes, gave dosing imperatives, diagnoses, treatment claims and named prescription drugs in 42 of 90 replies. Its rubric score is 0 because every base reply ran past the length limit.
- **R4 fails.** The DPO adapter emits an empty string for every prompt, so its 1.000 boundary score is not evidence of anything. The fix is in progress.
- **Caveat on SFT:** the SFT model collapsed to 6 template replies (13–23 words) with a leaked `to=user` channel token. That is the style of the offline template data it was trained on. It is safe and on-rubric, but not yet varied. Better training data comes next.

**Not built for this submission, stated plainly:** the GRPO intent policy, the reward model and the Jev question library are designed but not built. The harness reports them as `not_run`.

## Meta models

- **Muse Glimmer 30B** (`meta-models/Muse-Glimmer-30B`, Hugging Face weights) is the language model. We fine-tune it with SFT then DPO on Modal, using TRL and PEFT on transformers 5.22. It is loaded in 4-bit, and LoRA touches only the language tower, so the vision tower never runs.
- **Muse Spark** (via `MuseClient`, an OpenAI-compatible endpoint) realizes the synthetic member narratives when credentials are present. An offline template client stands in otherwise, so the build and CI never need a key.

## Repository layout

```
engine/                 the daypart package, tests, Modal deploy
docs/daypart/contracts/ build contracts (the only surface shared between test author and implementer)
docs/daypart/results/   committed benchmark and training records
syntropy/, tests/, LHM.md, docs/ (non-daypart)   pre-existing LHM SDK baseline, unmodified
```

## License

Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE) and [ATTRIBUTION.md](ATTRIBUTION.md).
