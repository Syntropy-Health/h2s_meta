# Contribution attribution — Daypart Engine (Meta / FHDIC 2026)

This repository separates **pre-existing Syntropy Health code** from **new hackathon work** by a git boundary.

## Baseline (pre-existing, not hackathon work)

| What | Where | Provenance |
|---|---|---|
| LHM — Longevity Health Metrics SDK (`syntropy.lhm`) | repo root: `syntropy/`, `tests/`, `docs/` (non-`daypart`), `pyproject.toml`, `uv.lock`, `LHM.md` (the SDK's README, moved from `README.md` unchanged) | Byte-identical to [`Syntropy-Health/lhm@4821dc7`](https://github.com/Syntropy-Health/lhm/commit/4821dc74406de21a0e5274892be433f00d6f0ad0), committed here as `f170784 Initial commit` (2026-11-02). Apache-2.0. |
| AI-RDLC scaffold | `.claude/`, `agency.yaml`, `config/` | Tooling config, not product code. |

The engine does **not** import the root copy. `engine/pyproject.toml` pins LHM from its upstream repo at the same commit (A&D D8), so the attribution line is also a dependency line.

## New hackathon work (from 2026-11-03)

Everything under:

- `engine/` — the `daypart` package (simmember, membrane, train, rm, memory, evidence, personalize, passport, eval, demo) and its tests.
- `docs/daypart/` — build contracts and technical docs.
- `.github/workflows/engine.yml` — CI, and CD of the hosted demo to Modal.
- `README.md` — the Daypart Engine README (the LHM README is preserved as `LHM.md`).
- `ATTRIBUTION.md` — this file.

Verify the boundary: `git diff --stat f170784 -- syntropy tests pyproject.toml uv.lock` is empty.

## Third-party components used by new work

| Component | Use | License |
|---|---|---|
| `syntropy-lhm` @4821dc7 (`[lhm-scoring]`) | claim/narration gates, record shapes, scoring, readiness | Apache-2.0 (Syntropy Health, pre-existing) |
| `fhaviary` (FutureHouse aviary) | RL environment framework (D16) | Apache-2.0 |
| numpy, pydantic, PyYAML | core | BSD / MIT |
| FastAPI, Streamlit, uvicorn | demo | MIT / Apache-2.0 / BSD |
| transformers, TRL, PEFT, bitsandbytes, torch (Modal image only) | training | Apache-2.0 / MIT / BSD |
| LOINC® 2.83 (18 codes, `engine/daypart/memory/registry_a4.json`, verified in `loinc_verify_v1.csv`) | concept codes and display names for the A4 registry | [LOINC licence](https://loinc.org/license/) (Regenstrief Institute) |

### LOINC

This material contains content from LOINC® (https://loinc.org). LOINC is copyright © 1995-2026, Regenstrief Institute, Inc. and the Logical Observation Identifiers Names and Codes (LOINC) Committee and is available at no cost under the license at https://loinc.org/license/. LOINC® is a registered United States trademark of Regenstrief Institute, Inc.

The codes are used for display of synthetic values only; no clinical interpretation. Licence: https://loinc.org/license/

## Model use

Muse Glimmer 30B (Meta, Hugging Face weights) fine-tuned with SFT then DPO as 4-bit QLoRA on Modal; Muse Spark for narrative realization when credentials are present (offline template otherwise). Fireworks was dropped (plan G6). GRPO is not built. Status of each arm is reported in `engine/eval` results, including arms that did not run.

## Data

Synthetic only. 25 curated profiles (agent-drafted, held out from all training, I1) plus 175 parameterized synthetic profiles from `daypart.simmember`. No real user data, no PHI.
