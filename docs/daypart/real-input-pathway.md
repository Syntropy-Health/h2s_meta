# Real-input pathway

Real-input mode is **hard off** in this build. The demo accepts synthetic members and the
packaged synthetic fixtures only.

Turning real input on is not a configuration change. Under design decision AD7 it needs all of:

1. a code change to the constant `REAL_INPUT_BUILD_ALLOWED` in
   `engine/daypart/governance/real_input.py` (it is `False` in this build);
2. a revised product vision and requirements (PVR) that covers real input;
3. a new architecture and design for it;
4. every gate in the checklist below closed, with a named owner and evidence.

Closing every gate on its own does not enable real input. The constant is checked first, so the
refusal reason stays `build_disallowed`.

## Gates

The source of truth is `engine/daypart/governance/real_input.yaml`. This table renders the same
12 rows; a unit test (`engine/tests/unit/test_real_input_unit.py`) fails if they differ.

| id | gate | owner | status |
|---|---|---|---|
| lawful_basis | Lawful basis for processing real health data | unassigned | open |
| consent | Informed consent from each person whose data is entered | unassigned | open |
| access_control | Access control for who can enter and view data | unassigned | open |
| encryption | Encryption of data in transit and at rest | unassigned | open |
| retention_deletion | Retention limits and deletion on request | unassigned | open |
| audit | Audit log of access and changes | unassigned | open |
| incident_response | Incident and breach response plan | unassigned | open |
| processor_terms | Terms with every data processor and model provider | unassigned | open |
| jurisdiction_review | Review of the rules in each jurisdiction served | unassigned | open |
| clinical_intended_use | Written clinical intended use and its limits | unassigned | open |
| qualified_oversight | Oversight by a qualified health professional | unassigned | open |
| regulatory_review | Regulatory classification review | unassigned | open |

## How the gate is enforced

- **The constant.** `REAL_INPUT_BUILD_ALLOWED = False`. `status().enabled` is
  `build_allowed and not open_gates`, so it is `False` whatever the checklist says.
  `request_enable()` always returns a refusal.
- **`GET /api/mode`** reports `{"mode": "demo", "real_input": {"enabled": false, "requested": …,
  "open_gates": [...]}}`.
- **`POST /api/mode`** always answers `403` with `{"error": "real_input_refused",
  "open_gates": [...], "reason": "build_disallowed"}`, whatever the body asks for.
- **The env var `DAYPART_REAL_INPUT`** (`1`, `on` or `true`) is read once at app start. The app
  still starts in demo mode, logs one warning line listing the open gate ids, and `/api/mode`
  reports `requested: true, enabled: false`.

This document records how the software is gated. It is not a legal, regulatory or clinical
assessment.
