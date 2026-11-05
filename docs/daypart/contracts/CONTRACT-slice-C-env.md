# CONTRACT slice C — `daypart.simmember.env` aviary environment

Plan task 1.4 (HAC-2). Sources: A&D D16, "Benchmarking & virtual environment", R6 (DSL violation = episode fail), I1. Depends on slice A (`Cohort`, `Profile`, `CuratedLeakError`, trajectory `STATES`) and slice B (`daypart.membrane.dsl.check`). Shared surface between test-author and implementer.

Framework: `aviary.core` (package `fhaviary`): `Environment`, `Message`, `Tool`, `ToolRequestMessage`, `ToolResponseMessage`, `ToolCall`.

## C1. Dynamics — `daypart.simmember.dynamics`

```python
INTENTS = ("none", "reminder", "encouragement", "barrier_problem_solving",
           "celebrate_streak", "check_in_question", "clinician_handoff")

def transition(latents: Latents, state: str, intent: str, rng: numpy.random.Generator) -> tuple[str, bool]
    # returns (next_state, adhered_today)
```
- Same chain semantics as slice A3 (`lapsed` → only `lapsed`/`recovering`, etc.). Unknown `intent` → `ValueError`.
- `intent="none"` is the baseline policy (slice A trajectories are generated with it).
- Intent effects are latent-conditioned and directional: averaged over the synthetic cohort, `barrier_problem_solving` yields a higher probability of leaving `strained`/`lapsed` than `none`. (Sealed test: over 200 seeded rollouts of 14 days from the synthetic profiles, mean adherence under always-`barrier_problem_solving` > under always-`none`.)

## C2. Scenarios — `daypart.simmember.env`

```python
class Scenario(pydantic.BaseModel, frozen=True):
    scenario_id: str          # f"scn-{seed}-{member_id}"
    member: Profile
    seed: int
    horizon: int = 14         # steps (days)

def make_scenarios(cohort: Cohort, n: int, seed: int) -> list[Scenario]
def scenario_manifest(scenarios: Sequence[Scenario]) -> dict   # {"scenarios": {scenario_id: sha256hex of canonical json}, "count": n}
```
- `make_scenarios` draws members only from `cohort.training_profiles()` (deterministic under `seed`); `n` larger than the synthetic pool → `ValueError`.
- **I1:** constructing a `Scenario` or `SimMemberEnv` with a curated (`cur-`) member raises `CuratedLeakError`.
- `scenario_manifest` is deterministic (canonical JSON `sort_keys=True, separators=(",", ":")`).

## C3. Environment

```python
class EnvState(pydantic.BaseModel):
    day: int
    member_state: str             # one of STATES
    adherence: list[bool]         # one per completed step
    done: bool
    violations: list[str]         # DSL rule_ids that ended the episode, else []

class SimMemberEnv(aviary.core.Environment[EnvState]):
    def __init__(self, scenario: Scenario) -> None
    async def reset(self) -> tuple[list[Message], list[Tool]]
    async def step(self, action: ToolRequestMessage) -> tuple[list[Message], float, bool, bool]
    state: EnvState
```
- `reset()` → (one observation `Message` describing the member's day-0 situation in synthetic template text — contains the scenario's `member_id` and no other identity, and no curated id; tools = exactly one `Tool` named `send_message` with params `text: str`, `intent: str`).
- `step(action)`: executes each `send_message` tool call in the request (at most one per step is honoured; extra calls are ignored with an error tool response). For the message:
  1. `daypart.membrane.dsl.check(text)` fails → reward `-1.0`, `done=True`, `state.violations` = fired rule_ids, `state.done=True`.
  2. `intent` not in `INTENTS` → tool response says invalid intent; treated as `"none"`.
  3. Otherwise advance one day via `transition(...)` using an rng seeded from `(scenario.seed, member index, day)`; reward = `1.0` if adhered else `0.0`.
  - Returns `([ToolResponseMessage with the member's synthetic reply observation], reward, done, truncated)`; `truncated=True` exactly when `day == horizon` (then `done=True` too).
  - A step with no tool calls → reward 0.0, no day advance, not done (aviary default no-tool-calls behaviour is acceptable).
  - `step` after `done` → `RuntimeError`.
- **Determinism:** two envs built from the same `Scenario`, driven with the same action sequence, produce identical rewards, dones, observation contents, and final `state`.
- `export_frame()` returns a `Frame` whose state includes `day`, `member_state`, `adherence`.
