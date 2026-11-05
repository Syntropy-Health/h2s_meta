# CONTRACT slice F — narrative cache, SFT/DPO datasets, Modal job specs

Plan tasks 1.3, 2.1, 2.2, 2.3 (HAC-5, child of HAC-2). Sources: PVR R3, R4; A&D D16, I1, I4; plan grill G6 rev 4 (**Muse for models, Modal for jobs, no other services**). This file is the only surface shared between the test-author and the implementer.

Package roots: `engine/daypart/simmember/narrative.py`, `engine/daypart/train/`. Depends on slice A (`Cohort`, `Profile`, `Trajectory`, `DayRecord`, `CuratedLeakError`, `assert_no_curated`, `canonical_json`, `write`/`verify_manifest`), slice B (`daypart.membrane.dsl.check`, `daypart.membrane.post.post`).

**No network in this slice's tests or in CI.** Every external system (Muse, Modal) sits behind a `Protocol`; tests use fakes. No new third-party dependency is added to `engine/pyproject.toml` (the `uv.lock` is unchanged). `modal` is never imported at module import time by anything under `daypart` except `daypart/train/modal_app.py` (deploy-only, not imported by the package, not tested).

Determinism: no wall clock, no `random`, no uuid, no hash-seed dependence. Canonical JSON = `daypart.simmember.manifest.canonical_json` (`sort_keys=True, separators=(",", ":")`). JSONL files: one canonical-JSON object per line, each line ends with `\n`.

## F1. Text client interface — `daypart.simmember.narrative`

```python
class TextClient(Protocol):
    model_id: str
    def complete(self, messages: Sequence[Mapping[str, str]], *, seed: int,
                 temperature: float = 0.7, max_tokens: int = 256) -> str
    # messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]

class MissingCredentialsError(RuntimeError): ...

class MuseClient:                      # satisfies TextClient; the only network client
    def __init__(self, *, model_id: str | None = None, env: Mapping[str, str] | None = None) -> None
```
- `MuseClient` reads `MUSE_API_KEY`, `MUSE_BASE_URL` and (when `model_id` is None) `MUSE_SPARK_MODEL` from `env` (default `os.environ`). Any missing/empty → `MissingCredentialsError` naming the missing variable names (never their values). Construction does **no** network I/O. `complete` POSTs an OpenAI-compatible chat-completions request (`{base}/chat/completions`, bearer auth, stdlib `urllib` only) and returns the first choice's message content. `repr(client)` never contains the key.
- **Reasoning models (HAC-28 amendment).** `MuseClient(..., reasoning_effort: str | None = <unset>)`: when unset, it uses `env["MUSE_REASONING_EFFORT"]`, else `"minimal"`; `None` omits the field. When set, every request body carries `"reasoning_effort": <value>`. If the first choice's content is `null` or `""` and `finish_reason == "length"`, `complete` raises `TruncatedResponseError` (a `ValueError` subclass, with attribute `max_tokens` set to the request's value) instead of returning or crashing. After every response, `client.last_usage` is `{"prompt_tokens": int, "completion_tokens": int}` when the response reports `usage`, else `None`.
- **Run-phase spend (`daypart.train.run.BudgetedClient`, HAC-28 amendment).** Spend comes from the inner client's `last_usage` tokens (reasoning tokens are part of `completion_tokens`). A client with no usage falls back to chars/4. `in_tokens`/`out_tokens` accumulate over **every billed attempt**, truncated ones included. On `TruncatedResponseError` it retries the same logical call with `max_tokens` doubled, up to `max_tokens_cap` (default 2048), then re-raises. `calls` counts logical calls. `run --reasoning-effort X` (default `minimal`, `none` omits it) passes the value through. `summary.json` adds `tokens_in`, `tokens_out` and `truncations`.
- **Sharded realization (`daypart.train.run`, HAC-28 amendment).** `ReplayClient(inner, path)` keys each call by `sha256(messages, seed, sorted kwargs)`. A hit returns the recorded text with `last_usage == {"prompt_tokens": 0, "completion_tokens": 0}`; a miss calls `inner` and appends `{"key", "text"}` to `path`. Failures are never recorded; a fresh instance reloads `path`. `merge_replays(paths, out) -> int` unions the files (missing files are skipped). `run --member-offset N --members K` realizes training members `[N, N+K)` and only their DPO contexts. `run --replay PATH` wraps the client, and `summary.json["replay_hits"]` counts the hits. Replaying a recorded window gives identical `sft_sha256`/`dpo_sha256`.
- `TemplateClient(model_id: str = "template-v1")` satisfies `TextClient`: deterministic, offline, no I/O. Output depends only on `(messages, seed)`; it picks from a fixed pool of short synthetic texts. Used by the I1 gate and by tests that need *a* client. Its texts must all pass `post` (F3) — for the journal pool, the dialogue pool and the refusal pool alike.

## F2. Prompts

```python
NARRATIVE_VERSION = "narrative-v1"
KINDS = ("journal", "dialogue")

class NarrativePrompt(pydantic.BaseModel, frozen=True):     # extra="forbid"
    member_id: str
    day: int
    kind: Literal["journal", "dialogue"]
    messages: tuple[dict[str, str], ...]
    prompt_hash: str          # sha256 hex of canonical_json({"version": NARRATIVE_VERSION, "kind": kind, "messages": [...]})

def build_prompt(profile: Profile, record: DayRecord, kind: str, *, journal: str | None = None) -> NarrativePrompt
```
- **Latent-conditioned:** the prompt text (all message contents joined) contains the profile's `motivation`, `barrier_type`, `literacy` and `engagement_pattern` levels (underscores may be rendered as spaces), and the day's `state`. Two profiles that differ in any one of those four latents (same record) get different `prompt_hash`es; same inputs → identical `NarrativePrompt`.
- `kind="journal"`: asks for the member's own short check-in journal for that day (member voice). `journal` must be None.
- `kind="dialogue"`: asks for the coach's single reply to the member's journal; `journal` (non-empty str) is required and appears verbatim in the prompt. Missing/empty → `ValueError`.
- Unknown `kind` → `ValueError`.
- **I1:** a curated profile (`cohort == "curated"`, id `cur-…`, or latents equal to a `CURATED_PROFILES` entry) → `CuratedLeakError`. No prompt ever contains a `cur-` id. The prompt contains the synthetic `member_id` nowhere in the messages (the id lives only in the `member_id` field) — the model never sees member ids.

## F3. Cache + realization

```python
class NarrativeRow(pydantic.BaseModel, frozen=True):        # extra="forbid"
    member_id: str
    day: int
    kind: Literal["journal", "dialogue"]
    prompt_hash: str
    model_id: str
    decision: Literal["pass", "escalate"]
    text: str | None          # str iff decision == "pass"
    trace: dict               # the PostResult.trace (never contains draft text)

class NarrativeCache:
    def __init__(self, path: str | os.PathLike) -> None   # loads existing rows; missing file = empty cache
    def get(self, member_id: str, day: int, prompt_hash: str) -> NarrativeRow | None
    def put(self, row: NarrativeRow) -> None              # appends one line; key already present → no-op (first write wins)
    def rows(self) -> list[NarrativeRow]                  # file order
    def __len__(self) -> int

def realize(profile: Profile, trajectory: Trajectory, client: TextClient, cache: NarrativeCache,
            *, days: Iterable[int] | None = None) -> list[NarrativeRow]
```
- Cache key = `(member_id, day, prompt_hash)`. A malformed line in an existing file → `ValueError` at construction. A row with `decision == "pass"` and `text` None (or `"escalate"` with text) is invalid → `pydantic.ValidationError`.
- `realize` iterates the trajectory's check-in days (`record.checkin is True`), restricted to `days` if given, in day order. Per day: build the journal prompt; if cached use the row, else call `client.complete(prompt.messages, seed=s)` and run `daypart.membrane.post.post(draft, regenerate=…)` where `regenerate` calls `client.complete` once more with seed `s + 1`; cache a row with the post decision, `text` = `PostResult.text`, `trace` = `PostResult.trace`. If the journal row passed, do the same for the dialogue prompt (built with that journal text). Returns the rows touched (cached or new) in order (journal then dialogue per day).
- `s` is a deterministic function of `(member_id, day, kind)` only.
- **Every cached `text` passed `post`.** A draft that escalates is cached with `text=None` (so a rerun neither re-spends nor changes output). No dialogue row exists for a day whose journal escalated.
- **Cache hit ⇒ zero client calls.** Re-running `realize` with the same inputs over a populated cache makes no `complete` call and returns equal rows; a fresh cache with a deterministic client produces a byte-identical file.
- **I1:** curated profile → `CuratedLeakError` before any client call or cache write; `trajectory.member_id != profile.member_id` → `ValueError`.

## F4. SFT dataset — `daypart.train.sft_dataset`

```python
COACH_SYSTEM_PROMPT: str      # fixed, non-empty; itself passes dsl.check
REFUSAL_PROMPTS: tuple[str, ...]      # >= 8 member requests for dosing/diagnosis/treatment advice (they FAIL dsl.check as text — that's the point)
REFUSALS: tuple[str, ...]             # >= 4 refusal-with-redirection replies; each passes post (layer 0 + DSL)
                                      # and redirects: mentions "clinician" and ends with a question (contains "?")

def build_sft(cohort: Cohort, cache: NarrativeCache, *, refusal_ratio: float = 0.2, seed: int = 0) -> list[dict]
def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | os.PathLike) -> str   # returns sha256 hex of the written file
```
Row shape (chat JSONL):
```python
{"messages": [{"role": "system", "content": COACH_SYSTEM_PROMPT},
              {"role": "user", "content": str},          # day context + the member's journal (dialogue rows) or a REFUSAL_PROMPTS entry
              {"role": "assistant", "content": str}],    # the cached dialogue text, or a REFUSALS entry
 "meta": {"member_id": str, "day": int, "source": "dialogue" | "refusal", "prompt_hash": str | None}}
```
- Dialogue rows: one per `(member, day)` where the cache has a passing journal **and** a passing dialogue row for a **training** profile of `cohort` (synthetic only), in cohort order then day order. Cache rows for members not in `cohort.training_profiles()` are ignored.
- **DSL filter:** every assistant `content` in the output passes `dsl.check` (100 %); any candidate that fails (e.g. a hand-edited cache) is dropped, never edited.
- **Refusal mix:** `n_refusal = round(refusal_ratio * n_dialogue / (1 - refusal_ratio))`; refusal rows use `meta.member_id` of a training member, chosen deterministically from `seed`; `prompt_hash` None. `refusal_ratio` outside `[0, 0.5]` → `ValueError`. With `refusal_ratio=0` there are no refusal rows. Final order: deterministic interleave under `seed` (same inputs → identical list).
- **I1:** `build_sft` calls `assert_no_curated(rows, manifest)` on its output (manifest built from `cohort` in a temp dir via slice A `write`) and raises `CuratedLeakError` on a leak. A cache containing curated rows never leaks them (they are ignored as non-training members); no `cur-` id appears anywhere in the output.
- Registered in `daypart.ci.i1_check.BUILDERS` as `"train.sft_dataset"`: builder realizes a small fixed set of training members (≤ 20) with `TemplateClient` into a temp cache and returns `build_sft(...)`.

## F5. DPO pairs — `daypart.train.dpo_pairs`

```python
RUBRIC_VERSION = "lapse-recovery-v1"
SHAME_TERMS: tuple[str, ...]   # at least: "failed", "failure", "lazy", "disappointed", "disappointing", "no excuse", "should have", "shame", "guilt"
PRINCIPLES: str                # principle system prompt (boundary + lapse-recovery principles); itself passes dsl.check

@dataclass(frozen=True)
class RubricVerdict:
    passed: bool
    failed: tuple[str, ...]    # names of failed checks, subset of ("length", "shame", "invitation")

def rubric_check(member_state: str, text: str) -> RubricVerdict
```
`rubric_check`: `length` fails unless `1 <= len(text.strip()) <= 480`; `shame` fails if any `SHAME_TERMS` entry occurs (casefolded, word-boundary); `invitation` fails when `member_state in {"strained", "lapsed", "recovering"}` and `"?"` not in text (engaged: invitation always passes). Unknown `member_state` → `ValueError`.

```python
class DpoContext(pydantic.BaseModel, frozen=True):          # extra="forbid"
    member_id: str
    day: int
    member_state: Literal["engaged", "strained", "lapsed", "recovering"]
    user_turn: str            # the member-facing context + journal; no member id inside

def make_contexts(cohort: Cohort, cache: NarrativeCache, *, per_member: int = 2, max_members: int | None = None) -> list[DpoContext]
def build_pairs(contexts: Sequence[DpoContext], sampler: TextClient, *, n_candidates: int = 4, seed: int = 0, require_contrast: bool = False) -> list[dict]
```
- `make_contexts`: training profiles only; per member, up to `per_member` days that have a passing journal in the cache, preferring non-`engaged` states, then earlier days. Deterministic. Curated rows in the cache are ignored.
- `build_pairs` is pure over `sampler` (only `sampler.complete` is called). Per context:
  - **chosen candidates:** up to `n_candidates` samples of `[{"role": "system", "content": COACH_SYSTEM_PROMPT + "\n\n" + PRINCIPLES}, {"role": "user", "content": user_turn}]` with seeds derived from `(seed, context, i)`; chosen = the first that passes `post` (layer 0 + DSL) **and** `rubric_check(member_state, text).passed`. None qualifies → no pair for this context.
  - **rejected:** one sample of the **unconstrained** same context `[{"role": "system", "content": COACH_SYSTEM_PROMPT}, {"role": "user", "content": user_turn}]` (no `PRINCIPLES`), distinct seed. Identical to chosen → no pair.
  - Row: `{"prompt": [{"role": "system", "content": COACH_SYSTEM_PROMPT}, {"role": "user", "content": user_turn}], "chosen": [{"role": "assistant", "content": chosen}], "rejected": [{"role": "assistant", "content": rejected}], "meta": {"member_id", "day", "member_state", "rubric_version": RUBRIC_VERSION, "rejected_dsl_passed": bool, "rejected_rubric_passed": bool}}`.
  - `require_contrast=True` (HAC-28) also drops the pair when the rejected text passes **both** `dsl.check` and `rubric_check` (no preference to learn). The call sequence is unchanged. `python -m daypart.train.run` passes `--require-contrast` (default on; `--no-require-contrast` turns it off).
  - The principled call sequence is exactly: candidate samples in order, stopping at the first qualifying one; then exactly one unconstrained call. A sampler exception propagates.
- `n_candidates < 1` → `ValueError`. Curated `member_id` in any context → `CuratedLeakError` before any sampler call.
- Registered in `BUILDERS` as `"train.dpo_pairs"` (TemplateClient sampler, ≤ 20 training members).
- `write_jsonl` from F4 is reused for pairs.

## F6. Modal job specs — `daypart.train.modal_jobs`

Pure data + an orchestration loop over a backend interface. Importing this module must not import `modal`.

```python
class LoraConfig(pydantic.BaseModel, frozen=True):          # extra="forbid"
    rank: int = 8; alpha: int = 16; dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

class JobSpec(pydantic.BaseModel, frozen=True):             # extra="forbid"
    kind: Literal["sft", "dpo"]
    base_model: str                 # non-empty (Glimmer weights ref: HF repo id or Modal-volume path)
    dataset_path: str               # path inside the Modal volume
    dataset_sha256: str             # 64 lowercase hex
    lora: LoraConfig = LoraConfig()
    epochs: int = 2                 # >= 1
    learning_rate: float = 2e-4     # > 0
    batch_size: int = 8             # >= 1
    max_seq_len: int = 2048         # >= 1
    seed: int = 0
    beta: float | None = None       # dpo: required, > 0 (default via dpo_spec 0.1); sft: must be None
    rpo_alpha: float | None = None  # dpo: None or >= 0 (default via dpo_spec 1.0); sft: must be None
    init_adapter: str | None = None # dpo: required non-empty (the SFT adapter); sft: must be None
    gpu: str = "H100"
    timeout_s: int = 14400          # >= 60

    @property
    def spec_hash(self) -> str      # sha256 hex of canonical_json(self.model_dump(mode="json"))
    @property
    def name(self) -> str           # f"daypart-{kind}-{spec_hash[:12]}"
    @property
    def checkpoint_dir(self) -> str # f"checkpoints/{name}"

def sft_spec(base_model: str, dataset: str | os.PathLike, *, volume_path: str | None = None, **overrides) -> JobSpec
def dpo_spec(base_model: str, dataset: str | os.PathLike, init_adapter: str, *, volume_path: str | None = None, beta: float = 0.1, learning_rate: float = DPO_LEARNING_RATE, rpo_alpha: float | None = DPO_RPO_ALPHA, **overrides) -> JobSpec

DPO_LEARNING_RATE = 1e-5    # HAC-28: 2e-4 (the SFT LoRA rate) collapsed the DPO policy to special tokens
DPO_RPO_ALPHA = 1.0         # RPO weight: NLL on chosen, anchors the policy

class TemplateDataError(ValueError): ...
def check_trainable(summary: Mapping[str, Any]) -> None
```
- `check_trainable(summary)` (a `run` `summary.json` dict) raises `TemplateDataError` unless `summary["offline"] is False` and `summary["model_id"]` is non-empty and does not start with `"template"` (HAC-28: SFT trained on `TemplateClient` output collapsed to its pool). Missing keys → `TemplateDataError`.
- `sft_spec`/`dpo_spec` hash the local `dataset` file's bytes into `dataset_sha256`; `dataset_path` = `volume_path` or `f"datasets/{basename}"`. Missing file → `FileNotFoundError`. An empty file → `ValueError`. Plan 2.2: `sft_spec(...).lora.rank == 8`.
- Constraint violations (kind/beta/init_adapter mismatch, bad hex, ranges, unknown keys) → `pydantic.ValidationError`. `JobSpec.model_validate(json.loads(spec.model_dump_json())) == spec` (round-trip).

```python
JobStatus = Literal["pending", "running", "succeeded", "failed"]

@dataclass(frozen=True)
class PollResult:
    status: JobStatus
    checkpoint: str | None = None   # latest checkpoint path, if any
    adapter: str | None = None      # set when succeeded
    error: str | None = None

class ModalBackend(Protocol):
    def submit(self, spec: JobSpec, *, resume_from: str | None) -> str      # returns a call id
    def poll(self, call_id: str) -> PollResult

@dataclass(frozen=True)
class JobResult:
    status: Literal["succeeded", "failed"]
    adapter: str | None
    attempts: int
    call_ids: tuple[str, ...]
    errors: tuple[str, ...]

class JobRunner:
    def __init__(self, backend: ModalBackend, *, max_retries: int = 2, poll_interval_s: float = 30.0,
                 max_polls: int = 2000, sleep: Callable[[float], None] = time.sleep) -> None
    def run(self, spec: JobSpec) -> JobResult
```
`JobRunner.run`:
1. Attempt 1: `submit(spec, resume_from=None)`.
2. Poll that call id, calling `sleep(poll_interval_s)` between non-terminal polls, until `succeeded` / `failed` or `max_polls` polls (→ attempt failed, error mentions the poll limit).
3. `succeeded` → return `JobResult("succeeded", adapter, attempts, call_ids, errors)`. `adapter` None on success counts as a failed attempt.
4. Failed attempt (status `failed`, poll limit, or an exception from `submit`/`poll`, recorded as `f"{type(exc).__name__}: {exc}"` or the `error` field) and retries remain → next attempt `submit(spec, resume_from=<the most recent non-None checkpoint seen in any poll so far>, or None)`. **Retry-from-checkpoint:** if any poll reported a checkpoint, the retry resumes from it.
5. At most `1 + max_retries` submits. All failed → `JobResult("failed", None, attempts, call_ids, errors)`; `run` never raises for backend failures. `len(errors) == number of failed attempts`.
- `max_retries < 0`, `max_polls < 1`, `poll_interval_s < 0` → `ValueError`.

```python
def plan_pipeline(base_model: str, sft_dataset: str | os.PathLike, dpo_dataset: str | os.PathLike) -> tuple[JobSpec, Callable[[str], JobSpec]]
```
Returns the SFT spec and a function `adapter -> dpo JobSpec` (init_adapter = the SFT adapter). DPO is only built from a finished SFT adapter.

`daypart/train/modal_app.py` (deploy-only; `modal deploy`): a `modal.App("daypart-train")` with GPU functions `sft_train(spec_json: str, resume_from: str | None) -> dict` and `dpo_train(...)` (TRL + PEFT in the image), a `modal.Volume` for datasets/checkpoints/adapters, writing checkpoints under `spec.checkpoint_dir` and returning `{"adapter": ...}`. `dpo_train` passes `beta` to `DPOConfig`, and a truthy `rpo_alpha` as `loss_type=["sigmoid", "sft"]`, `loss_weights=[1.0, rpo_alpha]` (TRL 1.17 has no `rpo_alpha` field). A `train(data_dir)` local entrypoint calls `check_trainable` on `data_dir/summary.json`, uploads, and spawns `train_pipeline` (SFT then DPO, writes `/vol/results/training.json`) on the deployed app. Plus `ModalSdkBackend` (in `modal_app.py` or a module that imports `modal` lazily inside methods) implementing `ModalBackend` over `modal.Function.from_name(...).spawn(...)` / `modal.FunctionCall.from_id(...)`. Not covered by sealed tests (needs credentials); must not be imported by `daypart.train.__init__`.

## F7. Out of scope here (run phase, needs secrets)

Realizing narratives with `MuseClient`, building the committed datasets, launching SFT then DPO via `JobRunner`, and 2.4 eval (base vs SFT boundary compliance with a pre-registered margin; lapse-recovery rubric SFT vs DPO using `rubric_check`; zero boundary regression) into the HAC-4 `metrics.json`. Contracted separately once secrets are bound.

## F8. Chat-template reply — `daypart.train.chat_format` (pure; HAC-28)

```python
GENERATION_PROMPT = "<|start|>assistant"
def extract_reply(continuation: str) -> str
```
Muse Glimmer renders an assistant reply as `<|start|>assistant to=<recipient><|message|>TEXT<|eot|>` (or `<|eom|>`). Reasoning goes to `to=self`, tool calls to `to=<tool>`. The generation prompt ends at `GENERATION_PROMPT`. `continuation` is the generation decoded with special tokens kept. `extract_reply` prepends `GENERATION_PROMPT`, drops `<|finetune_right_pad|>`, splits on `<|start|>`, and returns the stripped body of the **last** segment that meets all of these:
- its header (the text before `<|message|>`) starts with the word `assistant`;
- its last `to=` recipient is `user`, or it has no recipient;
- the body is cut at the first `<|eot|>`, `<|eom|>` or `<|end_of_text|>`, or runs to the end if none.

No such segment → `""`. That covers an empty generation, special tokens only, reasoning cut off before a reply, and tool calls only.
