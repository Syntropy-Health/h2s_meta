"""Deploy-only Modal app for Daypart SFT + DPO (``modal deploy daypart/train/modal_app.py``).

This is the ONLY module under ``daypart`` that imports ``modal``. It is never
imported by ``daypart.train`` and is not exercised by tests (needs Modal
credentials). The volume layout is::

    /vol/datasets/<file>.jsonl        uploaded by the run phase (or ``smoke``)
    /vol/checkpoints/<spec.name>/...  trainer checkpoints (resume source)
    /vol/adapters/<spec.name>/        final LoRA adapter
    /hf/                              Hugging Face cache (base weights, ~60 GB)

Base model: ``meta-models/Muse-Glimmer-30B`` (``MuseGlimmerForConditionalGeneration``,
29.8B bf16). It is trained text-only as 4-bit QLoRA (bitsandbytes nf4) on one
H100: LoRA sits on the language tower only (:func:`language_lora_targets`); the
vision tower, adapter and projection stay frozen in bf16 and never run.

2.4 boundary eval (base vs base+SFT vs base+DPO on held-out prompts, greedy, fixed seed);
writes ``/vol/results/boundary_eval.json`` and a local copy to commit::

    uv run --with modal modal run daypart/train/modal_app.py::boundary_eval --out ../docs/daypart/results/boundary_eval.json

Smoke step (2 SFT steps on 16 rows, then 2 DPO steps on 16 pairs from that
adapter), from ``engine/`` with Modal credentials::

    python -m daypart.train.run --offline --out /tmp/daypart-data
    uv run --with modal modal run daypart/train/modal_app.py::smoke --data-dir /tmp/daypart-data
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path, PurePosixPath

import modal

from daypart.train.modal_jobs import JobSpec, PollResult, language_lora_targets

APP_NAME = "daypart-train"
VOLUME_NAME = "daypart-train"
HF_CACHE_VOLUME_NAME = "daypart-hf-cache"
VOL = "/vol"
HF_HOME = "/hf"
BASE_MODEL = "meta-models/Muse-Glimmer-30B"
#: Modal list price for one H100, USD per second ($3.95/h). Used only for the smoke estimate.
H100_USD_PER_S = 0.001097

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.1",
        "transformers==5.22.1",  # first release line with model_type muse_glimmer
        "peft==0.21.2",
        "trl==1.17.1",  # its fused LM head applies Muse Glimmer's output_multiplier + logit softcap
        "accelerate==1.15.0",
        "bitsandbytes==0.50.2",
        "datasets==5.1.0",
        "pydantic>=2.7,<3",
    )
    .env({"HF_HOME": HF_HOME, "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("daypart")
)

_GPU = "H100"
_TIMEOUT_S = 14400
_VOLUMES = {VOL: volume, HF_HOME: hf_cache}
#: Kept in bf16 (not quantized): the output head and the unused vision side.
_KEEP_BF16 = ("lm_head", "vision_tower", "vision_adapter", "vision_projection")


def _vol_path(rel: str) -> str:
    return rel if rel.startswith("/") else str(PurePosixPath(VOL) / rel)


@app.function(image=image, volumes={HF_HOME: hf_cache}, cpu=4, memory=8192, timeout=3600)
def prefetch_base(base_model: str = BASE_MODEL) -> dict:
    """Download the base weights into the HF cache volume on CPU, so GPU time is not spent downloading."""
    from huggingface_hub import snapshot_download

    t0 = time.monotonic()
    path = snapshot_download(base_model)
    hf_cache.commit()
    return {"path": path, "seconds": round(time.monotonic() - t0, 1)}


def _load_base(spec: JobSpec):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(spec.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=list(_KEEP_BF16),
    )
    # MuseGlimmerForConditionalGeneration; fed input_ids only, so the vision tower never runs.
    model = AutoModelForImageTextToText.from_pretrained(
        spec.base_model, quantization_config=quant, dtype=torch.bfloat16, device_map={"": 0}
    )
    model.config.get_text_config().use_cache = False
    return model, tokenizer


def _lora_config(spec: JobSpec):
    from peft import LoraConfig as PeftLoraConfig

    return PeftLoraConfig(
        r=spec.lora.rank,
        lora_alpha=spec.lora.alpha,
        lora_dropout=spec.lora.dropout,
        target_modules=language_lora_targets(spec.lora.target_modules),
        task_type="CAUSAL_LM",
    )


def _common_args(spec: JobSpec, max_steps: int | None) -> dict:
    smoke = max_steps is not None
    return {
        "output_dir": _vol_path(spec.checkpoint_dir + ("-smoke" if smoke else "")),
        "num_train_epochs": spec.epochs,
        "max_steps": max_steps if smoke else -1,
        "learning_rate": spec.learning_rate,
        "per_device_train_batch_size": spec.batch_size,
        "max_length": spec.max_seq_len,
        "seed": spec.seed,
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "paged_adamw_8bit",
        "logging_steps": 1 if smoke else 10,
        "save_strategy": "no" if smoke else "steps",
        "save_steps": 100,
        "save_total_limit": 3,
        "report_to": [],
    }


def _load_rows(spec: JobSpec, keep: tuple[str, ...], max_rows: int | None):
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=_verify_dataset(spec), split="train")
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep])
    if max_rows is not None:
        dataset = dataset.select(range(min(max_rows, len(dataset))))
    return dataset


def _step_timer():
    from transformers import TrainerCallback

    class StepTimer(TrainerCallback):
        def __init__(self) -> None:
            self.step_s: list[float] = []
            self._t = 0.0

        def on_step_begin(self, args, state, control, **kwargs):
            self._t = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            self.step_s.append(round(time.monotonic() - self._t, 2))

    return StepTimer()


def _finish(trainer, tokenizer, spec: JobSpec, smoke: bool, timer, t0: float, t_loaded: float, t_trained: float) -> dict:
    import torch

    adapter_dir = _vol_path(f"adapters/{spec.name}" + ("-smoke" if smoke else ""))
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    volume.commit()
    return {
        "adapter": adapter_dir,
        "steps": trainer.state.global_step,
        "rows": len(trainer.train_dataset),
        "load_s": round(t_loaded - t0, 1),
        "train_s": round(t_trained - t_loaded, 1),
        "step_s": timer.step_s,
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "log_tail": trainer.state.log_history[-5:],
    }


@app.function(image=image, gpu=_GPU, volumes=_VOLUMES, memory=32768, timeout=_TIMEOUT_S)
def sft_train(
    spec_json: str, resume_from: str | None = None, max_steps: int | None = None, max_rows: int | None = None
) -> dict:
    """SFT. ``max_steps``/``max_rows`` set make it a smoke run (no checkpoints, ``-smoke`` adapter)."""
    from trl import SFTConfig, SFTTrainer

    t0 = time.monotonic()
    spec = JobSpec.model_validate(json.loads(spec_json))
    if spec.kind != "sft":
        raise ValueError("sft_train requires an sft spec")
    volume.reload()
    dataset = _load_rows(spec, ("messages",), max_rows)
    model, tokenizer = _load_base(spec)
    t_loaded = time.monotonic()
    timer = _step_timer()
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**_common_args(spec, max_steps)),
        train_dataset=dataset,
        processing_class=tokenizer,  # a tokenizer, not the processor: text-only training
        peft_config=_lora_config(spec),
        callbacks=[timer],
    )
    trainer.model.print_trainable_parameters()
    trainer.train(resume_from_checkpoint=_vol_path(resume_from) if resume_from else None)
    t_trained = time.monotonic()
    return _finish(trainer, tokenizer, spec, max_steps is not None, timer, t0, t_loaded, t_trained)


@app.function(image=image, gpu=_GPU, volumes=_VOLUMES, memory=32768, timeout=_TIMEOUT_S)
def dpo_train(
    spec_json: str, resume_from: str | None = None, max_steps: int | None = None, max_rows: int | None = None
) -> dict:
    """DPO from the SFT adapter. ``max_steps``/``max_rows`` as in :func:`sft_train`."""
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer

    t0 = time.monotonic()
    spec = JobSpec.model_validate(json.loads(spec_json))
    if spec.kind != "dpo" or not spec.init_adapter or spec.beta is None:
        raise ValueError("dpo_train requires a dpo spec with init_adapter and beta")
    volume.reload()
    dataset = _load_rows(spec, ("prompt", "chosen", "rejected"), max_rows)
    base, tokenizer = _load_base(spec)
    model = PeftModel.from_pretrained(base, _vol_path(spec.init_adapter), is_trainable=True)
    t_loaded = time.monotonic()
    timer = _step_timer()
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT: TRL copies the SFT adapter into a frozen "ref" adapter
        args=DPOConfig(beta=spec.beta, **_common_args(spec, max_steps)),
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[timer],
    )
    trainer.train(resume_from_checkpoint=_vol_path(resume_from) if resume_from else None)
    t_trained = time.monotonic()
    return _finish(trainer, tokenizer, spec, max_steps is not None, timer, t0, t_loaded, t_trained)


def _verify_dataset(spec: JobSpec) -> str:
    import hashlib

    path = _vol_path(spec.dataset_path)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if digest != spec.dataset_sha256:
        raise ValueError(f"dataset {spec.dataset_path} sha256 mismatch: {digest} != {spec.dataset_sha256}")
    return path


def upload_datasets(*paths: str | Path) -> None:
    """Upload local JSONL files to ``/vol/datasets/<name>`` (where :func:`sft_spec`/:func:`dpo_spec` point)."""
    with volume.batch_upload(force=True) as batch:
        for p in map(Path, paths):
            batch.put_file(str(p), f"/datasets/{p.name}")


def _cost_line(label: str, run: dict, rows_full: int, epochs: int, batch_size: int) -> str:
    steady = run["step_s"][-1]  # step 1 carries CUDA warm-up
    steps = -(-rows_full * epochs // batch_size)
    seconds = run["load_s"] + steps * steady
    return (
        f"{label}: {steps} steps x {steady}s + {run['load_s']}s load = {seconds / 60:.0f} min"
        f" ~ ${seconds * H100_USD_PER_S:.2f} (H100 at ${H100_USD_PER_S * 3600:.2f}/h)"
    )


@app.local_entrypoint()
def smoke(data_dir: str, base_model: str = BASE_MODEL, rows: int = 16, steps: int = 2, dpo: bool = True) -> None:
    """Upload data, prefetch weights, run ``steps`` SFT (+ DPO) steps on ``rows`` rows, print an estimate."""
    from daypart.train.modal_jobs import dpo_spec, sft_spec

    t0 = time.monotonic()
    sft_path, dpo_path = Path(data_dir) / "sft.jsonl", Path(data_dir) / "dpo.jsonl"
    upload_datasets(sft_path, dpo_path)
    print("prefetch:", prefetch_base.remote(base_model))
    sft = sft_spec(base_model, sft_path)
    out = sft_train.remote(sft.model_dump_json(), None, steps, rows)
    print("sft smoke:", json.dumps(out, indent=1))
    results = [("SFT", out, sum(1 for _ in sft_path.open()), sft)]
    if dpo:
        spec = dpo_spec(base_model, dpo_path, out["adapter"])
        out = dpo_train.remote(spec.model_dump_json(), None, steps, rows)
        print("dpo smoke:", json.dumps(out, indent=1))
        results.append(("DPO", out, sum(1 for _ in dpo_path.open()), spec))
    print(f"smoke wall clock: {time.monotonic() - t0:.0f}s (Modal billing: dashboard usage for app {APP_NAME})")
    for label, run, n, spec in results:
        print("estimate", _cost_line(label, run, n, spec.epochs, spec.batch_size))


SFT_ADAPTER = "adapters/daypart-sft-23866809bbd7"
DPO_ADAPTER = "adapters/daypart-dpo-1e2af88136ca"


@app.function(image=image, gpu=_GPU, volumes=_VOLUMES, memory=32768, timeout=_TIMEOUT_S)
def boundary_generate(
    prompts_json: str,
    adapters_json: str,
    base_model: str = BASE_MODEL,
    max_new_tokens: int = 192,
    batch_size: int = 16,
    seed: int = 0,
) -> dict:
    """Greedy responses per model: ``base`` (adapters disabled) plus one per named adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    t0 = time.monotonic()
    prompts = json.loads(prompts_json)
    adapters: dict[str, str] = json.loads(adapters_json)
    volume.reload()
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=list(_KEEP_BF16),
    )
    base = AutoModelForImageTextToText.from_pretrained(
        base_model, quantization_config=quant, dtype=torch.bfloat16, device_map={"": 0}
    )
    model = None
    for name, path in adapters.items():
        if model is None:
            model = PeftModel.from_pretrained(base, _vol_path(path), adapter_name=name)
        else:
            model.load_adapter(_vol_path(path), adapter_name=name)
    model = model if model is not None else base
    model.eval()
    model.config.get_text_config().use_cache = True
    t_loaded = time.monotonic()

    texts = [
        tokenizer.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True) for p in prompts
    ]

    def _generate() -> list[str]:
        out: list[str] = []
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(0) for k, v in enc.items()}
            with torch.no_grad():
                ids = model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
                )
            new = ids[:, enc["input_ids"].shape[1] :]
            out += [t.strip() for t in tokenizer.batch_decode(new, skip_special_tokens=True)]
        return out

    responses: dict[str, list[str]] = {}
    gen_s: dict[str, float] = {}
    for name in ["base", *adapters]:
        t = time.monotonic()
        if name == "base" and adapters:
            with model.disable_adapter():
                responses[name] = _generate()
        else:
            if name != "base":
                model.set_adapter(name)
            responses[name] = _generate()
        gen_s[name] = round(time.monotonic() - t, 1)
    return {
        "responses": responses,
        "load_s": round(t_loaded - t0, 1),
        "gen_s": gen_s,
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
    }


def _volume_jsonl(path: str) -> list[dict]:
    try:
        data = b"".join(volume.read_file(path))
    except Exception as exc:  # FileNotFoundError surfaces under several Modal error types
        raise SystemExit(f"cannot read {path} from volume {VOLUME_NAME}: {exc}") from exc
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


@app.local_entrypoint()
def boundary_eval(
    out: str = "boundary_eval.json",
    max_prompts: int = 300,
    max_new_tokens: int = 192,
    batch_size: int = 16,
    seed: int = 0,
    sft_adapter: str = SFT_ADAPTER,
    dpo_adapter: str = DPO_ADAPTER,
    base_model: str = BASE_MODEL,
    dry_run: bool = False,
) -> None:
    """2.4: held-out prompts (I1-gated), greedy base/SFT/DPO generation on one H100, R3/R4 report."""
    import io

    from daypart.eval.boundary import build_report, cost_estimate, heldout_prompts, score_response, training_user_turns
    from daypart.simmember.cohort import generate

    training = _volume_jsonl("datasets/sft.jsonl") + _volume_jsonl("datasets/dpo.jsonl")
    exclude = training_user_turns(training)
    prompts = heldout_prompts(generate(0), max_prompts=max_prompts, exclude_user_turns=exclude)
    est = cost_estimate(len(prompts), batch_size=batch_size, max_new_tokens=max_new_tokens, usd_per_s=H100_USD_PER_S)
    print(
        f"boundary_eval: {len(prompts)} held-out prompts x 3 models ({len(exclude)} training user turns excluded);"
        f" expected ~{est['seconds'] / 60:.0f} min on one H100 ~ ${est['usd']:.2f}"
        f" (H100 at ${H100_USD_PER_S * 3600:.2f}/h, {est['batches']} batches of {batch_size} x {max_new_tokens} tokens)"
    )
    if dry_run:
        return
    adapters = {"sft": sft_adapter, "dpo": dpo_adapter}
    gen = boundary_generate.remote(
        json.dumps(prompts), json.dumps(adapters), base_model, max_new_tokens, batch_size, seed
    )
    scored = {m: [score_response(p, t) for p, t in zip(prompts, gen["responses"][m])] for m in ("base", "sft", "dpo")}
    meta = {
        "base_model": base_model,
        "adapters": adapters,
        "seed": seed,
        "decoding": "greedy",
        "max_new_tokens": max_new_tokens,
        "n_prompts": len(prompts),
        "n_excluded_training_turns": len(exclude),
        "load_s": gen["load_s"],
        "gen_s": gen["gen_s"],
        "peak_gpu_gb": gen["peak_gpu_gb"],
        "estimate": est,
    }
    report = build_report(scored, meta=meta)
    text = json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(text.encode("utf-8")), "/results/boundary_eval.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    for m in ("base", "sft", "dpo"):
        agg = report["models"][m]
        print(f"{m:>4}: boundary {agg['boundary_pass_rate']:.3f}  rubric {agg['rubric_pass_rate']:.3f}  n={agg['n']}")
    print("R3:", json.dumps(report["r3"]))
    print("R4:", json.dumps(report["r4"]))
    print(f"wrote {out} and {VOLUME_NAME}:/results/boundary_eval.json")
    # The committed metrics.json must move with boundary_eval.json (QG F2; unit test guards it).
    from daypart.eval import dumps, load_runs, run

    results = Path(out).resolve().parent
    os.environ["DAYPART_RESULTS_DIR"] = str(results)
    (results / "metrics.json").write_text(dumps(run(load_runs(), root=Path(__file__).resolve().parents[2])), encoding="utf-8")
    print(f"regenerated {results.name}/metrics.json; commit it with {Path(out).name} and training.json")


class ModalSdkBackend:
    """:class:`~daypart.train.modal_jobs.ModalBackend` over the deployed ``daypart-train`` app."""

    def __init__(self, app_name: str = APP_NAME, volume_name: str = VOLUME_NAME) -> None:
        self.app_name = app_name
        self.volume_name = volume_name
        self._checkpoint_dirs: dict[str, str] = {}

    def submit(self, spec: JobSpec, *, resume_from: str | None) -> str:
        fn_name = "sft_train" if spec.kind == "sft" else "dpo_train"
        fn = modal.Function.from_name(self.app_name, fn_name)
        call = fn.spawn(spec.model_dump_json(), resume_from)
        self._checkpoint_dirs[call.object_id] = spec.checkpoint_dir
        return call.object_id

    def _latest_checkpoint(self, call_id: str) -> str | None:
        ckpt_dir = self._checkpoint_dirs.get(call_id)
        if ckpt_dir is None:
            return None
        vol = modal.Volume.from_name(self.volume_name)
        try:
            entries = vol.listdir(ckpt_dir)
        except Exception:  # directory not created yet
            return None
        steps = []
        for entry in entries:
            name = PurePosixPath(entry.path).name
            if name.startswith("checkpoint-") and name.split("-")[-1].isdigit():
                steps.append((int(name.split("-")[-1]), str(PurePosixPath(ckpt_dir) / name)))
        return max(steps)[1] if steps else None

    def poll(self, call_id: str) -> PollResult:
        call = modal.FunctionCall.from_id(call_id)
        try:
            out = call.get(timeout=0)
        except TimeoutError:
            return PollResult("running", checkpoint=self._latest_checkpoint(call_id))
        except Exception as exc:
            return PollResult(
                "failed",
                checkpoint=self._latest_checkpoint(call_id),
                error=f"{type(exc).__name__}: {exc}",
            )
        adapter = out.get("adapter") if isinstance(out, dict) else None
        return PollResult("succeeded", checkpoint=self._latest_checkpoint(call_id), adapter=adapter)


__all__ = [
    "APP_NAME",
    "BASE_MODEL",
    "ModalSdkBackend",
    "app",
    "boundary_eval",
    "boundary_generate",
    "dpo_train",
    "hf_cache",
    "prefetch_base",
    "sft_train",
    "smoke",
    "upload_datasets",
    "volume",
]
