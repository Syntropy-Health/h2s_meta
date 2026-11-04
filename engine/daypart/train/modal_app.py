"""Deploy-only Modal app for Daypart SFT + DPO (``modal deploy daypart/train/modal_app.py``).

This is the ONLY module under ``daypart`` that imports ``modal``. It is never
imported by ``daypart.train`` and is not exercised by tests (needs Modal
credentials). The volume layout is::

    /vol/datasets/<file>.jsonl        uploaded by the run phase
    /vol/checkpoints/<spec.name>/...  trainer checkpoints (resume source)
    /vol/adapters/<spec.name>/        final LoRA adapter
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath

import modal

from daypart.train.modal_jobs import JobSpec, PollResult

APP_NAME = "daypart-train"
VOLUME_NAME = "daypart-train"
VOL = "/vol"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.46.3",
        "peft==0.13.2",
        "trl==0.12.2",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "pydantic>=2.7,<3",
    )
    .add_local_python_source("daypart")
)

_GPU = "H100"
_TIMEOUT_S = 14400


def _vol_path(rel: str) -> str:
    return rel if rel.startswith("/") else str(PurePosixPath(VOL) / rel)


def _load_base(spec: JobSpec):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec.base_model, torch_dtype=torch.bfloat16)
    return model, tokenizer


def _verify_dataset(spec: JobSpec) -> str:
    import hashlib

    path = _vol_path(spec.dataset_path)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if digest != spec.dataset_sha256:
        raise ValueError(f"dataset {spec.dataset_path} sha256 mismatch: {digest} != {spec.dataset_sha256}")
    return path


@app.function(image=image, gpu=_GPU, volumes={VOL: volume}, timeout=_TIMEOUT_S)
def sft_train(spec_json: str, resume_from: str | None = None) -> dict:
    from datasets import load_dataset
    from peft import LoraConfig as PeftLoraConfig
    from trl import SFTConfig, SFTTrainer

    spec = JobSpec.model_validate(json.loads(spec_json))
    if spec.kind != "sft":
        raise ValueError("sft_train requires an sft spec")
    volume.reload()
    data_path = _verify_dataset(spec)
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "messages"])

    model, tokenizer = _load_base(spec)
    peft_config = PeftLoraConfig(
        r=spec.lora.rank,
        lora_alpha=spec.lora.alpha,
        lora_dropout=spec.lora.dropout,
        target_modules=list(spec.lora.target_modules),
        task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=_vol_path(spec.checkpoint_dir),
        num_train_epochs=spec.epochs,
        learning_rate=spec.learning_rate,
        per_device_train_batch_size=spec.batch_size,
        max_seq_length=spec.max_seq_len,
        seed=spec.seed,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=_vol_path(resume_from) if resume_from else None)
    adapter_dir = _vol_path(f"adapters/{spec.name}")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    volume.commit()
    return {"adapter": adapter_dir}


@app.function(image=image, gpu=_GPU, volumes={VOL: volume}, timeout=_TIMEOUT_S)
def dpo_train(spec_json: str, resume_from: str | None = None) -> dict:
    from datasets import load_dataset
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer

    spec = JobSpec.model_validate(json.loads(spec_json))
    if spec.kind != "dpo" or not spec.init_adapter or spec.beta is None:
        raise ValueError("dpo_train requires a dpo spec with init_adapter and beta")
    volume.reload()
    data_path = _verify_dataset(spec)
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in ("prompt", "chosen", "rejected")])

    base, tokenizer = _load_base(spec)
    model = PeftModel.from_pretrained(base, _vol_path(spec.init_adapter), is_trainable=True)
    args = DPOConfig(
        output_dir=_vol_path(spec.checkpoint_dir),
        beta=spec.beta,
        num_train_epochs=spec.epochs,
        learning_rate=spec.learning_rate,
        per_device_train_batch_size=spec.batch_size,
        max_length=spec.max_seq_len,
        seed=spec.seed,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT: the frozen base (adapter disabled) is the reference
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=_vol_path(resume_from) if resume_from else None)
    adapter_dir = _vol_path(f"adapters/{spec.name}")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    volume.commit()
    return {"adapter": adapter_dir}


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


__all__ = ["APP_NAME", "ModalSdkBackend", "app", "dpo_train", "sft_train", "volume"]
