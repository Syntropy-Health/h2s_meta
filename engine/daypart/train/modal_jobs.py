"""F6 — Modal job specs (pure data) and the retry-from-checkpoint job runner.

Importing this module never imports ``modal``: the runner drives any
:class:`ModalBackend`; the real SDK backend lives in the deploy-only
``daypart.train.modal_app``.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import pydantic
from pydantic import ConfigDict, Field, model_validator

from daypart.simmember.manifest import canonical_json

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class LoraConfig(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(default=8, ge=1)
    alpha: int = Field(default=16, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    @model_validator(mode="after")
    def _modules(self) -> "LoraConfig":
        if not self.target_modules or not all(m for m in self.target_modules):
            raise ValueError("target_modules must be non-empty strings")
        return self


class JobSpec(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["sft", "dpo"]
    base_model: str = Field(min_length=1)
    dataset_path: str = Field(min_length=1)
    dataset_sha256: str
    lora: LoraConfig = LoraConfig()
    epochs: int = Field(default=2, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0, allow_inf_nan=False)
    batch_size: int = Field(default=8, ge=1)
    max_seq_len: int = Field(default=2048, ge=1)
    seed: int = 0
    beta: float | None = Field(default=None, allow_inf_nan=False)
    #: DPO only: weight of the NLL-on-chosen term (TRL ``rpo_alpha``) that anchors the policy.
    rpo_alpha: float | None = Field(default=None, allow_inf_nan=False)
    init_adapter: str | None = None
    gpu: str = Field(default="H100", min_length=1)
    timeout_s: int = Field(default=14400, ge=60)

    @model_validator(mode="after")
    def _consistent(self) -> "JobSpec":
        if not self.base_model.strip():
            raise ValueError("base_model must be non-empty")
        if not _HEX64.match(self.dataset_sha256):
            raise ValueError("dataset_sha256 must be 64 lowercase hex characters")
        if self.kind == "dpo":
            if self.beta is None or not self.beta > 0:
                raise ValueError("dpo jobs require beta > 0")
            if not self.init_adapter or not self.init_adapter.strip():
                raise ValueError("dpo jobs require a non-empty init_adapter (the SFT adapter)")
            if self.rpo_alpha is not None and self.rpo_alpha < 0:
                raise ValueError("rpo_alpha must be >= 0")
        else:
            if self.beta is not None:
                raise ValueError("sft jobs must not set beta")
            if self.init_adapter is not None:
                raise ValueError("sft jobs must not set init_adapter")
            if self.rpo_alpha is not None:
                raise ValueError("sft jobs must not set rpo_alpha")
        return self

    @property
    def spec_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json")).encode("utf-8")).hexdigest()

    @property
    def name(self) -> str:
        return f"daypart-{self.kind}-{self.spec_hash[:12]}"

    @property
    def checkpoint_dir(self) -> str:
        return f"checkpoints/{self.name}"


def _hash_dataset(dataset: str | os.PathLike) -> tuple[Path, str]:
    path = Path(dataset)
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    data = path.read_bytes()
    if not data:
        raise ValueError(f"dataset is empty: {path}")
    return path, hashlib.sha256(data).hexdigest()


def sft_spec(
    base_model: str, dataset: str | os.PathLike, *, volume_path: str | None = None, **overrides
) -> JobSpec:
    path, digest = _hash_dataset(dataset)
    return JobSpec(
        kind="sft",
        base_model=base_model,
        dataset_path=volume_path or f"datasets/{path.name}",
        dataset_sha256=digest,
        **overrides,
    )


#: HAC-28: the first DPO run used the SFT LoRA rate (2e-4); chosen log-probs fell to about -60 and
#: the adapter emitted only special tokens. DPO wants a rate one to two orders lower, plus an SFT anchor.
DPO_LEARNING_RATE = 1e-5
DPO_RPO_ALPHA = 1.0


def dpo_spec(
    base_model: str,
    dataset: str | os.PathLike,
    init_adapter: str,
    *,
    volume_path: str | None = None,
    beta: float = 0.1,
    learning_rate: float = DPO_LEARNING_RATE,
    rpo_alpha: float | None = DPO_RPO_ALPHA,
    **overrides,
) -> JobSpec:
    path, digest = _hash_dataset(dataset)
    return JobSpec(
        kind="dpo",
        base_model=base_model,
        dataset_path=volume_path or f"datasets/{path.name}",
        dataset_sha256=digest,
        init_adapter=init_adapter,
        beta=beta,
        learning_rate=learning_rate,
        rpo_alpha=rpo_alpha,
        **overrides,
    )


class TemplateDataError(ValueError):
    """The datasets were realized by the offline ``TemplateClient``, not a language model."""


def check_trainable(summary: Mapping[str, Any]) -> None:
    """Refuse datasets whose ``summary.json`` is offline/template (HAC-28: SFT collapsed to the pool)."""
    model_id = str(summary.get("model_id") or "")
    if summary.get("offline") is not False or not model_id or model_id.startswith("template"):
        raise TemplateDataError(
            f"datasets come from {model_id or 'an unknown client'} (offline={summary.get('offline')!r}); "
            "realize narratives with Muse (python -m daypart.train.run without --offline) before training"
        )


JobStatus = Literal["pending", "running", "succeeded", "failed"]


@dataclass(frozen=True)
class PollResult:
    status: JobStatus
    checkpoint: str | None = None
    adapter: str | None = None
    error: str | None = None


class ModalBackend(Protocol):
    def submit(self, spec: JobSpec, *, resume_from: str | None) -> str: ...

    def poll(self, call_id: str) -> PollResult: ...


@dataclass(frozen=True)
class JobResult:
    status: Literal["succeeded", "failed"]
    adapter: str | None
    attempts: int
    call_ids: tuple[str, ...]
    errors: tuple[str, ...]


def _exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class JobRunner:
    """Submit, poll to a terminal state, and retry failed attempts from the latest checkpoint."""

    def __init__(
        self,
        backend: ModalBackend,
        *,
        max_retries: int = 2,
        poll_interval_s: float = 30.0,
        max_polls: int = 2000,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if max_polls < 1:
            raise ValueError("max_polls must be >= 1")
        if not poll_interval_s >= 0:
            raise ValueError("poll_interval_s must be >= 0")
        self.backend = backend
        self.max_retries = max_retries
        self.poll_interval_s = poll_interval_s
        self.max_polls = max_polls
        self.sleep = sleep

    def run(self, spec: JobSpec) -> JobResult:
        call_ids: list[str] = []
        errors: list[str] = []
        checkpoint: str | None = None
        attempts = 0
        for _ in range(1 + self.max_retries):
            attempts += 1
            try:
                call_id = self.backend.submit(spec, resume_from=checkpoint)
            except Exception as exc:  # backend failure: record, maybe retry
                errors.append(_exc(exc))
                continue
            call_ids.append(call_id)
            error: str | None = None
            adapter: str | None = None
            succeeded = False
            for n in range(1, self.max_polls + 1):
                try:
                    result = self.backend.poll(call_id)
                except Exception as exc:
                    error = _exc(exc)
                    break
                if result.checkpoint is not None:
                    checkpoint = result.checkpoint
                if result.status == "succeeded":
                    if result.adapter is None:
                        error = result.error or f"call {call_id} succeeded without an adapter"
                    else:
                        succeeded, adapter = True, result.adapter
                    break
                if result.status == "failed":
                    error = result.error or f"call {call_id} failed"
                    break
                if n < self.max_polls:
                    self.sleep(self.poll_interval_s)
            else:
                error = f"call {call_id} did not finish within the poll limit ({self.max_polls} polls)"
            if succeeded:
                return JobResult("succeeded", adapter, attempts, tuple(call_ids), tuple(errors))
            errors.append(error or f"call {call_id} failed")
        return JobResult("failed", None, attempts, tuple(call_ids), tuple(errors))


#: Decoder layers of the text tower inside a ``MuseGlimmerForConditionalGeneration`` checkpoint.
LANGUAGE_TOWER_LAYERS = "model.language_model.layers"
_MODULE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def language_lora_targets(target_modules: tuple[str, ...], *, prefix: str = LANGUAGE_TOWER_LAYERS) -> str:
    """PEFT ``target_modules`` regex that matches ``target_modules`` in the language tower only.

    The vision tower also has ``q_proj``/``k_proj``/``v_proj``, so bare names would put
    LoRA on it too. PEFT full-matches a string ``target_modules`` against module names.
    """
    if not target_modules or not all(_MODULE_NAME.match(m) for m in target_modules):
        raise ValueError(f"target_modules must be plain module names, got {target_modules!r}")
    names = "|".join(re.escape(m) for m in target_modules)
    return rf"{re.escape(prefix)}\.\d+\.(?:self_attn|mlp)\.(?:{names})"


def plan_pipeline(
    base_model: str, sft_dataset: str | os.PathLike, dpo_dataset: str | os.PathLike
) -> tuple[JobSpec, Callable[[str], JobSpec]]:
    """SFT spec now; DPO spec only once a finished SFT adapter exists."""
    sft = sft_spec(base_model, sft_dataset)
    _hash_dataset(dpo_dataset)  # fail fast on a missing/empty DPO dataset

    def dpo_from(adapter: str) -> JobSpec:
        return dpo_spec(base_model, dpo_dataset, adapter)

    return sft, dpo_from


__all__ = [
    "DPO_LEARNING_RATE",
    "DPO_RPO_ALPHA",
    "JobResult",
    "JobRunner",
    "JobSpec",
    "JobStatus",
    "LANGUAGE_TOWER_LAYERS",
    "LoraConfig",
    "ModalBackend",
    "PollResult",
    "TemplateDataError",
    "check_trainable",
    "dpo_spec",
    "language_lora_targets",
    "plan_pipeline",
    "sft_spec",
]
