"""G2 — summaries of the Modal training run record (``docs/daypart/results/training.json``).

Pure: the record is the dict ``modal_app._finish`` returns per stage (``sft``, ``dpo``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

STAGES: tuple[str, ...] = ("sft", "dpo")


def _stage(record: Mapping[str, Any], model: str) -> Mapping[str, Any]:
    if model not in STAGES:
        raise ValueError(f"model {model!r} not in {STAGES}")
    return record[model]  # KeyError propagates


def _final_log(stage: Mapping[str, Any]) -> Mapping[str, Any]:
    for entry in reversed(stage.get("log_tail") or []):
        if "train_loss" in entry:
            return entry
    return {}


def training_summary(record: Mapping[str, Any], model: str) -> dict:
    stage = _stage(record, model)
    final = _final_log(stage)
    out = {
        "adapter": stage["adapter"],
        "steps": stage["steps"],
        "rows": stage["rows"],
        "train_minutes": round(stage["train_s"] / 60, 1),
        "load_s": stage["load_s"],
        "peak_gpu_gb": stage["peak_gpu_gb"],
        "train_loss": final.get("train_loss"),
    }
    if model == "dpo":
        out["reward_margin"] = final.get("rewards/margins")
        out["reward_accuracy"] = final.get("rewards/accuracies")
    return out


def loss_curve(record: Mapping[str, Any], model: str) -> list[dict]:
    stage = _stage(record, model)
    points = [
        {"step": e["step"], "loss": e["loss"]}
        for e in stage.get("log_tail") or []
        if "step" in e and "loss" in e
    ]
    return sorted(points, key=lambda p: p["step"])


__all__ = ["STAGES", "loss_curve", "training_summary"]
