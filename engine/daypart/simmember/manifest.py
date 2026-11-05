"""A4 — the I1 manifest (held-out ledger) and the curated leak guard.

``write`` persists the cohort as canonical JSONL plus ``manifest.json``;
``assert_no_curated`` is the guard every training-data path runs before use.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pydantic

from daypart.simmember.cohort import Cohort
from daypart.simmember.latents import Latents

GENERATOR_VERSION = "simmember-1.0.0"

_CURATED_ID_RE = re.compile(r"\bcur-\d{3}\b")


class CuratedLeakError(Exception):
    """A curated (held-out) member, or its latents, reached a non-held-out path (I1)."""


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def latent_hash(latents: Latents | Mapping[str, Any]) -> str:
    """sha256 hex of the canonical JSON of ``latents.model_dump(mode="json")``."""
    if isinstance(latents, Latents):
        payload = latents.model_dump(mode="json")
    else:
        payload = dict(latents)
    return _sha256(canonical_json(payload).encode("utf-8"))


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(r) + "\n" for r in rows).encode("utf-8")


def write(cohort: Cohort, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles_bytes = _jsonl(p.model_dump(mode="json") for p in cohort.profiles)
    traj_bytes = _jsonl(
        cohort.trajectories[p.member_id].model_dump(mode="json") for p in cohort.profiles
    )
    curated = cohort.heldout_profiles()
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": cohort.seed,
        "counts": {"curated": len(curated), "synthetic": len(cohort.training_profiles())},
        "curated_ids": sorted(p.member_id for p in curated),
        "curated_latent_hashes": {p.member_id: latent_hash(p.latents) for p in curated},
        "files": {
            "profiles.jsonl": _sha256(profiles_bytes),
            "trajectories.jsonl": _sha256(traj_bytes),
        },
    }
    (out_dir / "profiles.jsonl").write_bytes(profiles_bytes)
    (out_dir / "trajectories.jsonl").write_bytes(traj_bytes)
    (out_dir / "manifest.json").write_bytes((canonical_json(manifest) + "\n").encode("utf-8"))
    return manifest


def verify_manifest(out_dir: Path) -> dict:
    """Re-hash the files. Profiles mismatch → ``CuratedLeakError``; other → ``ValueError``."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {"profiles.jsonl", "trajectories.jsonl"}:
        raise ValueError("manifest 'files' must list exactly profiles.jsonl and trajectories.jsonl")
    for name in ("profiles.jsonl", "trajectories.jsonl"):
        path = out_dir / name
        actual = _sha256(path.read_bytes()) if path.is_file() else None
        if actual != files[name]:
            msg = f"{name}: hash mismatch (manifest {files[name]}, actual {actual})"
            if name == "profiles.jsonl":
                raise CuratedLeakError(msg)
            raise ValueError(msg)
    return manifest


def _latent_hash_candidates(value: Any) -> set[str]:
    """Hashes of a ``latents`` mapping as-given and Latents-normalized (e.g. 0 vs 0.0)."""
    if isinstance(value, Latents):
        return {latent_hash(value)}
    out = {latent_hash(value)}
    try:
        out.add(latent_hash(Latents.model_validate(dict(value))))
    except (pydantic.ValidationError, TypeError, ValueError):
        pass
    return out


def assert_no_curated(records: Iterable[Mapping[str, Any]], manifest: Mapping[str, Any]) -> None:
    """Raise ``CuratedLeakError`` if any record leaks a curated id or curated latents."""
    curated_ids = tuple(manifest.get("curated_ids", ()))
    curated_hashes = set(dict(manifest.get("curated_latent_hashes", {})).values())

    def check_str(s: str) -> None:
        if _CURATED_ID_RE.search(s) or any(cid and cid in s for cid in curated_ids):
            raise CuratedLeakError(f"curated id found in record string {s[:80]!r}")

    def walk(node: Any) -> None:
        if isinstance(node, str):
            check_str(node)
        elif isinstance(node, pydantic.BaseModel):
            walk(node.model_dump(mode="json"))
        elif isinstance(node, Mapping):
            for key, val in node.items():
                if isinstance(key, str):
                    check_str(key)
                if key == "latents" and isinstance(val, (Mapping, Latents)):
                    if _latent_hash_candidates(val) & curated_hashes:
                        raise CuratedLeakError("record carries curated latents (relabeled profile)")
                walk(val)
        elif isinstance(node, (list, tuple, set, frozenset)):
            for item in node:
                walk(item)

    for record in records:
        walk(record)


__all__ = [
    "GENERATOR_VERSION",
    "CuratedLeakError",
    "assert_no_curated",
    "canonical_json",
    "latent_hash",
    "verify_manifest",
    "write",
]
