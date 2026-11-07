"""Fixture intake: the sha256 allow-list, size and type checks, sidecar proposals (A&D D-A2, G8).

Only the four bundled synthetic fixtures under ``daypart/demo/fixtures/`` are ever accepted. An
upload is checked in a fixed order: size (> :data:`MAX_UPLOAD_BYTES` → :class:`TooLarge`), magic
bytes (:func:`sniff_type`; anything but PNG/JPEG/CSV/JSON → :class:`UnsupportedType`), then the
sha256 allow-list (→ :class:`NotADemoFixture`). The file name and extension are never consulted.

Proposals come from each fixture's pre-recorded ``<file>.extract.json`` sidecar, never from a
runtime model; their flags come from the same :func:`~daypart.checkin.extract_text.detect_flags`
the text path uses.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Mapping
from functools import cache
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from types import MappingProxyType
from typing import Any

from daypart.checkin.extract_text import detect_flags
from daypart.checkin.models import METHOD_FIXTURE, SNIPPET_MAX, Proposal
from daypart.memory.registry import load_registry

MAX_UPLOAD_BYTES = 1024 * 1024

TYPES = ("png", "jpeg", "csv", "json")
KINDS = ("image_fixture", "file_fixture")
MEDIA_TYPES: Mapping[str, str] = MappingProxyType(
    {"png": "image/png", "jpeg": "image/jpeg", "csv": "text/csv; charset=utf-8", "json": "application/json"}
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_BOM = b"\xef\xbb\xbf"


class FixtureError(Exception):
    """Base for intake errors; ``error`` is the API error code."""

    error = "fixture_error"


class TooLarge(FixtureError):
    error = "too_large"


class UnsupportedType(FixtureError):
    error = "unsupported_type"


class NotADemoFixture(FixtureError):
    error = "not_a_demo_fixture"


class UnknownFixture(FixtureError):
    error = "unknown_fixture"


def _utf8(data: bytes) -> str | None:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _is_json(data: bytes) -> bool:
    body = data.removeprefix(_BOM)
    if body.lstrip()[:1] not in (b"{", b"["):
        return False
    text = _utf8(body)
    if text is None:
        return False
    try:
        json.loads(text)
    except (ValueError, RecursionError):  # deep nesting raises RecursionError: refuse, never crash
        return False
    return True


def sniff_type(data: bytes) -> str | None:
    """``"png" | "jpeg" | "json" | "csv"`` from the content alone, else None (PDF included)."""
    if data.startswith(_PNG_MAGIC):
        return "png"
    if data.startswith(_JPEG_MAGIC):
        return "jpeg"
    if _is_json(data):
        return "json"
    if b"\x00" in data:
        return None
    text = _utf8(data.removeprefix(_BOM))
    if not text:
        return None
    if "," in text.splitlines()[0]:
        return "csv"
    return None


def _fixtures_root() -> Traversable:
    return resources.files("daypart.demo").joinpath("fixtures")


def load_manifest(root: Traversable | Path) -> Mapping[str, Mapping[str, Any]]:
    """Read and verify ``manifest.json`` under ``root``; any drift raises (fail loudly)."""
    raw = json.loads(root.joinpath("manifest.json").read_text("utf-8"))
    out: dict[str, Mapping[str, Any]] = {}
    seen: set[str] = set()
    for fid, entry in raw.items():
        missing = {"file", "sha256", "type", "size", "label", "kind"} - set(entry)
        if missing:
            raise ValueError(f"manifest {fid!r}: missing {sorted(missing)}")
        if entry["type"] not in TYPES or entry["kind"] not in KINDS:
            raise ValueError(f"manifest {fid!r}: bad type/kind {entry['type']!r}/{entry['kind']!r}")
        path = root.joinpath(entry["file"])
        if not path.is_file():
            raise ValueError(f"manifest {fid!r}: file {entry['file']!r} is missing")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"manifest {fid!r}: sha256 of {entry['file']!r} is {digest}, manifest says {entry['sha256']}")
        if len(data) != entry["size"]:
            raise ValueError(f"manifest {fid!r}: size {len(data)} != {entry['size']}")
        if len(data) > MAX_UPLOAD_BYTES or sniff_type(data) != entry["type"]:
            raise ValueError(f"manifest {fid!r}: {entry['file']!r} would not pass intake")
        if digest in seen:
            raise ValueError(f"manifest {fid!r}: duplicate sha256")
        seen.add(digest)
        out[fid] = MappingProxyType(dict(entry))
    return MappingProxyType(out)


@cache
def manifest() -> Mapping[str, Mapping[str, Any]]:
    """The bundled, verified manifest: ``fixture_id -> {file, sha256, type, size, label, kind}``."""
    return load_manifest(_fixtures_root())


@cache
def _by_sha256() -> Mapping[str, str]:
    return MappingProxyType({e["sha256"]: fid for fid, e in manifest().items()})


def _entry(fixture_id: str) -> Mapping[str, Any]:
    entry = manifest().get(fixture_id)
    if entry is None:
        raise UnknownFixture(f"unknown fixture {fixture_id!r}")
    return entry


def fixtures_list() -> list[dict[str, str]]:
    """The picker view: ``[{fixture_id, label, kind, type}]`` in manifest order."""
    return [{"fixture_id": fid, "label": e["label"], "kind": e["kind"], "type": e["type"]}
            for fid, e in manifest().items()]


def fixture_bytes(fixture_id: str) -> tuple[bytes, str]:
    """``(bytes, media_type)`` of a bundled fixture; :class:`UnknownFixture` if it is not one."""
    entry = _entry(fixture_id)
    return _fixtures_root().joinpath(entry["file"]).read_bytes(), MEDIA_TYPES[entry["type"]]


def intake_upload(data: bytes) -> str:
    """The ``fixture_id`` of an uploaded body: size, then magic bytes, then the sha256 allow-list."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise TooLarge(f"upload is {len(data)} bytes; the limit is {MAX_UPLOAD_BYTES}")
    if sniff_type(data) is None:
        raise UnsupportedType("only PNG, JPEG, CSV and JSON are accepted")
    fid = _by_sha256().get(hashlib.sha256(data).hexdigest())
    if fid is None:
        raise NotADemoFixture("only the bundled demo fixtures are accepted")
    return fid


@cache
def _sidecar(fixture_id: str) -> Mapping[str, Any]:
    entry = _entry(fixture_id)
    raw = json.loads(_fixtures_root().joinpath(f"{entry['file']}.extract.json").read_text("utf-8"))
    if raw.get("fixture_id") != fixture_id or raw.get("method") != METHOD_FIXTURE:
        raise ValueError(f"sidecar for {fixture_id!r} does not match the manifest or method")
    return raw


def proposals_for(fixture_id: str, today: dt.date, *, checkin_id: str) -> tuple[Proposal, ...]:
    """Proposals ``p-1``, ``p-2``, ... from the fixture's sidecar, flags set by ``detect_flags``.

    ``checkin_id`` is accepted for symmetry with text extraction; ``source_ref`` is the fixture id.
    A null sidecar unit on a known concept takes the concept's default unit and is marked
    ``unit_assumed``, exactly as rules-v1 does for a number with no unit.
    """
    del checkin_id
    entry = _entry(fixture_id)
    reg = load_registry()
    props: list[Proposal] = []
    for i, item in enumerate(_sidecar(fixture_id)["items"], start=1):
        code = item["concept_code"]
        concept = reg.by_code(code) if code is not None else None
        unit = item["unit"]
        flags: tuple[str, ...] = ()
        if concept is not None and unit is None and item["value"] is not None:
            unit, flags = concept.unit, ("unit_assumed",)
        value = item["value"]
        props.append(
            Proposal(
                proposal_id=f"p-{i}",
                concept_code=code,
                display=concept.display if concept is not None else (item.get("display") or code or "unknown"),
                value=float(value) if value is not None else None,
                unit=unit,
                date=today + dt.timedelta(days=int(item["date_offset_days"])),
                source_kind=entry["kind"],
                source_ref=fixture_id,
                snippet=item["snippet"][:SNIPPET_MAX],
                method=METHOD_FIXTURE,
                confidence=item["confidence"],
                flags=flags,  # type: ignore[arg-type]
            )
        )
    return detect_flags(props)


__all__ = [
    "MAX_UPLOAD_BYTES",
    "MEDIA_TYPES",
    "FixtureError",
    "NotADemoFixture",
    "TooLarge",
    "UnknownFixture",
    "UnsupportedType",
    "fixture_bytes",
    "fixtures_list",
    "intake_upload",
    "load_manifest",
    "manifest",
    "proposals_for",
    "sniff_type",
]
