"""One-off generator for the four synthetic demo fixtures (build plan G8, P1 task 4).

Run from ``engine/``: ``uv run python tools/make_fixtures.py``. It writes into
``daypart/demo/fixtures/``: ``scale-photo.png``, ``bp-cuff.jpg``, ``wearable-export.csv`` and
``symptom-diary.json``, a ``<file>.extract.json`` sidecar (the pre-recorded extraction, D-A2) for
each, and ``manifest.json`` (``fixture_id -> {file, sha256, type, size, label, kind}``).

Everything is synthetic: the images are PIL primitives plus the default bitmap font (no photos,
no faces, no real devices) with a "SYNTHETIC DEMO" caption, and dates are relative offsets so the
files stay valid whatever the journal's ``today`` is.

Determinism: PNG output from PIL is deterministic for the same input; the JPEG is saved with an
explicit ``quality`` and ``subsampling`` so it is too (for one Pillow version). Regeneration only
matters for reproducibility, though: the *committed* bytes are what the sha256 allow-list pins, so
if a newer Pillow produces different bytes, commit the new files and manifest together.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "daypart" / "demo" / "fixtures"
METHOD = "fixture-sidecar-v1"
CAPTION = "SYNTHETIC DEMO"


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _scale_png() -> bytes:
    img = Image.new("RGB", (320, 200), (232, 236, 240))
    d = ImageDraw.Draw(img)
    font = _font()
    d.rounded_rectangle((20, 14, 300, 170), radius=24, fill=(70, 78, 90), outline=(40, 44, 52), width=3)
    d.rectangle((80, 50, 240, 110), fill=(196, 222, 170), outline=(30, 34, 40), width=2)
    d.text((112, 66), "72.4 kg", fill=(20, 24, 20), font=font)
    d.text((104, 88), "BATHROOM SCALE", fill=(40, 50, 40), font=font)
    for x in (60, 260):
        d.ellipse((x - 10, 140, x + 10, 160), fill=(110, 118, 130))
    d.text((112, 180), CAPTION, fill=(160, 20, 20), font=font)  # SYNTHETIC DEMO caption
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _bp_jpeg() -> bytes:
    img = Image.new("RGB", (320, 220), (240, 240, 236))
    d = ImageDraw.Draw(img)
    font = _font()
    d.rounded_rectangle((30, 12, 290, 190), radius=18, fill=(250, 250, 250), outline=(60, 60, 60), width=3)
    d.rectangle((60, 30, 260, 150), fill=(200, 214, 200), outline=(40, 40, 40), width=2)
    d.text((80, 46), "SYS 118  mmHg", fill=(10, 10, 10), font=font)
    d.text((80, 78), "DIA 76   mmHg", fill=(10, 10, 10), font=font)
    d.text((80, 110), "PULSE 64", fill=(10, 10, 10), font=font)
    d.ellipse((145, 160, 175, 182), fill=(70, 120, 200))
    d.text((112, 200), CAPTION, fill=(160, 20, 20), font=font)  # SYNTHETIC DEMO caption
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90, subsampling=0, optimize=False, progressive=False)
    return buf.getvalue()


CSV_ROWS = ((-2, 6450, 6.9, 63), (-1, 8120, 7.4, 61), (0, 2310, 6.1, 62))


def _csv() -> bytes:
    lines = ["date_offset_days,steps,sleep_hours,resting_heart_rate"]
    lines += [f"{o},{s},{h},{r}" for o, s, h, r in CSV_ROWS]
    return ("\n".join(lines) + "\n").encode("utf-8")


DIARY = {
    "synthetic": True,
    "entries": [
        {"date_offset_days": 0, "pain_0_10": 3, "note": "knee"},
        {"date_offset_days": -1, "phq2_total": 1},
    ],
}


def _json() -> bytes:
    return (json.dumps(DIARY, indent=2) + "\n").encode("utf-8")


def _item(code: str | None, value: float | None, unit: str | None, offset: int, snippet: str,
          confidence: str, display: str | None = None) -> dict:
    item = {"concept_code": code, "value": value, "unit": unit, "date_offset_days": offset,
            "snippet": snippet, "confidence": confidence}
    if display is not None:
        item["display"] = display
    return item


# fixture_id -> (file, type, kind, label, items)
FIXTURES: dict[str, tuple[str, str, str, str, list[dict]]] = {
    "demo-scale-photo": (
        "scale-photo.png", "png", "image_fixture", "Demo image: bathroom scale",
        [_item("29463-7", 72.4, "kg", 0, "display: 72.4 kg", "high")],
    ),
    "demo-bp-cuff": (
        "bp-cuff.jpg", "jpeg", "image_fixture", "Demo image: blood pressure monitor",
        [
            _item("8480-6", 118.0, "mm[Hg]", 0, "display: SYS 118 mmHg", "high"),
            _item("8462-4", 76.0, "mm[Hg]", 0, "display: DIA 76 mmHg", "high"),
            # The display shows no unit next to PULSE: the concept default is assumed (unit_assumed).
            _item("8867-4", 64.0, None, 0, "display: PULSE 64", "medium"),
        ],
    ),
    # One reading per metric (two values for one concept in one check-in would be a contradiction):
    # yesterday's steps (today's day is not over), last night's sleep, today's resting heart rate.
    "demo-wearable-csv": (
        "wearable-export.csv", "csv", "file_fixture", "Demo file: 3-day wearable export (CSV)",
        [
            _item("41950-7", 8120.0, "{steps}/(24.h)", -1, "row -1: steps=8120", "high"),
            _item("93832-4", 6.1, "h", 0, "row 0: sleep_hours=6.1", "high"),
            # The export's column has no unit: /min is assumed (unit_assumed).
            _item("8867-4", 62.0, None, 0, "row 0: resting_heart_rate=62", "medium"),
        ],
    ),
    "demo-symptom-json": (
        "symptom-diary.json", "json", "file_fixture", "Demo file: symptom diary (JSON)",
        [
            _item("72514-3", 3.0, "{score}", 0, 'entries[0]: "pain_0_10": 3', "high"),
            _item(None, None, None, 0, 'entries[0]: "note": "knee"', "low", display="knee"),
            _item("55758-7", 1.0, "{score}", -1, 'entries[1]: "phq2_total": 1', "high"),
        ],
    ),
}

_BUILDERS = {"png": _scale_png, "jpeg": _bp_jpeg, "csv": _csv, "json": _json}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for fid, (name, ftype, kind, label, items) in FIXTURES.items():
        data = _BUILDERS[ftype]()
        (OUT / name).write_bytes(data)
        sidecar = {"fixture_id": fid, "method": METHOD, "synthetic": True,
                   "note": "Pre-recorded extraction of a synthetic demo fixture.", "items": items}
        (OUT / f"{name}.extract.json").write_text(json.dumps(sidecar, indent=2) + "\n", "utf-8")
        manifest[fid] = {"file": name, "sha256": hashlib.sha256(data).hexdigest(), "type": ftype,
                         "size": len(data), "label": label, "kind": kind}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"wrote {len(manifest)} fixtures to {OUT}")


if __name__ == "__main__":
    main()
