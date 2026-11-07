"""P1 task 4 — fixture intake (A&D D-A2, "Fixture intake", upload failure rows; T3.1, T3.2).

Intake level only: the sha256 allow-list, the 1 MiB cap, magic-byte sniffing and the sidecar
proposals. Every fixture and every test body here is synthetic.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from datetime import timedelta
from importlib import resources
from pathlib import Path

import pytest
from PIL import Image

from daypart.checkin import fixtures as fx
from daypart.checkin.extract_text import detect_flags
from daypart.checkin.models import METHOD_FIXTURE, SNIPPET_MAX
from daypart.memory.registry import load_registry
from daypart.simmember.trajectory import START_DATE

TODAY = (START_DATE + timedelta(days=28)).date()  # G9
FIXTURE_IDS = ("demo-scale-photo", "demo-bp-cuff", "demo-wearable-csv", "demo-symptom-json")
UNIT_FLAGS = {"unit_assumed", "unit_mismatch"}
GENERATOR = Path(__file__).resolve().parents[2] / "tools" / "make_fixtures.py"


def _fixtures_dir():
    return resources.files("daypart.demo").joinpath("fixtures")


def _sidecar(fixture_id: str) -> dict:
    entry = fx.manifest()[fixture_id]
    return json.loads(_fixtures_dir().joinpath(entry["file"] + ".extract.json").read_text("utf-8"))


def _png(size: tuple[int, int] = (8, 8), color: tuple[int, int, int] = (10, 200, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- manifest


def test_manifest_lists_the_four_fixtures_with_required_fields() -> None:
    m = fx.manifest()
    assert set(m) == set(FIXTURE_IDS)
    kinds = {fid: e["kind"] for fid, e in m.items()}
    assert kinds["demo-scale-photo"] == kinds["demo-bp-cuff"] == "image_fixture"
    assert kinds["demo-wearable-csv"] == kinds["demo-symptom-json"] == "file_fixture"
    types = {e["type"] for e in m.values()}
    assert types == {"png", "jpeg", "csv", "json"}
    for e in m.values():
        assert set(e) >= {"file", "sha256", "type", "size", "label", "kind"}
        data = _fixtures_dir().joinpath(e["file"]).read_bytes()
        assert len(data) == e["size"] <= fx.MAX_UPLOAD_BYTES
        assert hashlib.sha256(data).hexdigest() == e["sha256"]
        assert fx.sniff_type(data) == e["type"]


def test_fixtures_list_is_the_picker_view() -> None:
    rows = fx.fixtures_list()
    assert {r["fixture_id"] for r in rows} == set(FIXTURE_IDS)
    for r in rows:
        assert set(r) == {"fixture_id", "label", "kind", "type"}
        assert r["label"]


def test_fixture_bytes_and_media_type() -> None:
    data, media = fx.fixture_bytes("demo-scale-photo")
    assert media == "image/png" and data.startswith(b"\x89PNG\r\n\x1a\n")
    assert fx.fixture_bytes("demo-bp-cuff")[1] == "image/jpeg"
    assert fx.fixture_bytes("demo-wearable-csv")[1].startswith("text/csv")
    assert fx.fixture_bytes("demo-symptom-json")[1] == "application/json"
    with pytest.raises(fx.UnknownFixture) as e:
        fx.fixture_bytes("demo-nope")
    assert e.value.error == "unknown_fixture"


def test_manifest_load_fails_loudly_on_sha256_drift(tmp_path: Path) -> None:
    src = Path(str(_fixtures_dir()))
    dst = tmp_path / "fixtures"
    shutil.copytree(src, dst)
    assert fx.load_manifest(dst) == fx.manifest()  # an untouched copy loads
    target = dst / fx.manifest()["demo-wearable-csv"]["file"]
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="sha256"):
        fx.load_manifest(dst)


def test_manifest_load_fails_on_missing_file(tmp_path: Path) -> None:
    dst = tmp_path / "fixtures"
    shutil.copytree(Path(str(_fixtures_dir())), dst)
    (dst / fx.manifest()["demo-bp-cuff"]["file"]).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        fx.load_manifest(dst)


# ---------------------------------------------------------------- sniff_type


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 16, "png"),
        (b"\xff\xd8\xff\xe0" + b"\x00" * 16, "jpeg"),
        (b'{"synthetic": true}', "json"),
        (b"  \n[1, 2, 3]\n", "json"),
        (b"\xef\xbb\xbf{\"a\": 1}", "json"),
        (b"a,b,c\n1,2,3\n", "csv"),
        (b"date_offset_days,steps\n-1,8000\n", "csv"),
        (b"%PDF-1.4\n%synthetic\n", None),
        (b"", None),
        (b"just some words\nno commas here\n", None),
        (b"a,b\x00,c\n", None),  # NUL bytes: not text
        (b"a,b\n\xff\xfe\n", None),  # not UTF-8
        (b'{"broken": ', None),  # starts like JSON, does not parse, no comma line: neither
        (b"GIF89a" + b"\x00" * 8, None),
    ],
)
def test_sniff_type(data: bytes, expected: str | None) -> None:
    assert fx.sniff_type(data) == expected


def test_sniff_type_valid_json_with_commas_is_json_not_csv() -> None:
    assert fx.sniff_type(b'{"a": 1, "b": 2}') == "json"
    assert fx.sniff_type(b"[1,2,3]") == "json"


# ---------------------------------------------------------------- intake (T3.2)


def test_every_bundled_fixture_passes_intake_back_to_its_id() -> None:
    for fid in FIXTURE_IDS:
        data, _ = fx.fixture_bytes(fid)
        assert fx.intake_upload(data) == fid


def test_unknown_png_is_not_a_demo_fixture() -> None:
    data = _png()
    assert fx.sniff_type(data) == "png"
    with pytest.raises(fx.NotADemoFixture) as e:
        fx.intake_upload(data)
    assert e.value.error == "not_a_demo_fixture"


def test_one_byte_altered_fixture_is_not_a_demo_fixture() -> None:
    data, _ = fx.fixture_bytes("demo-scale-photo")
    with pytest.raises(fx.NotADemoFixture):
        fx.intake_upload(data + b"\x00")


def test_too_large_is_checked_before_sniffing() -> None:
    body = b"\x89PNG\r\n\x1a\n" + b"\x00" * (fx.MAX_UPLOAD_BYTES + 1 - 8)
    assert len(body) == fx.MAX_UPLOAD_BYTES + 1
    with pytest.raises(fx.TooLarge) as e:
        fx.intake_upload(body)
    assert e.value.error == "too_large"
    pdf = b"%PDF-1.4\n" + b"0" * fx.MAX_UPLOAD_BYTES  # unsupported AND too large: size wins
    with pytest.raises(fx.TooLarge):
        fx.intake_upload(pdf)


def test_exactly_max_bytes_is_not_too_large() -> None:
    body = b"\x89PNG\r\n\x1a\n" + b"\x00" * (fx.MAX_UPLOAD_BYTES - 8)
    with pytest.raises(fx.NotADemoFixture):
        fx.intake_upload(body)


def test_pdf_is_unsupported_type() -> None:
    with pytest.raises(fx.UnsupportedType) as e:
        fx.intake_upload(b"%PDF-1.4\n1 0 obj\n<< /Synthetic true >>\nendobj\n%%EOF\n")
    assert e.value.error == "unsupported_type"


def test_unlisted_json_and_csv_are_not_demo_fixtures() -> None:
    with pytest.raises(fx.NotADemoFixture):
        fx.intake_upload(b'{"synthetic": true, "entries": []}')
    with pytest.raises(fx.NotADemoFixture):
        fx.intake_upload(b"date_offset_days,steps,sleep_hours,resting_heart_rate\n0,1,2,3\n")


def test_errors_share_a_base_with_the_error_code() -> None:
    codes = {
        fx.TooLarge: "too_large",
        fx.UnsupportedType: "unsupported_type",
        fx.NotADemoFixture: "not_a_demo_fixture",
        fx.UnknownFixture: "unknown_fixture",
    }
    for cls, code in codes.items():
        assert cls.error == code
        assert issubclass(cls, Exception)


# ---------------------------------------------------------------- proposals (T3.1)


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_each_fixture_yields_sidecar_proposals(fixture_id: str) -> None:
    props = fx.proposals_for(fixture_id, TODAY, checkin_id="c-1")
    kind = fx.manifest()[fixture_id]["kind"]
    assert len(props) >= 1
    assert [p.proposal_id for p in props] == [f"p-{i}" for i in range(1, len(props) + 1)]
    for p in props:
        assert p.method == METHOD_FIXTURE == "fixture-sidecar-v1"
        assert p.source_ref == fixture_id
        assert p.source_kind == kind
        assert p.date is not None and TODAY - timedelta(days=28) <= p.date <= TODAY
        assert len(p.snippet) <= SNIPPET_MAX
        assert p.blocking == bool(p.flags)


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_each_fixture_proposal_is_editable_through_correct(fixture_id: str) -> None:
    reg = load_registry()
    props = fx.proposals_for(fixture_id, TODAY, checkin_id="c-1")
    for p in props:
        concept = reg.by_code(p.concept_code) if p.concept_code else None
        unit = concept.unit if concept else None
        value = p.value if p.value is not None else (concept.plausible[0] if concept else 1.0)
        edited = [q.with_changes(value=value, unit=unit) if q.proposal_id == p.proposal_id else q for q in props]
        out = detect_flags(edited, corrected={p.proposal_id: frozenset({"value", "unit"})})
        mine = next(q for q in out if q.proposal_id == p.proposal_id)
        assert not UNIT_FLAGS & set(mine.flags), (fixture_id, mine)


def test_flags_match_text_detection() -> None:
    """Re-running detect_flags over a fixture's proposals changes nothing: one detector."""
    for fid in FIXTURE_IDS:
        props = fx.proposals_for(fid, TODAY, checkin_id="c-1")
        assert detect_flags(props) == props


def test_scale_photo_is_fully_clean() -> None:
    props = fx.proposals_for("demo-scale-photo", TODAY, checkin_id="c-1")
    assert props
    for p in props:
        assert p.flags == () and not p.blocking and p.confidence == "high"
    weight = next(p for p in props if p.concept_code == "29463-7")
    assert (weight.value, weight.unit, weight.date) == (72.4, "kg", TODAY)
    assert weight.display == load_registry().by_code("29463-7").display


def test_bp_cuff_values_and_assumed_pulse_unit() -> None:
    props = {p.concept_code: p for p in fx.proposals_for("demo-bp-cuff", TODAY, checkin_id="c-1")}
    assert props["8480-6"].value == 118 and props["8480-6"].flags == ()
    assert props["8462-4"].value == 76 and props["8462-4"].flags == ()
    pulse = props["8867-4"]
    assert pulse.value == 64 and pulse.unit == "/min"  # the concept default, assumed
    assert pulse.flags == ("unit_assumed",) and pulse.blocking


def test_symptom_diary_unknown_note_gets_unknown_code() -> None:
    props = fx.proposals_for("demo-symptom-json", TODAY, checkin_id="c-1")
    knee = next(p for p in props if p.concept_code is None)
    assert knee.display == "knee"
    assert "unknown_code" in knee.flags and knee.blocking
    phq2 = next(p for p in props if p.concept_code == "55758-7")
    assert phq2.date == TODAY - timedelta(days=1)


def test_dates_are_today_plus_offset() -> None:
    for fid in FIXTURE_IDS:
        items = _sidecar(fid)["items"]
        props = fx.proposals_for(fid, TODAY, checkin_id="c-1")
        assert [p.date for p in props] == [TODAY + timedelta(days=i["date_offset_days"]) for i in items]


def test_at_least_one_blocking_and_one_clean_fixture() -> None:
    blocking = [fid for fid in FIXTURE_IDS if any(p.blocking for p in fx.proposals_for(fid, TODAY, checkin_id="c"))]
    clean = [fid for fid in FIXTURE_IDS if not any(p.blocking for p in fx.proposals_for(fid, TODAY, checkin_id="c"))]
    assert blocking and clean


def test_proposals_for_unknown_fixture_raises() -> None:
    with pytest.raises(fx.UnknownFixture):
        fx.proposals_for("demo-nope", TODAY, checkin_id="c-1")


# ---------------------------------------------------------------- sidecars and synthetic marking


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_sidecar_shape(fixture_id: str) -> None:
    sc = _sidecar(fixture_id)
    assert sc["fixture_id"] == fixture_id
    assert sc["method"] == METHOD_FIXTURE
    assert sc["synthetic"] is True
    assert sc["items"]
    for item in sc["items"]:
        assert set(item) >= {"concept_code", "value", "unit", "date_offset_days", "snippet", "confidence"}
        assert len(item["snippet"]) <= SNIPPET_MAX
        assert item["confidence"] in ("high", "medium", "low")


def test_fixtures_are_marked_synthetic() -> None:
    diary = _fixtures_dir().joinpath(fx.manifest()["demo-symptom-json"]["file"]).read_text("utf-8")
    assert json.loads(diary)["synthetic"] is True
    for fid in FIXTURE_IDS:
        raw = _fixtures_dir().joinpath(fx.manifest()[fid]["file"] + ".extract.json").read_text("utf-8")
        assert "synthetic" in raw.lower()
    src = GENERATOR.read_text("utf-8")
    assert src.count("SYNTHETIC DEMO") >= 2  # the caption drawn on both images


def test_csv_uses_relative_offsets() -> None:
    text = _fixtures_dir().joinpath(fx.manifest()["demo-wearable-csv"]["file"]).read_text("utf-8")
    lines = text.strip().splitlines()
    assert lines[0] == "date_offset_days,steps,sleep_hours,resting_heart_rate"
    assert [ln.split(",")[0] for ln in lines[1:]] == ["-2", "-1", "0"]
