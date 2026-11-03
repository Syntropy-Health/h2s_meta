import json
import subprocess
import sys
from pathlib import Path

import pytest

from daypart.simmember.cohort import CURATED_PROFILES, generate
from daypart.simmember.manifest import (
    CuratedLeakError,
    assert_no_curated,
    latent_hash,
    verify_manifest,
    write,
)


@pytest.fixture(scope="module")
def cohort0():
    return generate(0)


@pytest.fixture()
def written(tmp_path, cohort0):
    return tmp_path, write(cohort0, tmp_path)


def test_write_shape_and_canonical(written, cohort0):
    out, manifest = written
    on_disk = (out / "manifest.json").read_text()
    assert json.loads(on_disk) == manifest
    assert on_disk == json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    assert manifest["counts"] == {"curated": 25, "synthetic": 175}
    assert manifest["seed"] == 0 and isinstance(manifest["generator_version"], str)
    assert manifest["curated_ids"] == sorted(p.member_id for p in CURATED_PROFILES)
    assert manifest["curated_latent_hashes"] == {p.member_id: latent_hash(p.latents) for p in CURATED_PROFILES}
    lines = (out / "profiles.jsonl").read_text().splitlines(keepends=True)
    assert len(lines) == 200 and all(l.endswith("\n") for l in lines)
    first = json.loads(lines[0])
    assert first == cohort0.profiles[0].model_dump(mode="json")
    assert lines[0] == json.dumps(first, sort_keys=True, separators=(",", ":")) + "\n"
    traj = [json.loads(l) for l in (out / "trajectories.jsonl").read_text().splitlines()]
    assert [t["member_id"] for t in traj] == [p.member_id for p in cohort0.profiles]
    assert verify_manifest(out) == manifest


def test_write_byte_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write(generate(4), a)
    write(generate(4), b)
    for name in ("profiles.jsonl", "trajectories.jsonl", "manifest.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_verify_manifest_tamper(written):
    out, _ = written
    p = out / "profiles.jsonl"
    original = p.read_bytes()
    p.write_bytes(original + b"\n")
    with pytest.raises(CuratedLeakError):
        verify_manifest(out)
    p.write_bytes(original)
    t = out / "trajectories.jsonl"
    t.write_bytes(t.read_bytes()[:-2] + b"\n")
    with pytest.raises(ValueError):
        verify_manifest(out)


def test_latent_hash_definition():
    import hashlib

    lat = CURATED_PROFILES[3].latents
    expect = hashlib.sha256(
        json.dumps(lat.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert latent_hash(lat) == expect


def test_assert_no_curated(written, cohort0):
    _, manifest = written
    syn_rows = [p.model_dump(mode="json") for p in cohort0.training_profiles()]
    assert assert_no_curated(syn_rows, manifest) is None
    assert assert_no_curated([], manifest) is None
    assert assert_no_curated(iter([{"a": [{"b": "syn-001 cure-0011 cur-01"}]}]), manifest) is None
    with pytest.raises(CuratedLeakError):  # containment of a manifest id is fail-closed
        assert_no_curated([{"b": "xcur-001x"}], manifest)

    leaks = [
        {"messages": [{"content": "hello cur-007 how are you"}]},
        {"deep": {"x": [1, 2, {"y": ("cur-025",)}]}},
        {"cur-003": 1},
    ]
    for rec in leaks:
        with pytest.raises(CuratedLeakError):
            assert_no_curated([syn_rows[0], rec], manifest)

    relabeled = CURATED_PROFILES[4].model_dump(mode="json")
    relabeled["member_id"] = "syn-999"
    relabeled["cohort"] = "synthetic"
    with pytest.raises(CuratedLeakError):
        assert_no_curated([{"wrapper": {"profile": relabeled}}], manifest)
    # int-vs-float normalisation still caught
    lat = dict(relabeled["latents"])
    if lat["affect_baseline"] == 0.0:
        lat["affect_baseline"] = 0
    with pytest.raises(CuratedLeakError):
        assert_no_curated([{"latents": lat}], manifest)


def test_cli(tmp_path):
    engine = Path(__file__).resolve().parents[2]
    out = tmp_path / "cli"
    res = subprocess.run([sys.executable, "-m", "daypart.simmember", "--seed", "0", "--out", str(out)],
                         cwd=engine, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    ref = tmp_path / "ref"
    write(generate(0), ref)
    for name in ("profiles.jsonl", "trajectories.jsonl", "manifest.json"):
        assert (out / name).read_bytes() == (ref / name).read_bytes()
