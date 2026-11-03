"""Sealed challenge suite — slice A: I1 manifest & curated leak guard.

Spec: docs/daypart/contracts/CONTRACT-slice-A-simmember.md section A4.

Ambiguities resolved (questions for the principal):
- "lines end with \\n" is applied strictly to the two JSONL files (every record
  line, including the last, is newline-terminated). For manifest.json the test
  accepts the canonical JSON with or without one trailing newline.
- File hashes are taken as sha256 hex of the exact file bytes on disk.
- The spec declares `class CuratedLeakError(Exception)`; a trajectories.jsonl
  hash mismatch must raise ValueError and is asserted NOT to be reported as a
  CuratedLeakError (the two failure classes are distinct).
"""

from __future__ import annotations

import hashlib
import json

import pytest

pytestmark = pytest.mark.category("integration/manifest")


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture()
def written(cohort0, tmp_path):
    from daypart.simmember.manifest import write

    out = tmp_path / "out"
    out.mkdir()
    manifest = write(cohort0, out)
    return out, manifest


class TestWrite:
    pytestmark = pytest.mark.category("integration/manifest-write")

    def test_three_files(self, written):
        out, _ = written
        assert {p.name for p in out.iterdir()} >= {"profiles.jsonl", "trajectories.jsonl", "manifest.json"}

    def test_returned_manifest_equals_file(self, written):
        out, m = written
        assert json.loads((out / "manifest.json").read_text()) == m

    def test_manifest_json_canonical(self, written):
        out, m = written
        text = (out / "manifest.json").read_text()
        assert text.rstrip("\n") == _canon(m)

    def test_manifest_fields(self, written, cohort0):
        _, m = written
        assert set(m) >= {"generator_version", "seed", "counts", "curated_ids", "curated_latent_hashes", "files"}
        assert isinstance(m["generator_version"], str) and m["generator_version"]
        assert m["seed"] == 0
        assert m["counts"] == {"curated": 25, "synthetic": 175}
        assert m["curated_ids"] == sorted(f"cur-{i:03d}" for i in range(1, 26))
        assert set(m["curated_latent_hashes"]) == set(m["curated_ids"])
        assert set(m["files"]) == {"profiles.jsonl", "trajectories.jsonl"}

    def test_curated_latent_hashes(self, written, cohort0):
        from daypart.simmember.manifest import latent_hash

        _, m = written
        for p in cohort0.heldout_profiles():
            assert m["curated_latent_hashes"][p.member_id] == latent_hash(p.latents)

    def test_file_hashes(self, written):
        out, m = written
        for name in ("profiles.jsonl", "trajectories.jsonl"):
            assert m["files"][name] == _sha((out / name).read_bytes())

    def test_profiles_jsonl_content(self, written, cohort0):
        out, _ = written
        expected = "".join(_canon(p.model_dump(mode="json")) + "\n" for p in cohort0.profiles)
        assert (out / "profiles.jsonl").read_text() == expected

    def test_trajectories_jsonl_content(self, written, cohort0):
        out, _ = written
        expected = "".join(
            _canon(cohort0.trajectories[p.member_id].model_dump(mode="json")) + "\n" for p in cohort0.profiles
        )
        assert (out / "trajectories.jsonl").read_text() == expected

    def test_synthetic_count_follows_cohort(self, tmp_path):
        from daypart.simmember.cohort import generate
        from daypart.simmember.manifest import write

        m = write(generate(2, n_synthetic=12, days=4), tmp_path)
        assert m["counts"] == {"curated": 25, "synthetic": 12}
        assert m["seed"] == 2


class TestLatentHash:
    pytestmark = pytest.mark.category("functional/latent-hash")

    def test_latent_hash_canonical(self, cohort0):
        from daypart.simmember.manifest import latent_hash

        for p in cohort0.profiles[:30]:
            expected = _sha(_canon(p.latents.model_dump(mode="json")).encode("utf-8"))
            assert latent_hash(p.latents) == expected

    def test_latent_hash_distinguishes(self, cohort0):
        from daypart.simmember.manifest import latent_hash

        hashes = {latent_hash(p.latents) for p in cohort0.heldout_profiles()}
        assert len(hashes) == len({p.latents for p in cohort0.heldout_profiles()})


class TestAssertNoCurated:
    pytestmark = pytest.mark.category("functional/leak-guard")

    def test_curated_id_nested_deep(self, written):
        from daypart.simmember.manifest import CuratedLeakError, assert_no_curated

        _, m = written
        rec = {"a": {"b": [{"c": [1, 2, {"d": "row for cur-007 here"}]}]}}
        with pytest.raises(CuratedLeakError):
            assert_no_curated([rec], m)

    def test_curated_id_as_bare_value(self, written):
        from daypart.simmember.manifest import CuratedLeakError, assert_no_curated

        _, m = written
        with pytest.raises(CuratedLeakError):
            assert_no_curated([{"member_id": "cur-025"}], m)

    def test_relabeled_curated_latents(self, written, cohort0):
        from daypart.simmember.manifest import CuratedLeakError, assert_no_curated

        _, m = written
        cur = cohort0.heldout_profiles()[3]
        rec = {
            "meta": {
                "profile": {
                    "member_id": "syn-001",
                    "cohort": "synthetic",
                    "latents": cur.latents.model_dump(mode="json"),
                }
            }
        }
        with pytest.raises(CuratedLeakError):
            assert_no_curated([rec], m)

    def test_curated_id_in_chat_message(self, written):
        from daypart.simmember.manifest import CuratedLeakError, assert_no_curated

        _, m = written
        rec = {
            "messages": [
                {"role": "system", "content": "You are a coach."},
                {"role": "user", "content": "Member cur-012 checked in today and felt fine."},
            ]
        }
        with pytest.raises(CuratedLeakError):
            assert_no_curated([rec], m)

    def test_leak_after_clean_records(self, written, cohort0):
        from daypart.simmember.manifest import CuratedLeakError, assert_no_curated

        _, m = written
        rows = [p.model_dump(mode="json") for p in cohort0.training_profiles()]
        rows.append({"x": "cur-001"})
        with pytest.raises(CuratedLeakError):
            assert_no_curated(iter(rows), m)

    def test_synthetic_rows_pass(self, written, cohort0):
        from daypart.simmember.manifest import assert_no_curated

        _, m = written
        rows = [p.model_dump(mode="json") for p in cohort0.training_profiles()]
        rows += [cohort0.trajectories[p.member_id].model_dump(mode="json") for p in cohort0.training_profiles()]
        rows.append({"messages": [{"role": "user", "content": "Member syn-012 says hi; ref note:syn-012:d3"}]})
        assert assert_no_curated(rows, m) is None

    def test_generator_input_passes(self, written, cohort0):
        from daypart.simmember.manifest import assert_no_curated

        _, m = written
        assert assert_no_curated((p.model_dump(mode="json") for p in cohort0.training_profiles()), m) is None

    def test_empty_iterable_passes(self, written):
        from daypart.simmember.manifest import assert_no_curated

        _, m = written
        assert assert_no_curated([], m) is None
        assert assert_no_curated(iter(()), m) is None

    def test_curated_profiles_themselves_fail(self, written, cohort0):
        from daypart.simmember.manifest import CuratedLeakError, assert_no_curated

        _, m = written
        with pytest.raises(CuratedLeakError):
            assert_no_curated([p.model_dump(mode="json") for p in cohort0.heldout_profiles()], m)

    def test_training_profiles_pass_guard(self, written, cohort0):
        from daypart.simmember.manifest import assert_no_curated

        _, m = written
        train = cohort0.training_profiles()
        assert not any(p.member_id.startswith("cur-") or p.cohort == "curated" for p in train)
        assert_no_curated([p.model_dump(mode="json") for p in train], m)


class TestVerifyManifest:
    pytestmark = pytest.mark.category("integration/manifest-verify")

    def test_untampered_returns_manifest(self, written):
        from daypart.simmember.manifest import verify_manifest

        out, m = written
        assert verify_manifest(out) == m

    def test_profiles_tamper_is_leak(self, written):
        from daypart.simmember.manifest import CuratedLeakError, verify_manifest

        out, _ = written
        p = out / "profiles.jsonl"
        lines = p.read_text().splitlines(keepends=True)
        p.write_text("".join(lines[:-1]))  # drop a profile
        with pytest.raises(CuratedLeakError):
            verify_manifest(out)

    def test_profiles_single_byte_tamper_is_leak(self, written):
        from daypart.simmember.manifest import CuratedLeakError, verify_manifest

        out, _ = written
        p = out / "profiles.jsonl"
        p.write_bytes(p.read_bytes() + b" ")
        with pytest.raises(CuratedLeakError):
            verify_manifest(out)

    def test_trajectories_tamper_is_value_error(self, written):
        from daypart.simmember.manifest import CuratedLeakError, verify_manifest

        out, _ = written
        t = out / "trajectories.jsonl"
        original = t.read_bytes()
        t.write_bytes(original[:-1] + b" \n")
        assert t.read_bytes() != original
        with pytest.raises(ValueError) as ei:
            verify_manifest(out)
        assert not isinstance(ei.value, CuratedLeakError)
