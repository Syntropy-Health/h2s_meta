"""Sealed: file and image check-in (R3) — T3.1, T3.2.

Spec: A&D HAC-33 D-A2 (pre-recorded sidecar extraction), Components (Fixture intake: allow-list
by sha256, size and type checks), API table (``fixture_id`` or multipart ``file`` + ``journal_id``;
413 ``too_large``, 415 ``unsupported_type``, 422 ``not_a_demo_fixture``), "Failure handling"
(1 MiB limit, magic bytes not extension), T3.1, T3.2; plan G7 (1 MiB + 1 byte -> 413) and G8
(``demo/fixtures/manifest.json``: fixture_id -> {file, sha256, type, size}).
"""

from __future__ import annotations

import hashlib

import pytest

from tests.sealed.conftest_chc import (
    MIB,
    PNG_MAGIC,
    body,
    chc_app,  # noqa: F401 - fixture
    checkin_ok,
    client,  # noqa: F401 - fixture
    concepts_by_code,
    discard_all_but,
    find_observations,
    fixture_bytes,
    fixture_kind,
    fixture_manifest,
    has_error,
    live_checkins,
    new_journal,
    post_fixture,
    post_upload,
    review,
    valid_correction,
    writes,  # noqa: F401 - fixture
)

pytestmark = pytest.mark.category("integration/chc-checkin-fixtures")


def _manifest_ids() -> list[str]:
    try:
        return sorted(fixture_manifest())
    except Exception:  # noqa: BLE001 - collection must not depend on the implementation
        return ["<manifest-missing>"]


@pytest.fixture(scope="module")
def manifest() -> dict[str, dict]:
    return fixture_manifest()


# --------------------------------------------------------------------------- #
# T3.1  bundled fixtures -> proposals, method fixture-sidecar-v1, editable
# --------------------------------------------------------------------------- #


class TestT31BundledFixtures:
    def test_t3_1_manifest_covers_image_and_data(self, manifest):
        """T3.1: >= 1 image fixture and >= 1 CSV or JSON fixture are bundled."""
        kinds = {fixture_kind(fid, e) for fid, e in manifest.items()}
        assert "image" in kinds, manifest
        assert "data" in kinds, manifest

    def test_t3_1_manifest_sha256_matches_bundled_bytes(self, manifest):
        """G8: the allow-list sha256 is the sha256 of the bundled file, which is under 1 MiB."""
        for fid, e in manifest.items():
            data = fixture_bytes(e)
            assert hashlib.sha256(data).hexdigest() == e["sha256"], fid
            assert len(data) <= MIB, fid

    @pytest.mark.parametrize("fixture_id", _manifest_ids())
    def test_t3_1_fixture_id_yields_sidecar_proposals(self, client, writes, fixture_id):
        """T3.1: each bundled fixture_id -> 201 with proposals whose method is fixture-sidecar-v1;
        nothing is written before review."""
        j = new_journal(client)
        writes.clear()
        c = checkin_ok(post_fixture(client, j["journal_id"], fixture_id))
        assert c["proposals"], fixture_id
        kind = fixture_kind(fixture_id, fixture_manifest()[fixture_id])
        for p in c["proposals"]:
            assert p["method"] == "fixture-sidecar-v1", p
            assert p["source_kind"] == ("image_fixture" if kind == "image" else "file_fixture"), p
            assert p["source_ref"] == fixture_id or fixture_id in str(p["source_ref"]), p
        assert len(writes) == 0

    @pytest.mark.parametrize("fixture_id", _manifest_ids())
    def test_t3_1_each_proposal_editable_through_correct(self, client, fixture_id):
        """T3.1: every mapped fixture proposal can be edited through `correct` (-> 200)."""
        concepts = concepts_by_code()
        j = new_journal(client)
        first = checkin_ok(post_fixture(client, j["journal_id"], fixture_id))
        mapped = [p for p in first["proposals"] if p.get("concept_code") in concepts]
        assert mapped, f"{fixture_id}: no mapped proposal to edit"
        for target in mapped:
            jj = new_journal(client)
            c = checkin_ok(post_fixture(client, jj["journal_id"], fixture_id))
            p = next(
                x
                for x in c["proposals"]
                if x.get("concept_code") == target["concept_code"] and x.get("snippet") == target.get("snippet")
            )
            fix = valid_correction(p, jj["today"], concepts)
            r = review(client, c["checkin_id"], discard_all_but(c["proposals"], {p["proposal_id"]: fix}))
            assert r.status_code == 200, (fixture_id, p, r.text)
            out = r.json()
            assert out["status"] == "saved"
            assert len(out["written"]) == 1, out
            recs = [o for o in find_observations(client, jj["journal_id"]) if o.get("event_id") in out["written"]]
            assert len(recs) == 1
            assert recs[0]["quality"] == "corrected"
            assert recs[0]["provenance"]["method"] == "fixture-sidecar-v1"

    def test_t3_1_bundled_bytes_upload_is_accepted(self, client, manifest):
        """Components (allow-list by sha256): uploading a bundled fixture's exact bytes -> 201."""
        fid, e = next((k, v) for k, v in sorted(manifest.items()) if fixture_kind(k, v) == "image")
        j = new_journal(client)
        name = str(e.get("file", fid)).rsplit("/", 1)[-1]
        ctype = "image/png" if name.lower().endswith(".png") else "image/jpeg"
        r = post_upload(client, j["journal_id"], fixture_bytes(e), name, ctype)
        c = checkin_ok(r)
        assert all(p["method"] == "fixture-sidecar-v1" for p in c["proposals"])


# --------------------------------------------------------------------------- #
# T3.2  foreign PNG -> 422; 1 MiB + 1 -> 413; PDF -> 415; journal unchanged
# --------------------------------------------------------------------------- #


def _foreign_png(size: int = 256) -> bytes:
    payload = PNG_MAGIC + b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    return payload + b"\x00sealed-foreign" * (size // 15)


PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"


class TestT32RefusedUploads:
    def _assert_unchanged(self, client, j, before_checkins, writes):
        assert live_checkins() == before_checkins
        assert len(writes) == 0
        assert find_observations(client, j["journal_id"], limit=10) == []

    def test_t3_2_png_not_in_allow_list(self, client, writes):
        """T3.2: a PNG not in the allow-list -> 422 not_a_demo_fixture; journal unchanged."""
        j = new_journal(client)
        before = live_checkins()
        writes.clear()
        r = post_upload(client, j["journal_id"], _foreign_png(), "scale-photo.png", "image/png")
        assert r.status_code == 422, r.text
        assert has_error(r, "not_a_demo_fixture"), body(r)
        self._assert_unchanged(client, j, before, writes)

    def test_t3_2_too_large(self, client, writes):
        """T3.2: a 1 MiB + 1 byte body -> 413 too_large; journal unchanged."""
        j = new_journal(client)
        before = live_checkins()
        writes.clear()
        data = _foreign_png(MIB + 64)[: MIB + 1]
        assert len(data) == MIB + 1
        r = post_upload(client, j["journal_id"], data, "big.png", "image/png")
        assert r.status_code == 413, r.text
        assert has_error(r, "too_large"), body(r)
        self._assert_unchanged(client, j, before, writes)

    def test_t3_2_pdf_unsupported(self, client, writes):
        """T3.2: a PDF -> 415 unsupported_type; journal unchanged."""
        j = new_journal(client)
        before = live_checkins()
        writes.clear()
        r = post_upload(client, j["journal_id"], PDF, "report.pdf", "application/pdf")
        assert r.status_code == 415, r.text
        assert has_error(r, "unsupported_type"), body(r)
        self._assert_unchanged(client, j, before, writes)

    def test_t3_2_extension_is_ignored(self, client, writes):
        """Failure handling: magic bytes decide, not the extension — a PDF named .png -> 415."""
        j = new_journal(client)
        before = live_checkins()
        writes.clear()
        r = post_upload(client, j["journal_id"], PDF, "photo.png", "image/png")
        assert r.status_code == 415, r.text
        assert has_error(r, "unsupported_type"), body(r)
        self._assert_unchanged(client, j, before, writes)

    def test_t3_2_unknown_fixture_id(self, client, writes):
        """API table: a fixture_id outside the manifest is refused (not a demo fixture)."""
        j = new_journal(client)
        before = live_checkins()
        writes.clear()
        r = post_fixture(client, j["journal_id"], "not-a-bundled-fixture")
        assert r.status_code in (404, 422), r.text
        self._assert_unchanged(client, j, before, writes)
