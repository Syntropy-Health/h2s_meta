"""Sealed: Clinical Human Clone concept registry (R5) — T5.1, T5.2 as amended by AM-1.

Spec: A&D HAC-33 "Components & interfaces" (Concept registry), 4.3, D-A5, T5.1, T5.2;
plan G3/G4 (the slice is "exactly the codes not in core()"); CTO AM-1 items 1, 2 and 5
(review label; optional, sourced-only ``reference_range``; a concept with no source has
``reference_range = null``).
"""

from __future__ import annotations

import math

import pytest

from tests.sealed.conftest_chc import (
    REVIEWED_LABEL,
    VERTICALS,
    registry_concepts,
    sources,
)

pytestmark = pytest.mark.category("functional/chc-registry")

# A&D 4.3: the seven A4 codes that core() already holds (they get registry metadata only).
ALREADY_CORE = frozenset({"72514-3", "44261-6", "70274-6", "8867-4", "29463-7", "55423-8", "93832-4"})
LOCAL_CODES = frozenset({"dp:adherence.protocol", "dp:symptom.severity"})
VALUE_TYPES = frozenset({"quantity", "integer", "ordinal"})
RANGE_FIELDS = ("low", "high", "unit", "population", "source_evidence_id")


def _is_loinc(c: dict) -> bool:
    return c.get("system") in ("loinc", "http://loinc.org")


@pytest.fixture(scope="module")
def concepts() -> list[dict]:
    return registry_concepts()


@pytest.fixture(scope="module")
def loinc_codes(concepts) -> set[str]:
    return {c["code"] for c in concepts if _is_loinc(c)}


# --------------------------------------------------------------------------- #
# T5.1  slice_ontology() = apply_update(core(), ...), version core-v1+u1
# --------------------------------------------------------------------------- #


class TestT51SliceOntology:
    def test_t5_1_version_and_single_history_record(self):
        """T5.1: slice_ontology().version == "core-v1+u1" and its history has exactly 1 record."""
        from daypart.memory.registry import slice_ontology

        s = slice_ontology()
        assert s.version == "core-v1+u1"
        assert len(s.history) == 1
        rec = s.history[0]
        assert rec.from_version == "core-v1"
        assert rec.to_version == "core-v1+u1"
        assert rec.gate == "passed"

    def test_t5_1_core_unchanged(self):
        """T5.1: core() is unchanged (equal digest, version core-v1) after building the slice."""
        from daypart.memory.registry import slice_ontology

        from daypart.memory.ontology import core

        before = core()
        digest_before = before.digest()
        slice_ontology()
        after = core()
        assert after.version == "core-v1"
        assert after.digest() == digest_before
        assert after == before
        assert after.history == ()

    def test_t5_1_core_nodes_preserved_in_slice(self):
        """T5.1 / I9: every core node is in the slice, unchanged (nodes are never replaced)."""
        from daypart.memory.registry import slice_ontology

        from daypart.memory.ontology import core

        c, s = core(), slice_ontology()
        for code, node in c.nodes.items():
            assert code in s.nodes, f"core node {code} missing from slice"
            assert s.nodes[code] == node, f"core node {code} altered in slice"
            assert s.is_core(code)

    def test_t5_1_added_nodes_are_exactly_the_non_core_a4_loinc_codes(self, loinc_codes):
        """T5.1 (plan G4): the update adds exactly the A4 LOINC codes not already in core()."""
        from daypart.memory.registry import slice_ontology

        from daypart.memory.ontology import LOINC_SYSTEM, core

        c, s = core(), slice_ontology()
        added = set(s.nodes) - set(c.nodes)
        expected = loinc_codes - set(c.nodes)
        assert added == expected
        assert added, "the update adds no nodes"
        for code in added:
            assert s.nodes[code].system == LOINC_SYSTEM, code
            assert not s.is_core(code), code
        # the seven A4 codes already in core() get registry metadata only
        assert ALREADY_CORE <= set(c.nodes)
        assert not (added & ALREADY_CORE)

    def test_t5_1_added_nodes_never_feed_a_score(self):
        """T5.1 / 4.4 (G3 unchanged): added LOINC nodes carry lhm_domain=None."""
        from daypart.memory.registry import slice_ontology

        from daypart.memory.ontology import core

        c, s = core(), slice_ontology()
        for code in set(s.nodes) - set(c.nodes):
            assert s.nodes[code].lhm_domain is None, code

    def test_t5_1_is_apply_update_of_core(self):
        """T5.1: the slice equals apply_update(core(), <its one update>) (same nodes, edges, digest)."""
        from daypart.memory.registry import slice_ontology

        from daypart.memory.ontology import OntologyUpdate, apply_update, core

        c, s = core(), slice_ontology()
        rec = s.history[0]
        added = tuple(s.nodes[k] for k in sorted(set(s.nodes) - set(c.nodes)))
        core_edges = {(e.source, e.target): e for e in c.edges}
        new_edges = tuple(e for e in s.edges if core_edges.get((e.source, e.target)) != e)
        rebuilt = apply_update(
            c,
            OntologyUpdate(proposer=rec.proposer, rationale=rec.rationale, add_nodes=added, set_edges=new_edges),
        )
        assert rebuilt.version == s.version
        assert dict(rebuilt.nodes) == dict(s.nodes)
        assert rebuilt.edges == s.edges
        assert rebuilt.digest() == s.digest()

    def test_t5_1_deterministic(self):
        """T5.1: slice_ontology() is equal on every call."""
        from daypart.memory.registry import slice_ontology

        a, b = slice_ontology(), slice_ontology()
        assert a == b
        assert a.digest() == b.digest()


# --------------------------------------------------------------------------- #
# T5.2  every A4 concept carries the full metadata (AM-1 label + reference_range)
# --------------------------------------------------------------------------- #


class TestT52ConceptMetadata:
    def test_t5_2_required_fields_present(self, concepts):
        """T5.2: concept_id, code, loinc_version (non-null for LOINC), display, unit, value_type,
        reviewer_status, allowed_use are present on every A4 concept."""
        for c in concepts:
            for key in ("concept_id", "code", "display", "unit", "value_type", "allowed_use", "vertical"):
                assert c.get(key) not in (None, ""), f"{c.get('code')}: {key} missing/empty"
            assert "loinc_version" in c, f"{c['code']}: loinc_version key missing"
            if _is_loinc(c):
                assert isinstance(c["loinc_version"], str) and c["loinc_version"], c["code"]
            assert c["value_type"] in VALUE_TYPES, c
            assert c["vertical"] in VERTICALS, c
            assert str(c["concept_id"]).startswith("chc:"), c["concept_id"]

    def test_t5_2_reviewer_status_is_board_sign_off(self, concepts):
        """T5.2 (AM-1 item 1): reviewer_status == "reviewed: board sign-off 2026-11-07"."""
        for c in concepts:
            assert c.get("reviewer_status") == REVIEWED_LABEL, (c["code"], c.get("reviewer_status"))

    def test_t5_2_label_never_claims_clinical_review(self, concepts):
        """AM-1 item 1: the copy never says "clinically reviewed"."""
        for c in concepts:
            for key in ("reviewer_status", "allowed_use"):
                text = str(c.get(key, "")).lower()
                assert "clinically reviewed" not in text, (c["code"], key)

    def test_t5_2_loinc_version_matches_the_one_constant(self, concepts):
        """T5.2 / D-A5: every LOINC concept's loinc_version equals registry.LOINC_VERSION;
        local codes carry none."""
        import daypart.memory.registry as reg

        version = getattr(reg, "LOINC_VERSION", None)
        assert isinstance(version, str) and version, "registry.LOINC_VERSION missing"
        for c in concepts:
            if _is_loinc(c):
                assert c["loinc_version"] == version, c["code"]
            else:
                assert c["loinc_version"] is None, c["code"]

    def test_t5_2_ids_unique(self, concepts):
        """T5.2: one concept per observable — unique concept_id and code."""
        ids = [c["concept_id"] for c in concepts]
        codes = [c["code"] for c in concepts]
        assert len(ids) == len(set(ids))
        assert len(codes) == len(set(codes))

    def test_t5_2_vertical_and_loinc_counts(self, concepts, loinc_codes):
        """T5.2: the 3 verticals have >= 4 concepts each; the LOINC count is between 15 and 25."""
        for v in VERTICALS:
            n = sum(1 for c in concepts if c["vertical"] == v)
            assert n >= 4, f"vertical {v} has {n} concepts"
        assert 15 <= len(loinc_codes) <= 25, len(loinc_codes)

    def test_t5_2_local_codes_present(self, concepts):
        """4.3: the slice keeps the two local codes dp:adherence.protocol and dp:symptom.severity."""
        codes = {c["code"] for c in concepts}
        assert LOCAL_CODES <= codes
        for c in concepts:
            if c["code"] in LOCAL_CODES:
                assert not _is_loinc(c)

    def test_t5_2_a4_codes_exist_in_slice_ontology(self, concepts):
        """4.3: every A4 concept code is a node of slice_ontology()."""
        from daypart.memory.registry import slice_ontology

        s = slice_ontology()
        for c in concepts:
            assert c["code"] in s.nodes, c["code"]


class TestT52ReferenceRange:
    """AM-1 item 2: optional reference_range, set only where a sources_v1.json entry supports it."""

    def test_t5_2_reference_range_key_present(self, concepts):
        """AM-1: every concept carries reference_range — an object, or null when it has none."""
        for c in concepts:
            assert "reference_range" in c, f"{c['code']}: reference_range key missing"
            assert c["reference_range"] is None or isinstance(c["reference_range"], dict), c["code"]

    def test_t5_2_reference_range_shape(self, concepts):
        """AM-1: a range is {low, high, unit, population, source_evidence_id} with real numbers."""
        for c in concepts:
            rr = c.get("reference_range")
            if rr is None:
                continue
            for key in RANGE_FIELDS:
                assert key in rr, f"{c['code']}: reference_range lacks {key}"
            nums = [rr["low"], rr["high"]]
            assert any(n is not None for n in nums), f"{c['code']}: range with no number"
            for n in nums:
                if n is not None:
                    assert isinstance(n, (int, float)) and not isinstance(n, bool), (c["code"], n)
                    assert math.isfinite(n), (c["code"], n)
            if rr["low"] is not None and rr["high"] is not None:
                assert rr["low"] <= rr["high"], c["code"]
            assert isinstance(rr["unit"], str) and rr["unit"], c["code"]
            assert isinstance(rr["population"], str) and rr["population"].strip(), c["code"]

    def test_t5_2_reference_range_source_resolves(self, concepts):
        """AM-1 / issue HAC-39: reference_range.source_evidence_id resolves to a sources_v1.json
        entry, and that entry applies to the concept."""
        srcs = sources()
        for c in concepts:
            rr = c.get("reference_range")
            if rr is None:
                continue
            sid = rr.get("source_evidence_id")
            assert isinstance(sid, str) and sid, f"{c['code']}: unsourced range"
            assert sid in srcs, f"{c['code']}: source_evidence_id {sid!r} not in sources_v1.json"
            applies = srcs[sid].get("applies_to") or []
            assert c["concept_id"] in applies, f"{c['code']}: source {sid} does not apply to {c['concept_id']}"

    def test_t5_2_concept_without_source_has_null_range(self, concepts):
        """AM-1 item 5: a concept that no sources_v1.json entry applies to has reference_range=null."""
        srcs = sources()
        covered = {cid for s in srcs.values() for cid in (s.get("applies_to") or [])}
        uncovered = [c for c in concepts if c["concept_id"] not in covered]
        for c in uncovered:
            assert c.get("reference_range") is None, f"{c['code']}: range without any source"

    def test_t5_2_plausible_is_distinct_from_reference_range(self, concepts):
        """AM-1 item 4: plausible (validation-only) bounds are not the reference_range."""
        for c in concepts:
            assert "plausible" in c, f"{c['code']}: plausible bounds missing"
