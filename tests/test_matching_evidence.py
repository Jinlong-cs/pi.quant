from pathlib import Path

import pytest

from piquant.contracts import EvidenceRecord
from piquant.evidence import JsonEvidenceStore
from piquant.matching import require_matches, select_modules


def test_module_selection_reports_include_and_exclude() -> None:
    coverage = select_modules(
        {"vision.patch": object(), "action_encoder": object(), "head": object()},
        ["*"],
        ["head"],
    )
    assert coverage.candidate_count == 3
    assert coverage.matched_names == ["action_encoder", "vision.patch"]
    assert coverage.excluded_names == ["head"]


def test_zero_match_fails_fast() -> None:
    coverage = select_modules({"head": object()}, ["vision.*"], [])
    with pytest.raises(ValueError, match="matched zero"):
        require_matches(coverage, ["vision.*"])


def test_evidence_store_round_trip(tmp_path: Path, synthetic_record: EvidenceRecord) -> None:
    path = tmp_path / "evidence.json"
    store = JsonEvidenceStore()
    store.write(synthetic_record, str(path))
    loaded = store.read(str(path))
    assert loaded.record_id == synthetic_record.record_id
    assert loaded.module_coverage.matched_count == synthetic_record.module_coverage.matched_count
