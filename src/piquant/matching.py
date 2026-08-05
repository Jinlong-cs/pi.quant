"""Deterministic module-name selection shared by recipes and backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase
from typing import Any

from piquant.contracts import ModuleCoverage


def select_modules(named_modules: Mapping[str, Any], include: Sequence[str], exclude: Sequence[str]) -> ModuleCoverage:
    """Select named modules with explicit include/exclude accounting."""

    candidate_names = sorted(named_modules)
    matched_names = [
        name
        for name in candidate_names
        if any(fnmatchcase(name, pattern) for pattern in include) and not any(fnmatchcase(name, pattern) for pattern in exclude)
    ]
    excluded_names = [
        name
        for name in candidate_names
        if any(fnmatchcase(name, pattern) for pattern in exclude) and any(fnmatchcase(name, pattern) for pattern in include)
    ]
    return ModuleCoverage(
        candidate_count=len(candidate_names),
        matched_count=len(matched_names),
        excluded_count=len(excluded_names),
        candidate_names=candidate_names,
        matched_names=matched_names,
        excluded_names=excluded_names,
    )


def require_matches(coverage: ModuleCoverage, include: Sequence[str]) -> ModuleCoverage:
    """Fail before calibration when a recipe does not select any module."""

    if coverage.matched_count == 0:
        raise ValueError(f"module selection matched zero candidates; include={list(include)!r}, candidates={coverage.candidate_names!r}")
    return coverage
