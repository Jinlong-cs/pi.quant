"""Deterministic, budget-controlled mixed-precision search primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Literal

from piquant.contracts import (
    CandidateMetrics,
    CandidateRecipe,
    CandidateRecord,
    ModelSpec,
    ParetoFrontRecord,
    SearchBoundary,
    SearchConstraint,
    SearchObjective,
    SearchPlan,
    SearchStudyRecord,
    SensitivitySignal,
    TargetFingerprint,
    candidate_recipe_hash,
    search_plan_hash,
    search_source_objectives,
)

SourceEvaluator = Callable[[CandidateRecipe], CandidateRecord]
TargetEvaluator = Callable[[CandidateRecord], CandidateRecord]


def make_candidate_recipe(
    *,
    recipe_id: str,
    control_kind: Literal["fp_control", "broad_quant", "manual_selective", "search"],
    precision_map: dict[str, Literal["fp32", "fp16", "bf16", "int8", "fp8", "nvfp4"]],
    parent_recipe_id: str | None = None,
    mutation: list[str] | None = None,
) -> CandidateRecipe:
    """Create a recipe with its canonical hash rather than accepting a caller hash."""

    resolved_mutation = () if mutation is None else tuple(mutation)
    resolved_hash = candidate_recipe_hash(
        recipe_id=recipe_id,
        control_kind=control_kind,
        precision_map=precision_map,
        parent_recipe_id=parent_recipe_id,
        mutation=resolved_mutation,
    )
    return CandidateRecipe(
        recipe_id=recipe_id,
        control_kind=control_kind,
        precision_map=precision_map,
        parent_recipe_id=parent_recipe_id,
        mutation=resolved_mutation,
        recipe_hash=resolved_hash,
    )


def resolve_search_plan(plan: SearchPlan) -> SearchPlan:
    """Attach the deterministic plan hash before serializing or executing a plan."""

    resolved_hash = search_plan_hash(plan)
    if plan.plan_hash == resolved_hash:
        return plan
    payload = plan.model_dump(mode="python")
    payload["plan_hash"] = resolved_hash
    return SearchPlan.model_validate(payload)


def _signal_priority(signal: SensitivitySignal) -> tuple[float, float, float, str, str]:
    score = signal.quality_recovery / max(signal.latency_cost_ms, 1e-12)
    return (-score, -signal.quality_recovery, signal.latency_cost_ms, signal.group, signal.to_precision)


def _sorted_signals(signals: Iterable[SensitivitySignal]) -> list[SensitivitySignal]:
    return sorted(signals, key=_signal_priority)


def generate_candidate_recipes(plan: SearchPlan) -> list[CandidateRecipe]:
    """Generate controls plus a bounded deterministic beam of group restorations."""

    resolved_plan = resolve_search_plan(plan)
    if len(resolved_plan.controls) > resolved_plan.budget.max_source_candidates:
        raise ValueError("control recipes exceed max_source_candidates")

    recipes = list(resolved_plan.controls)
    recipe_by_precision = {tuple(sorted(recipe.precision_map.items())): recipe for recipe in recipes}
    if len(recipe_by_precision) != len(recipes):
        raise ValueError("SearchPlan control recipes must have distinct precision maps")
    broad = [recipe for recipe in recipes if recipe.control_kind == "broad_quant"]
    beams = [(recipe, 0.0) for recipe in broad]
    if not beams:
        raise ValueError("SearchPlan requires a broad_quant control")

    for signal in _sorted_signals(resolved_plan.sensitivity_signals):
        score = signal.quality_recovery / max(signal.latency_cost_ms, 1e-12)
        expanded = list(beams)
        for parent, parent_score in beams:
            if parent.precision_map.get(signal.group) != signal.from_precision:
                continue
            precision_map = dict(parent.precision_map)
            precision_map[signal.group] = signal.to_precision
            precision_key = tuple(sorted(precision_map.items()))
            recipe = recipe_by_precision.get(precision_key)
            if recipe is None:
                safe_group = signal.group.replace("/", "_").replace(".", "_")
                recipe_id = f"{parent.recipe_id}__{safe_group}__{signal.to_precision}"
                mutation = [*parent.mutation, f"{signal.group}:{signal.from_precision}->{signal.to_precision}"]
                recipe = make_candidate_recipe(
                    recipe_id=recipe_id,
                    control_kind="search",
                    precision_map=precision_map,
                    parent_recipe_id=parent.recipe_id,
                    mutation=mutation,
                )
                recipe_by_precision[precision_key] = recipe
                recipes.append(recipe)
                if len(recipes) >= resolved_plan.budget.max_source_candidates:
                    return recipes[: resolved_plan.budget.max_source_candidates]
            expanded.append((recipe, parent_score + score))
        best_by_precision: dict[tuple[tuple[str, str], ...], tuple[CandidateRecipe, float]] = {}
        for recipe, recipe_score in expanded:
            precision_key = tuple(sorted(recipe.precision_map.items()))
            current = best_by_precision.get(precision_key)
            if current is None or (recipe_score, recipe.recipe_id) > (current[1], current[0].recipe_id):
                best_by_precision[precision_key] = (recipe, recipe_score)
        beams = sorted(best_by_precision.values(), key=lambda item: (-item[1], len(item[0].mutation), item[0].recipe_id))[
            : resolved_plan.budget.beam_width
        ]
    return recipes[: resolved_plan.budget.max_source_candidates]


def _manifest_fingerprint(plan: SearchPlan, name: str) -> str:
    try:
        return plan.split_audit.manifest_fingerprints[name]
    except KeyError as error:
        raise ValueError(f"SearchPlan split audit is missing {name!r} manifest fingerprint") from error


def _validate_candidate_identity(plan: SearchPlan, candidate: CandidateRecord, recipe: CandidateRecipe, *, target: bool) -> None:
    if candidate.candidate_id != recipe.recipe_id:
        raise ValueError(f"candidate {candidate.candidate_id!r} must use its generated recipe_id")
    if candidate.recipe != recipe:
        raise ValueError(f"candidate {candidate.candidate_id!r} recipe identity differs from generated recipe")
    if candidate.model != plan.model or candidate.target != plan.target:
        raise ValueError(f"candidate {candidate.candidate_id!r} model or target identity differs from SearchPlan")
    if candidate.search_plan_hash != search_plan_hash(plan):
        raise ValueError(f"candidate {candidate.candidate_id!r} search plan hash differs from SearchPlan")
    expected = {
        "calibration": _manifest_fingerprint(plan, "calibration"),
        "sensitivity": _manifest_fingerprint(plan, "sensitivity"),
        "search_validation": _manifest_fingerprint(plan, "search_validation"),
        "promotion_reserved": _manifest_fingerprint(plan, "promotion_reserved"),
    }
    actual = {
        "calibration": candidate.calibration_manifest_fingerprint,
        "sensitivity": candidate.sensitivity_manifest_fingerprint,
        "search_validation": candidate.search_validation_manifest_fingerprint,
        "promotion_reserved": candidate.promotion_reserved_manifest_fingerprint,
    }
    if actual != expected:
        raise ValueError(f"candidate {candidate.candidate_id!r} split identity differs from SearchPlan")
    if target and candidate.status == "measured" and candidate.target_metrics is None:
        raise ValueError(f"target candidate {candidate.candidate_id!r} is missing target metrics")
    if not target and candidate.target_metrics is not None:
        raise ValueError(f"source candidate {candidate.candidate_id!r} unexpectedly contains target metrics")


def _completed_candidate(existing: Sequence[CandidateRecord], recipe: CandidateRecipe) -> CandidateRecord | None:
    matches = [candidate for candidate in existing if candidate.recipe.recipe_hash == recipe.recipe_hash]
    if len(matches) > 1:
        raise ValueError(f"resume state contains duplicate recipe hash {recipe.recipe_hash}")
    if not matches:
        return None
    candidate = matches[0]
    if candidate.recipe.recipe_id != recipe.recipe_id:
        raise ValueError("resume state maps one recipe hash to a different recipe_id")
    if candidate.status in {"measured", "rejected", "unsupported"}:
        return candidate
    return None


def run_source_search(
    plan: SearchPlan,
    evaluator: SourceEvaluator,
    *,
    existing: Sequence[CandidateRecord] = (),
) -> list[CandidateRecord]:
    """Evaluate generated source recipes, resuming only identity-complete records."""

    resolved_plan = resolve_search_plan(plan)
    records: list[CandidateRecord] = []
    for recipe in generate_candidate_recipes(resolved_plan):
        resumed = _completed_candidate(existing, recipe)
        if resumed is not None:
            _validate_candidate_identity(resolved_plan, resumed, recipe, target=False)
            records.append(_apply_constraints(resolved_plan, resumed, boundary="source"))
            continue
        candidate = evaluator(recipe)
        if candidate.candidate_id != recipe.recipe_id:
            raise ValueError("source evaluator must return candidate_id equal to recipe_id")
        _validate_candidate_identity(resolved_plan, candidate, recipe, target=False)
        records.append(_apply_constraints(resolved_plan, candidate, boundary="source"))
    return records


def _metric_value(metrics: CandidateMetrics, name: str) -> float:
    value = getattr(metrics, name)
    if value is None:
        raise ValueError(f"candidate metrics are missing Pareto objective {name!r}")
    return float(value)


def _metric_uncertainty(metrics: CandidateMetrics, name: str) -> float:
    return float(metrics.uncertainty.get(name, 0.0))


def _constraint_failure(
    metrics: CandidateMetrics,
    constraints: Sequence[SearchConstraint],
    *,
    boundary: SearchBoundary,
    fp_control: bool,
) -> SearchConstraint | None:
    if fp_control:
        return None
    for constraint in constraints:
        if constraint.boundary != boundary:
            continue
        value = _metric_value(metrics, constraint.name)
        uncertainty = _metric_uncertainty(metrics, constraint.name)
        if constraint.operator == "le" and value + uncertainty > constraint.threshold:
            return constraint
        if constraint.operator == "ge" and value - uncertainty < constraint.threshold:
            return constraint
    return None


def _apply_constraints(plan: SearchPlan, candidate: CandidateRecord, *, boundary: SearchBoundary) -> CandidateRecord:
    if candidate.status != "measured":
        return candidate
    metrics = candidate.source_metrics if boundary == "source" else candidate.target_metrics
    if metrics is None:
        raise ValueError(f"measured {boundary} candidate {candidate.candidate_id!r} is missing metrics")
    failed = _constraint_failure(
        metrics,
        plan.constraints,
        boundary=boundary,
        fp_control=candidate.recipe.control_kind == "fp_control",
    )
    if failed is None:
        return candidate
    payload = candidate.model_dump(mode="python")
    payload.update(
        status="rejected",
        reason_code=f"{boundary.upper()}_CONSTRAINT_{failed.name.upper()}",
        reason=f"{failed.name} {failed.operator} {failed.threshold} failed with pre-registered uncertainty",
    )
    return CandidateRecord.model_validate(payload)


def apply_search_constraints(plan: SearchPlan, candidate: CandidateRecord, *, boundary: SearchBoundary) -> CandidateRecord:
    """Apply the pre-registered hard constraints to an identity-matched candidate."""

    resolved_plan = resolve_search_plan(plan)
    _validate_candidate_identity(resolved_plan, candidate, candidate.recipe, target=boundary == "target")
    return _apply_constraints(resolved_plan, candidate, boundary=boundary)


def pareto_front(
    candidates: Sequence[CandidateRecord],
    *,
    boundary: SearchBoundary,
    objectives: Sequence[SearchObjective],
    model: ModelSpec,
    target: TargetFingerprint,
    front_id: str,
    search_plan_hash: str,
) -> ParetoFrontRecord:
    """Compute an uncertainty-aware non-dominated front without a weighted aggregate."""

    eligible: list[CandidateRecord] = []
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Pareto input candidate IDs must be unique")
    for candidate in candidates:
        if candidate.status != "measured":
            continue
        if candidate.model != model or candidate.target != target:
            raise ValueError(f"candidate {candidate.candidate_id!r} model or target differs from Pareto identity")
        if candidate.search_plan_hash != search_plan_hash:
            raise ValueError(f"candidate {candidate.candidate_id!r} search plan hash differs from Pareto identity")
        metrics = candidate.source_metrics if boundary == "source" else candidate.target_metrics
        if metrics is None:
            continue
        for objective in objectives:
            _metric_value(metrics, objective.name)
        eligible.append(candidate)

    values: dict[str, dict[str, float]] = {}
    uncertainty: dict[str, dict[str, float]] = {}
    for candidate in eligible:
        metrics = candidate.source_metrics if boundary == "source" else candidate.target_metrics
        if metrics is None:
            raise ValueError(f"candidate {candidate.candidate_id!r} is missing {boundary} metrics")
        values[candidate.candidate_id] = {objective.name: _metric_value(metrics, objective.name) for objective in objectives}
        uncertainty[candidate.candidate_id] = {objective.name: _metric_uncertainty(metrics, objective.name) for objective in objectives}

    def interval(candidate_id: str, name: str) -> tuple[float, float]:
        value = values[candidate_id][name]
        radius = uncertainty[candidate_id][name]
        return value - radius, value + radius

    def dominates(left: str, right: str) -> bool:
        no_worse = True
        strictly_better = False
        for objective in objectives:
            left_low, left_high = interval(left, objective.name)
            right_low, right_high = interval(right, objective.name)
            if objective.direction == "minimize":
                if left_high > right_low:
                    no_worse = False
                if left_high < right_low:
                    strictly_better = True
            else:
                if left_low < right_high:
                    no_worse = False
                if left_low > right_high:
                    strictly_better = True
        return no_worse and strictly_better

    def intervals_overlap(left: str, right: str) -> bool:
        return all(
            interval(left, objective.name)[0] <= interval(right, objective.name)[1]
            and interval(right, objective.name)[0] <= interval(left, objective.name)[1]
            for objective in objectives
        )

    candidate_ids = sorted(values)
    non_dominated = [candidate_id for candidate_id in candidate_ids if not any(dominates(other, candidate_id) for other in candidate_ids)]
    dominated = [candidate_id for candidate_id in candidate_ids if candidate_id not in non_dominated]
    neighbours = {
        candidate_id: {other for other in non_dominated if other != candidate_id and intervals_overlap(candidate_id, other)}
        for candidate_id in non_dominated
    }
    tie_groups: list[list[str]] = []

    def maximal_ties(group: set[str], candidates: set[str], excluded: set[str]) -> None:
        if not candidates and not excluded:
            if len(group) > 1:
                tie_groups.append(sorted(group))
            return
        for candidate_id in sorted(tuple(candidates)):
            maximal_ties(
                group | {candidate_id},
                candidates & neighbours[candidate_id],
                excluded & neighbours[candidate_id],
            )
            candidates.remove(candidate_id)
            excluded.add(candidate_id)

    maximal_ties(set(), set(non_dominated), set())
    tie_groups.sort()

    ranking_reasons: dict[str, list[str]] = {}
    for candidate_id in candidate_ids:
        if candidate_id in non_dominated:
            reasons = ["non_dominated"]
            reasons.extend(f"tie:{','.join(group)}" for group in tie_groups if candidate_id in group)
            ranking_reasons[candidate_id] = reasons
        else:
            ranking_reasons[candidate_id] = [
                f"dominated_by:{other}" for other in candidate_ids if other != candidate_id and dominates(other, candidate_id)
            ]

    return ParetoFrontRecord(
        front_id=front_id,
        search_plan_hash=search_plan_hash,
        boundary=boundary,
        model=model,
        target=target,
        objectives=list(objectives),
        candidate_ids=candidate_ids,
        objective_values=values,
        objective_uncertainty=uncertainty,
        non_dominated_candidate_ids=sorted(non_dominated),
        dominated_candidate_ids=sorted(dominated),
        tie_groups=tie_groups,
        ranking_reasons=ranking_reasons,
    )


def select_target_candidates(
    plan: SearchPlan,
    source_candidates: Sequence[CandidateRecord],
) -> list[CandidateRecord]:
    """Preserve measured controls, then add bounded source-Pareto search candidates."""

    resolved_plan = resolve_search_plan(plan)
    validate_resume_identity(resolved_plan, source_candidates)
    limit = resolved_plan.budget.target_compile_limit
    eligible: list[CandidateRecord] = []
    controls: dict[str, CandidateRecord] = {}
    for candidate in source_candidates:
        metrics = candidate.source_metrics
        if metrics is None:
            continue
        if candidate.recipe.control_kind != "search":
            if candidate.status == "measured" or (
                candidate.status == "rejected" and (candidate.reason_code or "").startswith("SOURCE_CONSTRAINT_")
            ):
                controls[candidate.recipe.control_kind] = candidate
            if candidate.status != "measured":
                continue
        elif candidate.status != "measured":
            continue
        if candidate.recipe.control_kind != "fp_control" and metrics.quant_coverage <= 0.0:
            continue
        if (
            _constraint_failure(
                metrics,
                resolved_plan.constraints,
                boundary="source",
                fp_control=candidate.recipe.control_kind == "fp_control",
            )
            is not None
        ):
            continue
        eligible.append(candidate)
    front = pareto_front(
        eligible,
        boundary="source",
        objectives=search_source_objectives(resolved_plan),
        model=resolved_plan.model,
        target=resolved_plan.target,
        front_id="source-preselection",
        search_plan_hash=search_plan_hash(resolved_plan),
    )
    required_controls = ["fp_control", "broad_quant", "manual_selective"]
    if missing := [kind for kind in required_controls if kind not in controls]:
        raise ValueError(f"source gate did not produce required control candidates: {missing!r}")
    selected = [controls[kind] for kind in required_controls]
    search_candidates = [
        candidate
        for candidate in eligible
        if candidate.recipe.control_kind == "search" and candidate.candidate_id in front.non_dominated_candidate_ids
    ]
    recipe_order = {recipe.recipe_hash: index for index, recipe in enumerate(generate_candidate_recipes(resolved_plan))}
    ordered_search = sorted(search_candidates, key=lambda candidate: recipe_order[candidate.recipe.recipe_hash])
    return [*selected, *ordered_search[: limit - len(selected)]]


def run_target_search(
    plan: SearchPlan,
    source_candidates: Sequence[CandidateRecord],
    evaluator: TargetEvaluator,
    *,
    existing: Sequence[CandidateRecord] = (),
) -> list[CandidateRecord]:
    """Compile/evaluate only the explicit top-N source survivors for one target."""

    resolved_plan = resolve_search_plan(plan)
    selected = select_target_candidates(resolved_plan, source_candidates)
    records: list[CandidateRecord] = []
    for source in selected:
        resumed = _completed_candidate(existing, source.recipe)
        if resumed is not None:
            _validate_candidate_identity(resolved_plan, resumed, source.recipe, target=True)
            if resumed.parent_candidate_id != source.candidate_id:
                raise ValueError("resume target candidate parent identity differs from source candidate")
            if resumed.source_metrics != source.source_metrics:
                raise ValueError("resume target candidate source metrics differ from its parent")
            records.append(_apply_constraints(resolved_plan, resumed, boundary="target"))
            continue
        candidate = evaluator(source)
        if candidate.status == "pending":
            raise ValueError("target evaluator must return a terminal measured, rejected, or unsupported candidate")
        if candidate.recipe != source.recipe or candidate.parent_candidate_id != source.candidate_id:
            raise ValueError("target evaluator must preserve recipe and set the source parent_candidate_id")
        if candidate.source_metrics != source.source_metrics:
            raise ValueError("target evaluator must preserve source metrics")
        _validate_candidate_identity(resolved_plan, candidate, source.recipe, target=True)
        records.append(_apply_constraints(resolved_plan, candidate, boundary="target"))
    return records


def run_search(
    plan: SearchPlan,
    source_evaluator: SourceEvaluator,
    *,
    target_evaluator: TargetEvaluator | None = None,
    existing_source: Sequence[CandidateRecord] = (),
    existing_target: Sequence[CandidateRecord] = (),
    study_id: str = "search-study",
) -> SearchStudyRecord:
    """Run source search and optional target compilation under one immutable budget."""

    resolved_plan = resolve_search_plan(plan)
    source_candidates = run_source_search(resolved_plan, source_evaluator, existing=existing_source)
    source_front = pareto_front(
        source_candidates,
        boundary="source",
        objectives=search_source_objectives(resolved_plan),
        model=resolved_plan.model,
        target=resolved_plan.target,
        front_id=f"{study_id}-source",
        search_plan_hash=search_plan_hash(resolved_plan),
    )
    target_candidates: list[CandidateRecord] = []
    target_front: ParetoFrontRecord | None = None
    compiler_build_count = 0
    if target_evaluator is not None:
        target_candidates = run_target_search(
            resolved_plan,
            source_candidates,
            target_evaluator,
            existing=existing_target,
        )
        compiler_build_count = len(target_candidates)
        measured_target = [candidate for candidate in target_candidates if candidate.status == "measured"]
        if measured_target:
            target_front = pareto_front(
                measured_target,
                boundary="target",
                objectives=resolved_plan.objectives,
                model=resolved_plan.model,
                target=resolved_plan.target,
                front_id=f"{study_id}-target",
                search_plan_hash=search_plan_hash(resolved_plan),
            )
    statuses = [candidate.status for candidate in [*source_candidates, *target_candidates]]
    study_status: Literal["pending", "measured", "rejected"]
    if target_evaluator is None or any(status == "pending" for status in statuses):
        study_status = "pending"
    elif any(candidate.status == "measured" for candidate in target_candidates):
        study_status = "measured"
    else:
        study_status = "rejected"
    return SearchStudyRecord(
        study_id=study_id,
        status=study_status,
        plan=resolved_plan,
        plan_hash=search_plan_hash(resolved_plan),
        source_candidates=source_candidates,
        target_candidates=target_candidates,
        source_front=source_front,
        target_front=target_front,
        compiler_build_count=compiler_build_count,
        gate40_count=0,
        full400_count=0,
        notes=[
            "Search ranking is not human acceptance.",
            "Target Pareto values are target-local measurements; source quality and target cost remain separate evidence lanes.",
        ],
    )


def validate_resume_identity(plan: SearchPlan, candidates: Sequence[CandidateRecord]) -> None:
    """Fail fast if a resumable record is not tied to the current plan and split identities."""

    resolved_plan = resolve_search_plan(plan)
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    recipe_hashes = [candidate.recipe.recipe_hash for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("resume candidate IDs must be unique")
    if len(recipe_hashes) != len(set(recipe_hashes)):
        raise ValueError("resume recipe hashes must be unique")
    recipe_by_hash = {recipe.recipe_hash: recipe for recipe in generate_candidate_recipes(resolved_plan)}
    for candidate in candidates:
        expected_recipe = recipe_by_hash.get(candidate.recipe.recipe_hash)
        if expected_recipe is None:
            raise ValueError("resume candidate recipe was not generated by the current SearchPlan")
        if candidate.recipe != expected_recipe:
            raise ValueError("resume candidate recipe hash maps to a different generated recipe")
        _validate_candidate_identity(
            resolved_plan,
            candidate,
            candidate.recipe,
            target=candidate.parent_candidate_id is not None or candidate.target_metrics is not None,
        )


__all__ = [
    "SourceEvaluator",
    "TargetEvaluator",
    "apply_search_constraints",
    "generate_candidate_recipes",
    "make_candidate_recipe",
    "pareto_front",
    "resolve_search_plan",
    "run_search",
    "run_source_search",
    "run_target_search",
    "select_target_candidates",
    "validate_resume_identity",
]
