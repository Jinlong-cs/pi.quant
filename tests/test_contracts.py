from pathlib import Path

import pytest

from piquant.contracts import (
    ActionSchema,
    ArtifactRef,
    BenchmarkProtocol,
    CandidateMetrics,
    CandidateRecipe,
    CandidateRecord,
    CaptureSpec,
    GoldenCaptureManifest,
    ModelSpec,
    OptimizationPlan,
    PromotionEvidence,
    SearchBudget,
    SearchConstraint,
    SearchObjective,
    SearchPlan,
    SearchSplitAudit,
    SearchStudyRecord,
    SensitivitySignal,
    TargetFingerprint,
    fingerprint,
    load_plan,
    search_plan_hash,
    search_source_objectives,
)
from piquant.promotion import apply_promotion_evidence, build_promotion_plan, next_promotion_gate
from piquant.search import (
    generate_candidate_recipes,
    make_candidate_recipe,
    pareto_front,
    resolve_search_plan,
    run_source_search,
    run_target_search,
    select_target_candidates,
    validate_resume_identity,
)


def _artifact(kind: str, token: str) -> ArtifactRef:
    return ArtifactRef(kind=kind, path=f"/external/{kind}", sha256=token * 64)


def _search_plan() -> SearchPlan:
    controls = [
        make_candidate_recipe(
            recipe_id="fp",
            control_kind="fp_control",
            precision_map={"vision": "fp16", "language": "fp16", "action": "fp16"},
        ),
        make_candidate_recipe(
            recipe_id="broad",
            control_kind="broad_quant",
            precision_map={"vision": "int8", "language": "int8", "action": "int8"},
        ),
        make_candidate_recipe(
            recipe_id="manual",
            control_kind="manual_selective",
            precision_map={"vision": "int8", "language": "int8", "action": "fp16"},
        ),
    ]
    signals = [
        SensitivitySignal(
            group="action",
            from_precision="int8",
            to_precision="fp16",
            quality_recovery=0.8,
            latency_cost_ms=2.0,
            source_evidence=_artifact("action-source", "1"),
            target_cost_evidence=_artifact("action-target", "2"),
        ),
        SensitivitySignal(
            group="vision",
            from_precision="int8",
            to_precision="fp16",
            quality_recovery=0.4,
            latency_cost_ms=1.0,
            source_evidence=_artifact("vision-source", "3"),
            target_cost_evidence=_artifact("vision-target", "4"),
        ),
        SensitivitySignal(
            group="language",
            from_precision="int8",
            to_precision="fp16",
            quality_recovery=0.2,
            latency_cost_ms=1.0,
            source_evidence=_artifact("language-source", "5"),
            target_cost_evidence=_artifact("language-target", "6"),
        ),
    ]
    return SearchPlan(
        plan_id="v05-test",
        model=ModelSpec(model_id="pi05", family="vla", framework="torch", revision="test", action_dim=7, action_horizon=50),
        target=TargetFingerprint(platform="Linux-x86_64", python_version="3.12", device="cuda:0", gpu_name="RTX 5090"),
        benchmark=BenchmarkProtocol(
            protocol_id="standalone-50x200",
            timing_boundary="standalone",
            warmup=50,
            repeat=200,
            synchronization="cuda-event-plus-host-wall",
        ),
        objective="minimize_target_full_policy_latency",
        objectives=[
            SearchObjective(name="quality_drift", direction="minimize"),
            SearchObjective(name="latency_p95_ms", direction="minimize"),
            SearchObjective(name="memory_mib", direction="minimize"),
            SearchObjective(name="quant_coverage", direction="maximize"),
            SearchObjective(name="closed_loop_evidence_level", direction="maximize"),
        ],
        constraints=[
            SearchConstraint(boundary="source", name="quality_drift", operator="le", threshold=0.5),
            SearchConstraint(boundary="source", name="quant_coverage", operator="ge", threshold=0.5),
            SearchConstraint(boundary="target", name="quality_drift", operator="le", threshold=0.5),
            SearchConstraint(boundary="target", name="latency_p95_ms", operator="le", threshold=30.0),
            SearchConstraint(boundary="target", name="memory_mib", operator="le", threshold=200.0),
            SearchConstraint(boundary="target", name="quant_coverage", operator="ge", threshold=0.5),
        ],
        precision_space=["fp16", "int8"],
        semantic_groups=["vision", "language", "action"],
        controls=controls,
        sensitivity_signals=signals,
        split_audit=SearchSplitAudit(
            calibration_episode_ids=["suite/task/episode-0"],
            sensitivity_episode_ids=["suite/task/episode-1"],
            search_validation_episode_ids=["suite/task/episode-2"],
            promotion_reserved_episode_ids=["suite/task/episode-3"],
            calibration_seed_ids=[10],
            sensitivity_seed_ids=[11],
            search_validation_seed_ids=[12],
            promotion_reserved_seed_ids=[13],
            stage_rule_hash="a" * 64,
            manifest_fingerprints={
                "calibration": "b" * 64,
                "sensitivity": "c" * 64,
                "search_validation": "d" * 64,
                "promotion_reserved": "e" * 64,
            },
        ),
        budget=SearchBudget(
            beam_width=2,
            max_source_candidates=8,
            max_compiler_builds=4,
            max_gate40=1,
            max_full400=1,
            target_compile_limit=4,
        ),
        seed=17,
    )


def _candidate(plan: SearchPlan, recipe_index: int, *, target_metrics: CandidateMetrics | None = None) -> CandidateRecord:
    recipe = plan.controls[recipe_index]
    fingerprints = plan.split_audit.manifest_fingerprints
    source_metrics = CandidateMetrics(
        shape_match=True,
        finite=True,
        quality_drift=0.0 if recipe.control_kind == "fp_control" else 0.1,
        action_drift=0.0 if recipe.control_kind == "fp_control" else 0.1,
        quant_coverage=0.0 if recipe.control_kind == "fp_control" else 0.8,
    )
    return CandidateRecord(
        candidate_id=recipe.recipe_id,
        search_plan_hash=search_plan_hash(plan),
        recipe=recipe,
        model=plan.model,
        target=plan.target,
        calibration_manifest_fingerprint=fingerprints["calibration"],
        sensitivity_manifest_fingerprint=fingerprints["sensitivity"],
        search_validation_manifest_fingerprint=fingerprints["search_validation"],
        promotion_reserved_manifest_fingerprint=fingerprints["promotion_reserved"],
        source_metrics=source_metrics,
        target_metrics=target_metrics,
        compiler_evidence=[] if target_metrics is None else [_artifact(f"compiler-{recipe.recipe_id}", str(recipe_index + 1))],
        timing_evidence=[] if target_metrics is None else [_artifact(f"timing-{recipe.recipe_id}", str(recipe_index + 4))],
        status="measured",
    )


def test_recipe_loads_through_one_schema() -> None:
    plan = load_plan(Path("recipes/synthetic/flow-vla-int8.yaml"))
    assert isinstance(plan, OptimizationPlan)
    assert plan.backend == "modelopt"
    assert plan.calibration.stages == ["approach", "grasp", "lift", "place"]
    assert fingerprint(plan) == fingerprint(plan.model_copy(deep=True))


def test_pi05_recipes_share_the_audited_capture_and_calibration_contract() -> None:
    control = load_plan(Path("recipes/pi05/libero-fp-control.yaml"))
    broad = load_plan(Path("recipes/pi05/libero-int8-broad.yaml"))
    assert control.capture_points == broad.capture_points
    assert len(control.capture_points) == 33
    assert control.calibration == broad.calibration
    assert control.representation == "fp_control"
    assert broad.representation == "fake_quant"


def test_unknown_recipe_fields_fail_fast() -> None:
    with pytest.raises(ValueError):
        OptimizationPlan.model_validate(
            {
                "plan_id": "bad",
                "backend": "modelopt",
                "calibration": {
                    "dataset_id": "synthetic",
                    "sample_count": 1,
                    "seed": 1,
                    "stages": ["approach"],
                    "input_fields": ["observation"],
                    "unexpected": True,
                },
                "seed": 1,
                "capture_points": ["action"],
                "timing_boundary": "offline",
            }
        )


def test_golden_manifest_rejects_duplicate_sample_lineage() -> None:
    with pytest.raises(ValueError, match="sample IDs must be unique"):
        GoldenCaptureManifest(
            manifest_id="duplicate",
            model=ModelSpec(model_id="model", family="test", framework="test", action_dim=7, action_horizon=1),
            action_schema=ActionSchema(
                model_action_dim=7,
                output_action_dim=7,
                horizon=1,
                denoise_steps=1,
                translation_indices=[0, 1, 2],
                rotation_indices=[3, 4, 5],
                gripper_index=6,
                postprocess="identity",
            ),
            holdout_manifest_fingerprint="0" * 64,
            capture_specs=[CaptureSpec(logical_id="action", backend_path="$action", component="head", kind="action")],
            chunks=[
                {"chunk_id": "a", "sample_ids": ["sample"], "artifact": {"kind": "capture", "path": "/a", "sha256": "1" * 64}},
                {"chunk_id": "b", "sample_ids": ["sample"], "artifact": {"kind": "capture", "path": "/b", "sha256": "2" * 64}},
            ],
            seed=1,
            status="measured",
        )


def test_v05_search_plan_is_deterministic_and_split_disjoint() -> None:
    plan = _search_plan()
    resolved = resolve_search_plan(plan)
    first = generate_candidate_recipes(resolved)
    second = generate_candidate_recipes(resolved)
    assert resolved.plan_hash == resolve_search_plan(plan.model_copy(deep=True)).plan_hash
    assert [(recipe.recipe_id, recipe.recipe_hash) for recipe in first] == [(recipe.recipe_id, recipe.recipe_hash) for recipe in second]
    assert len(first) <= plan.budget.max_source_candidates
    assert len({tuple(sorted(recipe.precision_map.items())) for recipe in first}) == len(first)
    with pytest.raises(TypeError):
        plan.controls[0].precision_map["vision"] = "int8"  # type: ignore[index]
    overlapping = plan.split_audit.model_dump(mode="json")
    overlapping["sensitivity_episode_ids"] = plan.split_audit.calibration_episode_ids
    with pytest.raises(ValueError, match="episode overlap"):
        SearchSplitAudit.model_validate(overlapping)
    duplicated = plan.split_audit.model_dump(mode="json")
    duplicated["calibration_episode_ids"] = ["suite/task/episode-0", "suite/task/episode-0"]
    with pytest.raises(ValueError, match="duplicate episode IDs"):
        SearchSplitAudit.model_validate(duplicated)
    missing_latency_gate = plan.model_dump(mode="json", exclude={"plan_hash"})
    missing_latency_gate["constraints"] = [
        constraint
        for constraint in missing_latency_gate["constraints"]
        if not (constraint["boundary"] == "target" and constraint["name"] == "latency_p95_ms")
    ]
    with pytest.raises(ValueError, match="required hard constraints"):
        SearchPlan.model_validate(missing_latency_gate)

    rollout_payload = plan.model_dump(mode="json", exclude={"plan_hash"})
    rollout_payload["objectives"][0]["name"] = "rollout_drift"
    for constraint in rollout_payload["constraints"]:
        if constraint["name"] == "quality_drift":
            constraint["name"] = "rollout_drift"
    rollout_plan = SearchPlan.model_validate(rollout_payload)
    assert [objective.name for objective in search_source_objectives(rollout_plan)] == ["rollout_drift", "quant_coverage"]


def test_v05_source_search_resumes_only_matching_completed_records() -> None:
    plan = _search_plan()
    calls: list[str] = []

    def evaluate(recipe: CandidateRecipe) -> CandidateRecord:
        calls.append(recipe.recipe_id)
        fingerprints = plan.split_audit.manifest_fingerprints
        quality_drift = {"fp_control": 0.0, "broad_quant": 0.6}.get(recipe.control_kind, 0.1)
        return CandidateRecord(
            candidate_id=recipe.recipe_id,
            search_plan_hash=search_plan_hash(plan),
            recipe=recipe,
            model=plan.model,
            target=plan.target,
            calibration_manifest_fingerprint=fingerprints["calibration"],
            sensitivity_manifest_fingerprint=fingerprints["sensitivity"],
            search_validation_manifest_fingerprint=fingerprints["search_validation"],
            promotion_reserved_manifest_fingerprint=fingerprints["promotion_reserved"],
            source_metrics=CandidateMetrics(
                shape_match=True,
                finite=True,
                quality_drift=quality_drift,
                quant_coverage=0.0 if recipe.control_kind == "fp_control" else 0.75,
            ),
            status="measured",
        )

    first = run_source_search(plan, evaluate)
    assert len(calls) == len(first)
    assert first[1].status == "rejected"
    assert first[1].reason_code == "SOURCE_CONSTRAINT_QUALITY_DRIFT"
    calls.clear()
    resumed = run_source_search(plan, evaluate, existing=first)
    assert calls == []
    assert resumed == first
    invalid_candidate_id = first[0].model_dump(mode="json")
    invalid_candidate_id["candidate_id"] = "different-candidate"
    with pytest.raises(ValueError, match="candidate_id must equal recipe_id"):
        CandidateRecord.model_validate(invalid_candidate_id)
    selected = select_target_candidates(plan, first)
    assert [candidate.recipe.control_kind for candidate in selected[:3]] == ["fp_control", "broad_quant", "manual_selective"]
    assert selected[3].recipe.control_kind == "search"

    target_calls: list[str] = []

    def compile_candidate(source: CandidateRecord) -> CandidateRecord:
        target_calls.append(source.candidate_id)
        coverage = source.source_metrics.quant_coverage if source.source_metrics else 0.0
        payload = source.model_dump(mode="python")
        payload.update(
            {
                "parent_candidate_id": source.candidate_id,
                "target_metrics": CandidateMetrics(
                    shape_match=True,
                    finite=True,
                    implementation_parity_passed=True,
                    build_succeeded=True,
                    quality_drift=0.0 if source.recipe.control_kind == "fp_control" else 0.1,
                    latency_p95_ms=20.0,
                    memory_mib=100,
                    quant_coverage=coverage,
                ),
                "compiler_evidence": [_artifact(f"compiler-{source.candidate_id}", "7")],
                "timing_evidence": [_artifact(f"timing-{source.candidate_id}", "8")],
                "status": "measured",
                "reason_code": None,
                "reason": None,
            }
        )
        return CandidateRecord.model_validate(payload)

    target = run_target_search(plan, first, compile_candidate)
    assert target_calls == [candidate.candidate_id for candidate in selected]
    assert all(candidate.parent_candidate_id == candidate.candidate_id for candidate in target)
    target_calls.clear()
    assert run_target_search(plan, first, compile_candidate, existing=target) == target
    assert target_calls == []

    bad = first[0].model_copy(update={"search_validation_manifest_fingerprint": "f" * 64})
    with pytest.raises(ValueError, match="split identity differs"):
        run_source_search(plan, evaluate, existing=[bad, *first[1:]])
    duplicate_recipe = first[0].model_copy(update={"candidate_id": "duplicate"})
    with pytest.raises(ValueError, match="resume recipe hashes must be unique"):
        validate_resume_identity(plan, [first[0], duplicate_recipe])
    overflow = [first[0] for _ in range(9)]
    with pytest.raises(ValueError, match="source candidate count"):
        SearchStudyRecord(
            study_id="overflow",
            status="pending",
            plan=plan,
            plan_hash=search_plan_hash(plan),
            source_candidates=overflow,
            compiler_build_count=0,
            gate40_count=0,
            full400_count=0,
        )


def test_v05_pareto_ties_and_promotion_approval_boundary() -> None:
    plan = _search_plan()
    broad = _candidate(
        plan,
        1,
        target_metrics=CandidateMetrics(
            shape_match=True,
            finite=True,
            implementation_parity_passed=True,
            build_succeeded=True,
            quality_drift=0.1,
            latency_p95_ms=20.0,
            memory_mib=100,
            quant_coverage=0.8,
            uncertainty={"quality_drift": 0.02, "latency_p95_ms": 0.2},
        ),
    )
    manual = _candidate(
        plan,
        2,
        target_metrics=CandidateMetrics(
            shape_match=True,
            finite=True,
            implementation_parity_passed=True,
            build_succeeded=True,
            quality_drift=0.11,
            latency_p95_ms=20.1,
            memory_mib=100,
            quant_coverage=0.8,
            uncertainty={"quality_drift": 0.02, "latency_p95_ms": 0.2},
        ),
    )
    fp = _candidate(
        plan,
        0,
        target_metrics=CandidateMetrics(
            shape_match=True,
            finite=True,
            implementation_parity_passed=True,
            build_succeeded=True,
            quality_drift=0.2,
            latency_p95_ms=25.0,
            memory_mib=120,
            quant_coverage=0.0,
        ),
    )
    front = pareto_front(
        [broad, manual, fp],
        boundary="target",
        objectives=plan.objectives,
        model=plan.model,
        target=plan.target,
        front_id="target-front",
        search_plan_hash=search_plan_hash(plan),
    )
    assert front.non_dominated_candidate_ids == ["broad", "manual"]
    assert front.dominated_candidate_ids == ["fp"]
    assert front.tie_groups == [["broad", "manual"]]

    promotion = build_promotion_plan(manual, baseline=fp, target_front=front, search_plan=plan)
    assert next_promotion_gate(promotion).gate_id == "mechanical"
    with pytest.raises(ValueError, match="matched FP control"):
        build_promotion_plan(manual, baseline=broad, target_front=front, search_plan=plan)
    with pytest.raises(ValueError, match="non-dominated"):
        build_promotion_plan(fp, baseline=fp, target_front=front, search_plan=plan)
    no_promotion_budget = plan.model_dump(mode="json", exclude={"plan_hash"})
    no_promotion_budget["budget"]["max_gate40"] = 0
    no_promotion_budget["budget"]["max_full400"] = 0
    no_promotion_plan = SearchPlan.model_validate(no_promotion_budget)
    with pytest.raises(ValueError, match="reserve one gate40"):
        build_promotion_plan(manual, baseline=fp, target_front=front, search_plan=no_promotion_plan)
    stale_front = front.model_copy(
        update={"objective_values": {**front.objective_values, "manual": {**front.objective_values["manual"], "latency_p95_ms": 1.0}}}
    )
    with pytest.raises(ValueError, match="Pareto values differ"):
        build_promotion_plan(manual, baseline=fp, target_front=stale_front, search_plan=plan)
    mechanical = PromotionEvidence(
        evidence_id="mechanical-pass",
        promotion_plan_id=promotion.plan_id,
        candidate_id=manual.candidate_id,
        gate_id="mechanical",
        status="measured",
        artifacts=[_artifact("mechanical", "7")],
        verified_by="machine",
        verified_at="2026-08-07T00:00:00Z",
    )
    promotion = apply_promotion_evidence(promotion, mechanical)
    assert next_promotion_gate(promotion).gate_id == "offline"
    invalid_status = promotion.model_dump(mode="json")
    invalid_status["status"] = "measured"
    with pytest.raises(ValueError, match="promotion plan status"):
        type(promotion).model_validate(invalid_status)
    invalid_order = promotion.model_dump(mode="json")
    invalid_order["gates"][1].update(status="rejected", reason_code="OFFLINE_REGRESSION")
    invalid_order["gates"][2].update(status="unsupported", reason_code="TARGET_UNAVAILABLE")
    invalid_order["status"] = "rejected"
    with pytest.raises(ValueError, match="must remain pending"):
        type(promotion).model_validate(invalid_order)
    with pytest.raises(ValueError, match="approval_record"):
        PromotionEvidence(
            evidence_id="gate40-without-approval",
            promotion_plan_id=promotion.plan_id,
            candidate_id=manual.candidate_id,
            gate_id="gate40",
            status="measured",
            artifacts=[_artifact("gate40", "8")],
            verified_by="machine",
            verified_at="2026-08-07T00:00:00Z",
        )
    with pytest.raises(ValueError, match="approval_record"):
        PromotionEvidence(
            evidence_id="gate40-failed-without-approval",
            promotion_plan_id=promotion.plan_id,
            candidate_id=manual.candidate_id,
            gate_id="gate40",
            status="rejected",
            reason_code="CLOSED_LOOP_REGRESSION",
        )
    with pytest.raises(ValueError):
        PromotionEvidence(
            evidence_id="accepted-is-not-a-machine-state",
            promotion_plan_id=promotion.plan_id,
            candidate_id=manual.candidate_id,
            gate_id="offline",
            status="accepted",
        )
