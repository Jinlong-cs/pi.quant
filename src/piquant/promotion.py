"""Pending-first promotion gate planning for mixed-precision candidates."""

from __future__ import annotations

from collections.abc import Sequence

from piquant.contracts import (
    ArtifactRef,
    CandidateRecord,
    ParetoFrontRecord,
    PromotionEvidence,
    PromotionGate,
    PromotionGateId,
    PromotionPlan,
    SearchCandidateStatus,
    SearchPlan,
    search_plan_hash,
)
from piquant.search import validate_resume_identity

GATE_ORDER: tuple[PromotionGateId, ...] = (
    "mechanical",
    "offline",
    "target_latency",
    "server_client_smoke",
    "gate40",
    "full400",
)


def build_promotion_plan(
    candidate: CandidateRecord,
    *,
    baseline: CandidateRecord,
    target_front: ParetoFrontRecord,
    search_plan: SearchPlan,
    plan_id: str | None = None,
) -> PromotionPlan:
    """Create a plan with every gate pending and high-cost approvals withheld."""

    if candidate.status != "measured":
        raise ValueError("promotion plan requires a measured candidate")
    if candidate.target_metrics is None:
        raise ValueError("promotion plan requires target metrics")
    if baseline.status != "measured" or baseline.target_metrics is None:
        raise ValueError("promotion baseline requires measured target metrics")
    if baseline.recipe.control_kind != "fp_control":
        raise ValueError("promotion baseline must be the matched FP control")
    if search_plan.budget.max_gate40 < 1 or search_plan.budget.max_full400 < 1:
        raise ValueError("SearchPlan budget must reserve one gate40 and one full400 attempt before promotion planning")
    plan_hash = search_plan_hash(search_plan)
    if candidate.search_plan_hash != plan_hash or candidate.model != search_plan.model or candidate.target != search_plan.target:
        raise ValueError("promotion candidate identity differs from SearchPlan")
    if baseline.search_plan_hash != plan_hash or baseline.model != search_plan.model or baseline.target != search_plan.target:
        raise ValueError("promotion baseline identity differs from SearchPlan")
    if target_front.boundary != "target" or target_front.search_plan_hash != plan_hash:
        raise ValueError("promotion Pareto front identity differs from SearchPlan")
    if (
        target_front.model != search_plan.model
        or target_front.target != search_plan.target
        or target_front.objectives != search_plan.objectives
    ):
        raise ValueError("promotion Pareto front model, target, or objectives differ from SearchPlan")
    if candidate.candidate_id not in target_front.non_dominated_candidate_ids:
        raise ValueError("promotion candidate must be non-dominated on the target Pareto front")
    if baseline.candidate_id not in target_front.candidate_ids:
        raise ValueError("promotion baseline must be measured in the target Pareto front")
    validate_resume_identity(search_plan, [candidate, baseline])
    for record in (candidate, baseline):
        metrics = record.target_metrics
        if metrics is None:
            raise ValueError("promotion candidate and baseline require target metrics")
        expected_values: dict[str, float] = {}
        for objective in search_plan.objectives:
            value = getattr(metrics, objective.name)
            if value is None:
                raise ValueError(f"promotion target metrics are missing objective {objective.name!r}")
            expected_values[objective.name] = float(value)
        expected_uncertainty = {objective.name: float(metrics.uncertainty.get(objective.name, 0.0)) for objective in search_plan.objectives}
        if target_front.objective_values[record.candidate_id] != expected_values:
            raise ValueError("promotion Pareto values differ from candidate target metrics")
        if target_front.objective_uncertainty[record.candidate_id] != expected_uncertainty:
            raise ValueError("promotion Pareto uncertainty differs from candidate target metrics")
    quality_constraints = {
        f"{constraint.boundary}:{constraint.name}:{constraint.operator}": constraint.threshold
        for constraint in search_plan.constraints
        if constraint.name != "latency_p95_ms"
    }
    quality_constraints.update({f"promotion:{name}": value for name, value in search_plan.promotion_constraints.items()})
    latency_constraints = {
        f"{constraint.boundary}:{constraint.name}:{constraint.operator}": constraint.threshold
        for constraint in search_plan.constraints
        if constraint.name == "latency_p95_ms"
    }
    gates = [
        PromotionGate(gate_id="mechanical", status="pending"),
        PromotionGate(gate_id="offline", status="pending"),
        PromotionGate(gate_id="target_latency", status="pending"),
        PromotionGate(gate_id="server_client_smoke", status="pending"),
        PromotionGate(gate_id="gate40", status="pending", approval_required=True),
        PromotionGate(gate_id="full400", status="pending", approval_required=True),
    ]
    return PromotionPlan(
        plan_id=plan_id or f"promotion-{candidate.candidate_id}",
        search_plan_hash=plan_hash,
        target_front_id=target_front.front_id,
        candidate_id=candidate.candidate_id,
        candidate_recipe_hash=candidate.recipe.recipe_hash,
        baseline_candidate_id=baseline.candidate_id,
        baseline_recipe_hash=baseline.recipe.recipe_hash,
        model=candidate.model,
        target=candidate.target,
        benchmark=search_plan.benchmark,
        quality_constraints=quality_constraints,
        latency_constraints=latency_constraints,
        gates=gates,
        notes=[
            "Promotion gates are ordered and pending by default.",
            "The Agent may record machine evidence but cannot mark a deployment candidate accepted.",
        ],
    )


def next_promotion_gate(plan: PromotionPlan) -> PromotionGate:
    """Return the first not-measured gate; callers must inspect approval before running it."""

    if plan.status == "rejected":
        raise ValueError("rejected promotion plans have no executable next gate")
    for gate in plan.gates:
        if gate.status != "measured":
            return gate
    raise ValueError("promotion plan has no pending gate")


def apply_promotion_evidence(plan: PromotionPlan, evidence: PromotionEvidence) -> PromotionPlan:
    """Attach one gate result while preserving the pending/measured/rejected boundary."""

    if evidence.promotion_plan_id != plan.plan_id or evidence.candidate_id != plan.candidate_id:
        raise ValueError("promotion evidence identity differs from plan")
    gate_index = next((index for index, gate in enumerate(plan.gates) if gate.gate_id == evidence.gate_id), None)
    if gate_index is None:
        raise ValueError(f"promotion evidence references unknown gate {evidence.gate_id!r}")
    current = plan.gates[gate_index]
    if current.status != "pending":
        raise ValueError(f"promotion gate {evidence.gate_id!r} is already terminal")
    if evidence.status == "pending":
        raise ValueError("pending evidence does not advance a promotion plan")
    prior = plan.gates[:gate_index]
    if any(gate.status != "measured" for gate in prior):
        raise ValueError("promotion gates must be completed in order")
    if current.approval_required and evidence.status == "measured" and evidence.approval_record is None:
        raise ValueError("high-cost promotion evidence requires an approval record")

    updated_gate_payload = current.model_dump(mode="python")
    updated_gate_payload.update(
        status=evidence.status,
        approval_record=evidence.approval_record,
        evidence=[*current.evidence, *evidence.artifacts],
        reason_code=evidence.reason_code,
    )
    updated_gate = PromotionGate.model_validate(updated_gate_payload)
    gates = list(plan.gates)
    gates[gate_index] = updated_gate
    statuses = [gate.status for gate in gates]
    status = (
        "rejected"
        if any(value in {"rejected", "unsupported"} for value in statuses)
        else "measured"
        if all(value == "measured" for value in statuses)
        else "pending"
    )
    updated_plan_payload = plan.model_dump(mode="python")
    updated_plan_payload.update(gates=gates, status=status)
    return PromotionPlan.model_validate(updated_plan_payload)


def validate_promotion_evidence(plan: PromotionPlan, evidence: Sequence[PromotionEvidence]) -> PromotionGate:
    """Validate a resumable gate history and return the next executable boundary."""

    current = plan
    seen: set[str] = set()
    for item in evidence:
        if item.evidence_id in seen:
            raise ValueError(f"duplicate promotion evidence ID {item.evidence_id!r}")
        seen.add(item.evidence_id)
        current = apply_promotion_evidence(current, item)
    if current.status == "rejected":
        return next(gate for gate in current.gates if gate.status in {"rejected", "unsupported"})
    return next_promotion_gate(current)


def promotion_evidence(
    *,
    evidence_id: str,
    promotion_plan_id: str,
    candidate_id: str,
    gate_id: PromotionGateId,
    status: SearchCandidateStatus,
    artifacts: Sequence[ArtifactRef] = (),
    metrics: dict[str, float] | None = None,
    reason_code: str | None = None,
    approval_record: ArtifactRef | None = None,
    verified_by: str = "",
    verified_at: str = "",
    commands: Sequence[str] = (),
) -> PromotionEvidence:
    """Construct a gate evidence record without providing an accepted state."""

    return PromotionEvidence(
        evidence_id=evidence_id,
        promotion_plan_id=promotion_plan_id,
        candidate_id=candidate_id,
        gate_id=gate_id,
        status=status,
        metrics={} if metrics is None else dict(metrics),
        artifacts=list(artifacts),
        approval_record=approval_record,
        reason_code=reason_code,
        commands=list(commands),
        verified_by=verified_by,
        verified_at=verified_at,
    )


__all__ = [
    "GATE_ORDER",
    "apply_promotion_evidence",
    "build_promotion_plan",
    "next_promotion_gate",
    "promotion_evidence",
    "validate_promotion_evidence",
]
