"""MoE routing metrics and bounded rule-based diagnosis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

from .trace import TraceBatch, TraceError


@dataclass(frozen=True, slots=True)
class Finding:
    code: str
    severity: str
    message: str


@dataclass(frozen=True, slots=True)
class ClinicReport:
    experts: int
    batches: int
    tokens: int
    assignments: int
    mean_top_k: float
    alpha: float
    z_coeff: float
    load_balance_loss: float
    z_loss: float | None
    assignment_shares: tuple[float, ...]
    probability_means: tuple[float, ...]
    normalized_entropy: float
    load_cv: float
    load_gini: float
    dead_experts: tuple[int, ...]
    max_expert: int
    max_share: float
    capacity_overflow: int
    dropped_assignments: int
    severity: str
    findings: tuple[Finding, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["findings"] = [asdict(finding) for finding in self.findings]
        return payload


def _gini(values: tuple[float, ...]) -> float:
    ordered = sorted(values)
    total = sum(ordered)
    if total == 0.0:
        return 0.0
    count = len(ordered)
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2.0 * weighted) / (count * total) - (count + 1.0) / count


def _severity(findings: list[Finding]) -> str:
    if any(finding.severity == "collapse" for finding in findings):
        return "collapse"
    if findings:
        return "warning"
    return "healthy"


def analyze(
    batches: Iterable[TraceBatch],
    *,
    alpha: float = 0.01,
    z_coeff: float = 1e-4,
    dead_threshold: float = 0.01,
) -> ClinicReport:
    traces = tuple(batches)
    if not traces:
        raise TraceError("at least one trace batch is required")
    if not (math.isfinite(alpha) and alpha >= 0.0):
        raise ValueError("alpha must be finite and non-negative")
    if not (math.isfinite(z_coeff) and z_coeff >= 0.0):
        raise ValueError("z_coeff must be finite and non-negative")
    if not (math.isfinite(dead_threshold) and 0.0 <= dead_threshold < 1.0):
        raise ValueError("dead_threshold must be between 0 and 1")

    experts = traces[0].experts
    if any(trace.experts != experts for trace in traces):
        raise TraceError("all batches must use the same expert count")
    counts = tuple(sum(trace.assignment_counts[i] for trace in traces) for i in range(experts))
    probability_sums = tuple(sum(trace.probability_sums[i] for trace in traces) for i in range(experts))
    tokens = sum(trace.token_count for trace in traces)
    assignments = sum(counts)
    if tokens <= 0 or assignments <= 0:
        raise TraceError("trace must contain tokens and assignments")

    shares = tuple(count / assignments for count in counts)
    means = tuple(value / tokens for value in probability_sums)
    auxiliary = alpha * experts * sum(fraction * probability for fraction, probability in zip(shares, means))

    lse_sum = sum(trace.logsumexp_sq_sum or 0.0 for trace in traces)
    logit_tokens = sum(trace.logit_token_count for trace in traces)
    z_loss = z_coeff * lse_sum / logit_tokens if logit_tokens else None

    entropy = -sum(value * math.log(value) for value in shares if value > 0.0)
    normalized_entropy = entropy / math.log(experts) if experts > 1 else 1.0
    mean_load = assignments / experts
    variance = sum((count - mean_load) ** 2 for count in counts) / experts
    load_cv = math.sqrt(variance) / mean_load
    gini = _gini(shares)
    dead = tuple(index for index, share in enumerate(shares) if share < dead_threshold)
    max_expert = max(range(experts), key=shares.__getitem__)
    max_share = shares[max_expert]
    overflow = sum(
        sum(max(0, count - trace.capacity) for count in trace.assignment_counts)
        for trace in traces
        if trace.capacity is not None
    )
    dropped = sum(trace.dropped_assignments for trace in traces)

    findings: list[Finding] = []
    dead_fraction = len(dead) / experts
    if dead_fraction >= 0.25:
        findings.append(Finding("dead-experts", "collapse", f"{len(dead)}/{experts} experts receive less than {dead_threshold:.1%} of assignments"))
    elif dead:
        findings.append(Finding("dead-experts", "warning", f"experts {', '.join(map(str, dead))} receive less than {dead_threshold:.1%} of assignments"))
    collapse_concentration = max(0.50, 1.50 / experts)
    warning_concentration = max(0.35, 1.25 / experts)
    if max_share >= collapse_concentration:
        findings.append(Finding("router-concentration", "collapse", f"expert {max_expert} receives {max_share:.1%} of assignments"))
    elif max_share >= warning_concentration:
        findings.append(Finding("router-concentration", "warning", f"expert {max_expert} receives {max_share:.1%} of assignments"))
    if normalized_entropy < 0.65:
        findings.append(Finding("low-entropy", "collapse", f"normalized assignment entropy is {normalized_entropy:.3f}"))
    elif normalized_entropy < 0.85:
        findings.append(Finding("low-entropy", "warning", f"normalized assignment entropy is {normalized_entropy:.3f}"))
    if load_cv > 0.50:
        findings.append(Finding("load-variance", "warning", f"expert load coefficient of variation is {load_cv:.3f}"))
    if overflow:
        findings.append(Finding("capacity-overflow", "warning", f"{overflow} assignments exceed recorded expert capacities"))
    if dropped:
        findings.append(Finding("dropped-assignments", "warning", f"trace reports {dropped} dropped assignments"))

    return ClinicReport(
        experts=experts,
        batches=len(traces),
        tokens=tokens,
        assignments=assignments,
        mean_top_k=assignments / tokens,
        alpha=alpha,
        z_coeff=z_coeff,
        load_balance_loss=auxiliary,
        z_loss=z_loss,
        assignment_shares=shares,
        probability_means=means,
        normalized_entropy=normalized_entropy,
        load_cv=load_cv,
        load_gini=gini,
        dead_experts=dead,
        max_expert=max_expert,
        max_share=max_share,
        capacity_overflow=overflow,
        dropped_assignments=dropped,
        severity=_severity(findings),
        findings=tuple(findings),
    )
