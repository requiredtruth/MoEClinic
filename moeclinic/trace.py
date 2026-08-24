"""Strict JSONL ingestion for raw and compact MoE router traces."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable


class TraceError(ValueError):
    """Raised when router trace data is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class TraceBatch:
    step: int
    token_count: int
    assignment_counts: tuple[int, ...]
    probability_sums: tuple[float, ...]
    logsumexp_sq_sum: float | None = None
    logit_token_count: int = 0
    dropped_assignments: int = 0
    capacity: int | None = None

    @property
    def experts(self) -> int:
        return len(self.assignment_counts)


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TraceError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise TraceError(f"{name} must be finite and >= {minimum}")
    return result


def _rows(value: object, name: str) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise TraceError(f"{name} must be a non-empty array of rows")
    result: list[list[float]] = []
    width: int | None = None
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or not row:
            raise TraceError(f"{name}[{row_index}] must be a non-empty array")
        parsed = [_finite(item, f"{name}[{row_index}]") for item in row]
        width = len(parsed) if width is None else width
        if len(parsed) != width:
            raise TraceError(f"{name} rows must have the same expert count")
        result.append(parsed)
    return result


def _logsumexp(row: list[float]) -> float:
    maximum = max(row)
    return maximum + math.log(sum(math.exp(value - maximum) for value in row))


def _from_raw(record: dict[str, object], step: int) -> TraceBatch:
    probabilities = _rows(record.get("router_probs"), "router_probs")
    experts = len(probabilities[0])
    for index, row in enumerate(probabilities):
        if any(value < 0.0 or value > 1.0 for value in row):
            raise TraceError(f"router_probs[{index}] values must be between 0 and 1")
        if not math.isclose(sum(row), 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise TraceError(f"router_probs[{index}] must sum to 1")

    selected_value = record.get("selected_experts")
    selections: list[list[int]] = []
    if selected_value is None:
        selections = [[max(range(experts), key=row.__getitem__)] for row in probabilities]
    else:
        if not isinstance(selected_value, list) or len(selected_value) != len(probabilities):
            raise TraceError("selected_experts must contain one entry per token")
        for token, selected in enumerate(selected_value):
            items = selected if isinstance(selected, list) else [selected]
            if not items:
                raise TraceError(f"selected_experts[{token}] cannot be empty")
            parsed = [_integer(item, f"selected_experts[{token}]") for item in items]
            if len(set(parsed)) != len(parsed):
                raise TraceError(f"selected_experts[{token}] contains a duplicate expert")
            if any(item >= experts for item in parsed):
                raise TraceError(f"selected_experts[{token}] is outside the expert range")
            selections.append(parsed)

    counts = [0] * experts
    for selection in selections:
        for expert in selection:
            counts[expert] += 1
    probability_sums = tuple(sum(row[i] for row in probabilities) for i in range(experts))

    logits_value = record.get("router_logits")
    lse_sq_sum: float | None = None
    logit_tokens = 0
    if logits_value is not None:
        logits = _rows(logits_value, "router_logits")
        if len(logits) != len(probabilities) or len(logits[0]) != experts:
            raise TraceError("router_logits must match router_probs shape")
        lse_sq_sum = sum(_logsumexp(row) ** 2 for row in logits)
        logit_tokens = len(logits)

    capacity_value = record.get("capacity")
    capacity = None if capacity_value is None else _integer(capacity_value, "capacity", minimum=1)
    return TraceBatch(
        step=step,
        token_count=len(probabilities),
        assignment_counts=tuple(counts),
        probability_sums=probability_sums,
        logsumexp_sq_sum=lse_sq_sum,
        logit_token_count=logit_tokens,
        dropped_assignments=_integer(record.get("dropped_assignments", 0), "dropped_assignments"),
        capacity=capacity,
    )


def _from_aggregate(record: dict[str, object], step: int) -> TraceBatch:
    token_count = _integer(record.get("token_count"), "token_count", minimum=1)
    raw_counts = record.get("assignment_counts")
    raw_sums = record.get("probability_sums")
    if not isinstance(raw_counts, list) or len(raw_counts) < 2:
        raise TraceError("assignment_counts must contain at least two experts")
    if not isinstance(raw_sums, list) or len(raw_sums) != len(raw_counts):
        raise TraceError("probability_sums must match assignment_counts")
    counts = tuple(_integer(value, "assignment_counts") for value in raw_counts)
    if sum(counts) == 0:
        raise TraceError("assignment_counts cannot all be zero")
    sums = tuple(_finite(value, "probability_sums", minimum=0.0) for value in raw_sums)
    if not math.isclose(sum(sums), token_count, rel_tol=1e-3, abs_tol=1e-3):
        raise TraceError("probability_sums must total token_count")

    lse_value = record.get("logsumexp_sq_sum")
    lse_sq_sum = None if lse_value is None else _finite(lse_value, "logsumexp_sq_sum", minimum=0.0)
    logit_tokens = 0
    if lse_sq_sum is not None:
        logit_tokens = _integer(record.get("logit_token_count", token_count), "logit_token_count", minimum=1)
    elif "logit_token_count" in record:
        raise TraceError("logit_token_count requires logsumexp_sq_sum")

    capacity_value = record.get("capacity")
    capacity = None if capacity_value is None else _integer(capacity_value, "capacity", minimum=1)
    return TraceBatch(
        step=step,
        token_count=token_count,
        assignment_counts=counts,
        probability_sums=sums,
        logsumexp_sq_sum=lse_sq_sum,
        logit_token_count=logit_tokens,
        dropped_assignments=_integer(record.get("dropped_assignments", 0), "dropped_assignments"),
        capacity=capacity,
    )


def parse_record(record: object, line_number: int) -> TraceBatch:
    if not isinstance(record, dict):
        raise TraceError("record must be a JSON object")
    step = _integer(record.get("step", line_number), "step")
    if "router_probs" in record:
        return _from_raw(record, step)
    return _from_aggregate(record, step)


def load_jsonl(path: str | Path) -> tuple[TraceBatch, ...]:
    source = Path(path)
    batches: list[TraceBatch] = []
    try:
        lines: Iterable[str] = source.open("r", encoding="utf-8")
    except OSError as exc:
        raise TraceError(f"cannot open trace: {source}") from exc
    try:
        with lines as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    batches.append(parse_record(record, line_number))
                except (json.JSONDecodeError, TraceError) as exc:
                    raise TraceError(f"line {line_number}: {exc}") from exc
    except OSError as exc:
        raise TraceError(f"cannot read trace: {source}") from exc
    if not batches:
        raise TraceError("trace contains no records")
    experts = batches[0].experts
    if any(batch.experts != experts for batch in batches):
        raise TraceError("expert count changes inside the trace")
    return tuple(batches)
