"""Command line interface for router trace diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .metrics import ClinicReport, analyze
from .trace import TraceError, load_jsonl


_RANK = {"healthy": 0, "warning": 1, "collapse": 2}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moeclinic",
        description="Measure load balancing and collapse signals from MoE router JSONL.",
    )
    parser.add_argument("trace", help="raw or compact router trace in JSONL format")
    parser.add_argument("--alpha", type=float, default=0.01, help="auxiliary loss coefficient")
    parser.add_argument("--z-coeff", type=float, default=1e-4, help="router z-loss coefficient")
    parser.add_argument("--dead-threshold", type=float, default=0.01)
    parser.add_argument("--json", action="store_true", help="print machine-readable report")
    parser.add_argument("--output", help="also write the JSON report to this path")
    parser.add_argument(
        "--fail-on",
        choices=("warning", "collapse", "never"),
        default="collapse",
        help="return exit status 1 at or above this diagnosis",
    )
    return parser


def _human(report: ClinicReport) -> str:
    lines = [
        f"MoEClinic: {report.severity.upper()}",
        f"experts={report.experts} batches={report.batches} tokens={report.tokens} assignments={report.assignments}",
        f"aux_loss={report.load_balance_loss:.8f} z_loss={'unavailable' if report.z_loss is None else f'{report.z_loss:.8f}'}",
        f"entropy={report.normalized_entropy:.4f} cv={report.load_cv:.4f} gini={report.load_gini:.4f}",
        f"max_expert={report.max_expert} max_share={report.max_share:.2%} dead={list(report.dead_experts)}",
    ]
    lines.extend(f"- [{finding.severity}] {finding.code}: {finding.message}" for finding in report.findings)
    if not report.findings:
        lines.append("- no configured collapse signal crossed its threshold")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = analyze(
            load_jsonl(args.trace),
            alpha=args.alpha,
            z_coeff=args.z_coeff,
            dead_threshold=args.dead_threshold,
        )
    except (TraceError, ValueError) as exc:
        print(f"moeclinic: {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.output:
        try:
            Path(args.output).write_text(payload + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"moeclinic: cannot write report: {exc}", file=sys.stderr)
            return 2
    print(payload if args.json else _human(report))
    if args.fail_on != "never" and _RANK[report.severity] >= _RANK[args.fail_on]:
        return 1
    return 0
