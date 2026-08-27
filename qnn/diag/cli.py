"""CLI entry: ``python -m qnn.diag [analyze | --checkpoint …]``

Two modes
---------
``python -m qnn.diag analyze --run-dir <run> --cache-dir <cache> …``
    Phase-2 unified per-head analysis (see ``qnn.diag.analyze``).

``python -m qnn.diag --checkpoint <ckpt> …``  (legacy, unchanged)
    Existing diagnostic suite (``qnn.diag.report.run_report``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qnn.diag.report import run_report, render_markdown
from qnn.diag.target_location_probe import add_cli_parser as _add_target_location_probe_parser
from qnn.diag.target_location_probe import run_cli as _run_target_location_probe


# ---------------------------------------------------------------------------
# analyze subcommand
# ---------------------------------------------------------------------------

def _add_analyze_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "analyze",
        help="Run the unified per-head analysis against a trained run + cache.",
        description=(
            "Load a policy from a run directory and run per-head analysis "
            "functions, collecting results into a single JSON report."
        ),
    )
    p.add_argument(
        "--run-dir", type=Path, required=True,
        help="Run directory (must contain config/probe.json + checkpoints/).",
    )
    p.add_argument(
        "--cache-dir", type=Path, required=True,
        help="Root cache directory; resident source is built from "
             "<cache-dir>/precomputed_val.",
    )
    p.add_argument(
        "--heads", type=str, default="attack,look,move",
        help="Comma-separated list of heads to run "
             "(default: attack,look,move). weapon is included if available.",
    )
    p.add_argument(
        "--segment", choices=["engaged", "all"], default="engaged",
        help="Frame segment: 'engaged' (act.target != 0) or 'all' (no filter).",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Torch device string (e.g. cpu, cuda). Auto-detected if omitted.",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        metavar="REPORT.json",
        help="Write the JSON report to this path.",
    )


def _run_analyze(args) -> None:
    from qnn.diag.analyze import run_analyze

    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    run_analyze(
        run_dir=args.run_dir,
        cache_dir=args.cache_dir,
        heads=heads,
        segment=args.segment,
        device=args.device,
        out=args.out,
    )


# ---------------------------------------------------------------------------
# Legacy diagnostic subcommand (run_report)
# ---------------------------------------------------------------------------

def _add_diagnostic_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "diagnostic",
        help="Run the checkpoint-based diagnostic suite (legacy run_report).",
        description=(
            "Run the full diagnostic suite on a trained QNN policy checkpoint."
        ),
    )
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to the best-model checkpoint (best_<run_id>.pth).",
    )
    p.add_argument("--data-dir", type=Path, default=Path("artifacts/collect/qwd"))
    p.add_argument(
        "--run-dir", type=Path, default=None,
        help="Run directory (for history). Defaults to checkpoint's grandparent.",
    )
    p.add_argument("--max-val-shards", type=int, default=1)
    p.add_argument(
        "--max-val-episodes", type=int, default=8,
        help="Episodes to use for ablation/gradients (smaller = faster).",
    )
    p.add_argument(
        "--skip", default="",
        help=(
            "Comma-separated sections to skip: "
            "history,rank,ablation,gradients,participation,attention,pruning,linear_probe"
        ),
    )
    p.add_argument(
        "--include", default="",
        help="Comma-separated slow sections to enable (default off): pruning,linear_probe",
    )
    p.add_argument(
        "--report-out", type=Path, default=None,
        help="Write markdown report here. JSON sibling is written automatically.",
    )


def _run_diagnostic(args) -> None:
    run_dir = args.run_dir
    if run_dir is None and args.checkpoint.exists():
        run_dir = args.checkpoint.parent.parent

    skip    = {s.strip() for s in args.skip.split(",")    if s.strip()}
    include = {s.strip() for s in args.include.split(",") if s.strip()}

    report = run_report(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        run_dir=run_dir,
        max_val_shards=args.max_val_shards,
        max_val_episodes=args.max_val_episodes,
        skip=skip,
        include=include,
    )

    md = render_markdown(report)
    if args.report_out is None:
        print(md)
    else:
        args.report_out.write_text(md)
        json_out = args.report_out.with_suffix(".json")
        json_out.write_text(json.dumps(report, indent=2, default=str))
        print(f"[diag] Wrote {args.report_out}")
        print(f"[diag] Wrote {json_out}")


# ---------------------------------------------------------------------------
# Top-level parser — supports both subcommand and legacy flat-arg styles
# ---------------------------------------------------------------------------

def main() -> None:
    # ── peek at argv to decide routing ──────────────────────────────────────
    import sys
    argv = sys.argv[1:]

    # If the first non-flag argument is a known subcommand, use subparser routing.
    known_subcmds = {"analyze", "diagnostic", "target-location-probe"}
    first_positional = next((a for a in argv if not a.startswith("-")), None)

    if first_positional in known_subcmds:
        # Subcommand mode.
        top = argparse.ArgumentParser(
            description="QNN diagnostic tools.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        sub = top.add_subparsers(dest="subcommand", required=True)
        _add_analyze_parser(sub)
        _add_diagnostic_parser(sub)
        _add_target_location_probe_parser(sub)
        args = top.parse_args(argv)
        if args.subcommand == "analyze":
            _run_analyze(args)
        elif args.subcommand == "diagnostic":
            _run_diagnostic(args)
        elif args.subcommand == "target-location-probe":
            _run_target_location_probe(args)
        return

    # ── Legacy flat-arg mode (backward compatibility) ────────────────────────
    # Kept so that existing callers of ``python -m qnn.diag --checkpoint …``
    # continue to work without modification.
    parser = argparse.ArgumentParser(
        description="Run diagnostic suite on a trained QNN policy checkpoint.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to the best-model checkpoint (best_<run_id>.pth)",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/collect/qwd"))
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Run directory (for history). Defaults to checkpoint's grandparent.",
    )
    parser.add_argument("--max-val-shards", type=int, default=1)
    parser.add_argument(
        "--max-val-episodes", type=int, default=8,
        help="Episodes to use for ablation/gradients (smaller = faster).",
    )
    parser.add_argument(
        "--skip", default="",
        help=(
            "Comma-separated sections to skip: "
            "history,rank,ablation,gradients,participation,attention,pruning,linear_probe"
        ),
    )
    parser.add_argument(
        "--include", default="",
        help="Comma-separated slow sections to enable (default off): pruning,linear_probe",
    )
    parser.add_argument(
        "--report-out", type=Path, default=None,
        help="Write markdown report here. JSON sibling is written automatically.",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    if run_dir is None and args.checkpoint.exists():
        run_dir = args.checkpoint.parent.parent

    skip    = {s.strip() for s in args.skip.split(",")    if s.strip()}
    include = {s.strip() for s in args.include.split(",") if s.strip()}

    report = run_report(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        run_dir=run_dir,
        max_val_shards=args.max_val_shards,
        max_val_episodes=args.max_val_episodes,
        skip=skip,
        include=include,
    )

    md = render_markdown(report)
    if args.report_out is None:
        print(md)
    else:
        args.report_out.write_text(md)
        json_out = args.report_out.with_suffix(".json")
        json_out.write_text(json.dumps(report, indent=2, default=str))
        print(f"[diag] Wrote {args.report_out}")
        print(f"[diag] Wrote {json_out}")


if __name__ == "__main__":
    main()
