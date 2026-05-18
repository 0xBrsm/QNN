"""CLI entry: ``python -m qnn.diag.diagnose --checkpoint … --data-dir …``"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qnn.diag.report import run_report, render_markdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run diagnostic suite on a trained QNN policy checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to bc_best_model.pth")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/collect/qwd"))
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Run directory (for history). Defaults to checkpoint's grandparent.")
    parser.add_argument("--max-val-shards", type=int, default=1)
    parser.add_argument("--max-val-episodes", type=int, default=8,
                        help="Episodes to use for ablation/gradients (smaller = faster).")
    parser.add_argument("--skip", default="",
                        help="Comma-separated sections to skip: history,rank,ablation,gradients,"
                             "participation,attention,pruning,linear_probe")
    parser.add_argument("--include", default="",
                        help="Comma-separated slow sections to enable (default off): pruning,linear_probe")
    parser.add_argument("--report-out", type=Path, default=None,
                        help="Write markdown report here. JSON sibling is written automatically.")
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None and args.checkpoint.exists():
        # ../.. of bc_best_model.pth = run dir
        run_dir = args.checkpoint.parent.parent

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
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
