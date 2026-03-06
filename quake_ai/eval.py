"""CLI for policy evaluation."""

from __future__ import annotations

import argparse

from quake_ai.evaluation import EvalConfig, run_evaluation
from quake_ai.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Quake policy")
    parser.add_argument("--config", required=True, help="Path to JSON-compatible YAML config")
    parser.add_argument("--ckpt", default=None, help="Optional checkpoint override")
    parser.add_argument("--device", default=None, help="Optional torch device override (auto, gpu, cpu, cuda, cuda:0, rocm, mps)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.ckpt:
        cfg["checkpoint_path"] = args.ckpt
    if args.device:
        cfg["device"] = args.device
    config = EvalConfig(**cfg)
    metrics = run_evaluation(config)
    print(metrics)


if __name__ == "__main__":
    main()
