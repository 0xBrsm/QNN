"""CLI for behavior cloning training."""

from __future__ import annotations

import argparse

from quake_ai.training_bc import BCConfig, run_behavior_cloning
from quake_ai.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train behavior cloning policy for Quake E1M1")
    parser.add_argument("--config", required=True, help="Path to JSON-compatible YAML config")
    parser.add_argument("--device", default=None, help="Optional torch device override (auto, gpu, cpu, cuda, cuda:0, rocm, mps)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    config = BCConfig(**cfg)
    metrics = run_behavior_cloning(config)
    print(metrics)


if __name__ == "__main__":
    main()
