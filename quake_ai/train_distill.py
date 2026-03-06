"""CLI for policy distillation from PPO rollouts."""

from __future__ import annotations

import argparse

from quake_ai.training_distill import DistillConfig, run_distillation
from quake_ai.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill a sampled teacher policy into a stable student checkpoint")
    parser.add_argument("--config", required=True, help="Path to JSON-compatible YAML config")
    parser.add_argument("--teacher_ckpt", default=None, help="Optional override for teacher checkpoint")
    parser.add_argument("--device", default=None, help="Optional torch device override (auto, gpu, cpu, cuda, cuda:0, rocm, mps)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.teacher_ckpt:
        cfg["teacher_ckpt"] = args.teacher_ckpt
    if args.device:
        cfg["device"] = args.device
    config = DistillConfig(**cfg)
    metrics = run_distillation(config)
    print(metrics)


if __name__ == "__main__":
    main()
