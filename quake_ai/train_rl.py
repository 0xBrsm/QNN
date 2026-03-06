"""CLI for PPO fine-tuning."""

from __future__ import annotations

import argparse

from quake_ai.training_rl import PPOConfig, run_ppo
from quake_ai.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Quake policy with PPO")
    parser.add_argument("--config", required=True, help="Path to JSON-compatible YAML config")
    parser.add_argument("--init_ckpt", default=None, help="Optional override for initial BC checkpoint")
    parser.add_argument("--device", default=None, help="Optional torch device override (auto, gpu, cpu, cuda, cuda:0, rocm, mps)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.init_ckpt:
        cfg["init_ckpt"] = args.init_ckpt
    if args.device:
        cfg["device"] = args.device
    config = PPOConfig(**cfg)
    metrics = run_ppo(config)
    print(metrics)


if __name__ == "__main__":
    main()
