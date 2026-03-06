"""CLI for reporting the current torch accelerator runtime."""

from __future__ import annotations

import argparse
import json

from quake_ai.utils.device import describe_torch_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the torch accelerator runtime visible to Quake AI")
    parser.add_argument("--device", default="auto", help="Requested device override (auto, gpu, cpu, cuda, cuda:0, rocm, mps)")
    args = parser.parse_args()
    print(json.dumps(describe_torch_runtime(args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
