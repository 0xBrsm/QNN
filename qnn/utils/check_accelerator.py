"""CLI for reporting the current torch accelerator runtime."""

from __future__ import annotations

import argparse
import json

from qnn.utils.device import describe_torch_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description="Report the torch accelerator runtime visible to Quake AI")
    parser.add_argument("--device", default="auto", help="Requested device override (auto, gpu, cpu, cuda, cuda:0, rocm, mps)")
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit non-zero when the requested accelerator is unavailable",
    )
    args = parser.parse_args()
    runtime = describe_torch_runtime(args.device)
    print(json.dumps(runtime, indent=2, sort_keys=True))
    if args.fail_on_error and runtime.get("error"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
