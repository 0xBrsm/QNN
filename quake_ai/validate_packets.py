"""CLI for packet/telemetry alignment validation."""

from __future__ import annotations

import argparse

from quake_ai.data.packet_validation import validate_packet_alignment
from quake_ai.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate packet and telemetry alignment")
    parser.add_argument("--telemetry", required=True, help="Telemetry NDJSON path")
    parser.add_argument("--packets", required=True, help="Packets NDJSON path")
    parser.add_argument("--out", default=None, help="Optional JSON report path")
    parser.add_argument("--tick_window", type=int, default=2, help="Tick match tolerance")
    args = parser.parse_args()

    report = validate_packet_alignment(
        telemetry_path=args.telemetry,
        packets_path=args.packets,
        tick_window=args.tick_window,
    )
    payload = report.to_dict()
    if args.out:
        write_json(args.out, payload)
    print(payload)


if __name__ == "__main__":
    main()
