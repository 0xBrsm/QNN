"""CLI for demo collection."""

from __future__ import annotations

import argparse

from quake_ai.data.collector import collect_from_demos
from quake_ai.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect telemetry/packets from Quake demos")
    parser.add_argument("--map", required=True, dest="map_id", help="Map ID, e.g. E1M1")
    parser.add_argument("--demo_dir", required=True, help="Directory containing .dem demo files")
    parser.add_argument("--out", required=True, help="Output artifact directory")
    parser.add_argument("--map_path", default=None, help="Optional BSP or JSON map metadata path")
    args = parser.parse_args()

    artifacts = collect_from_demos(map_id=args.map_id, demo_dir=args.demo_dir, out_dir=args.out, map_path=args.map_path)
    write_json(f"{args.out}/collect_manifest.json", artifacts)


if __name__ == "__main__":
    main()
