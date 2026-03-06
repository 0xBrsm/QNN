#!/usr/bin/env python3
"""Materialize a map-specific demo subset from the corpus manifest."""

from __future__ import annotations

import argparse
import json

from quake_ai.data.corpus import materialize_corpus_subset


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize corpus demos for a single map")
    parser.add_argument("--manifest", default="../artifacts/corpus/netquake/meta/download_manifest.ndjson")
    parser.add_argument("--map", required=True, dest="map_id")
    parser.add_argument("--out", required=True)
    parser.add_argument("--remote-username", default="guest")
    parser.add_argument("--remote-password", default="guest")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on materialized demos")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = materialize_corpus_subset(
        manifest_path=args.manifest,
        output_dir=args.out,
        map_id=args.map_id,
        remote_username=args.remote_username,
        remote_password=args.remote_password,
        limit=args.limit or None,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
