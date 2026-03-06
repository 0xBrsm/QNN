#!/usr/bin/env python3
"""Build per-level counts from NetQuake manifest."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

URL_RE = [
    re.compile(r"/quake/demos/([^/]+)/([^/?#]+)$", re.IGNORECASE),
    re.compile(r"/quake/contests/demos/([^/?#]+)$", re.IGNORECASE),
]


def parse_level(url: str) -> tuple[str, str]:
    u = url.lower()
    m = URL_RE[0].search(u)
    if m:
        category = m.group(1).upper()
        filename = m.group(2)
    else:
        m2 = URL_RE[1].search(u)
        if not m2:
            return "unknown", "UNKNOWN"
        category = "CONTEST"
        filename = m2.group(1)

    base = filename.split(".")[0]
    level = base.split("_")[0]
    return level, category


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="../artifacts/corpus/netquake/meta/download_manifest.ndjson")
    parser.add_argument("--out", default="../artifacts/corpus/netquake/meta/level_index.json")
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    level_counts = Counter()
    category_counts = Counter()
    backend_counts = Counter()
    level_to_categories = defaultdict(Counter)

    for row in rows:
        url = str(row.get("url", ""))
        level, category = parse_level(url)
        level_counts[level] += 1
        category_counts[category] += 1
        backend_counts[str(row.get("storage_backend", "unknown"))] += 1
        level_to_categories[level][category] += 1

    payload = {
        "manifest_rows": len(rows),
        "levels": len(level_counts),
        "top_levels": [{"level": k, "count": v, "categories": dict(level_to_categories[k])} for k, v in level_counts.most_common(args.top)],
        "category_counts": dict(category_counts),
        "backend_counts": dict(backend_counts),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "manifest_rows": payload["manifest_rows"],
        "levels": payload["levels"],
        "top_level": payload["top_levels"][0] if payload["top_levels"] else None,
    }, indent=2))


if __name__ == "__main__":
    main()
