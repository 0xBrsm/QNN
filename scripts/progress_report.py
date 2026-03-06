#!/usr/bin/env python3
"""Emit a corpus progress snapshot as JSON and optionally append to a log."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


CANONICAL_MAP_IDS: list[str] = [
    *(f"e1m{i}" for i in range(1, 9)),
    *(f"e2m{i}" for i in range(1, 8)),
    *(f"e3m{i}" for i in range(1, 8)),
    *(f"e4m{i}" for i in range(1, 9)),
    "end",
    *(f"dm{i}" for i in range(1, 7)),
]
CANONICAL_MAP_SET = set(CANONICAL_MAP_IDS)


def _read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _iter_manifest_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _source_key(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if "speeddemosarchive.com" in host:
        if path.startswith("/quake/qdq/demos/") or path.startswith("/quake/qdq/hobby/") or path.startswith("/quake/projects/hobby/"):
            return "sda_qdq"
        if path == "/quake/downloads/quake-light.zip":
            return "sda_quake_light"
        return "sda"
    if "archive.org" in host:
        if path == "/download/idgames-2-archive/demos.zip":
            return "archive_idgames2_bundle"
        if path in {"/download/sda_quake_demos/maps.zip", "/download/quake_sda_collection/maps.zip"}:
            return "archive_sda_bundle"
        return "archive_org"
    if "demos.igmdb.org" in host and path.startswith("/challengetv/demostorage/quake%201/"):
        return "igmdb_chtv_netquake"
    if "demos.igmdb.org" in host and path.startswith("/challengetv/chtv.quakeworld.nu/netquake/"):
        return "igmdb_chtv_legacy_netquake"
    if "quaketastic.com" in host and path.startswith("/files/demos/"):
        return "quaketastic_demos"
    if "quaketerminus.com" in host:
        return "quaketerminus"
    if "quaddicted.com" in host and path.startswith("/files/idgames2/demos/"):
        return "idgames_mirror"
    if any(host.endswith(suffix) for suffix in ("gamers.org", "braindrainlan.nu", "infania.net", "fu-berlin.de")) and "/idgames2/demos/" in path:
        return "idgames_mirror"
    if "quaddicted.com" in host and path.startswith("/files/demos/"):
        return "quaddicted_demos"
    return host or "unknown"


def _map_id(url: str) -> str:
    path = urlparse(url).path
    name = path.split("/")[-1].lower()
    if "." in name:
        stem = name.rsplit(".", 1)[0]
    else:
        stem = name
    if "_" in stem:
        stem = stem.split("_", 1)[0]
    match = re.match(r"^(e[1-4]m[1-8]|dm[1-6]|end)", stem)
    if match:
        return match.group(1)
    return stem


def build_snapshot(summary_path: Path, manifest_path: Path, target_downloads: int) -> dict:
    summary = _read_summary(summary_path)
    rows = _iter_manifest_rows(manifest_path)

    by_url: dict[str, dict] = {}
    for row in rows:
        url = str(row.get("url", ""))
        if url and url not in by_url:
            by_url[url] = row

    source_counts = Counter()
    status_counts = Counter()
    map_counts = Counter()
    unknown_map_urls = 0
    downloaded_rows: list[dict] = []

    for url, row in by_url.items():
        source_counts[_source_key(url)] += 1
        status = str(row.get("status", ""))
        status_counts[status] += 1
        if status == "downloaded":
            downloaded_rows.append(row)

        map_id = _map_id(url)
        if map_id in CANONICAL_MAP_SET:
            map_counts[map_id] += 1
        else:
            unknown_map_urls += 1

    downloads = len(downloaded_rows)
    total_bytes = sum(int((row.get("size", 0) or 0)) for row in downloaded_rows)
    unique_hashes = len({str(row.get("sha256", "")) for row in downloaded_rows if str(row.get("sha256", ""))})
    progress_pct = (downloads / float(target_downloads) * 100.0) if target_downloads > 0 else 0.0

    return {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "target_downloads": target_downloads,
        "downloads": downloads,
        "progress_pct": round(progress_pct, 3),
        "unique_hashes": unique_hashes,
        "total_bytes": total_bytes,
        "remote_available": bool(summary.get("remote_available", False)),
        "remote_latched_to_local": bool(summary.get("remote_latched_to_local", False)),
        "unique_urls": len(by_url),
        "source_counts": dict(source_counts),
        "status_counts": dict(status_counts),
        "unknown_noncanonical_urls": unknown_map_urls,
        "canonical_map_counts": {map_id: map_counts.get(map_id, 0) for map_id in CANONICAL_MAP_IDS},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit corpus progress snapshot")
    parser.add_argument("--summary", default="../artifacts/corpus/netquake/meta/summary.json")
    parser.add_argument("--manifest", default="../artifacts/corpus/netquake/meta/download_manifest.ndjson")
    parser.add_argument("--target-downloads", type=int, default=600000)
    parser.add_argument("--append", default="", help="Optional path to append JSONL snapshots")
    args = parser.parse_args()

    snapshot = build_snapshot(Path(args.summary), Path(args.manifest), args.target_downloads)
    print(json.dumps(snapshot, sort_keys=True))

    if args.append:
        out = Path(args.append)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, sort_keys=True) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
