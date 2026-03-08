#!/usr/bin/env python3
"""Classify NetQuake demos from the corpus manifest and write a local label manifest."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from engine.adapter import DemoEpisode
from quake_ai.data.corpus import extract_demo_bytes, load_manifest_rows, manifest_row_map_id
from quake_ai.data.demo_classifier import classify_competitive
from quake_ai.data.demo_metadata import DemoMetadata, build_demo_metadata
from quake_ai.data.netquake_demo import parse_netquake_demo_metadata

_SINGLEPLAYER_MAP_RE = re.compile(r"^(e[1-4]m[1-8]|end)$", re.IGNORECASE)


def _sorted_int_key_dict(values: dict[int, int] | dict[int, str]) -> dict[str, int] | dict[str, str]:
    return {str(key): values[key] for key in sorted(values)}


def _normalize_remote_prefix(storage_root: str, remote_prefix: str) -> str:
    prefix = remote_prefix.replace("/", "\\").strip("\\")
    if not prefix:
        return ""
    normalized_root = storage_root.replace("/", "\\").rstrip("\\").lower()
    if normalized_root.endswith("\\" + prefix.lower()) or normalized_root == prefix.lower():
        return ""
    return remote_prefix


def _default_output_paths(manifest_path: Path) -> tuple[Path, Path, Path]:
    stem = manifest_path.stem
    label_stem = stem.replace("download_manifest", "label_manifest", 1)
    if label_stem == stem:
        label_stem = f"{stem}_labels"
    failure_stem = label_stem.replace("label_manifest", "label_failures", 1)
    summary_stem = label_stem.replace("label_manifest", "label_summary", 1)
    return (
        manifest_path.with_name(f"{label_stem}.ndjson"),
        manifest_path.with_name(f"{failure_stem}.ndjson"),
        manifest_path.with_name(f"{summary_stem}.json"),
    )


def _safe_stem(source_ref: str, manifest_index: int) -> str:
    leaf = source_ref.replace("\\", "/").split("/")[-1]
    stem = Path(leaf).stem or f"demo_{manifest_index:06d}"
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in stem)


def _label_row(row: dict[str, object], meta: DemoMetadata, manifest_index: int) -> dict[str, object]:
    source_ref = str(row.get("extracted_dem_path", "")) or str(row.get("local_path", "")) or str(row.get("url", ""))
    return {
        "manifest_index": manifest_index,
        "url": str(row.get("url", "")),
        "sha256": str(row.get("sha256", "")),
        "size": int(row.get("size", 0) or 0),
        "content_type": str(row.get("content_type", "")),
        "storage_backend": str(row.get("storage_backend", "")),
        "storage_root": str(row.get("storage_root", "")),
        "local_path": str(row.get("local_path", "")),
        "extracted_dem_path": str(row.get("extracted_dem_path", "")),
        "source_ref": source_ref,
        "episode_id": meta.episode_id,
        "map_id": meta.map_id,
        "protocol": meta.protocol,
        "game_dir": meta.game_dir,
        "maxclients": meta.maxclients,
        "named_player_count": len(meta.player_names),
        "player_names": _sorted_int_key_dict(meta.player_names),
        "player_colors": _sorted_int_key_dict(meta.player_colors),
        "frag_update_count": len(meta.frag_updates),
        "final_frags": _sorted_int_key_dict(meta.final_frags),
        "deathmatch": meta.deathmatch,
        "teamplay": meta.teamplay,
        "fraglimit": meta.fraglimit,
        "timelimit": meta.timelimit,
        "text_flags": {key: bool(meta.text_flags[key]) for key in sorted(meta.text_flags)},
        "tick_count": meta.tick_count,
        "duration_s": meta.duration_s,
        "classification": meta.classification,
        "classification_confidence": meta.classification_confidence,
        "mode": meta.mode,
    }


def _metadata_episode(demo_path: Path, map_hint: str) -> DemoEpisode:
    parsed = parse_netquake_demo_metadata(demo_path, map_id=map_hint)
    return DemoEpisode(
        episode_id=parsed.episode_id,
        map_id=parsed.map_id,
        ticks=[],
        metadata={
            "serverinfo": dict(parsed.serverinfo),
            "cvars": dict(parsed.cvars),
            "player_names": dict(parsed.player_names),
            "player_colors": dict(parsed.player_colors),
            "frag_updates": list(parsed.frag_updates),
            "text_flags": dict(parsed.text_flags),
            "maxclients": int(parsed.maxclients),
            "duration_s": float(parsed.duration_s),
            "tick_count": int(parsed.tick_count),
        },
    )


def _failure_row(row: dict[str, object], manifest_index: int, source_ref: str, map_hint: str, error: Exception) -> dict[str, object]:
    return {
        "manifest_index": manifest_index,
        "url": str(row.get("url", "")),
        "storage_backend": str(row.get("storage_backend", "")),
        "storage_root": str(row.get("storage_root", "")),
        "local_path": str(row.get("local_path", "")),
        "extracted_dem_path": str(row.get("extracted_dem_path", "")),
        "source_ref": source_ref,
        "map_hint": map_hint,
        "error": str(error),
    }


def _manifest_only_non_competitive_row(row: dict[str, object], manifest_index: int, source_ref: str, map_hint: str) -> dict[str, object]:
    episode_id = f"{manifest_index:06d}_{_safe_stem(source_ref, manifest_index)}"
    return {
        "manifest_index": manifest_index,
        "url": str(row.get("url", "")),
        "sha256": str(row.get("sha256", "")),
        "size": int(row.get("size", 0) or 0),
        "content_type": str(row.get("content_type", "")),
        "storage_backend": str(row.get("storage_backend", "")),
        "storage_root": str(row.get("storage_root", "")),
        "local_path": str(row.get("local_path", "")),
        "extracted_dem_path": str(row.get("extracted_dem_path", "")),
        "source_ref": source_ref,
        "episode_id": episode_id,
        "map_id": map_hint.lower(),
        "protocol": None,
        "game_dir": None,
        "maxclients": None,
        "named_player_count": 0,
        "player_names": {},
        "player_colors": {},
        "frag_update_count": 0,
        "final_frags": {},
        "deathmatch": None,
        "teamplay": None,
        "fraglimit": None,
        "timelimit": None,
        "text_flags": {},
        "tick_count": 0,
        "duration_s": 0.0,
        "classification": "non_competitive",
        "classification_confidence": 0.9,
        "mode": "non_competitive",
    }


def _process_row(task: tuple[int, dict[str, object], str, str, str]) -> tuple[str, dict[str, object]]:
    manifest_index, row, remote_username, remote_password, remote_prefix_arg = task
    source_ref = str(row.get("extracted_dem_path", "")) or str(row.get("local_path", "")) or str(row.get("url", ""))
    map_hint = manifest_row_map_id(row) or "unknown"
    if _SINGLEPLAYER_MAP_RE.fullmatch(map_hint):
        return "ok", _manifest_only_non_competitive_row(row, manifest_index, source_ref, map_hint)
    remote_prefix = _normalize_remote_prefix(str(row.get("storage_root", "")), remote_prefix_arg)

    try:
        with tempfile.TemporaryDirectory(prefix=f"quake_ai_label_{manifest_index:06d}_") as temp_dir:
            temp_demo_path = Path(temp_dir) / f"{manifest_index:06d}_{_safe_stem(source_ref, manifest_index)}.dem"
            demo_bytes = extract_demo_bytes(
                row,
                remote_username=remote_username,
                remote_password=remote_password,
                remote_prefix=remote_prefix,
            )
            temp_demo_path.write_bytes(demo_bytes)
            episode = _metadata_episode(temp_demo_path, map_hint)
            meta = classify_competitive(
                build_demo_metadata(
                    episode,
                    source_path=source_ref,
                    source_url=str(row.get("url", "")) or None,
                )
            )
            return "ok", _label_row(row, meta, manifest_index)
    except Exception as exc:
        return "error", _failure_row(row, manifest_index, source_ref, map_hint, exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Label NetQuake demos from the corpus manifest")
    parser.add_argument("--manifest", default="../artifacts/corpus/netquake/meta/download_manifest_full.ndjson")
    parser.add_argument("--out")
    parser.add_argument("--failures-out")
    parser.add_argument("--summary-out")
    parser.add_argument("--remote-username", default="guest")
    parser.add_argument("--remote-password", default="guest")
    parser.add_argument("--remote-prefix", default="netquake")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on manifest rows")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    default_out, default_failures, default_summary = _default_output_paths(manifest_path)
    out_path = Path(args.out) if args.out else default_out
    failures_path = Path(args.failures_out) if args.failures_out else default_failures
    summary_path = Path(args.summary_out) if args.summary_out else default_summary

    out_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_manifest_rows(manifest_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    class_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    map_counts: Counter[str] = Counter()
    failed_rows = 0

    tasks = [
        (manifest_index, row, args.remote_username, args.remote_password, args.remote_prefix)
        for manifest_index, row in enumerate(rows, start=1)
    ]

    with out_path.open("w", encoding="utf-8", buffering=1) as out_handle, failures_path.open(
        "w", encoding="utf-8", buffering=1
    ) as failure_handle:
        if args.workers <= 1:
            next_write_index = 1
            for processed, task in enumerate(tasks, start=1):
                manifest_index = task[0]
                status, payload = _process_row(task)
                if status == "ok":
                    labeled = payload
                    class_counts[labeled["classification"]] += 1
                    mode_counts[labeled["mode"]] += 1
                    map_counts[labeled["map_id"]] += 1
                    out_handle.write(json.dumps(labeled, sort_keys=True) + "\n")
                else:
                    failed_rows += 1
                    failure_handle.write(json.dumps(payload, sort_keys=True) + "\n")
                next_write_index = manifest_index + 1
                if args.progress_every > 0 and processed % args.progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "processed": processed,
                                "labeled": processed - failed_rows,
                                "failed": failed_rows,
                                "written": next_write_index - 1,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        else:
            next_write_index = 1
            pending_results: dict[int, tuple[str, dict[str, object]]] = {}
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_to_index = {executor.submit(_process_row, task): task[0] for task in tasks}
                for processed, future in enumerate(as_completed(future_to_index), start=1):
                    manifest_index = future_to_index[future]
                    status, payload = future.result()
                    pending_results[manifest_index] = (status, payload)
                    if status == "ok":
                        labeled = payload
                        class_counts[labeled["classification"]] += 1
                        mode_counts[labeled["mode"]] += 1
                        map_counts[labeled["map_id"]] += 1
                    else:
                        failed_rows += 1

                    while next_write_index in pending_results:
                        write_status, write_payload = pending_results.pop(next_write_index)
                        if write_status == "ok":
                            out_handle.write(json.dumps(write_payload, sort_keys=True) + "\n")
                        else:
                            failure_handle.write(json.dumps(write_payload, sort_keys=True) + "\n")
                        next_write_index += 1

                    if args.progress_every > 0 and processed % args.progress_every == 0:
                        print(
                            json.dumps(
                                {
                                    "processed": processed,
                                    "labeled": processed - failed_rows,
                                    "failed": failed_rows,
                                    "written": next_write_index - 1,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )

    summary = {
        "manifest_path": str(manifest_path),
        "output_path": str(out_path),
        "failures_path": str(failures_path),
        "rows_requested": len(rows),
        "rows_labeled": int(sum(class_counts.values())),
        "rows_failed": failed_rows,
        "classification_counts": dict(sorted(class_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "map_counts": dict(sorted(map_counts.items())),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
