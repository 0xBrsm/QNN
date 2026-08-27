"""Measure how much dense BSP geometry survives the spatial-token bottleneck.

The engine diagnostic (QNN_SPATIAL_DIAG) writes the production quantized
depth-atlas codes — the exact wire payload — plus the pre-quantization
center-ray distances, beside an independent dense hull-1 trace field on
the same grid.  This module reconstructs directional depth from only the
payload and scores it against the traces.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from typing import Any, Iterable

import numpy as np

from qnn.engine_norm import ATLAS_DEPTH_LEVELS, ATLAS_MISS_CODE
from qnn.utils.io import write_json

# "atlas" scores the production 4-bit codes (the gate target); "atlas_float"
# scores the pre-quantization center-ray distances (the representation's
# upper bound, isolating quantization loss).
LAYOUTS = ("atlas", "atlas_float")

_LEVELS = np.asarray(ATLAS_DEPTH_LEVELS, dtype=np.float64)


def _direction_max(record: dict[str, Any], elevation_deg: float) -> float:
    max_distance = float(record["max_horiz"])
    sine = abs(math.sin(math.radians(elevation_deg)))
    if sine > 1e-5:
        max_distance = min(max_distance, float(record["max_vert"]) / sine)
    return max_distance


def _record_limits(record: dict[str, Any]) -> tuple[list[float], int, np.ndarray]:
    elevations = [float(value) for value in record["elevations"]]
    yaw_count = 360 // int(record["yaw_step"])
    limits = np.asarray(
        [[_direction_max(record, elevation)] * yaw_count for elevation in elevations],
        dtype=np.float64,
    )
    return elevations, yaw_count, limits


def reconstruct_record(
    record: dict[str, Any], *, layout: str = "atlas",
) -> np.ndarray:
    """Reconstruct the dense (elevation, yaw) depth field from the payload."""
    elevations, yaw_count, limits = _record_limits(record)
    shape = (len(elevations), yaw_count)
    if layout == "atlas":
        codes = np.asarray(record["atlas_code"], dtype=np.int64)
        if codes.shape != shape:
            raise ValueError(f"atlas_code shape {codes.shape}, expected {shape}")
        hit = codes != ATLAS_MISS_CODE
        # Cap at the per-direction trace limit, never below it: the scorer
        # reads values within one unit of the limit as open, and a real hit
        # at the instrument's range boundary must decode to the same side of
        # that threshold as the truth field records it (calibration control:
        # a perfect engine dump must score zero false blocks).
        decoded = np.minimum(_LEVELS[np.minimum(codes, len(_LEVELS) - 1)], limits)
    elif layout == "atlas_float":
        distances = np.asarray(record["atlas_distance"], dtype=np.float64)
        if distances.shape != shape:
            raise ValueError(
                f"atlas_distance shape {distances.shape}, expected {shape}"
            )
        hit = distances >= 0.0
        decoded = np.minimum(distances, limits)
    else:
        raise ValueError(f"unknown spatial layout {layout!r}")
    return np.where(hit, decoded, limits)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _near_obstacle_metrics(
    truth: np.ndarray, prediction: np.ndarray, limits: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    """Score preservation of movement-critical obstruction distances."""
    result: dict[str, dict[str, float | int | None]] = {}
    for threshold in (32.0, 64.0, 128.0, 256.0):
        # A capped no-hit value is only evidence of "not near" when the
        # diagnostic trace extended beyond the threshold being tested.
        eligible = limits > threshold + 1.0
        truth_near = truth <= threshold
        prediction_near = prediction <= threshold
        tp = int(np.count_nonzero(eligible & truth_near & prediction_near))
        fn = int(np.count_nonzero(eligible & truth_near & ~prediction_near))
        fp = int(np.count_nonzero(eligible & ~truth_near & prediction_near))
        tn = int(np.count_nonzero(eligible & ~truth_near & ~prediction_near))
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        result[f"{int(threshold)}u"] = {
            "eligible_samples": tp + fn + fp + tn,
            "precision": precision,
            "recall": recall,
            "f1": (
                2.0 * precision * recall / (precision + recall)
                if precision is not None and recall is not None
                and precision + recall > 0.0 else None
            ),
            "accuracy": _safe_ratio(tp + tn, tp + fn + fp + tn),
        }
    return result


def _metrics(truth: np.ndarray, prediction: np.ndarray, limits: np.ndarray) -> dict[str, Any]:
    error = prediction - truth
    truth_hit = truth < limits - 1.0
    pred_hit = prediction < limits - 1.0
    true_positive = int(np.count_nonzero(truth_hit & pred_hit))
    false_negative = int(np.count_nonzero(truth_hit & ~pred_hit))
    false_positive = int(np.count_nonzero(~truth_hit & pred_hit))
    true_negative = int(np.count_nonzero(~truth_hit & ~pred_hit))
    open_count = false_positive + true_negative
    false_block_given_open = _safe_ratio(false_positive, open_count)
    return {
        "samples": int(truth.size),
        "truth_hits": true_positive + false_negative,
        "truth_open": open_count,
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "p90_abs_error": float(np.percentile(np.abs(error), 90)),
        "hit_precision": _safe_ratio(true_positive, true_positive + false_positive),
        "hit_recall": _safe_ratio(true_positive, true_positive + false_negative),
        "missed_obstacle_rate": _safe_ratio(false_negative, int(np.count_nonzero(truth_hit))),
        # Preserve the original key for existing consumers, while making its
        # conditional denominator explicit and adding the population rate.
        "false_block_rate": false_block_given_open,
        "false_block_rate_given_open": false_block_given_open,
        "false_block_rate_all": false_positive / int(truth.size),
        "blocked_early_gt_32_rate": float(np.mean(error < -32.0)),
        "missed_depth_gt_32_rate": float(np.mean(error > 32.0)),
        "near_obstacle": _near_obstacle_metrics(truth, prediction, limits),
    }


def score_records(
    records: Iterable[dict[str, Any]], *, layout: str = "atlas",
) -> dict[str, Any]:
    records = list(records)
    if not records:
        raise ValueError("no spatial reconstruction records")

    truths: list[np.ndarray] = []
    planes: list[np.ndarray] = []
    limits: list[np.ndarray] = []
    by_elevation: dict[str, dict[str, list[np.ndarray]]] = {}
    for record in records:
        elevations, yaw_count, record_limits = _record_limits(record)
        truth = np.asarray(record["truth"], dtype=np.float64).reshape(len(elevations), yaw_count)
        reconstructed = reconstruct_record(record, layout=layout)
        truths.append(truth.reshape(-1))
        planes.append(reconstructed.reshape(-1))
        limits.append(record_limits.reshape(-1))
        for ei, elevation in enumerate(elevations):
            bucket = by_elevation.setdefault(
                f"{elevation:g}", {"truth": [], "plane": [], "limit": []}
            )
            bucket["truth"].append(truth[ei])
            bucket["plane"].append(reconstructed[ei])
            bucket["limit"].append(record_limits[ei])

    truth_all = np.concatenate(truths)
    plane_all = np.concatenate(planes)
    limit_all = np.concatenate(limits)
    plane_metrics = _metrics(truth_all, plane_all, limit_all)
    return {
        "layout": layout,
        "records": len(records),
        "directions_per_record": int(truths[0].size),
        "plane": plane_metrics,
        "by_elevation": {
            elevation: _metrics(
                np.concatenate(bucket["truth"]),
                np.concatenate(bucket["plane"]),
                np.concatenate(bucket["limit"]),
            )
            for elevation, bucket in by_elevation.items()
        },
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def threshold_failures(
    summary: dict[str, Any], *, max_mae: float | None = None,
    max_missed_obstacle_rate: float | None = None,
    max_level_mae: float | None = None,
    max_false_block_rate_all: float | None = None,
    max_blocked_early_gt_32_rate: float | None = None,
) -> list[str]:
    """Return human-readable acceptance failures for optional CI gates."""
    checks = (
        ("plane.mae", summary["plane"]["mae"], max_mae),
        ("plane.missed_obstacle_rate", summary["plane"]["missed_obstacle_rate"],
         max_missed_obstacle_rate),
        ("plane.false_block_rate_all", summary["plane"]["false_block_rate_all"],
         max_false_block_rate_all),
        ("plane.blocked_early_gt_32_rate",
         summary["plane"]["blocked_early_gt_32_rate"],
         max_blocked_early_gt_32_rate),
        ("by_elevation.0.mae", summary["by_elevation"]["0"]["mae"], max_level_mae),
    )
    return [
        f"{name}={actual:.6g} exceeds {limit:.6g}"
        for name, actual, limit in checks
        if limit is not None and actual is not None and actual > limit
    ]


def run_worker(
    *, worker: Path, assets: Path, demo: Path, sidecar: Path,
    tick_hz: int, play_end: int, stride: int, max_samples: int,
) -> str:
    """Play one demo and collect the engine's independent trace sidecar."""
    from qnn.wire import ACTION_SIZE, FLAG_DONE, OBS_BUFFER_SIZE, TICK_HEADER_SIZE

    sidecar.parent.mkdir(parents=True, exist_ok=True)
    # Match qnn.bc.collect: Quake's -game directory is the demo's parent,
    # relative to QUAKE_BASEDIR. The id1 paks remain the fallback search path
    # for maps; no temporary symlink or asset-tree mutation is needed.
    game_dir = os.path.relpath(demo.resolve().parent, assets.resolve())
    env = {
        **os.environ,
        "QUAKE_BASEDIR": str(assets),
        "QNN_SPATIAL_DIAG": str(sidecar),
        "QNN_SPATIAL_DIAG_STRIDE": str(stride),
        "QNN_SPATIAL_DIAG_MAX": str(max_samples),
    }
    stderr_text = ""
    with tempfile.TemporaryFile() as stderr_file:
        try:
            proc = subprocess.Popen(
                [str(worker), "-game", game_dir],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=stderr_file, env=env,
            )
            assert proc.stdin is not None and proc.stdout is not None
            hello = {
                "op": "hello", "map_id": "start",
                "tick_hz": tick_hz, "resample_hz": tick_hz,
            }
            proc.stdin.write((json.dumps(hello) + "\n").encode())
            proc.stdin.flush()
            response = proc.stdout.readline()
            if b'"ok":true' not in response:
                raise RuntimeError(f"worker hello failed: {response!r}")
            collect = {
                "op": "collect", "demo_path": demo.name, "seed": 0,
                "trim_match": 0, "play_end": play_end,
            }
            proc.stdin.write((json.dumps(collect) + "\n").encode())
            proc.stdin.flush()

            payload_size = TICK_HEADER_SIZE + OBS_BUFFER_SIZE + ACTION_SIZE
            while True:
                magic = proc.stdout.read(4)
                if len(magic) < 4:
                    break
                if magic.startswith(b"{"):
                    message = magic + proc.stdout.readline()
                    raise RuntimeError(f"worker returned JSON during collect: {message!r}")
                if magic != b"QOBS":
                    raise RuntimeError(f"unexpected worker record magic: {magic!r}")
                payload = proc.stdout.read(payload_size)
                if len(payload) != payload_size:
                    raise RuntimeError("truncated QOBS record")
                flags = struct.unpack_from("<IIIHH", payload, 0)[3]
                if flags & FLAG_DONE:
                    break

            try:
                proc.stdin.write(b'{"op":"shutdown"}\n')
                proc.stdin.flush()
            except BrokenPipeError:
                pass
            return_code = proc.wait(timeout=10)
            stderr_file.seek(0)
            stderr_text = stderr_file.read().decode(errors="replace")
            if return_code != 0:
                raise RuntimeError(
                    f"worker exited {return_code}:\n{stderr_text[-4000:]}"
                )
        except Exception as exc:
            stderr_file.seek(0)
            stderr_text = stderr_file.read().decode(errors="replace")
            if stderr_text:
                raise RuntimeError(f"{exc}\nworker stderr:\n{stderr_text[-4000:]}") from exc
            raise
    return stderr_text


def _write_summary(summary: dict[str, Any], output: Path | None) -> None:
    if output is not None:
        write_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score_parser = subparsers.add_parser("score", help="score an existing JSONL sidecar")
    score_parser.add_argument("sidecars", type=Path, nargs="+")
    score_parser.add_argument("--output", type=Path)

    run_parser = subparsers.add_parser("run", help="play a demo, collect truth, and score it")
    run_parser.add_argument("--worker", type=Path, required=True)
    run_parser.add_argument("--assets", type=Path, required=True)
    run_parser.add_argument("--demo", type=Path, required=True)
    run_parser.add_argument("--sidecar", type=Path, required=True)
    run_parser.add_argument("--output", type=Path)
    run_parser.add_argument("--tick-hz", type=int, default=20)
    run_parser.add_argument("--play-end", type=int, default=5000)
    run_parser.add_argument("--stride", type=int, default=20)
    run_parser.add_argument("--max-samples", type=int, default=256)
    for subparser in (score_parser, run_parser):
        subparser.add_argument("--layout", choices=LAYOUTS, default="atlas")
        subparser.add_argument("--max-mae", type=float)
        subparser.add_argument("--max-missed-obstacle-rate", type=float)
        subparser.add_argument("--max-false-block-rate-all", type=float)
        subparser.add_argument("--max-blocked-early-gt-32-rate", type=float)
        subparser.add_argument("--max-level-mae", type=float)
    args = parser.parse_args(argv)

    if args.command == "run":
        run_worker(
            worker=args.worker, assets=args.assets, demo=args.demo,
            sidecar=args.sidecar, tick_hz=args.tick_hz,
            play_end=args.play_end, stride=args.stride,
            max_samples=args.max_samples,
        )
    sidecars = [args.sidecar] if args.command == "run" else args.sidecars
    summary = score_records(
        (record for sidecar in sidecars for record in load_records(sidecar)),
        layout=args.layout,
    )
    _write_summary(summary, args.output)
    failures = threshold_failures(
        summary, max_mae=args.max_mae,
        max_missed_obstacle_rate=args.max_missed_obstacle_rate,
        max_false_block_rate_all=args.max_false_block_rate_all,
        max_blocked_early_gt_32_rate=args.max_blocked_early_gt_32_rate,
        max_level_mae=args.max_level_mae,
    )
    if failures:
        print("spatial reconstruction acceptance failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
