"""Validate collected BC shards for live actor / target-label sanity.

Stricter than shape checking — catches the class of silent breakage
where target labels look valid because actor tokens exist, but those
actors are stale baseline/player-idx ghosts rather than live players.

Works on the current variable-length flat entity schema where each
shard exposes:

    obs.entity_count[T]               — # entities emitted for frame t
    obs.entity_types[N]               — flat per-token type stream
    obs.entity_player_id[N]           — flat per-token player slot (u8)
    obs.entity_rel[N, 3]              — flat per-token relative pos
    obs.entity_vel[N, 3]              — flat per-token velocity
    act.target_probs[T, 17]           — labeler dist; index 0 = NO_TARGET
    act.move[T]                       — bit-packed move byte
                                          (bit 0 = attack press)

Per-frame entity-slot k (k in 0..15) is at the flat index
``offsets[t] + k`` when ``k < entity_count[t]``; the argmax of
``target_probs[t, 1:]`` gives the predicted slot.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qnn.bc.target_labeler import TARGET_IGNORE
from qnn.vocab import TOKEN_ACTOR


@dataclass
class SplitStats:
    split: str
    frames: int = 0
    live_actor_frames: int = 0
    live_actor_tokens: int = 0
    moving_actor_tokens: int = 0
    target_valid: int = 0
    target_actor: int = 0
    target_pid: int = 0
    target_rel_nonzero: int = 0
    target_fire: int = 0
    max_unique_pids: int = 0
    suspect_episode_count: int = 0

    def merge(self, other: "SplitStats") -> None:
        self.frames += other.frames
        self.live_actor_frames += other.live_actor_frames
        self.live_actor_tokens += other.live_actor_tokens
        self.moving_actor_tokens += other.moving_actor_tokens
        self.target_valid += other.target_valid
        self.target_actor += other.target_actor
        self.target_pid += other.target_pid
        self.target_rel_nonzero += other.target_rel_nonzero
        self.target_fire += other.target_fire
        self.max_unique_pids = max(self.max_unique_pids, other.max_unique_pids)
        self.suspect_episode_count += other.suspect_episode_count


def _load_corpus_manifest(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for idx, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        out[idx] = json.loads(line)
    return out


def _iter_shards(split_dir: Path) -> list[dict[str, Any]]:
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "sharded_v1":
        raise ValueError(f"{manifest_path}: expected sharded_v1 manifest")
    return list(manifest.get("shards", []))


def _load_shard_arrays(split_dir: Path, shard: dict[str, Any]) -> dict[str, np.ndarray]:
    obs = shard["obs"]
    actions = shard["actions"]
    return {
        "entity_count": np.load(split_dir / obs["entity_count"], mmap_mode="r"),
        "entity_types": np.load(split_dir / obs["entity_types"], mmap_mode="r"),
        "entity_player_id": np.load(split_dir / obs["entity_player_id"], mmap_mode="r"),
        "entity_rel": np.load(split_dir / obs["entity_rel"], mmap_mode="r"),
        "entity_vel": np.load(split_dir / obs["entity_vel"], mmap_mode="r"),
        "target_probs": np.load(split_dir / actions["target_probs"], mmap_mode="r"),
        "move": np.load(split_dir / actions["move"], mmap_mode="r"),
    }


def _episode_expected_opponents(
    demo_idx: int,
    corpus_manifest: dict[int, dict[str, Any]],
) -> int | None:
    entry = corpus_manifest.get(int(demo_idx))
    if not entry:
        return None
    maxclients = entry.get("maxclients")
    if isinstance(maxclients, int) and maxclients > 1:
        return maxclients - 1
    return None


def _validate_shard(
    split: str,
    split_dir: Path,
    shard: dict[str, Any],
    corpus_manifest: dict[int, dict[str, Any]],
    unique_pid_slack: int,
) -> SplitStats:
    arrays = _load_shard_arrays(split_dir, shard)
    counts = arrays["entity_count"].astype(np.int64)
    types = arrays["entity_types"]
    pids = arrays["entity_player_id"]
    rel = arrays["entity_rel"].astype(np.float32)
    vel = arrays["entity_vel"].astype(np.float32)
    target_probs = arrays["target_probs"].astype(np.float32)
    move = arrays["move"]

    T = int(counts.shape[0])
    if target_probs.shape[0] != T:
        raise ValueError(
            f"{split_dir}: row mismatch: {T} entity_count rows vs "
            f"{target_probs.shape[0]} target_probs rows"
        )

    # Frame offsets into flat per-token arrays
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    # Per-token: ACTOR with positive player_id == "live actor"
    actor_mask = (types == TOKEN_ACTOR)
    live_actor = actor_mask & (pids > 0)

    # Per-token: live actor with non-zero velocity (moving)
    vel_abs = np.abs(vel).sum(axis=1)
    moving_actor = live_actor & (vel_abs > 1e-6)

    # Per-frame: any live actor visible?
    cum_live = np.concatenate([[0], np.cumsum(live_actor.astype(np.int64))])
    live_per_frame = cum_live[offsets[1:]] - cum_live[offsets[:-1]]
    live_actor_frames_mask = live_per_frame > 0

    # Per-frame: # distinct positive player_ids among live actors
    unique_pids = np.zeros(T, dtype=np.int32)
    for t in range(T):
        s = int(offsets[t])
        e = int(offsets[t + 1])
        if e == s:
            continue
        slice_live = live_actor[s:e]
        if not slice_live.any():
            continue
        slot_pids = pids[s:e][slice_live]
        unique_pids[t] = int(np.unique(slot_pids).size)

    # Target validity from target_probs:
    # - "valid target" = present mass >= 0.5 (1 - NO_TARGET prob)
    # - target_slot = argmax of idx_dist (entity-stream slot, 0..15)
    present = (1.0 - target_probs[:, 0]) >= 0.5
    idx_dist = target_probs[:, 1:]
    target_slot = idx_dist.argmax(axis=1).astype(np.int64)

    # For each "valid" frame, look up the entity at that slot.
    # Slot is within emitted bounds iff target_slot < entity_count.
    target_in_bounds = present & (target_slot < counts)
    flat_idx = offsets[:-1] + target_slot
    # Clamp to safe range for indexing where in_bounds is False (those
    # results are masked out below).
    flat_idx_safe = np.minimum(flat_idx, max(types.shape[0] - 1, 0)).astype(np.int64)
    type_at_target = types[flat_idx_safe]
    pid_at_target = pids[flat_idx_safe]
    rel_at_target = np.abs(rel[flat_idx_safe]).sum(axis=1)

    target_actor_mask = target_in_bounds & (type_at_target == TOKEN_ACTOR)
    target_pid_mask = target_in_bounds & (pid_at_target > 0)
    target_rel_mask = target_in_bounds & (rel_at_target > 1e-6)

    fire = (move & 0x01).astype(np.int64)

    stats = SplitStats(split=split)
    stats.frames = T
    stats.live_actor_frames = int((live_actor_frames_mask & present).sum())
    stats.live_actor_tokens = int(live_actor.sum())
    stats.moving_actor_tokens = int(moving_actor.sum())
    stats.target_valid = int(present.sum())
    stats.target_actor = int(target_actor_mask.sum())
    stats.target_pid = int(target_pid_mask.sum())
    stats.target_rel_nonzero = int(target_rel_mask.sum())
    stats.target_fire = int((fire & present.astype(np.int64)).sum())
    stats.max_unique_pids = int(unique_pids.max()) if unique_pids.size else 0

    # Per-episode max-unique-pids check vs manifest maxclients
    cursor = 0
    for length, demo_idx in zip(
        shard.get("episode_lengths", []),
        shard.get("demo_idxs", []),
    ):
        end = cursor + int(length)
        expected = _episode_expected_opponents(int(demo_idx), corpus_manifest)
        if expected is not None:
            ep_max = int(unique_pids[cursor:end].max(initial=0))
            if ep_max > expected + unique_pid_slack:
                stats.suspect_episode_count += 1
        cursor = end

    return stats


def validate_collect(
    data_dir: Path,
    *,
    corpus_manifest: Path | None = None,
    min_live_actor_frame_rate: float = 0.95,
    min_moving_actor_token_rate: float = 0.001,
    min_target_pid_rate: float = 0.99,
    min_target_rel_rate: float = 0.95,
    unique_pid_slack: int = 0,
    demo_dir: Path | None = None,
    validate_labels: bool = True,
    drift_ref: Path | None = None,
) -> int:
    corpus = _load_corpus_manifest(corpus_manifest)
    failures: list[str] = []

    for split in ("precomputed_train", "precomputed_val"):
        split_dir = data_dir / split
        shards = _iter_shards(split_dir)
        if not shards:
            print(f"{split}: no shards")
            continue

        total = SplitStats(split=split)
        for shard in shards:
            total.merge(_validate_shard(split, split_dir, shard, corpus, unique_pid_slack))

        live_rate = total.live_actor_frames / max(total.target_valid, 1)
        moving_rate = total.moving_actor_tokens / max(total.live_actor_tokens, 1)
        target_pid_rate = total.target_pid / max(total.target_valid, 1)
        target_actor_rate = total.target_actor / max(total.target_valid, 1)
        target_rel_rate = total.target_rel_nonzero / max(total.target_valid, 1)

        print(
            f"{split}: frames={total.frames} live_actor_frames={total.live_actor_frames} "
            f"live_rate={live_rate:.4f} live_actor_tokens={total.live_actor_tokens} "
            f"moving_actor_rate={moving_rate:.6f} target_valid={total.target_valid} "
            f"target_pid_rate={target_pid_rate:.4f} target_actor_rate={target_actor_rate:.4f} "
            f"target_rel_rate={target_rel_rate:.4f} max_unique_pids={total.max_unique_pids} "
            f"suspect_episodes={total.suspect_episode_count}"
        )

        if live_rate < min_live_actor_frame_rate:
            failures.append(f"{split}: live actor frame rate {live_rate:.4f} < {min_live_actor_frame_rate:.4f}")
        if moving_rate < min_moving_actor_token_rate:
            failures.append(f"{split}: moving actor token rate {moving_rate:.6f} < {min_moving_actor_token_rate:.6f}")
        if target_pid_rate < min_target_pid_rate:
            failures.append(f"{split}: target pid rate {target_pid_rate:.4f} < {min_target_pid_rate:.4f}")
        if target_actor_rate < min_target_pid_rate:
            failures.append(f"{split}: target actor rate {target_actor_rate:.4f} < {min_target_pid_rate:.4f}")
        if target_rel_rate < min_target_rel_rate:
            failures.append(f"{split}: target rel nonzero rate {target_rel_rate:.4f} < {min_target_rel_rate:.4f}")
        if total.suspect_episode_count:
            failures.append(f"{split}: {total.suspect_episode_count} episodes exceed manifest maxclients")

    # Attack/jump label validation against the demo svc_sound byte truth.
    # Needs the source demos + the corpus manifest (collect has both). Runs
    # the canonical qnn.bc.validate_labels gate (default 5% tolerance) and
    # folds any deviation into the same FAILED summary.
    if validate_labels and demo_dir is not None and corpus_manifest is not None:
        from qnn.bc import validate_labels as _vl
        print()
        rc = _vl.validate_labels(
            data_dir,
            demo_dir=demo_dir,
            manifest_path=corpus_manifest,
        )
        if rc != 0:
            failures.append("attack/jump label validation failed (see above)")

    # Look turn-delta grid drift (informational): EMD of this collect's theta
    # distribution vs a reference collect's. Rate changes legitimately shift it,
    # so this never fails the gate — it flags when a pinned grid may be stale.
    if drift_ref is not None and drift_ref.resolve() != data_dir.resolve():
        import json as _json
        try:
            from qnn.model import look_grid as _lg
            this_lg = _json.loads((data_dir / "collect_metadata.json").read_text()).get("look_grid")
            ref_lg = _json.loads((drift_ref / "collect_metadata.json").read_text()).get("look_grid")
            if this_lg and ref_lg:
                emd = _lg.emd_theta(this_lg["theta_hist"]["counts"], ref_lg["theta_hist"]["counts"])
                print()
                print(f"look-grid drift vs {drift_ref}: theta EMD={emd:.2f}deg  "
                      f"hold_frac {ref_lg['hold_frac']:.3f}->{this_lg['hold_frac']:.3f}"
                      + ("  [LARGE — refit/repin the grid for this corpus]" if emd > 2.0 else ""))
        except (FileNotFoundError, KeyError, TypeError):
            pass  # reference or look_grid block absent — skip drift readout

    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("OK")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate BC collect actor and target labels")
    parser.add_argument("data_dir", nargs="?", default="artifacts/collect/qwd")
    parser.add_argument("--corpus-manifest", default="artifacts/corpus/qwd_manifest.ndjson")
    parser.add_argument("--demo-dir", default="artifacts/corpus/qwd",
                        help="Source demo dir for attack/jump label validation")
    parser.add_argument("--no-validate-labels", dest="validate_labels",
                        action="store_false", default=True,
                        help="Skip the attack/jump label byte-truth gate")
    parser.add_argument("--min-live-actor-frame-rate", type=float, default=0.95)
    parser.add_argument("--min-moving-actor-token-rate", type=float, default=0.001)
    parser.add_argument("--min-target-pid-rate", type=float, default=0.99)
    parser.add_argument("--min-target-rel-rate", type=float, default=0.95)
    parser.add_argument("--unique-pid-slack", type=int, default=0)
    parser.add_argument("--drift-ref", default="artifacts/collect/qwd",
                        help="Reference collect for look-grid theta drift readout "
                             "(informational; empty to skip)")
    args = parser.parse_args()

    manifest = Path(args.corpus_manifest) if args.corpus_manifest else None
    raise SystemExit(validate_collect(
        Path(args.data_dir),
        corpus_manifest=manifest,
        drift_ref=Path(args.drift_ref) if args.drift_ref else None,
        min_live_actor_frame_rate=args.min_live_actor_frame_rate,
        min_moving_actor_token_rate=args.min_moving_actor_token_rate,
        min_target_pid_rate=args.min_target_pid_rate,
        min_target_rel_rate=args.min_target_rel_rate,
        unique_pid_slack=args.unique_pid_slack,
        demo_dir=Path(args.demo_dir) if args.demo_dir else None,
        validate_labels=args.validate_labels,
    ))


if __name__ == "__main__":
    main()
