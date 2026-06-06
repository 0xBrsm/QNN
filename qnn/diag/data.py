"""Lightweight val-data loader for diagnostics.

Loads a small slice of precomputed BC val data — enough to compute losses, gradients,
and ablation deltas reliably without paying full-corpus eval cost.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


def load_val_episodes(
    data_dir: Path,
    *,
    split: str = "val",
    max_shards: int | None = None,
    max_episodes: int | None = None,
) -> list[dict]:
    """Load episodes from a precomputed shard directory.

    Returns list of dicts with keys ``obs`` (dict of np arrays), ``actions`` (dict),
    and ``n_samples`` (int). Episodes are sliced from shard memmaps; no copy.
    """
    base = Path(data_dir) / f"precomputed_{split}"
    manifest = json.loads((base / "manifest.json").read_text())
    if manifest.get("format") != "sharded_v1":
        raise RuntimeError(f"{base}/manifest.json: expected sharded_v1 format")

    shards = manifest.get("shards", [])
    if max_shards is not None:
        shards = shards[:max_shards]

    episodes: list[dict] = []
    for shard in shards:
        obs_arrays = {k: np.load(base / v, mmap_mode="r") for k, v in shard["obs"].items()}
        action_arrays = {k: np.load(base / v, mmap_mode="r") for k, v in shard["actions"].items()}
        # Unpack packed move byte → (T, 3) axis class indices like the trainer does.
        # Format: one-hot-per-direction (mirror of input_mask byte):
        #   bit 0 = attack, bit 1 = fb_neg, bit 2 = fb_pos,
        #   bit 3 = lr_neg, bit 4 = lr_pos, bit 5 = ud_neg, bit 6 = ud_pos,
        #   bit 7 = jump.
        # Per-axis class = 1 + pos_bit - neg_bit (in {0=neg, 1=none, 2=pos}).
        if "move" in action_arrays:
            arr = np.asarray(action_arrays["move"], dtype=np.uint8)
            if arr.ndim == 1:
                fb = (1 + ((arr >> 2) & 1) - ((arr >> 1) & 1)).astype(np.uint8)
                lr = (1 + ((arr >> 4) & 1) - ((arr >> 3) & 1)).astype(np.uint8)
                ud = (1 + ((arr >> 6) & 1) - ((arr >> 5) & 1)).astype(np.uint8)
                action_arrays["move"] = np.ascontiguousarray(np.stack([fb, lr, ud], axis=-1))

        ep_lengths = shard.get("episode_lengths", [])
        cursor = 0
        for n_samples in ep_lengths:
            end = cursor + int(n_samples)
            obs = {k: v[cursor:end] for k, v in obs_arrays.items()}
            actions = {k: v[cursor:end] for k, v in action_arrays.items()}
            episodes.append({"obs": obs, "actions": actions, "n_samples": int(n_samples)})
            cursor = end
            if max_episodes is not None and len(episodes) >= max_episodes:
                return episodes
    return episodes


def episode_to_torch(episode: dict, device: torch.device, *, seq_first: bool = True) -> tuple[dict, dict]:
    """Convert one episode's numpy arrays to torch tensors on device.

    Returns (obs_tensors, action_tensors). Obs are unsqueezed to (T, 1, ...) batch=1
    for sequence-first model forward; actions stay (T, ...) — caller reshapes as needed.
    """
    obs_t: dict[str, torch.Tensor] = {}
    for key, arr in episode["obs"].items():
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
        if seq_first:
            t = t.unsqueeze(1)  # (T, 1, ...)
        obs_t[key] = t
    act_t: dict[str, torch.Tensor] = {
        k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in episode["actions"].items()
    }
    return obs_t, act_t


def iter_episodes(episodes: list[dict], device: torch.device) -> Iterator[tuple[dict, dict]]:
    """Yield (obs_tensors, action_tensors) for each episode."""
    for ep in episodes:
        yield episode_to_torch(ep, device)
