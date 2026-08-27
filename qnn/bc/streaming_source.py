"""Metadata-first access to sharded BC caches.

``StreamingSource`` opens shard arrays lazily as memmaps, keeps only
episode metadata in memory, and can read row ranges on demand without
constructing episode objects or copying whole shards through the parent
process.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from qnn.bc.train import (
    _NATIVE_TOKEN_INDEXED_OBS_FIELDS,
    _build_indptr,
    _filter_referenced_keys,
    _mask_target_probs_for_tokens,
    _mask_token_array,
    _shard_segments,
    _token_keep_mask,
    _unpack_attack_bit,
    _unpack_move_axes,
)


@dataclass(frozen=True, slots=True)
class EpisodeRef:
    """A logical episode or train-time segment inside one shard."""

    shard_idx: int
    row_start: int
    row_end: int
    token_start: int
    token_end: int
    sort_key: tuple[int, int, int]

    @property
    def n_samples(self) -> int:
        return self.row_end - self.row_start


@dataclass(slots=True)
class ShardView:
    """Thread-local mmap view of one shard."""

    obs: dict[str, np.ndarray]
    actions: dict[str, np.ndarray]
    indptr: np.ndarray | None
    token_keep: np.ndarray | None


class StreamingSource:
    """Lazy, mmap-backed BC cache reader."""

    def __init__(
        self,
        cache_dir: Path,
        manifest: dict[str, Any],
        episodes: list[EpisodeRef],
        token_mask: dict | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.manifest = manifest
        self._shards = list(manifest.get("shards", []))
        self.episodes = episodes
        self._token_mask = token_mask
        self._local = threading.local()

    @classmethod
    def from_cache_dir(
        cls,
        cache_dir: Path,
        *,
        segment_mask: dict | None = None,
        token_mask: dict | None = None,
    ) -> "StreamingSource":
        cache_dir = Path(cache_dir)
        manifest = json.loads((cache_dir / "manifest.json").read_text())
        if not isinstance(manifest, dict) or manifest.get("format") != "sharded_v1":
            raise RuntimeError(f"{cache_dir}/manifest.json: expected sharded_v1 format")
        if manifest.get("format_version") != "native_v1":
            raise RuntimeError(
                f"{cache_dir}/manifest.json: expected format_version='native_v1', "
                f"got {manifest.get('format_version')!r}"
            )
        episodes: list[EpisodeRef] = []
        fallback_idx = 0
        for shard_idx, shard in enumerate(manifest.get("shards", [])):
            refs = _build_shard_episode_refs(
                cache_dir,
                shard_idx,
                shard,
                fallback_idx,
                segment_mask=segment_mask,
                token_mask=token_mask,
            )
            episodes.extend(refs)
            fallback_idx += len(shard.get("episode_lengths", []))
        episodes.sort(key=lambda ref: ref.sort_key)
        return cls(cache_dir, manifest, episodes, token_mask=token_mask)

    def open_shard(self, shard_idx: int) -> ShardView:
        """Open one shard as thread-local memmaps."""

        cache: dict[int, ShardView] | None = getattr(self._local, "cache", None)
        if cache is None:
            cache = {}
            self._local.cache = cache
        view = cache.get(shard_idx)
        if view is not None:
            return view

        shard = self._shards[shard_idx]
        obs = {
            key: np.load(self.cache_dir / fname, mmap_mode="r")
            for key, fname in shard["obs"].items()
        }
        actions = {
            key: np.load(self.cache_dir / fname, mmap_mode="r")
            for key, fname in shard["actions"].items()
        }
        indptr = None
        token_keep = None
        if "entity_count" in obs:
            indptr = _build_indptr(obs["entity_count"])
            token_keep = _token_keep_mask(obs, self._token_mask)
            if token_keep is not None and "target_probs" in actions:
                actions = dict(actions)
                actions["target_probs"] = _mask_target_probs_for_tokens(
                    actions["target_probs"], indptr, token_keep,
                )
        view = ShardView(obs=obs, actions=actions, indptr=indptr, token_keep=token_keep)
        cache[shard_idx] = view
        return view

    def read_rows(
        self,
        ref: EpisodeRef,
        row_lo: int,
        row_hi: int,
        keys: Iterable[str],
    ) -> dict[str, np.ndarray]:
        """Read a source-relative row range from an ``EpisodeRef``.

        ``row_lo`` / ``row_hi`` are relative to ``ref``.  Row-indexed
        fields slice the shard row axis.  Token-indexed entity fields
        slice the flat token axis using ``entity_count`` / ``indptr``.
        Action fields use the ``act.`` prefix.
        """

        if row_lo < 0 or row_hi < row_lo or row_hi > ref.n_samples:
            raise IndexError(
                f"invalid row slice [{row_lo}, {row_hi}) for episode of length {ref.n_samples}"
            )
        view = self.open_shard(ref.shard_idx)
        abs_lo = ref.row_start + row_lo
        abs_hi = ref.row_start + row_hi
        out: dict[str, np.ndarray] = {}

        want = list(keys)
        if any(k in _NATIVE_TOKEN_INDEXED_OBS_FIELDS for k in want) and "entity_count" not in want:
            want.append("entity_count")

        for key in want:
            if key.startswith("act."):
                head = key[4:]
                if head == "attack" and "attack" not in view.actions and "move" in view.actions:
                    out[key] = _unpack_attack_bit(view.actions["move"][abs_lo:abs_hi])
                    continue
                out[key] = view.actions[head][abs_lo:abs_hi]
                continue
            if key in _NATIVE_TOKEN_INDEXED_OBS_FIELDS and view.indptr is not None:
                tok_lo = int(view.indptr[abs_lo])
                tok_hi = int(view.indptr[abs_hi])
                arr = view.obs[key][tok_lo:tok_hi]
                if view.token_keep is not None:
                    arr = _mask_token_array(key, arr, view.token_keep[tok_lo:tok_hi])
                out[key] = arr
                continue
            out[key] = view.obs[key][abs_lo:abs_hi]
        return out


def _build_shard_episode_refs(
    cache_dir: Path,
    shard_idx: int,
    shard: dict[str, Any],
    fallback_idx_start: int,
    *,
    segment_mask: dict | None,
    token_mask: dict | None,
) -> list[EpisodeRef]:
    if "entity_count" not in shard.get("obs", {}):
        raise RuntimeError(f"{cache_dir} shard {shard_idx} is missing obs.entity_count")
    if "target_probs" not in shard.get("actions", {}):
        raise RuntimeError(f"{cache_dir} shard {shard_idx} is missing act.target_probs")
    obs = {
        key: np.load(cache_dir / fname, mmap_mode="r")
        for key, fname in shard["obs"].items()
    }
    actions = {
        key: np.load(cache_dir / fname, mmap_mode="r")
        for key, fname in shard["actions"].items()
    }
    if "move" in actions:
        refs = _filter_referenced_keys(segment_mask)
        if "act.move" in refs or "act.attack" in refs:
            move_packed = actions["move"]
            actions["move"] = _unpack_move_axes(move_packed)
            actions["attack"] = _unpack_attack_bit(move_packed)

    indptr = _build_indptr(obs["entity_count"])
    token_keep = _token_keep_mask(obs, token_mask)
    if token_keep is not None:
        actions["target_probs"] = _mask_target_probs_for_tokens(
            actions["target_probs"], indptr, token_keep,
        )

    segments = _shard_segments(
        shard, fallback_idx_start, obs, actions, indptr, segment_mask,
    )
    return [
        EpisodeRef(
            shard_idx=shard_idx,
            row_start=seg.src_row_start,
            row_end=seg.src_row_end,
            token_start=int(indptr[seg.src_row_start]),
            token_end=int(indptr[seg.src_row_end]),
            sort_key=seg.meta.sort_key,
        )
        for seg in segments
    ]


def _token_range(indptr: np.ndarray | None, row_start: int, row_end: int) -> tuple[int, int]:
    if indptr is None:
        return (0, 0)
    return (int(indptr[row_start]), int(indptr[row_end]))
