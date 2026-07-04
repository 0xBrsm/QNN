"""Reusable BC source containers.

This module owns the resident/streaming corpus bundle used by normal BC
training and by the dynamic ablation daemon. It deliberately avoids job
scheduling or socket concerns; those live in ``qnn.bc.ablation_daemon``.
"""

from __future__ import annotations

import json
import os
import time as _time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from qnn.model.policy import HEAD_LOSS_WEIGHTS
from qnn.utils.device import resolve_torch_device


@dataclass(slots=True)
class BCSourceBundle:
    """Reusable train/val sources built from one BC cache configuration."""

    train_source: Any
    val_source: Any
    actual_fingerprint: Mapping[str, Any] | None
    compatibility_key: tuple[Any, ...]

    @property
    def sample_counts(self) -> dict[str, int]:
        return {
            "train": int(self.train_source.n_total_rows),
            "val": int(self.val_source.n_total_rows),
        }

    def release_device_tensors(self) -> None:
        self.train_source.release_device_tensors()
        self.val_source.release_device_tensors()


def effective_head_loss_weights(raw: str) -> Dict[str, float]:
    weights = dict(HEAD_LOSS_WEIGHTS)
    if not raw:
        return weights
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"head_loss_weights must be a JSON object, got {type(parsed).__name__}")
    for head, value in parsed.items():
        weights[str(head)] = float(value)
    return weights


def _require_action_files(
    action_files: Mapping[str, str],
    required_actions: frozenset[str],
    *,
    cache_dir: Path,
) -> None:
    missing = sorted(required_actions.difference(action_files))
    if not missing:
        return
    raise RuntimeError(
        f"{cache_dir} is missing required action arrays {missing}. "
        "Recollect BC data on this branch before training."
    )


def validate_cache_for_training(
    cache_dir: Path,
    *,
    required_actions: frozenset[str],
) -> None:
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("format") != "sharded_v1":
        raise RuntimeError(
            f"{cache_dir}/manifest.json: expected sharded_v1 format. "
            "Recollect BC data with the current collector."
        )
    format_version = manifest.get("format_version")
    if format_version != "native_v1":
        raise RuntimeError(
            f"{cache_dir}/manifest.json: expected format_version='native_v1', "
            f"got {format_version!r}. Legacy f16 caches must be recollected "
            f"with the current collector — no silent migration."
        )
    required = set(required_actions)
    required.add("target_probs")
    for shard in manifest.get("shards", []):
        _require_action_files(shard["actions"], frozenset(required), cache_dir=cache_dir)
        if "entity_count" not in shard.get("obs", {}):
            raise RuntimeError(
                f"{cache_dir} contains a native_v1 shard without obs.entity_count; "
                "recollect with the current collector."
            )


def _bc_cache_paths(config: Any) -> tuple[Path, Path, Path]:
    bc_data_dir = Path(config.bc_data_dir) if hasattr(config, "bc_data_dir") else Path(config.output_dir).parent
    return (
        bc_data_dir,
        bc_data_dir / "precomputed_train",
        bc_data_dir / "precomputed_val",
    )


def _required_actions_for_config(
    config: Any,
    head_loss_weights: Mapping[str, float],
) -> frozenset[str]:
    required_actions_set: set[str] = set()
    if config.model.use_weapon_head and float(head_loss_weights.get("weapon", 1.0)) > 0.0:
        required_actions_set.add("weapon")
    # input_mask is required when labels/derived loss targets depend on
    # op-frame semantics. Corpora without it still train cleanly when
    # those toggles are off.
    if config.input_mask or config.attack_label_shift:
        required_actions_set.add("input_mask")
    return frozenset(required_actions_set)


def _needs_move_hazard(config: Any, head_loss_weights: Mapping[str, float] | None = None) -> bool:
    """a25: does this run train the move-hazard head? Derived from the run's
    head_loss_weights so it is correct even when the caller (e.g. the ablation
    daemon's source-bundle build) passes only ``config`` — config.head_loss_weights
    is a raw JSON string, so parse it via effective_head_loss_weights."""
    weights = head_loss_weights if head_loss_weights is not None else effective_head_loss_weights(
        config.head_loss_weights
    )
    return bool(float((weights or {}).get("move_hazard", 0.0)) > 0.0)


def _source_compatibility_key(
    config: Any,
    *,
    required_actions: frozenset[str],
) -> tuple[Any, ...]:
    def _json_key(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    device_spec = resolve_torch_device(str(config.device))
    return (
        str(Path(config.bc_data_dir).resolve()),
        device_spec.resolved,
        device_spec.backend,
        bool(config.streaming),
        str(config.dtype),
        _json_key(config.segment_mask),
        _json_key(config.token_mask),
        tuple(sorted(required_actions)),
        float(config.engagement_ema_alpha),
        # a25: hazard runs carry extra derived obs/act columns, so they must not
        # share a cached source bundle with hazard-less runs (and vice versa).
        _needs_move_hazard(config),
    )


def source_compatibility_key_for_config(config: Any) -> tuple[Any, ...]:
    head_loss_weights = effective_head_loss_weights(config.head_loss_weights)
    required_actions = _required_actions_for_config(config, head_loss_weights)
    return _source_compatibility_key(config, required_actions=required_actions)


def validate_source_bundle_compatible(
    config: Any,
    bundle: BCSourceBundle,
    *,
    head_loss_weights: Mapping[str, float] | None = None,
) -> None:
    weights = head_loss_weights if head_loss_weights is not None else effective_head_loss_weights(config.head_loss_weights)
    required_actions = _required_actions_for_config(config, weights)
    expected = _source_compatibility_key(config, required_actions=required_actions)
    if bundle.compatibility_key != expected:
        raise RuntimeError(
            "BC source bundle is incompatible with this run config. "
            f"bundle={bundle.compatibility_key!r} expected={expected!r}"
        )


def build_behavior_cloning_sources(
    config: Any,
    *,
    head_loss_weights: Mapping[str, float] | None = None,
) -> BCSourceBundle:
    """Build reusable train/val sources for BC.

    The returned bundle can be passed to compatible calls to
    ``run_behavior_cloning`` to avoid rematerializing the corpus for
    sequential or in-process parallel ablations.
    """

    from qnn import collection_fingerprint
    from qnn.bc.loop import (
        make_resident_source,
        make_resident_source_from_cache,
        make_streaming_source,
    )

    weights = head_loss_weights if head_loss_weights is not None else effective_head_loss_weights(config.head_loss_weights)
    bc_data_dir, train_cache, val_cache = _bc_cache_paths(config)
    if not train_cache.exists():
        raise RuntimeError(
            f"BC training data not found at {train_cache}. "
            "Run python -m qnn.bc.collect first."
        )

    actual_fp = collection_fingerprint.verify(
        expected_fingerprint=config.collection_fingerprint,
        data_dir=bc_data_dir,
    )
    print(f"  [bc] collection fingerprint: {actual_fp['fingerprint']}")

    required_actions = _required_actions_for_config(config, weights)
    validate_cache_for_training(train_cache, required_actions=required_actions)
    if val_cache.exists():
        validate_cache_for_training(val_cache, required_actions=required_actions)

    print(f"  [bc] Loading training data: {train_cache}")
    if config.segment_mask:
        print(f"  [bc] segment_mask: {config.segment_mask}")
    if config.token_mask:
        print(f"  [bc] token_mask: {config.token_mask}")

    _t0 = _time.monotonic()
    eng_alpha = float(config.engagement_ema_alpha)
    device = resolve_torch_device(str(config.device)).device
    # a25: derive the move-hazard columns on the fly only when the run trains
    # that head. Derived from config so it is correct even when called WITHOUT
    # head_loss_weights (the ablation daemon's source-bundle build passes only
    # config) — that gap is exactly what KeyError'd 'move_held_class' before.
    needs_move_hazard = _needs_move_hazard(config, head_loss_weights)
    if bool(config.streaming):
        print(f"  [bc] streaming=true: lazy mmap reads from {train_cache}")
        train_source = make_streaming_source(
            train_cache, device,
            segment_mask=config.segment_mask, token_mask=config.token_mask,
            prefetch_depth=max(2, int(config.prefetch)),
            engagement_ema_alpha=eng_alpha,
            needs_move_hazard=needs_move_hazard,
        )
        val_source = (
            make_streaming_source(
                val_cache, device,
                segment_mask=config.segment_mask, token_mask=config.token_mask,
                prefetch_depth=max(2, int(config.prefetch)),
                engagement_ema_alpha=eng_alpha,
                needs_move_hazard=needs_move_hazard,
            ) if val_cache.exists()
            else make_resident_source([], device, engagement_ema_alpha=eng_alpha)
        )
    else:
        import gc as _gc

        print(f"  [bc] preload: materializing train cache directly to {device}")
        train_source = make_resident_source_from_cache(
            train_cache, device,
            segment_mask=config.segment_mask,
            token_mask=config.token_mask,
            chunk_rows=int(os.environ.get("QNN_BC_RESIDENT_PRELOAD_CHUNK_ROWS", "65536")),
            engagement_ema_alpha=eng_alpha,
            compact_dequantized=True,
            needs_move_hazard=needs_move_hazard,
        )
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if val_cache.exists():
            print(f"  [bc] preload: materializing val cache directly to {device}")
            val_source = make_resident_source_from_cache(
                val_cache, device,
                segment_mask=config.segment_mask,
                token_mask=config.token_mask,
                chunk_rows=int(os.environ.get("QNN_BC_RESIDENT_PRELOAD_CHUNK_ROWS", "65536")),
                engagement_ema_alpha=eng_alpha,
                compact_dequantized=True,
                needs_move_hazard=needs_move_hazard,
            )
        else:
            val_source = make_resident_source([], device, engagement_ema_alpha=eng_alpha)
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"  [bc] source ready in {_time.monotonic() - _t0:.1f}s")

    return BCSourceBundle(
        train_source=train_source,
        val_source=val_source,
        actual_fingerprint=actual_fp,
        compatibility_key=_source_compatibility_key(config, required_actions=required_actions),
    )
