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
        """Drop the sources' device tensors AND hand their blocks back.

        ``ResidentSource.release_device_tensors`` only clears the obs/actions
        dicts.  That drops the references, so the tensors are freed to torch's
        caching allocator as *reserved* — held by this process, not returned to
        the device.  Reuse hides it whenever the next bundle wants
        identically-shaped blocks, which is every reload of the same corpus, so
        the daemon has always looked clean across `reset`.  A differently sized
        corpus cannot reuse them: the allocator requests fresh segments while
        the previous corpus still occupies its own, and the device OOMs at
        roughly the sum of the two.  That reads as a mysterious OOM on a corpus
        that fits comfortably on its own.

        ``gc.collect()`` first — a tensor still reachable through a reference
        cycle is not freed by ``clear()`` alone, and ``empty_cache`` can only
        release blocks with no live tensors in them.
        """
        import gc as _gc

        self.train_source.release_device_tensors()
        self.val_source.release_device_tensors()
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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


# Head names whose HeadNodeSpec.hz is meaningful today (mirrors the "hz"
# override key qnn.model.graph.spec.HeadNodeSpec.from_dict allows only for
# name == "look_seg"). Every other head's hz field is unreachable from JSON
# (from_dict rejects the key) and stays at the dataclass default 0, so
# checking it would false-positive on every graph. Add a name here if another
# head type gains its own hz parameterization.
_HZ_BEARING_HEADS = ("look_seg",)


def validate_head_hz_against_fixed_tick_hz(graph: Any, fixed_tick_hz: int) -> None:
    """Fail loud if a head's corpus-tick-rate parameter disagrees with the
    run's ``fixed_tick_hz``.

    Two independent Hz paths exist and nothing else cross-checks them: this
    train config's ``fixed_tick_hz`` (drives the load-time corpus resample,
    qnn.bc.resample, and what the loader actually feeds the model) and each
    head's own ``HeadNodeSpec.hz`` (drives which ``look_seg_bins`` table
    ``look_seg`` labels/decodes against, resolved via
    ``look_seg_bins.resolve_hz``). A probe.json that adds a ``look_seg`` head
    without stamping ``"hz"`` silently resolves to ``LEGACY_HZ`` (10) — this
    exact bug class (wrong-Hz quantiles) shipped once already (commit
    7836cac1) before ``hz`` existed at all.

    ``graph`` is ``None`` for the plain (non-bench) BC path, which has no
    ``look_seg`` head to check — nothing to validate.

    Deliberately RESOLVES hz (via ``resolve_hz``) before comparing, rather
    than exempting raw ``hz == 0``: exempting it would just re-open the same
    silent-default hole this guard exists to close. ``resolve_hz(0)`` is only
    correct for a head that was actually trained at ``LEGACY_HZ`` — if this
    run's ``fixed_tick_hz`` differs, an unstamped head is exactly as wrong as
    a mis-stamped one, and must fail the same way.
    """
    if graph is None:
        return
    from qnn.model.look_seg_bins import resolve_hz

    target = int(fixed_tick_hz)
    for head in getattr(graph, "heads", ()):
        if head.name not in _HZ_BEARING_HEADS:
            continue
        raw_hz = int(getattr(head, "hz", 0) or 0)
        resolved = resolve_hz(raw_hz)
        if resolved != target:
            raise ValueError(
                f"head {head.name!r} hz={raw_hz} resolves to {resolved} Hz, "
                f"but this run trains at fixed_tick_hz={target}. Stamp "
                f'"hz": {target} on heads.{head.name} in probe.json (or fix '
                "fixed_tick_hz) — an unresolved mismatch silently trains "
                "this head's labels against the wrong look_seg_bins table."
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
    *,
    graph: Any = None,
) -> frozenset[str]:
    required_actions_set: set[str] = set()
    if float(head_loss_weights.get("attack", 1.0)) > 0.0:
        required_actions_set.add("attack")
    # input_mask is required when labels/derived loss targets depend on
    # op-frame semantics. Corpora without it still train cleanly when
    # those toggles are off.
    if config.input_mask or config.attack_label_shift:
        required_actions_set.add("input_mask")
    # Fire-at-alignment objective (agents/plans/fire-at-alignment-objective.md):
    # the attack_with selector's align_weight_gamma knob reads
    # actions["align_hbw"] (qnn.bc.cache_align_hbw). Fail loud at container
    # startup rather than silently training with the knob inert — the plain
    # (non-bench) BC path has no graph/head specs to inspect, so this only
    # ever fires for bench probes that set the knob.
    for head in getattr(graph, "heads", ()):
        if head.name == "attack" and float(getattr(head, "align_weight_gamma", 0.0)) > 0.0:
            required_actions_set.add("align_hbw")
    return frozenset(required_actions_set)


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
        # The pinned corpus identity. build_behavior_cloning_sources verifies
        # this against the corpus's fingerprint.json — but ONLY when it builds.
        # On the daemon's reuse path the build is skipped, so without this a run
        # pinning a different fingerprint silently trained on the cached bundle
        # of a different collect. (Disk changing under a bundle with the SAME
        # pinned value is caught by the re-verify in
        # validate_source_bundle_compatible.)
        str(getattr(config, "collection_fingerprint", "") or ""),
        # Load-time temporal resample is derived from fixed_tick_hz against the
        # corpus's collected tick_hz (build_behavior_cloning_sources ->
        # qnn.bc.resample), so it bakes into the resident tensors. Without this
        # a 10 Hz run and a 20 Hz run on one corpus SHARE a bundle and one of
        # them trains on the wrong frame grouping.
        int(getattr(config, "fixed_tick_hz", 0) or 0),
    )


def source_compatibility_key_for_config(config: Any, *, graph: Any = None) -> tuple[Any, ...]:
    head_loss_weights = effective_head_loss_weights(config.head_loss_weights)
    required_actions = _required_actions_for_config(config, head_loss_weights, graph=graph)
    return _source_compatibility_key(config, required_actions=required_actions)


def validate_source_bundle_compatible(
    config: Any,
    bundle: BCSourceBundle,
    *,
    head_loss_weights: Mapping[str, float] | None = None,
    graph: Any = None,
) -> None:
    weights = head_loss_weights if head_loss_weights is not None else effective_head_loss_weights(config.head_loss_weights)
    required_actions = _required_actions_for_config(config, weights, graph=graph)
    expected = _source_compatibility_key(config, required_actions=required_actions)
    if bundle.compatibility_key != expected:
        raise RuntimeError(
            "BC source bundle is incompatible with this run config. "
            f"bundle={bundle.compatibility_key!r} expected={expected!r}"
        )
    # Corpus-drift check on the REUSE path. The key above pins the CONFIG's
    # fingerprint, which catches "different run, different pinned corpus". It
    # cannot catch "same pinned value, corpus recollected on disk while a bundle
    # is live" — and build_behavior_cloning_sources' verify only runs when it
    # BUILDS, so on reuse nothing looked at the corpus at all.
    #
    # The bundle already records what it baked (`actual_fingerprint`, set at
    # build); it was simply never compared. Compare it. One small JSON read.
    from qnn import collection_fingerprint
    baked = bundle.actual_fingerprint
    if baked is not None:
        bc_data_dir, _train_cache, _val_cache = _bc_cache_paths(config)
        current = collection_fingerprint.load(bc_data_dir)
        if current is not None:
            # Schema-aware: baked and current can legitimately be different
            # fingerprint SCHEMA versions (e.g. the corpus was recollected
            # under a newer qnn.collection_fingerprint that added
            # components) without the underlying corpus having changed at
            # all. A raw composite-hash `!=` can't tell those apart and
            # would misreport every schema bump as corpus drift.
            cmp = collection_fingerprint.compare(baked, current)
            if not cmp["equivalent"]:
                detail = (
                    f"schema v{cmp['schema_a']} vs v{cmp['schema_b']}, "
                    f"differing components: {cmp['differing_components']}"
                    if cmp["schema_mismatch"] else
                    f"differing components: {cmp['differing_components']}"
                )
                raise RuntimeError(
                    "BC source bundle was built from a different collect than the "
                    f"corpus now on disk at {bc_data_dir}: bundle baked "
                    f"{baked.get('fingerprint')!r}, disk now has "
                    f"{current.get('fingerprint')!r} ({detail}). The corpus was "
                    "recollected while this bundle was cached — reset the daemon "
                    "(or restart it) so the next submit rebuilds from the current data."
                )
            elif cmp["schema_mismatch"]:
                print(
                    "  [bc] note: collection fingerprint schema upgraded "
                    f"(v{cmp['schema_a']} -> v{cmp['schema_b']}) on disk at "
                    f"{bc_data_dir}, but the corpus content is unchanged "
                    "(common components match) — not treated as drift"
                )


def build_behavior_cloning_sources(
    config: Any,
    *,
    head_loss_weights: Mapping[str, float] | None = None,
    graph: Any = None,
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

    required_actions = _required_actions_for_config(config, weights, graph=graph)
    validate_cache_for_training(train_cache, required_actions=required_actions)
    if val_cache.exists():
        validate_cache_for_training(val_cache, required_actions=required_actions)

    # Load-time temporal resample: if the run's fixed_tick_hz is below the
    # corpus's collected rate, group every corpus_hz//target_hz frames into one
    # at load (qnn.bc.resample) — no second collect. ratio 1 = no-op default.
    resample_ratio = 1
    _meta_path = bc_data_dir / "collect_metadata.json"
    _corpus_hz = (json.loads(_meta_path.read_text()).get("tick_hz")
                  if _meta_path.exists() else None)
    _tgt_hz = int(getattr(config, "fixed_tick_hz", 0) or 0)
    if _corpus_hz and _tgt_hz and _tgt_hz < _corpus_hz:
        from qnn.bc.resample import resample_ratio as _resample_ratio
        resample_ratio = _resample_ratio(_corpus_hz, _tgt_hz)
        print(f"  [bc] load-time resample: corpus {_corpus_hz} Hz -> {_tgt_hz} Hz "
              f"(group {resample_ratio} frames/tick)")

    print(f"  [bc] Loading training data: {train_cache}")
    if config.segment_mask:
        print(f"  [bc] segment_mask: {config.segment_mask}")
    if config.token_mask:
        print(f"  [bc] token_mask: {config.token_mask}")

    _t0 = _time.monotonic()
    eng_alpha = float(config.engagement_ema_alpha)
    device = resolve_torch_device(str(config.device)).device
    if bool(config.streaming):
        print(f"  [bc] streaming=true: lazy mmap reads from {train_cache}")
        train_source = make_streaming_source(
            train_cache, device,
            segment_mask=config.segment_mask, token_mask=config.token_mask,
            prefetch_depth=max(2, int(config.prefetch)),
            engagement_ema_alpha=eng_alpha,
            resample_ratio=resample_ratio,
        )
        val_source = (
            make_streaming_source(
                val_cache, device,
                segment_mask=config.segment_mask, token_mask=config.token_mask,
                prefetch_depth=max(2, int(config.prefetch)),
                engagement_ema_alpha=eng_alpha,
                    resample_ratio=resample_ratio,
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
            resample_ratio=resample_ratio,
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
                    resample_ratio=resample_ratio,
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
