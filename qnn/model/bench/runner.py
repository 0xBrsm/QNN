"""Head-probe runner — thin adapter over the canonical BC trainer.

Head probes are *just alternate models* plugged into the BC training
pipeline. Shard loading, segment_mask filtering, fingerprint
verification, episode shuffling, BPTT batching, eval cadence,
checkpointing, and ``bc_history.json`` all come from ``qnn.bc.train``
unchanged.

This runner reads the run-dir's three config blobs:

  * ``train.json``  — every BCConfig training knob. Same schema as
                       ``qnn.bc.templates/train.json`` (the BC training
                       template). ``head_loss_weights`` lives here too,
                       so each probe run-dir owns the loss-weight choice
                       — zero out the heads the model doesn't predict.
  * ``machine.json`` — same machine knobs the BC runner uses.
  * ``probe.json``   — head-specific architecture knobs (which head,
                       MLP shape, feature list, obs-embedding width, …),
                       all consumed by ``HeadSpec.build``.

It asks the per-head ``HeadSpec.build`` callable to produce a
``(ModelConfig, model_factory)`` pair, then calls
``run_behavior_cloning`` with the injected factory.

No defaults live in this file — every BCConfig field must be present
in the run-dir config blobs. ``BCConfig(**cfg)`` enforces the schema:
missing or extra keys raise loud at construction time.
"""

from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Any

from qnn.model.bench.heads import HEADS
from qnn.bc.train import BCConfig, run_behavior_cloning
from qnn.run.common import RunnerContext, base_results, finalize_results, prepare_bc_run_outputs
from qnn.run.config import (
    _require_key,
    _require_mapping,
    _require_string,
    run_output_dirs,
)


def _build_head_probe_bc_config(
    run_cfg: dict[str, Any],
    device: str,
) -> tuple[BCConfig, "Callable[[int, Any], Any]"]:  # noqa: F821 — Callable resolved at call time
    """Translate a head_probe run-dir config into BCConfig + model_factory.

    Mirrors ``qnn.run.config.build_run_bc_config``: copy train.json
    verbatim, augment with the output_dir + bc_data_dir + machine
    knobs, attach the synthesized ModelConfig. BCConfig's dataclass
    constructor then enforces the schema — any train.json or machine.json
    field missing from the run-dir surfaces as a TypeError there. No
    Python-level defaults live in this runner.
    """
    train = _require_mapping(run_cfg, "train", "run config")
    machine = _require_mapping(run_cfg, "machine", "run config")
    probe = _require_mapping(run_cfg, "probe", "run config")

    head_name = _require_string(probe, "head", "probe.json")
    if head_name not in HEADS:
        raise RuntimeError(
            f"probe.json.head={head_name!r} not registered; "
            f"available: {sorted(HEADS)}"
        )
    model_config, model_factory = HEADS[head_name].build(probe)

    bc_cfg: dict[str, Any] = dict(train)
    bc_cfg["model"] = model_config
    bc_cfg["output_dir"] = str(run_output_dirs(run_cfg)["checkpoints"])
    bc_cfg["bc_data_dir"] = _require_string(machine, "bc_data_dir", "machine.json")
    bc_cfg["device"] = device
    if "microbatch_size" in machine:
        raise ValueError(
            "machine.json: 'microbatch_size' was renamed — use 'batch_size'. "
            "For recurrent training, batch_size is the parallel lane count."
        )
    bc_cfg["batch_size"] = int(_require_key(machine, "batch_size", "machine.json"))
    bc_cfg["pin_memory"] = bool(_require_key(machine, "pin_memory", "machine.json"))
    bc_cfg["prefetch"] = int(_require_key(machine, "prefetch", "machine.json"))
    bc_cfg["snapshot_interval"] = int(_require_key(machine, "snapshot_interval", "machine.json"))
    bc_cfg["streaming"] = bool(_require_key(machine, "streaming", "machine.json"))
    # Optional per-run knob: engagement EMA decay rate. Defaults to 0.5
    # to preserve historical behavior for run-dirs that don't set it.
    bc_cfg["engagement_ema_alpha"] = float(probe.get("engagement_ema_alpha", 0.5))
    # Optional per-run knob: when true, the attack-head LOSS is computed
    # against a +1 op-frame shifted attack label (val metrics still use
    # the original label). Off by default — bit-identical to historical
    # runs that don't set it. See BCConfig.attack_label_shift.
    bc_cfg["attack_label_shift"] = bool(probe.get("attack_label_shift", False))

    return BCConfig(**bc_cfg), model_factory


def run(ctx: RunnerContext) -> dict[str, object]:
    """Router entry — drive a head-probe run-dir through canonical BC."""
    results = base_results(ctx)
    stage_timings: dict[str, float] = {}

    bc_config, model_factory = _build_head_probe_bc_config(ctx.run_cfg, ctx.device)
    prepare_bc_run_outputs(ctx.run_cfg, resume=ctx.resume)

    bc_data_dir = Path(bc_config.bc_data_dir)
    train_cache = bc_data_dir / "precomputed_train"
    if not train_cache.exists():
        raise RuntimeError(
            f"BC training data not found at {train_cache}. "
            f"Run python -m qnn.bc.collect first."
        )

    started = _time.monotonic()
    seed_checkpoint = str(ctx.run_cfg.get("checkpoint_path", "") or "")
    results["bc"] = run_behavior_cloning(
        bc_config, seed_checkpoint=seed_checkpoint, model_factory=model_factory,
    )
    stage_timings["bc"] = _time.monotonic() - started
    results["stage_timings"] = stage_timings
    return finalize_results(ctx, results, stage_timings)
