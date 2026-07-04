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
  * ``probe.json``   — a model-graph delta: a ``base`` graph name plus
                       ``overrides`` (see ``qnn.model.graph``). The merged
                       GraphSpec is built by ``build_network`` and persisted
                       into the checkpoint (self-describing).

It resolves probe.json into a ``GraphSpec``, derives the bridge ``ModelConfig``,
and calls ``run_behavior_cloning`` with the graph attached (no model_factory —
the graph is the single build path).

No defaults live in this file — every BCConfig field must be present
in the run-dir config blobs. ``BCConfig(**cfg)`` enforces the schema:
missing or extra keys raise loud at construction time.
"""

from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Any

from qnn.bc.train import BCConfig, run_behavior_cloning
from qnn.run.common import RunnerContext, base_results, finalize_results, prepare_bc_run_outputs
from qnn.run.config import (
    _require_key,
    _require_mapping,
    _require_string,
    run_output_dirs,
)


_GRAPH_PROBE_KEYS = ("base", "overrides", "engagement_ema_alpha", "attack_label_shift")


def _resolve_probe_model(probe: dict[str, Any]):
    """Resolve probe.json into ``(model_config, model_factory, graph)``.

    A probe is a delta on a committed base graph:

        {"base": "full_5head", "overrides": {...}}

    ``overrides`` deep-merges onto the base (``null`` deletes a node) and
    the merged graph is validated/built by ``qnn.model.graph``. The graph
    is persisted into every checkpoint (self-describing — no probe.json
    rehydration on load). ``model_factory`` is always ``None`` (the graph
    is the single build path; the legacy ``{"head": ...}`` HEADS registry
    was retired once the rc lineage migrated to self-describing graphs).
    """
    from qnn.model.graph import (
        GraphSpec, GraphSpecError, base_graph_dict, merge_overrides, model_config_from_graph,
    )
    from qnn.model.graph.spec import _reject_unknown
    if "base" not in probe:
        raise RuntimeError(
            "probe.json must declare a 'base' graph (+ optional 'overrides'); "
            "the legacy '{\"head\": ...}' schema was retired")
    try:
        _reject_unknown(probe, _GRAPH_PROBE_KEYS, "probe.json")
    except GraphSpecError as exc:
        # Same validation as the graph spec; runner callers expect RuntimeError.
        raise RuntimeError(str(exc)) from None
    raw = merge_overrides(base_graph_dict(str(probe["base"])), probe.get("overrides") or {})
    graph = GraphSpec.from_dict(raw)
    return model_config_from_graph(graph), None, graph


def _build_head_probe_bc_config(
    run_cfg: dict[str, Any],
    device: str,
) -> tuple[BCConfig, "Callable[[int, Any], Any] | None", "Any | None"]:  # noqa: F821
    """Translate a head_probe run-dir config into BCConfig + model source.

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

    model_config, model_factory, graph = _resolve_probe_model(dict(probe))

    # Pin the look turn-delta grid this run trains/decodes with. The grid lives
    # in the run dir (config/look_grid.json) — written by run.init from the
    # corpus's data-fit grid (or the materialized code-default for retained
    # models). No implicit default: a head_probe run-dir must carry its grid.
    # Installing here (once, before the model factory runs) keeps the loss
    # target and the offline decode on the SAME centers. See
    # qnn.model.look_bins.install_polar_grid + qnn.model.look_grid.
    import json as _json
    from qnn.model import look_bins as _look_bins
    run_dir = Path(run_output_dirs(run_cfg)["checkpoints"]).parent
    grid_path = run_dir / "config" / "look_grid.json"
    if not grid_path.exists():
        raise FileNotFoundError(
            f"{grid_path} is absent — every run must pin its look grid "
            "(run.init writes it from the corpus fit; no implicit default).")
    _grid = _json.loads(grid_path.read_text())
    _look_bins.install_polar_grid(_grid["mag_centers_rad"], _grid.get("dir_centers_rad"))
    print(f"  [head_probe] look grid pinned: source={_grid.get('source','?')} "
          f"hold_max={_grid['hold_max_rad']:.5f} rad "
          f"(fit rms {_grid.get('fit_rms_deg', '?')} vs default {_grid.get('default_rms_deg', '?')})")

    bc_cfg: dict[str, Any] = dict(train)
    bc_cfg["model"] = model_config
    bc_cfg["run_id"] = str(_require_mapping(run_cfg, "manifest", "run config").get("run_id", ""))
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

    return BCConfig(**bc_cfg), model_factory, graph


def run(ctx: RunnerContext) -> dict[str, object]:
    """Router entry — drive a head-probe run-dir through canonical BC."""
    results = base_results(ctx)
    stage_timings: dict[str, float] = {}

    bc_config, model_factory, graph = _build_head_probe_bc_config(ctx.run_cfg, ctx.device)
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
    from qnn.model.bench.side_channels import bench_side_channel_scope
    results["bc"] = run_behavior_cloning(
        bc_config, seed_checkpoint=seed_checkpoint, model_factory=model_factory,
        graph=graph, side_channel_provider=bench_side_channel_scope,
    )
    stage_timings["bc"] = _time.monotonic() - started
    results["stage_timings"] = stage_timings
    return finalize_results(ctx, results, stage_timings)
