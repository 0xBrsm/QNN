"""PPO pipeline: run directory → training job → post-train eval."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from qnn.env.planning import validate_native_mod_assets
from qnn.eval.run import EvalConfig, run_evaluation
from qnn.run.common import (
    base_results,
    best_checkpoint,
    ensure_arena_workers,
    ensure_worker,
    finalize_results,
    latest_ppo_checkpoint,
    prepare_eval_checkpoint,
    prepare_eval_outputs,
    prepare_ppo_run_outputs,
    require_cfg_list,
    require_cfg_mapping,
    require_cfg_string,
    require_cfg_value,
)
from qnn.run.config import build_run_ppo_eval_config
from qnn.utils.checkpoint_converter import checkpoint_model_graph
from qnn.utils.io import read_json, trusted_torch_load


def _scenario_config_json_from_cfg(config: dict[str, Any]) -> str:
    path_value = str(require_cfg_value(config, "scenario_config_path", "PPO config")).strip()
    if not path_value:
        return ""
    payload = read_json(path_value)
    scenarios = payload.get("scenarios", payload)
    if not isinstance(scenarios, list) or not scenarios:
        raise RuntimeError(f"scenario_config_path must define a non-empty scenarios list: {path_value}")
    return json.dumps(scenarios)


def _validate_warm_start_arch(init_ckpt: str, ppo_cfg: dict[str, Any]) -> None:
    if not init_ckpt or not Path(init_ckpt).exists():
        return
    arch_keys = (
        "encoder_hidden", "d_gru", "use_gru", "d_model", "n_heads",
        "n_layers", "d_ffn", "attn_dropout",
    )
    try:
        payload = trusted_torch_load(init_ckpt, map_location="cpu")
    except Exception:
        return
    meta = None
    if isinstance(payload, dict):
        if "meta" in payload and isinstance(payload["meta"], dict):
            meta = payload["meta"]
        elif "encoder_hidden" in payload:
            meta = payload
    if meta is None:
        return
    mismatches = []
    for key in arch_keys:
        ckpt_val = meta.get(key)
        if ckpt_val is None:
            continue
        cfg_val = ppo_cfg.get(key)
        if cfg_val is None:
            continue
        if type(ckpt_val) != type(cfg_val):
            try:
                ckpt_val = type(cfg_val)(ckpt_val)
            except (TypeError, ValueError):
                pass
        if ckpt_val != cfg_val:
            mismatches.append(f"  {key}: checkpoint={ckpt_val!r}  ppo_cfg={cfg_val!r}")
    if mismatches:
        raise RuntimeError(
            "Architecture mismatch between warm-start checkpoint and PPO config.\n"
            f"Checkpoint: {init_ckpt}\n"
            + "\n".join(mismatches)
            + "\n\nEnsure the run's frozen model config matches the warm-start checkpoint metadata."
        )


def _warm_start_model_graph(init_ckpt: str) -> dict[str, Any] | None:
    """The warm-start checkpoint's declarative model graph, if it carries one.

    Thin alias over ``checkpoint_model_graph`` (sidecar read, tolerant of
    missing/corrupt) — kept because ``select_seed.py`` imports it from here.
    """
    return checkpoint_model_graph(init_ckpt)


def _resolve_model_graph(ppo_cfg: dict[str, Any], init_checkpoint: str) -> dict[str, Any] | None:
    """Warm-start graph for this job; validates model.json agrees with it.

    Graph-described seeds carry their architecture; the run's frozen
    model.json must match (fail loud, never silently diverge). Multi-seed
    Legacy multi-seed configs read the first seed; all seeds must already
    share one architecture.
    """
    graph = _warm_start_model_graph(init_checkpoint)
    if graph is None:
        ckpts = ppo_cfg.get("init_ckpts") or []
        if ckpts:
            graph = _warm_start_model_graph(str(ckpts[0]))
    if graph is None:
        return None

    from qnn.model.graph import GraphSpec, model_config_from_graph

    bridge = model_config_from_graph(GraphSpec.from_dict(graph))
    expectations: dict[str, Any] = {
        "d_model": bridge.d_model, "n_heads": bridge.n_heads,
        "n_layers": bridge.n_layers, "d_ffn": bridge.d_ffn,
        "attn_dropout": bridge.attn_dropout,
        "d_gru": bridge.d_gru, "use_gru": bridge.use_gru,
    }
    mismatches = []
    for key, graph_val in expectations.items():
        cfg_val = ppo_cfg.get(key)
        if cfg_val is None:
            continue
        try:
            cfg_val = type(graph_val)(cfg_val)
        except (TypeError, ValueError):
            pass
        if cfg_val != graph_val:
            mismatches.append(f"  {key}: model.json={ppo_cfg.get(key)!r}  graph={graph_val!r}")
    if mismatches:
        raise RuntimeError(
            "PPO model.json disagrees with the warm-start checkpoint's model graph.\n"
            + "\n".join(mismatches)
            + "\n\nUpdate the run's frozen model.json to match the seed checkpoint."
        )
    return graph


def _detect_obs_dim_from_checkpoint(ppo_cfg: dict[str, Any], checkpoint_path: str | Path | None = None) -> int:
    init_ckpt = str(checkpoint_path or require_cfg_string(ppo_cfg, "init_ckpt", "PPO config"))
    # Sidecar first — QNNPolicy.save writes the full meta there, so the
    # multi-MB payload need not be deserialized just to read one int.
    sidecar = Path(init_ckpt).with_suffix(".json")
    if sidecar.is_file():
        try:
            meta = read_json(sidecar)
            if isinstance(meta, dict) and "obs_dim" in meta:
                return int(meta["obs_dim"])
        except (OSError, ValueError):
            pass
    try:
        payload = trusted_torch_load(str(init_ckpt), map_location="cpu")
        if isinstance(payload, dict) and "meta" in payload:
            meta = payload["meta"]
            if isinstance(meta, dict) and "obs_dim" in meta:
                return int(meta["obs_dim"])
    except Exception as exc:
        raise RuntimeError(f"Unable to read obs_dim from warm-start checkpoint metadata: {init_ckpt}") from exc
    raise RuntimeError(f"Warm-start checkpoint metadata is missing obs_dim: {init_ckpt}")


def run_training_job(
    ppo_cfg: dict[str, Any],
    resolved_asset_root: Path,
    worker_path: Path,
    device: str,
) -> dict[str, Any]:
    """Execute a single native PPO training job.

    The trainer consumes the flat ppo_cfg directly and saves native
    QNNPolicy checkpoints (best/best_model.pth) — there is no post-hoc
    format conversion step anymore.
    """
    from qnn.ppo.train import run_native_ppo

    init_checkpoint = str(ppo_cfg.get("init_ckpt", ""))
    # Fail loud when the run's frozen model.json disagrees with the seed
    # checkpoint's declarative graph — never silently diverge.
    _resolve_model_graph(ppo_cfg, init_checkpoint)

    ppo_result = run_native_ppo(ppo_cfg, resolved_asset_root, worker_path, device)

    best = Path(str(require_cfg_string(ppo_cfg, "output_dir", "PPO config"))) / "best" / "best_model.pth"
    if best.exists():
        ppo_result["best_model_path"] = str(best)
    return ppo_result


def run_pipeline(ctx: Any, *, post_train_eval: bool = True, write_report: bool = True) -> dict[str, Any]:
    """Execute PPO pipeline: train + optional post-train eval."""
    results = base_results(ctx)
    stage_timings: dict[str, float] = {}

    will_resume_ppo = prepare_ppo_run_outputs(ctx.run_cfg, resume=ctx.resume)
    ppo_cfg, eval_cfg = build_run_ppo_eval_config(ctx.run_cfg, ctx.device)
    seed_ckpt = str(ctx.run_cfg.get("checkpoint_path", "") or "")
    ppo_cfg["resume"] = will_resume_ppo
    if seed_ckpt and not will_resume_ppo and not Path(seed_ckpt).exists():
        raise FileNotFoundError(f"Seed checkpoint from run.json does not exist: {seed_ckpt}")
    if will_resume_ppo:
        latest_ckpt = latest_ppo_checkpoint(ctx.run_cfg)
        if latest_ckpt is None:
            raise RuntimeError("PPO resume requested, but no latest checkpoint could be located")
        results["ppo_resume_from"] = str(latest_ckpt)
    elif seed_ckpt:
        ppo_cfg["init_ckpt"] = seed_ckpt
        results["ppo_init_ckpt"] = seed_ckpt
        if ctx.resume:
            results["ppo_resume_fallback"] = "No existing PPO checkpoint found; started from seed checkpoint."
        _validate_warm_start_arch(seed_ckpt, ppo_cfg)
    else:
        results["ppo_init_ckpt"] = ""
        results["ppo_random_init"] = True

    validate_native_mod_assets(
        ctx.asset_root,
        ppo_cfg.get("native_args") if isinstance(ppo_cfg.get("native_args"), list) else None,
    )
    worker_path = ensure_worker(ctx.worker_binary, rebuild=False)
    if (
        str(ppo_cfg.get("env_backend", "process")) == "arena_grid"
        or str(eval_cfg.get("env_backend", "process")) == "arena_grid"
    ):
        arena_server, arena_client = ensure_arena_workers(
            Path(str(ppo_cfg["arena_server_binary"])),
            Path(str(ppo_cfg["arena_client_binary"])),
            rebuild=False,
        )
        arena_bsp = ctx.asset_root / "id1" / "maps" / f"{ppo_cfg['arena_map_id']}.bsp"
        if not arena_bsp.is_file():
            raise FileNotFoundError(
                f"Grouped PPO arena map is missing: {arena_bsp}. "
                "Run scripts/build_arena_grid.py first."
            )
        ppo_cfg["arena_server_binary"] = str(arena_server)
        ppo_cfg["arena_client_binary"] = str(arena_client)
        eval_cfg["arena_server_binary"] = str(arena_server)
        eval_cfg["arena_client_binary"] = str(arena_client)

    started = time.monotonic()
    ppo_cfg["native_env"] = {"QUAKE_BASEDIR": str(ctx.asset_root)}
    ppo_cfg["native_executable"] = str(worker_path)
    results["ppo"] = run_training_job(ppo_cfg, ctx.asset_root, worker_path, ctx.device)
    stage_timings["ppo"] = time.monotonic() - started

    if post_train_eval:
        started = time.monotonic()
        prepare_eval_outputs(ctx.run_cfg, resume=False)
        eval_cfg["native_env"] = {"QUAKE_BASEDIR": str(ctx.asset_root)}
        eval_cfg["native_executable"] = str(worker_path)
        eval_ckpt = best_checkpoint(ctx.output_dirs["checkpoints"])
        if eval_ckpt is None:
            print("[training] No checkpoints found for post-train eval, skipping.")
        else:
            eval_cfg["checkpoint_path"] = prepare_eval_checkpoint(
                str(eval_ckpt), str(eval_cfg["output_dir"]),
            )
            if not eval_cfg.get("decode_regime"):
                raise RuntimeError(
                    "PPO post-train eval requires eval_decode_regime in the "
                    "run's frozen train.json; model architecture is not a "
                    "decode-regime default"
                )
            results["eval"] = run_evaluation(EvalConfig(**eval_cfg))
        stage_timings["eval"] = time.monotonic() - started

    results["stage_timings"] = stage_timings
    if not write_report:
        return results
    return finalize_results(ctx, results, stage_timings)


def run(ctx: Any) -> dict[str, Any]:
    """Runner entry point called by run.router."""
    return run_pipeline(ctx, post_train_eval=True, write_report=True)
