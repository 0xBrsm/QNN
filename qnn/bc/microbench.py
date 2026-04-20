"""Synthetic throughput benchmark for the BC model.

Strips data loading, prefetch, and metrics entirely. Feeds pre-materialised
GPU tensors through supervised_step in a tight loop and reports rows/sec.

Answers: is the 17k rows/sec floor from the model itself or the data pipeline?

Usage:
  python -m qnn.bc.microbench --run-dir runs/bc/bc_bs_test_256 \
      --bs 256 --chunk 256 --iters 50 --warmup 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qnn.actions import ACTION_HEADS
from qnn.model.policy import QNNPolicy

ACTION_HEADS_SIZE = ACTION_HEADS


_OBS_SHAPES: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "action_history": ((8, 8), np.float32),
    "entity_event_actions": ((16, 4), np.int32),
    "entity_event_counts": ((16,), np.uint8),
    "entity_event_sources": ((16, 4), np.int32),
    "entity_ids": ((16, 3), np.int32),
    "entity_scalars_raw": ((16, 19), np.float32),
    "entity_types": ((16,), np.int32),
    "self_armor_type_id": ((1,), np.int32),
    "self_movement_id": ((1,), np.int32),
    "self_powerup_ids": ((5,), np.int32),
    "self_scalars": ((14,), np.float32),
    "self_weapon_id": ((1,), np.int32),
    "spatial_scalars": ((9, 13), np.float32),
}

_ACT_SHAPES: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "fire": ((), np.int32),
    "switch": ((), np.int32),
    "recall_0": ((), np.int32),
    "recall_1": ((), np.int32),
    "recall_2": ((), np.int32),
    "recall_3": ((), np.int32),
    "look": ((3,), np.float32),
    "move": ((3,), np.float32),
}


def _make_policy(run_dir: Path) -> QNNPolicy:
    model_cfg = json.loads((run_dir / "config" / "model.json").read_text())
    # obs_dim is ignored by the token-policy forward path (uses obs dict shapes).
    # A real checkpoint load would be more faithful but unnecessary for perf.
    obs_dim = 1
    return QNNPolicy(
        obs_dim=obs_dim,
        trunk_hidden=int(model_cfg["trunk_hidden"]),
        gru_hidden=int(model_cfg["gru_hidden"]),
        use_gru=bool(model_cfg["use_gru"]),
        seed=0,
        device="auto",
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        n_layers=int(model_cfg["n_layers"]),
        ffn_dim=int(model_cfg["ffn_dim"]),
        attn_dropout=float(model_cfg.get("attn_dropout", 0.0)),
        readout=str(model_cfg.get("readout", "self")),
        action_history_tokens=int(model_cfg.get("action_history_tokens", 0)),
    )


def _build_obs(chunk: int, bs: int, device: torch.device) -> dict[str, torch.Tensor]:
    obs: dict[str, torch.Tensor] = {}
    for key, (tail_shape, dtype) in _OBS_SHAPES.items():
        shape = (chunk, bs, *tail_shape)
        if np.issubdtype(dtype, np.floating):
            t = torch.randn(*shape, dtype=torch.float32, device=device) * 0.1
        else:
            t = torch.zeros(*shape, dtype=torch.int64, device=device)
        obs[key] = t
    return obs


def _build_actions(chunk: int, bs: int, device: torch.device) -> dict[str, torch.Tensor]:
    act: dict[str, torch.Tensor] = {}
    for key, (tail_shape, dtype) in _ACT_SHAPES.items():
        shape = (chunk, bs, *tail_shape)
        if np.issubdtype(dtype, np.floating):
            t = torch.randn(*shape, dtype=torch.float32, device=device) * 0.1
        else:
            t = torch.zeros(*shape, dtype=torch.int64, device=device)
        act[key] = t
    return act


def _autocast_ctx(dtype_name: str, device_type: str):
    if dtype_name == "fp32":
        return torch.amp.autocast(device_type=device_type, enabled=False)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    return torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=True)


def _run_components(policy: QNNPolicy, bs: int, chunk: int, iters: int, warmup: int, dtype_name: str = "fp32") -> dict:
    """Time forward sub-stages (trunk, GRU, heads), loss, and backward separately.
    Uses CUDA events for sub-millisecond accuracy on GPU work."""
    import torch.nn.functional as F
    device = policy.device
    obs = _build_obs(chunk, bs, device)
    actions = _build_actions(chunk, bs, device)
    hidden = torch.zeros((bs, policy.gru_hidden), dtype=torch.float32, device=device) if policy.use_gru else None
    optimizer = policy._optimizer("bc", policy.model.parameters(), 1e-3)

    seq_len = chunk
    flat_obs = {
        key: value.reshape(seq_len * bs, *value.shape[2:])
        for key, value in obs.items()
    }

    def event_pair():
        return (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if device.type == "cuda" else (None, None)

    stages = ["trunk", "gru", "heads", "loss", "backward", "step"]
    totals = {s: 0.0 for s in stages}

    for it in range(warmup + iters):
        optimizer.zero_grad()
        evs = {s: event_pair() for s in stages}

        with _autocast_ctx(dtype_name, device.type):
            if device.type == "cuda": evs["trunk"][0].record()
            trunk_features = policy.model.trunk(flat_obs).reshape(seq_len, bs, policy.d_model)
            if device.type == "cuda": evs["trunk"][1].record()

            if device.type == "cuda": evs["gru"][0].record()
            if policy.use_gru:
                gru_features, next_hidden = policy.model._run_gru(trunk_features, hidden, masks=None)
                features = torch.cat([trunk_features, gru_features], dim=-1)
            else:
                features = trunk_features
            if device.type == "cuda": evs["gru"][1].record()

            if device.type == "cuda": evs["heads"][0].record()
            flat_features = features.reshape(seq_len * bs, -1)
            logits = {
                head: layer(flat_features).reshape(seq_len, bs, ACTION_HEADS_SIZE[head])
                for head, layer in policy.model.policy_heads.items()
            }
            if device.type == "cuda": evs["heads"][1].record()

            if device.type == "cuda": evs["loss"][0].record()
            losses, _is_real, _metrics = policy._compute_head_losses_and_metrics(
                logits, actions, head_loss_weights=None,
                focal_gamma=0.0, sparse_discrete=False, look_deadzone=0.0, look_turn_alpha=0.0,
                compute_metrics=False,
            )
            loss = torch.stack(losses).mean()
            if device.type == "cuda": evs["loss"][1].record()

        if device.type == "cuda": evs["backward"][0].record()
        loss.backward()
        if device.type == "cuda": evs["backward"][1].record()

        if device.type == "cuda": evs["step"][0].record()
        optimizer.step()
        if device.type == "cuda": evs["step"][1].record()

        if device.type == "cuda":
            torch.cuda.synchronize()
            if it >= warmup:
                for s in stages:
                    totals[s] += evs[s][0].elapsed_time(evs[s][1])

    per_step = {s: totals[s] / iters for s in stages}
    return {
        "bs": bs,
        "chunk": chunk,
        "iters": iters,
        "dtype": dtype_name,
        "ms_per_step": {k: round(v, 2) for k, v in per_step.items()},
        "total_ms": round(sum(per_step.values()), 2),
        "rows_per_sec": round(bs * chunk * 1000.0 / sum(per_step.values()), 1),
    }


def _run_compiled(policy: QNNPolicy, bs: int, chunk: int, iters: int, warmup: int, dtype_name: str) -> dict:
    """Wrap model with torch.compile and time end-to-end."""
    device = policy.device
    obs = _build_obs(chunk, bs, device)
    actions = _build_actions(chunk, bs, device)
    hidden = torch.zeros((bs, policy.gru_hidden), dtype=torch.float32, device=device) if policy.use_gru else None

    # Compile the inner net. Default mode (vs reduce-overhead) usually wins
    # on small models since reduce-overhead adds CUDAGraph capture cost.
    policy.model = torch.compile(policy.model, dynamic=False)
    optimizer = policy._optimizer("bc", policy.model.parameters(), 1e-3)

    def step():
        optimizer.zero_grad()
        with _autocast_ctx(dtype_name, device.type):
            metrics = policy.supervised_step(
                obs, actions, class_weights={}, lr=1e-3,
                hidden=hidden, accumulate_only=False,
                head_loss_weights=None, focal_gamma=0.0, sparse_discrete=False,
                look_deadzone=0.0, look_turn_alpha=0.0, loss_scale=1.0,
                compute_metrics=False,
            )

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.monotonic()
    for _ in range(iters):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - t0
    rows = iters * bs * chunk
    return {
        "bs": bs, "chunk": chunk, "iters": iters, "mode": "compile", "dtype": dtype_name,
        "elapsed_s": round(elapsed, 3),
        "rows_per_sec": round(rows / elapsed, 1),
        "sec_per_step": round(elapsed / iters, 4),
    }


def _run_cudagraph(policy: QNNPolicy, bs: int, chunk: int, iters: int, warmup: int, dtype_name: str) -> dict:
    """Capture full step into a CUDA/HIP graph and replay each iteration."""
    device = policy.device
    if device.type != "cuda":
        return {"bs": bs, "mode": "cudagraph", "error": "requires cuda device"}
    obs = _build_obs(chunk, bs, device)
    actions = _build_actions(chunk, bs, device)
    hidden = torch.zeros((bs, policy.gru_hidden), dtype=torch.float32, device=device) if policy.use_gru else None
    optimizer = policy._optimizer("bc", policy.model.parameters(), 1e-3)
    # Adam requires capturable=True to allow optimizer.step() inside a CUDA graph.
    for pg in optimizer.param_groups:
        pg["capturable"] = True

    def step():
        # set_to_none=False so grad tensors stay allocated; graph requires
        # persistent buffers. accumulate_only=True skips supervised_step's own
        # zero_grad/step so we control them with capture-friendly settings.
        optimizer.zero_grad(set_to_none=False)
        with _autocast_ctx(dtype_name, device.type):
            policy.supervised_step(
                obs, actions, class_weights={}, lr=1e-3,
                hidden=hidden, accumulate_only=True,
                head_loss_weights=None, focal_gamma=0.0, sparse_discrete=False,
                look_deadzone=0.0, look_turn_alpha=0.0, loss_scale=1.0,
                compute_metrics=False,
            )
        optimizer.step()

    # Warmup pre-capture.
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    torch.cuda.synchronize()

    t0 = time.monotonic()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - t0
    rows = iters * bs * chunk
    return {
        "bs": bs, "chunk": chunk, "iters": iters, "mode": "cudagraph", "dtype": dtype_name,
        "elapsed_s": round(elapsed, 3),
        "rows_per_sec": round(rows / elapsed, 1),
        "sec_per_step": round(elapsed / iters, 4),
    }


def _run_profile(policy: QNNPolicy, bs: int, chunk: int, iters: int, warmup: int, dtype_name: str, trace_path: Path) -> dict:
    """Capture a Chrome trace using torch.profiler — for inspecting per-kernel
    times and idle gaps."""
    from torch.profiler import profile, ProfilerActivity, record_function
    device = policy.device
    obs = _build_obs(chunk, bs, device)
    actions = _build_actions(chunk, bs, device)
    hidden = torch.zeros((bs, policy.gru_hidden), dtype=torch.float32, device=device) if policy.use_gru else None
    optimizer = policy._optimizer("bc", policy.model.parameters(), 1e-3)

    def step():
        optimizer.zero_grad()
        with _autocast_ctx(dtype_name, device.type):
            policy.supervised_step(
                obs, actions, class_weights={}, lr=1e-3,
                hidden=hidden, accumulate_only=False,
                head_loss_weights=None, focal_gamma=0.0, sparse_discrete=False,
                look_deadzone=0.0, look_turn_alpha=0.0, loss_scale=1.0,
                compute_metrics=False,
            )

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
        for _ in range(iters):
            with record_function("step"):
                step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    prof.export_chrome_trace(str(trace_path))
    table = prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=20)
    print(table, flush=True)
    return {"bs": bs, "chunk": chunk, "iters": iters, "mode": "profile", "dtype": dtype_name, "trace": str(trace_path)}


def _run_graph_debug(policy: QNNPolicy, bs: int, chunk: int, dtype_name: str) -> dict:
    """Try HIP graph capture progressively to isolate the breaking op.
    Captures forward, then fwd+loss, then fwd+loss+backward, then +optimizer.
    Returns the first stage that fails."""
    device = policy.device
    if device.type != "cuda":
        return {"error": "requires cuda"}
    obs = _build_obs(chunk, bs, device)
    actions = _build_actions(chunk, bs, device)
    hidden = torch.zeros((bs, policy.gru_hidden), dtype=torch.float32, device=device) if policy.use_gru else None
    optimizer = policy._optimizer("bc", policy.model.parameters(), 1e-3)

    # Warm up & init optimizer state with one real step.
    with _autocast_ctx(dtype_name, device.type):
        m = policy.supervised_step(
            obs, actions, class_weights={}, lr=1e-3,
            hidden=hidden, accumulate_only=False,
            head_loss_weights=None, focal_gamma=0.0, sparse_discrete=False,
            look_deadzone=0.0, look_turn_alpha=0.0, loss_scale=1.0,
            compute_metrics=False,
        )
    torch.cuda.synchronize()

    # Pre-flatten obs (matches what trunk/forward sees inside _forward_tensors).
    sample = obs["self_scalars"]
    seq_len, batch = int(sample.shape[0]), int(sample.shape[1])
    flat_obs = {k: v.reshape(seq_len * batch, *v.shape[2:]) for k, v in obs.items()}

    results: dict[str, str] = {}
    stages = [
        ("trunk_only", lambda: policy.model.trunk(flat_obs)),
        ("trunk+gru", lambda: _gd_trunk_gru(policy, flat_obs, hidden, seq_len, batch)),
        ("trunk+gru+heads", lambda: _gd_trunk_gru_heads(policy, flat_obs, hidden, seq_len, batch)),
        ("forward (full model)", lambda: _gd_forward(policy, obs, hidden, dtype_name)),
        ("forward+loss", lambda: _gd_forward_loss(policy, obs, actions, hidden, dtype_name)),
        ("forward+loss+backward", lambda: _gd_full_no_step(policy, obs, actions, hidden, optimizer, dtype_name)),
        ("forward+loss+backward+step", lambda: _gd_full(policy, obs, actions, hidden, optimizer, dtype_name)),
    ]
    for name, fn in stages:
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            torch.cuda.synchronize()
            results[name] = "OK"
        except Exception as exc:
            results[name] = f"FAIL: {repr(exc)[:200]}"
            print(f"[{name}] {results[name]}", flush=True)
            break
        else:
            print(f"[{name}] OK", flush=True)
    return {"bs": bs, "chunk": chunk, "dtype": dtype_name, "graph_debug": results}


def _gd_trunk_gru(policy, flat_obs, hidden, seq_len, batch):
    trunk_features = policy.model.trunk(flat_obs).reshape(seq_len, batch, policy.d_model)
    if policy.use_gru:
        policy.model._run_gru(trunk_features, hidden, masks=None)


def _gd_trunk_gru_heads(policy, flat_obs, hidden, seq_len, batch):
    trunk_features = policy.model.trunk(flat_obs).reshape(seq_len, batch, policy.d_model)
    if policy.use_gru:
        gru_features, _ = policy.model._run_gru(trunk_features, hidden, masks=None)
        features = torch.cat([trunk_features, gru_features], dim=-1)
    else:
        features = trunk_features
    flat_features = features.reshape(seq_len * batch, -1)
    {head: layer(flat_features) for head, layer in policy.model.policy_heads.items()}


def _gd_forward(policy, obs, hidden, dtype_name):
    with _autocast_ctx(dtype_name, "cuda"):
        policy.model(obs, hidden=hidden)


def _gd_forward_loss(policy, obs, actions, hidden, dtype_name):
    with _autocast_ctx(dtype_name, "cuda"):
        _, logits, _, _ = policy.model(obs, hidden=hidden)
        losses, _, _ = policy._compute_head_losses_and_metrics(
            logits, actions, head_loss_weights=None,
            focal_gamma=0.0, sparse_discrete=False,
            look_deadzone=0.0, look_turn_alpha=0.0, compute_metrics=False,
        )
        torch.stack(losses).mean()


def _gd_full_no_step(policy, obs, actions, hidden, optimizer, dtype_name):
    optimizer.zero_grad(set_to_none=False)
    with _autocast_ctx(dtype_name, "cuda"):
        _, logits, _, _ = policy.model(obs, hidden=hidden)
        losses, _, _ = policy._compute_head_losses_and_metrics(
            logits, actions, head_loss_weights=None,
            focal_gamma=0.0, sparse_discrete=False,
            look_deadzone=0.0, look_turn_alpha=0.0, compute_metrics=False,
        )
        loss = torch.stack(losses).mean()
    loss.backward()


def _gd_full(policy, obs, actions, hidden, optimizer, dtype_name):
    optimizer.zero_grad(set_to_none=False)
    with _autocast_ctx(dtype_name, "cuda"):
        _, logits, _, _ = policy.model(obs, hidden=hidden)
        losses, _, _ = policy._compute_head_losses_and_metrics(
            logits, actions, head_loss_weights=None,
            focal_gamma=0.0, sparse_discrete=False,
            look_deadzone=0.0, look_turn_alpha=0.0, compute_metrics=False,
        )
        loss = torch.stack(losses).mean()
    loss.backward()
    optimizer.step()


def _run_one(policy: QNNPolicy, bs: int, chunk: int, iters: int, warmup: int) -> dict:
    device = policy.device
    obs = _build_obs(chunk, bs, device)
    actions = _build_actions(chunk, bs, device)
    hidden = torch.zeros((bs, policy.gru_hidden), dtype=torch.float32, device=device) if policy.use_gru else None

    for _ in range(warmup):
        policy.supervised_step(
            obs, actions, class_weights={}, lr=1e-3,
            hidden=hidden, accumulate_only=False,
            head_loss_weights=None, focal_gamma=0.0, sparse_discrete=False,
            look_deadzone=0.0, look_turn_alpha=0.0, loss_scale=1.0,
            compute_metrics=False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.monotonic()
    for _ in range(iters):
        policy.supervised_step(
            obs, actions, class_weights={}, lr=1e-3,
            hidden=hidden, accumulate_only=False,
            head_loss_weights=None, focal_gamma=0.0, sparse_discrete=False,
            look_deadzone=0.0, look_turn_alpha=0.0, loss_scale=1.0,
            compute_metrics=False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - t0

    rows = iters * bs * chunk
    return {
        "bs": bs, "chunk": chunk, "iters": iters,
        "elapsed_s": round(elapsed, 3),
        "rows": rows,
        "rows_per_sec": round(rows / elapsed, 1),
        "sec_per_step": round(elapsed / iters, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--bs", type=int, default=None, help="single batch size (omit if --sweep)")
    ap.add_argument("--sweep", type=str, default=None, help="comma-separated batch sizes, e.g. 64,128,256,384,512")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None, help="optional JSONL output for results")
    ap.add_argument("--components", action="store_true", help="time forward sub-stages, loss, backward separately")
    ap.add_argument("--mode", default="full", choices=["full", "components", "compile", "cudagraph", "profile", "graph_debug"], help="bench mode")
    ap.add_argument("--trace", type=Path, default=None, help="profile mode chrome trace output path")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"], help="autocast dtype")
    args = ap.parse_args()

    if args.sweep:
        bs_list = [int(x) for x in args.sweep.split(",")]
    elif args.bs is not None:
        bs_list = [args.bs]
    else:
        bs_list = [256]

    policy = _make_policy(args.run_dir)
    print(f"device={policy.device} chunk={args.chunk} iters={args.iters} warmup={args.warmup} bs_list={bs_list}", flush=True)

    out_fh = open(args.out, "a", buffering=1) if args.out else None  # line-buffered
    try:
        for bs in bs_list:
            print(f"--- bs={bs} starting ---", flush=True)
            try:
                mode = "components" if args.components else args.mode
                if mode == "components":
                    result = _run_components(policy, bs, args.chunk, args.iters, args.warmup, dtype_name=args.dtype)
                elif mode == "compile":
                    result = _run_compiled(policy, bs, args.chunk, args.iters, args.warmup, dtype_name=args.dtype)
                elif mode == "cudagraph":
                    result = _run_cudagraph(policy, bs, args.chunk, args.iters, args.warmup, dtype_name=args.dtype)
                elif mode == "profile":
                    trace_path = args.trace or Path(f"/tmp/microbench_trace_bs{bs}_{args.dtype}.json")
                    result = _run_profile(policy, bs, args.chunk, args.iters, args.warmup, dtype_name=args.dtype, trace_path=trace_path)
                elif mode == "graph_debug":
                    result = _run_graph_debug(policy, bs, args.chunk, dtype_name=args.dtype)
                else:
                    result = _run_one(policy, bs, args.chunk, args.iters, args.warmup)
            except Exception as exc:
                line = json.dumps({"bs": bs, "error": repr(exc)})
                print(line, flush=True)
                if out_fh:
                    out_fh.write(line + "\n")
                continue
            line = json.dumps(result)
            print(line, flush=True)
            if out_fh:
                out_fh.write(line + "\n")
    finally:
        if out_fh:
            out_fh.close()


if __name__ == "__main__":
    main()
