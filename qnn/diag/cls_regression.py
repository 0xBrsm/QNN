"""Ridge regression probes: CLS readout → continuous self-state scalars.

Fits Ridge(alpha=1) from cls_readout → each scalar and reports R² on
held-out val data.  Answers: what game-state variables are linearly
decodable from the CLS token?

Scalars probed (7 total):
  state:   health, effective_armor   (self_state_scalars[:,0:2])
  arsenal: attack_finished           (self_arsenal_scalars[:,0])
  motion:  vel_x, vel_y, vel_z, view_pitch (self_motion_scalars[:,0:4])
"""
from __future__ import annotations

import numpy as np
import torch

_SCALAR_NAMES = ["health", "armor", "attack_fin", "vel_x", "vel_y", "vel_z", "view_pitch"]


def _ridge_r2(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    alpha: float = 1.0,
) -> float:
    """Fit Ridge(alpha) via the normal equations; return R² on val."""
    n, d = X_train.shape
    A = X_train.T @ X_train + alpha * np.eye(d)
    b = X_train.T @ y_train
    w = np.linalg.solve(A, b)
    y_pred = X_val @ w
    ss_res = np.sum((y_val - y_pred) ** 2)
    ss_tot = np.sum((y_val - y_val.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _inject_view_pitch(obs_ep: dict) -> dict:
    """Backfill view_pitch from spatial_dir for legacy shards (no-op if present)."""
    if "view_pitch" in obs_ep or "spatial_dir" not in obs_ep:
        return obs_ep
    sin_p = np.asarray(obs_ep["spatial_dir"][:, 7, 0], dtype=np.float32) / 127.0
    np.clip(sin_p, -1.0, 1.0, out=sin_p)
    pitch_norm = np.arcsin(sin_p) * (2.0 / np.pi)
    out = dict(obs_ep)
    out["view_pitch"] = np.round(pitch_norm * 127.0).clip(-127, 127).astype(np.int8)
    return out


def _pad_entities(obs_ep: dict) -> dict:
    """Pad flat packed entity arrays to (T, MAX_TOKEN_OBJECTS) (no-op if already 2D)."""
    et = obs_ep.get("entity_types")
    if et is None or np.asarray(et).ndim == 2:
        return obs_ep
    from qnn.bc.train import _materialize_padded_entity
    from qnn.vocab import MAX_TOKEN_OBJECTS
    return _materialize_padded_entity(obs_ep, MAX_TOKEN_OBJECTS)


def _prep_batch(obs_ep: dict, start: int, end: int, device: torch.device) -> dict:
    out = {}
    for k, v in obs_ep.items():
        t = torch.from_numpy(np.ascontiguousarray(v[start:end]))
        if t.dtype == torch.float16:
            t = t.float()
        out[k] = t.to(device)
    return out


def _unpack_scalars(obs_t: dict) -> tuple[np.ndarray, ...]:
    """Return (state, arsenal, motion) numpy arrays from an obs batch.

    Handles both post-dequant (self_state_scalars present) and pre-dequant
    native-field obs dicts (health / effective_armor / vel / etc.).
    Regression R² is invariant to the linear normalization difference.
    """
    if "self_state_scalars" in obs_t:
        state   = obs_t["self_state_scalars"].cpu().float().numpy()
        arsenal = obs_t["self_arsenal_scalars"].cpu().float().numpy()
        motion  = obs_t["self_motion_scalars"].cpu().float().numpy()
    else:
        # Native engine fields — use raw values (R² is scale-invariant)
        health  = obs_t["health"].cpu().float().numpy()[:, None]
        armor   = obs_t["effective_armor"].cpu().float().numpy()[:, None]
        state   = np.concatenate([health, armor], axis=1)
        arsenal = obs_t["attack_finished"].cpu().float().numpy()[:, None]
        vel     = obs_t["vel"].cpu().float().numpy()            # (B, 3)
        pitch   = obs_t["view_pitch"].cpu().float().numpy()[:, None]
        motion  = np.concatenate([vel, pitch], axis=1)
    return state, arsenal, motion


def extract_cls_and_scalars(
    policy,
    episodes: list[dict],
    *,
    max_frames: int | None = None,
    batch_size: int = 64,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run frozen forward passes; return (cls_vecs, scalars).

    cls_vecs shape: (N, d_model)
    scalars: dict of name → (N,) float32
    """
    cls_list: list[np.ndarray] = []
    scalar_lists: dict[str, list[np.ndarray]] = {n: [] for n in _SCALAR_NAMES}

    frames_seen = 0
    with torch.inference_mode():
        for ep in episodes:
            if max_frames is not None and frames_seen >= max_frames:
                break
            ep = _inject_view_pitch(ep)
            ep = _pad_entities(ep)
            n = next(iter(ep.values())).shape[0]
            for start in range(0, n, batch_size):
                if max_frames is not None and frames_seen >= max_frames:
                    break
                end = min(start + batch_size, n)
                obs_t = _prep_batch(ep, start, end, policy.device)

                enc_out = policy.model.encoder(policy.model.obs_embedding(obs_t))
                cls_out = enc_out.self_readout
                cls_list.append(cls_out.cpu().float().numpy())

                state, arsenal, motion = _unpack_scalars(obs_t)
                scalar_lists["health"].append(state[:, 0])
                scalar_lists["armor"].append(state[:, 1])
                scalar_lists["attack_fin"].append(arsenal[:, 0])
                scalar_lists["vel_x"].append(motion[:, 0])
                scalar_lists["vel_y"].append(motion[:, 1])
                scalar_lists["vel_z"].append(motion[:, 2])
                scalar_lists["view_pitch"].append(motion[:, 3])
                frames_seen += end - start

    cls_arr = np.concatenate(cls_list, axis=0)
    scalars = {k: np.concatenate(v) for k, v in scalar_lists.items()}
    return cls_arr, scalars


def extract_cls_gru_and_scalars(
    policy,
    episodes: list[dict],
    *,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Run trunk + GRU over full episode sequences; return (cls, gru_h, scalars).

    Each episode is processed as a single sequence so the GRU sees proper
    temporal context (h₀=0 at episode start, matching BC training).

    cls shape:   (N, d_model)
    gru_h shape: (N, gru_hidden)
    scalars:     dict of name → (N,) float32
    """
    if not policy.model.use_gru:
        raise RuntimeError("model was trained without a GRU (use_gru=False)")

    cls_list: list[np.ndarray] = []
    gru_list: list[np.ndarray] = []
    scalar_lists: dict[str, list[np.ndarray]] = {n: [] for n in _SCALAR_NAMES}

    frames_seen = 0
    with torch.inference_mode():
        for ep in episodes:
            if max_frames is not None and frames_seen >= max_frames:
                break
            ep = _inject_view_pitch(ep)
            ep = _pad_entities(ep)
            T = next(iter(ep.values())).shape[0]
            take = T if max_frames is None else min(T, max_frames - frames_seen)

            obs_t = _prep_batch(ep, 0, take, policy.device)

            # Encoder: all frames at once (no recurrence in transformer)
            enc_out = policy.model.encoder(policy.model.obs_embedding(obs_t))
            cls_out = enc_out.self_readout                         # (T, d_model)

            # GRU: sequence-first, batch=1, h₀=zeros
            gru_in = cls_out.unsqueeze(1)                          # (T, 1, d_model)
            gru_out, _ = policy.model.gru(gru_in)                  # (T, 1, gru_hidden)
            gru_out = gru_out.squeeze(1)                           # (T, gru_hidden)

            cls_list.append(cls_out.cpu().float().numpy())
            gru_list.append(gru_out.cpu().float().numpy())

            state, arsenal, motion = _unpack_scalars(obs_t)
            scalar_lists["health"].append(state[:, 0])
            scalar_lists["armor"].append(state[:, 1])
            scalar_lists["attack_fin"].append(arsenal[:, 0])
            scalar_lists["vel_x"].append(motion[:, 0])
            scalar_lists["vel_y"].append(motion[:, 1])
            scalar_lists["vel_z"].append(motion[:, 2])
            scalar_lists["view_pitch"].append(motion[:, 3])
            frames_seen += take

    cls_arr = np.concatenate(cls_list, axis=0)
    gru_arr = np.concatenate(gru_list, axis=0)
    scalars = {k: np.concatenate(v) for k, v in scalar_lists.items()}
    return cls_arr, gru_arr, scalars


def compare_cls_gru_report(
    policy,
    train_episodes: list[dict],
    val_episodes: list[dict],
    *,
    max_train_frames: int = 50_000,
    max_val_frames: int = 20_000,
) -> dict[str, dict[str, float]]:
    """Fit Ridge for both CLS and GRU h_t; return {scalar: {cls: R², gru: R²}}."""
    train_cls, train_gru, train_sc = extract_cls_gru_and_scalars(
        policy, train_episodes, max_frames=max_train_frames,
    )
    val_cls, val_gru, val_sc = extract_cls_gru_and_scalars(
        policy, val_episodes, max_frames=max_val_frames,
    )

    out: dict[str, dict[str, float]] = {}
    for name in _SCALAR_NAMES:
        try:
            out[name] = {
                "cls": _ridge_r2(train_cls, train_sc[name], val_cls, val_sc[name]),
                "gru": _ridge_r2(train_gru, train_sc[name], val_gru, val_sc[name]),
            }
        except Exception as e:  # noqa: BLE001
            out[name] = {"cls": float("nan"), "gru": float("nan"), "_error": str(e)}
    return out


def cls_regression_report(
    policy,
    train_episodes: list[dict],
    val_episodes: list[dict],
    *,
    max_train_frames: int = 50_000,
    max_val_frames: int = 20_000,
    batch_size: int = 64,
) -> dict[str, float]:
    """Fit Ridge(alpha=1) cls→scalar per scalar. Returns dict of name→R²."""
    train_cls, train_sc = extract_cls_and_scalars(
        policy, train_episodes, max_frames=max_train_frames, batch_size=batch_size,
    )
    val_cls, val_sc = extract_cls_and_scalars(
        policy, val_episodes, max_frames=max_val_frames, batch_size=batch_size,
    )

    out: dict[str, float] = {}
    for name in _SCALAR_NAMES:
        try:
            out[name] = _ridge_r2(train_cls, train_sc[name], val_cls, val_sc[name])
        except Exception as e:  # noqa: BLE001
            out[f"{name}_error"] = str(e)
    return out
