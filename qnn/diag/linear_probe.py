"""Linear-probe per head: how separable are the trunk-side features?

Trains a single linear classifier (logistic regression) on frozen head-input
features and reports macro F1. Compare to the full trained head's F1 from
``bc_history.json`` to answer:

  - If probe F1 ≈ trained-head F1 → the trunk is already producing
    head-separable features; the head's nonlinearity adds little.
  - If trained-head F1 ≫ probe F1 → the head's bottleneck+ReLU is doing
    real (non-linear) work the trunk doesn't.

Memory note: full-corpus probe was previously too slow (~2hr). Default here
is 1 train shard / 1 val shard for speed; pass ``--max-train-shards N``
for more if needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


def _first_linear(head: nn.Module) -> nn.Linear:
    for m in head.modules():
        if isinstance(m, nn.Linear):
            return m
    raise RuntimeError("no Linear in head")


def extract_head_input_features(
    policy,
    val_episodes: list[dict],
    *,
    max_frames: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run frozen forward passes; capture each head's input via a hook.

    Returns (features dict, labels dict) keyed by head name.
    Labels are unified per-head: discretised move axes, fire {0,1}, weapon class.
    """
    captures: dict[str, list[np.ndarray]] = {"move": [], "look": [], "fire": [], "weapon": []}
    move_lab: list[np.ndarray] = []
    fire_lab: list[np.ndarray] = []
    weapon_lab: list[np.ndarray] = []

    def hook_factory(name: str):
        def hook(_m, inp, _out):
            captures[name].append(inp[0].detach().cpu().numpy())
        return hook

    handles = []
    for name in ("move", "look", "fire", "weapon"):
        head = getattr(policy.model, f"{name}_head", None)
        if head is None:
            continue
        handles.append(_first_linear(head).register_forward_hook(hook_factory(name)))

    frames_seen = 0
    with torch.inference_mode():
        for ep in val_episodes:
            n = ep["n_samples"]
            if max_frames is not None and frames_seen >= max_frames:
                break
            obs_t = {
                k: torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(1).to(policy.device)
                for k, v in ep["obs"].items()
            }
            policy._forward_tensors(obs_t)
            move_lab.append(np.asarray(ep["actions"]["move"]))
            fire_lab.append(np.asarray(ep["actions"]["fire"]))
            if "weapon_slot" in ep["actions"]:
                weapon_lab.append(np.asarray(ep["actions"]["weapon_slot"]))
            elif "weapon" in ep["actions"]:
                weapon_lab.append(np.asarray(ep["actions"]["weapon"]))
            frames_seen += n

    for h in handles:
        h.remove()

    feats = {k: np.concatenate(v, axis=0) if v else np.empty((0, 0)) for k, v in captures.items()}
    # Flatten leading (T, B=1) → (T,) per the sequence-first model forward.
    for k, v in list(feats.items()):
        if v.ndim == 3 and v.shape[1] == 1:
            feats[k] = v.reshape(v.shape[0], v.shape[2])

    labels: dict[str, np.ndarray] = {
        "move": np.concatenate(move_lab, axis=0) if move_lab else np.empty((0, 3), dtype=np.int64),
        "fire": np.concatenate(fire_lab, axis=0) if fire_lab else np.empty((0,), dtype=np.int64),
    }
    if weapon_lab:
        labels["weapon"] = np.concatenate(weapon_lab, axis=0)
    return feats, labels


def fit_linear_probe(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    val_feats: np.ndarray,
    val_labels: np.ndarray,
    *,
    multi_class: bool = True,
) -> float:
    """Fit a logistic regression and return macro F1 on val."""
    if not HAVE_SKLEARN:
        raise RuntimeError("sklearn not available — pip install scikit-learn")
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial" if multi_class else "ovr",
        n_jobs=-1,
    )
    clf.fit(train_feats, train_labels)
    preds = clf.predict(val_feats)
    return float(f1_score(val_labels, preds, average="macro", zero_division=0))


def linear_probe_report(
    policy,
    train_episodes: list[dict],
    val_episodes: list[dict],
    *,
    max_train_frames: int = 100_000,
    max_val_frames: int = 50_000,
) -> dict[str, float]:
    """End-to-end linear probe across all heads. Returns dict of head → macro F1."""
    if not HAVE_SKLEARN:
        return {"_error": "sklearn not available"}

    train_feats, train_labels = extract_head_input_features(
        policy, train_episodes, max_frames=max_train_frames,
    )
    val_feats, val_labels = extract_head_input_features(
        policy, val_episodes, max_frames=max_val_frames,
    )

    out: dict[str, float] = {}

    # Move: probe each axis (fb, lr, ud) separately, average their F1s
    if train_labels["move"].size and val_labels["move"].size:
        axis_f1s = []
        for axis_idx, axis_name in enumerate(("fb", "lr", "ud")):
            try:
                f1 = fit_linear_probe(
                    train_feats["move"], train_labels["move"][:, axis_idx],
                    val_feats["move"], val_labels["move"][:, axis_idx],
                    multi_class=True,
                )
                out[f"move_{axis_name}"] = f1
                axis_f1s.append(f1)
            except Exception as e:  # noqa: BLE001
                out[f"move_{axis_name}_error"] = str(e)
        if axis_f1s:
            out["move_macro"] = float(np.mean(axis_f1s))

    if train_labels["fire"].size and val_labels["fire"].size:
        try:
            out["fire"] = fit_linear_probe(
                train_feats["fire"], train_labels["fire"].astype(int),
                val_feats["fire"], val_labels["fire"].astype(int),
                multi_class=False,
            )
        except Exception as e:  # noqa: BLE001
            out["fire_error"] = str(e)

    if "weapon" in train_labels and "weapon" in val_labels:
        try:
            out["weapon"] = fit_linear_probe(
                train_feats["weapon"], train_labels["weapon"].astype(int),
                val_feats["weapon"], val_labels["weapon"].astype(int),
                multi_class=True,
            )
        except Exception as e:  # noqa: BLE001
            out["weapon_error"] = str(e)

    return out
