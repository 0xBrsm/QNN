"""Linear-probe per head: how separable are the encoder-side features?

Trains a single linear classifier (logistic regression) on frozen head-input
features and reports macro F1. Compare to the full trained head's F1 from
``bc_history.json`` to answer:

  - If probe F1 ≈ trained-head F1 → the encoder is already producing
    head-separable features; the head's nonlinearity adds little.
  - If trained-head F1 ≫ probe F1 → the head's bottleneck+ReLU is doing
    real (non-linear) work the encoder doesn't.

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
    Labels are unified per-head: discretised move axes and attack class 0..8.
    """
    captures: dict[str, list[np.ndarray]] = {"move": [], "look": [], "attack": []}
    move_lab: list[np.ndarray] = []
    attack_lab: list[np.ndarray] = []

    def hook_factory(name: str):
        def hook(_m, inp, _out):
            captures[name].append(inp[0].detach().cpu().numpy())
        return hook

    handles = []
    for name in ("move", "look", "attack"):
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
            attack_lab.append(np.asarray(ep["actions"]["attack"]))
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
        "attack": np.concatenate(attack_lab, axis=0) if attack_lab else np.empty((0,), dtype=np.int64),
    }
    return feats, labels


def fit_linear_probe(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    val_feats: np.ndarray,
    val_labels: np.ndarray,
    *,
    multi_class: bool = True,
    feature_slice: tuple[int, int] | slice | None = None,
) -> float:
    """Fit a logistic regression and return macro F1 on val.

    ``feature_slice`` restricts the probe to a contiguous range of input
    dims. Pass ``(0, d_model)`` to probe only the self/weapon half of the
    canonical ``cat(self_readout, target_feat)`` head input, or
    ``(d_model, 2*d_model)`` for the target-feat half. ``None`` (default)
    probes the full feature.
    """
    if not HAVE_SKLEARN:
        raise RuntimeError("sklearn not available — pip install scikit-learn")
    if feature_slice is not None:
        if isinstance(feature_slice, slice):
            sl = feature_slice
        else:
            start, end = feature_slice
            sl = slice(start, end)
        train_feats = train_feats[:, sl]
        val_feats = val_feats[:, sl]
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


def attack_slice_probe_report(
    policy,
    train_episodes: list[dict],
    val_episodes: list[dict],
    *,
    self_slice: tuple[int, int],
    target_slice: tuple[int, int],
    max_train_frames: int = 100_000,
    max_val_frames: int = 50_000,
) -> dict[str, float]:
    """Three attack-head linear probes: full, self/weapon half only, target half only.

    Pair with the bench look/attack parity ablations to answer: how much
    of the fire signal lives in the self/weapon-token half of the head
    input vs the target_feat half? F1 difference between the full probe
    and the half probes localises where the linear-decodable signal is.

    ``self_slice`` and ``target_slice`` are ``(start, end)`` ranges on
    the head's first-Linear input. For the default look/attack parity
    setup (``d_model=64``, no temporal), both halves are 64-wide:
    self=``(0, 64)``, target=``(64, 128)``.
    """
    if not HAVE_SKLEARN:
        return {"_error": "sklearn not available"}

    train_feats, train_labels = extract_head_input_features(
        policy, train_episodes, max_frames=max_train_frames,
    )
    val_feats, val_labels = extract_head_input_features(
        policy, val_episodes, max_frames=max_val_frames,
    )
    if not train_labels["attack"].size or not val_labels["attack"].size:
        return {"_error": "no attack labels"}

    out: dict[str, float] = {}
    for name, sl in (
        ("attack_full", None),
        ("attack_self_half", self_slice),
        ("attack_target_half", target_slice),
    ):
        try:
            out[name] = fit_linear_probe(
                train_feats["attack"], train_labels["attack"].astype(int),
                val_feats["attack"], val_labels["attack"].astype(int),
                multi_class=False,
                feature_slice=sl,
            )
        except Exception as e:  # noqa: BLE001
            out[f"{name}_error"] = str(e)
    return out


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

    if train_labels["attack"].size and val_labels["attack"].size:
        try:
            out["attack"] = fit_linear_probe(
                train_feats["attack"], train_labels["attack"].astype(int),
                val_feats["attack"], val_labels["attack"].astype(int),
                multi_class=True,
            )
        except Exception as e:  # noqa: BLE001
            out["attack_error"] = str(e)

    return out
