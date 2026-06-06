"""Prior + MLP-residual decomposition for look and attack heads.

Both bench head variants emit a structured forward output:

  LookHead / WeaponAimLookHead:
      pred_look  = normalize(base_look + delta_look)
      base_look  = geometric prior (target-anchored or aim_vec)
      delta_look = MLP residual

  LookStyleAttackHead:
      attack_logit = prior_logit + delta_attack
      prior_logit  = alignment_scale * base_look[..., 0]
      delta_attack = MLP residual (zero-initialised)

The trainer logs only ``cos_sim_look`` and ``f1_attack`` — single-number
mean-cosine / mean-F1. Those numbers average across frames, so they hide
two things:

  * how much of the score comes from the geometric prior alone, vs the
    MLP residual ("is the head doing real work?")
  * regime-specific behaviour (per held weapon, per masked-fire state),
    which is where the residual usually matters most.

This module decomposes both heads per held weapon and per fire state,
and reports prior-only vs full-head scores side-by-side so the MLP's
contribution is visible.

Cost: one forward pass per episode in the supplied val set. With
``max_frames=100_000`` (default) it takes a few seconds on the trainer
GPU.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from qnn.vocab import self_weapon_id_to_impulse


_WEAPON_NAMES = ("NONE", "AXE", "SG", "SSG", "NG", "SNG", "GL", "RL", "LG")


def _collect_head_outputs(
    policy,
    source,
    *,
    batch_size: int,
    max_frames: int,
) -> dict[str, torch.Tensor]:
    """Run forward passes via a BC ``Source``; gather head logits + labels.

    Takes a pre-built ``Source`` (the same object the BC trainer feeds:
    ``qnn.bc.loop.make_resident_source(episodes, device)``). Iterates
    frame-shuffled batches, runs the forward, and returns per-frame
    tensors needed by ``look_decomposition`` / ``attack_decomposition``.
    """
    from qnn.bc.supervised_loop import frame_shuffled_batches

    bufs: dict[str, list[torch.Tensor]] = {
        k: [] for k in (
            "pred_look", "base_look", "delta_look", "look_target_unit", "look_valid",
            "attack_logit", "prior_logit", "delta_attack",
            "masked_fire", "weapon_impulse",
        )
    }
    seen = 0
    with torch.inference_mode(), policy._autocast():
        for thunk in frame_shuffled_batches(source, batch_size=batch_size, training=False, rng=None):
            if seen >= max_frames:
                break
            b = thunk()
            # Renormalised target_probs_idx for the GT pointer — matches
            # supervised_step / evaluate_supervised.
            td = policy._tensor(b.actions["target_probs"], dtype=torch.float32)
            pres = (1.0 - td[..., 0]).clamp(min=1e-6)
            tpi = td[..., 1:] / pres.unsqueeze(-1)
            _, logits, _, _, _, _ = policy._forward_tensors(
                b.obs, hidden=b.hidden, masks=b.masks,
                target_gt=b.actions.get("target"),
                target_probs_idx=tpi,
            )
            # --- look ---
            if "look" in logits and "_look_base" in logits and "_look_delta" in logits:
                bufs["pred_look"].append(logits["look"].reshape(-1, 3).float().cpu())
                bufs["base_look"].append(logits["_look_base"].reshape(-1, 3).float().cpu())
                bufs["delta_look"].append(logits["_look_delta"].reshape(-1, 3).float().cpu())
                lt = policy._tensor(b.actions["look"], dtype=torch.float32).reshape(-1, 3)
                lt_n = torch.linalg.vector_norm(lt, dim=-1, keepdim=True)
                lt_u = (lt / lt_n.clamp(min=1e-6)).cpu()
                bufs["look_target_unit"].append(lt_u)
                bufs["look_valid"].append((lt_n.squeeze(-1) > 1e-6).cpu())
            # --- attack ---
            if "attack" in logits:
                al = logits["attack"].reshape(-1).float().cpu()
                pl = logits.get("_attack_prior")
                dl = logits.get("_attack_delta")
                pl = pl.reshape(-1).float().cpu() if pl is not None else torch.zeros_like(al)
                dl = dl.reshape(-1).float().cpu() if dl is not None else torch.zeros_like(al)
                bufs["attack_logit"].append(al)
                bufs["prior_logit"].append(pl)
                bufs["delta_attack"].append(dl)
            # --- shared per-frame conditioners ---
            fire = policy._tensor(b.actions["attack"], dtype=torch.float32).reshape(-1).cpu()
            if "input_mask" in b.actions:
                im = policy._tensor(b.actions["input_mask"], dtype=torch.long).reshape(-1).cpu()
                feas = (im & 1).float()
            else:
                feas = torch.ones_like(fire)
            bufs["masked_fire"].append(feas * fire)
            wid = policy._tensor(b.obs["self_weapon_id"], dtype=torch.long).reshape(-1).cpu()
            bufs["weapon_impulse"].append(self_weapon_id_to_impulse(wid).long())
            seen += int(b.rows)
    return {k: torch.cat(v) if v else torch.empty(0) for k, v in bufs.items()}


def _angular_summary(cos: torch.Tensor) -> dict[str, float]:
    deg = torch.acos(cos.clamp(-1.0, 1.0)) * (180.0 / math.pi)
    return {
        "mean": float(cos.mean()),
        "median": float(cos.median()),
        "p10": float(cos.quantile(0.1)),
        "deg_median": float(deg.median()),
        "deg_p90": float(deg.quantile(0.9)),
        "deg_p99": float(deg.quantile(0.99)),
    }


def look_decomposition(
    policy,
    source,
    *,
    batch_size: int = 4096,
    max_frames: int = 100_000,
    min_bucket: int = 100,
) -> dict:
    """Prior-vs-MLP cos_sim decomposition for the look head.

    ``source`` is a BC :class:`Source` — build it with
    ``qnn.bc.loop.make_resident_source(_load_precomputed(val_cache), device)``.

    Returns:
      overall:                   prior-alone vs full-head cos summaries + |delta_look|
      per_weapon:                same, conditioned on held weapon (impulse-indexed)
      per_fire_state:            same, conditioned on masked_fire {0, 1}
      mlp_gain:                  full - prior, top-line scalar
    """
    data = _collect_head_outputs(policy, source, batch_size=batch_size, max_frames=max_frames)
    valid = data["look_valid"].bool()
    pred = data["pred_look"][valid]
    base = data["base_look"][valid]
    delta = data["delta_look"][valid]
    tu = data["look_target_unit"][valid]
    wid = data["weapon_impulse"][valid]
    mf = data["masked_fire"][valid]

    cos_pred = (pred * tu).sum(-1)
    cos_base = (base * tu).sum(-1)
    mag_delta = torch.linalg.vector_norm(delta, dim=-1)

    out: dict = {
        "n_frames": int(valid.sum()),
        "overall": {
            "cos_pred": _angular_summary(cos_pred),
            "cos_base": _angular_summary(cos_base),
            "mlp_gain_cos": float((cos_pred - cos_base).mean()),
            "mag_delta_mean": float(mag_delta.mean()),
            "mag_delta_p90": float(mag_delta.quantile(0.9)),
        },
        "per_weapon": {},
        "per_fire_state": {},
    }
    for w in range(len(_WEAPON_NAMES)):
        mask = (wid == w)
        if int(mask.sum()) < min_bucket:
            continue
        out["per_weapon"][_WEAPON_NAMES[w]] = {
            "n": int(mask.sum()),
            "cos_pred_mean": float(cos_pred[mask].mean()),
            "cos_base_mean": float(cos_base[mask].mean()),
            "mlp_gain_cos": float((cos_pred[mask] - cos_base[mask]).mean()),
            "mag_delta_mean": float(mag_delta[mask].mean()),
        }
    for label, sub in (("masked_fire=1", mf > 0.5), ("masked_fire=0", mf <= 0.5)):
        if int(sub.sum()) < min_bucket:
            continue
        out["per_fire_state"][label] = {
            "n": int(sub.sum()),
            "cos_pred_mean": float(cos_pred[sub].mean()),
            "cos_base_mean": float(cos_base[sub].mean()),
            "mlp_gain_cos": float((cos_pred[sub] - cos_base[sub]).mean()),
            "mag_delta_mean": float(mag_delta[sub].mean()),
        }
    return out


def _binary_metrics(logit: torch.Tensor, label: torch.Tensor, threshold: float = 0.0) -> dict[str, float]:
    pred = (logit > threshold).long()
    tp = int(((pred == 1) & (label == 1)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    tn = int(((pred == 0) & (label == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    bce = F.binary_cross_entropy_with_logits(
        logit.float(), label.float(), reduction="mean",
    ).item()
    return {
        "n": int(label.numel()),
        "pos_rate": float(label.float().mean()),
        "pred_rate": float((pred == 1).float().mean()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "bce": bce,
        "logit_mean_pos": float(logit[label == 1].mean()) if (label == 1).any() else float("nan"),
        "logit_mean_neg": float(logit[label == 0].mean()) if (label == 0).any() else float("nan"),
    }


def attack_decomposition(
    policy,
    source,
    *,
    batch_size: int = 4096,
    max_frames: int = 100_000,
    min_bucket: int = 100,
) -> dict:
    """Prior-vs-MLP BCE/F1 decomposition for ``LookStyleAttackHead``-style attack.

    ``source`` is a BC :class:`Source` (same as ``look_decomposition``).

    ``attack_logit = prior_logit + delta_attack``. The decomposition
    reports BCE/F1 and TP/FP/FN at threshold 0 for:

      * the full attack head (``attack_logit``)
      * the prior alone (``prior_logit`` with ``delta_attack`` zeroed)

    Plus mean prior/delta magnitudes and the per-class logit means
    (logit-separation between masked_fire=1 vs masked_fire=0 frames),
    overall + per weapon + per fire-state. Compare to ``look_decomposition``
    to see whether the attack MLP is adding work analogous to look's.
    """
    data = _collect_head_outputs(policy, source, batch_size=batch_size, max_frames=max_frames)
    if data["attack_logit"].numel() == 0:
        return {"_error": "no attack head logits"}

    al = data["attack_logit"]
    pl = data["prior_logit"]
    dl = data["delta_attack"]
    mf = data["masked_fire"].to(torch.long)
    wid = data["weapon_impulse"]

    out: dict = {
        "n_frames": int(al.numel()),
        "overall": {
            "full_head": _binary_metrics(al, mf),
            "prior_only": _binary_metrics(pl, mf),
            "delta_mean": float(dl.mean()),
            "delta_abs_mean": float(dl.abs().mean()),
            "delta_abs_p90": float(dl.abs().quantile(0.9)),
            "prior_mean": float(pl.mean()),
            "prior_abs_mean": float(pl.abs().mean()),
        },
        "per_weapon": {},
        "per_fire_state": {},
    }
    for w in range(len(_WEAPON_NAMES)):
        mask = (wid == w)
        if int(mask.sum()) < min_bucket:
            continue
        full = _binary_metrics(al[mask], mf[mask])
        prior = _binary_metrics(pl[mask], mf[mask])
        out["per_weapon"][_WEAPON_NAMES[w]] = {
            "n": int(mask.sum()),
            "full_f1": full["f1"], "prior_f1": prior["f1"],
            "full_bce": full["bce"], "prior_bce": prior["bce"],
            "mlp_gain_f1": full["f1"] - prior["f1"],
            "logit_sep_full": full["logit_mean_pos"] - full["logit_mean_neg"],
            "logit_sep_prior": prior["logit_mean_pos"] - prior["logit_mean_neg"],
        }
    for label, sub in (("masked_fire=1", mf == 1), ("masked_fire=0", mf == 0)):
        if int(sub.sum()) < min_bucket:
            continue
        out["per_fire_state"][label] = {
            "n": int(sub.sum()),
            "logit_full_mean": float(al[sub].mean()),
            "logit_prior_mean": float(pl[sub].mean()),
            "logit_delta_mean": float(dl[sub].mean()),
        }
    return out
