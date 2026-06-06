"""Diagnostics for the trained target_head_probe.

Loads a checkpoint and answers two questions about what the probe is
actually learning:

  1. Modality vs lookahead split — for each bucket B frame (labeler chose
     a pid != engine slot 0), classify it as:
       MODALITY:  labeler's pid is in obs at this frame AND its recency > 0.
                  Engine could not have it as sticky because of the
                  SIGHT/PROXIMITY filter. The causal signal is present.
       LOOKAHEAD: labeler's pid recency = 0 (in SIGHT now) but engine
                  slot 0 is a different pid. The label is justified by a
                  future fire that a causal model cannot see.
     Reports probe accuracy in each split.

  2. Feature ablation — re-evaluate while zeroing out one feature group
     at a time, to identify which signal carries the probe's gains:
       --ablate recency         zero out per-slot recency scalar (offset 18)
       --ablate rel             zero out per-slot rel vector (offset 3..5)
       --ablate look            zero out the look vector
       --ablate fire            zero out the fire flag
       --ablate self_scalars    zero out the self scalar block
       --ablate slot_enemy      zero out the enemy mask
       --ablate everything_else keep only per-slot scalars

Usage:
    PYTHONPATH=src python -m qnn.labeler.probes.target_head_diag \
        --data-dir artifacts/collect/qwd \
        --ckpt    runs/probe/full_w0_15_gpu/latest.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from qnn.labeler.probes.target_head_probe import (
    N_SLOTS, N_SLOT_SCALARS, TargetHeadProbe, _ChunkedDataset,
    _load_split, _to_device, _accumulate, BucketStats,
)

RECENCY_OFFSET = 18
REL_OFFSET = 3
REL_LEN = 3


def _build_model(ckpt: dict, device: torch.device) -> TargetHeadProbe:
    c = ckpt["cfg"]
    model = TargetHeadProbe(
        channels=c["channels"], n_layers=c["n_layers"],
        kernel_size=c["kernel_size"], p_drop=c.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _eval_split(
    model: TargetHeadProbe, loader: DataLoader, device: torch.device,
    label: str, ablations: list[str] | None = None,
) -> None:
    """Run inference and report:
       - argmax accuracy on bucket A / B
       - bucket B split into MODALITY vs LOOKAHEAD by inspecting per-frame
         features (recency on the labeled slot).
    """
    abl = set(ablations or [])

    a = BucketStats(); b = BucketStats(); allb = BucketStats()
    # Sub-bucket B: by modality.
    b_mod = BucketStats(); b_look = BucketStats()
    # Confidence-bucket histograms over bucket B correct / incorrect.
    conf_correct: list[float] = []
    conf_incorrect: list[float] = []

    # For threshold-policy at one practical tau.
    tau = 0.9
    tau_overrides = 0
    tau_override_correct = 0
    tau_correct_a = 0
    tau_correct_b = 0
    tau_n_a = 0
    tau_n_b = 0

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)

            slot_scalars = batch["slot_scalars"].clone()  # (B,T,16,19)
            self_scalars = batch["self_scalars"].clone()
            slot_enemy   = batch["slot_enemy"].clone()
            look         = batch["look"].clone()
            fire         = batch["fire"].clone()

            # Apply ablations.
            if "recency" in abl:
                slot_scalars[..., RECENCY_OFFSET] = 0
            if "rel" in abl:
                slot_scalars[..., REL_OFFSET:REL_OFFSET + REL_LEN] = 0
            if "self_scalars" in abl:
                self_scalars[...] = 0
            if "slot_enemy" in abl:
                slot_enemy[...] = 0
            if "look" in abl:
                look[...] = 0
            if "fire" in abl:
                fire[...] = 0
            if "slot_scalars" in abl:
                slot_scalars[...] = 0

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                logits = model(
                    slot_scalars, batch["slot_types"], slot_enemy,
                    self_scalars, batch["movement_oh"],
                    batch["weapon_id"], look, fire,
                )
            logits = logits.float()
            pred = logits.argmax(dim=-1)
            probs = F.softmax(logits, dim=-1)

            target = batch["target"]
            _accumulate(allb, a, b, target, pred)

            # Sub-bucket the bucket B frames by recency of the labeled slot.
            mask = target != -100
            tg = target[mask]
            pr = pred[mask]
            pp = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)[mask]
            # slot_scalars[mask, lbl_slot, RECENCY_OFFSET] gives recency for the
            # labeled slot. We need to index the labeled slot at each frame.
            # ss has shape (B, T, 16, 19); apply mask -> (N, 16, 19)
            ss_masked = batch["slot_scalars"][mask]
            lbl_rec = ss_masked.gather(
                1, tg.view(-1, 1, 1).expand(-1, 1, N_SLOT_SCALARS)
            ).squeeze(1)[:, RECENCY_OFFSET]

            is_b = tg != 0
            tg_b = tg[is_b]
            pr_b = pr[is_b]
            pp_b = pp[is_b]
            rec_b = lbl_rec[is_b]

            # MODALITY: labeled pid currently has recency > 0 (it's in SOUND-only
            # or stale-SIGHT modality). LOOKAHEAD: recency = 0 (in SIGHT now but
            # engine chose a different sticky).
            mod_mask = rec_b > 0
            look_mask = ~mod_mask
            for sub_stats, sub_mask in [(b_mod, mod_mask), (b_look, look_mask)]:
                if not sub_mask.any():
                    continue
                t_sub = tg_b[sub_mask]
                p_sub = pr_b[sub_mask]
                n = int(sub_mask.sum().item())
                correct = int((t_sub == p_sub).sum().item())
                pred_zero = int((p_sub == 0).sum().item())
                sub_stats.n += n
                sub_stats.correct += correct
                sub_stats.pred_zero += pred_zero

            # Confidence histogram on bucket B correct/incorrect.
            correct_b = (tg_b == pr_b)
            conf_correct.extend(pp_b[correct_b].cpu().tolist())
            conf_incorrect.extend(pp_b[~correct_b].cpu().tolist())

            # Threshold-policy at tau=0.9.
            is_a = tg == 0
            override = (pr != 0) & (pp >= tau)
            policy_pred = torch.where(override, pr, torch.zeros_like(pr))
            tau_n_a += int(is_a.sum().item())
            tau_n_b += int(is_b.sum().item())
            tau_correct_a += int(((policy_pred == tg) & is_a).sum().item())
            tau_correct_b += int(((policy_pred == tg) & is_b).sum().item())
            tau_overrides += int(override.sum().item())
            tau_override_correct += int(((policy_pred == tg) & override).sum().item())

    bucket_b_mod_acc = b_mod.correct / max(b_mod.n, 1) * 100
    bucket_b_look_acc = b_look.correct / max(b_look.n, 1) * 100

    print(f"\n=== {label} ===")
    print(f"argmax:")
    print(f"  bucket A (target=0)    n={a.n:7d}  correct={a.correct:7d}  ({a.acc():.2f}%)")
    print(f"  bucket B (target!=0)   n={b.n:7d}  correct={b.correct:7d}  ({b.acc():.2f}%)")
    print(f"  bucket B / MODALITY    n={b_mod.n:7d}  correct={b_mod.correct:7d}  "
          f"({bucket_b_mod_acc:.2f}%)  pred_zero={b_mod.pred_zero}")
    print(f"  bucket B / LOOKAHEAD   n={b_look.n:7d}  correct={b_look.correct:7d}  "
          f"({bucket_b_look_acc:.2f}%)  pred_zero={b_look.pred_zero}")
    print(f"τ=0.9 policy:")
    A_acc = 100 * tau_correct_a / max(tau_n_a, 1)
    B_acc = 100 * tau_correct_b / max(tau_n_b, 1)
    overall = 100 * (tau_correct_a + tau_correct_b) / max(tau_n_a + tau_n_b, 1)
    ov_acc = 100 * tau_override_correct / max(tau_overrides, 1)
    print(f"  A={A_acc:.2f}%  B={B_acc:.2f}%  overall={overall:.2f}%  "
          f"overrides={tau_overrides}  override_acc={ov_acc:.2f}%")

    # Confidence stats.
    if conf_correct:
        cc = np.array(conf_correct)
        print(f"argmax confidence on bucket B / CORRECT:   "
              f"mean={cc.mean():.3f}  median={np.median(cc):.3f}  "
              f"p90={np.quantile(cc, 0.9):.3f}")
    if conf_incorrect:
        ci = np.array(conf_incorrect)
        print(f"argmax confidence on bucket B / INCORRECT: "
              f"mean={ci.mean():.3f}  median={np.median(ci):.3f}  "
              f"p90={np.quantile(ci, 0.9):.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--ckpt",     type=Path, required=True)
    p.add_argument("--chunk-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-shards-val", type=int, default=None,
                   help="limit val to first N shards")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = _build_model(ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded ckpt epoch={ckpt['epoch']}  params={n_params:,}")

    val_shards = _load_split(args.data_dir / "precomputed_val", args.max_shards_val)
    val_ds     = _ChunkedDataset(val_shards, args.chunk_len)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    print(f"val chunks: {len(val_ds)}")

    # Baseline eval (no ablation).
    _eval_split(model, val_loader, device, "baseline (no ablation)")

    # Single-feature ablations.
    for ablation in [
        ["recency"], ["rel"], ["look"], ["fire"],
        ["self_scalars"], ["slot_enemy"], ["slot_scalars"],
    ]:
        _eval_split(model, val_loader, device,
                    f"ablate {','.join(ablation)}", ablations=ablation)


if __name__ == "__main__":
    main()
