"""Per-checkpoint duration calibration for the a25 move commitment decode.

The segment head's duration posterior is right-censoring biased: training
drops right-censored segments (disproportionately the LONG ones, cut by
encounter boundaries), so sampled dwells run short and the closed-loop
change-point rate inflates (research/move-head.md §8 rhythm addendum). The
correction is ``move.commit_dur_tilt``: an exponential tilt
``+tilt * bucket_index`` on the bucket logits, one scalar per axis, fit by
moment-matching the decode-realistic expected duration to the human event
mean on the engaged val cache.

The tilt is a PER-CHECKPOINT constant — each model's posterior has its own
censoring geometry. Never carry fitted values between checkpoints; refit
(decode-fit pipeline stage 2 does this automatically for move_seg runs).

Kernel for both the decode-fit pipeline (qnn.eval.decode_fit_pipeline,
stage 2) and the standalone audit CLI
(scripts/analysis/_move_seg_dur_calibration.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from qnn.model.bench.a25.decode import _BUCKET_HI, _BUCKET_LO
from qnn.model.bench.a25.move_seg_head import N_BUCKETS, derive_segment_targets

AXES = ("fb", "lr")
# expected duration of a uniform-in-bucket draw, per bucket
_MID = (np.asarray(_BUCKET_LO, dtype=np.float64)
        + np.asarray(_BUCKET_HI, dtype=np.float64)) / 2.0


@torch.inference_mode()
def collect_events(policy, source):
    """Per-axis onset events: joint logits (30,), true target, prev class."""
    from qnn.model.bench.side_channels import bench_side_channel_scope

    offsets = np.asarray(source.episode_offsets, dtype=np.int64)
    rows = {ax: {"logit": [], "tgt": [], "prev": []} for ax in AXES}
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e - s < 3:
            continue
        obs = {k: v[s:e] for k, v in source.obs.items()}
        act = {k: v[s:e] for k, v in source.actions.items()}
        # Tensorize through the policy's inference dequant (idempotent):
        # resident/collect obs may carry packed spatial_atlas, and the raw
        # forward path below bypasses act()'s dequant boundary.
        obs_t = {k: v.unsqueeze(1)
                 for k, v in policy._obs_tensors_dequant(obs).items()}
        act_t = {k: torch.as_tensor(np.asarray(v)) for k, v in act.items()}
        with bench_side_channel_scope(act_t, None):
            _, logits, _, _, _ = policy._forward_tensors(obs_t, hidden=None, masks=None)
        # (T, n_axes, 3*N_BUCKETS): movearch seg heads carry a third water-ud
        # axis; dur_tilt is fb/lr-only by definition, so slice the first two.
        n_rows = len(np.asarray(act["move"]))
        seg = logits["move_seg"].reshape(n_rows, -1, 3 * N_BUCKETS)[:, :2, :]
        mv = torch.as_tensor(np.asarray(act["move"])).long().reshape(-1, 1, 3)
        valid = torch.ones(len(mv), 1, dtype=torch.bool)
        tgt = derive_segment_targets(mv, valid).reshape(-1, 2)
        mv2 = mv.reshape(-1, 3)
        for ai, ax in enumerate(AXES):
            idx = np.where((tgt[:, ai] != -100).numpy())[0]
            if not len(idx):
                continue
            rows[ax]["logit"].append(seg[idx, ai].float())
            rows[ax]["tgt"].append(tgt[idx, ai])
            # class held BEFORE the onset (the decode's expiry-masked class)
            prev = torch.where(torch.as_tensor(idx) > 0,
                               mv2[np.maximum(idx - 1, 0), ai], torch.tensor(-1))
            rows[ax]["prev"].append(prev)
    return rows


def masked_bucket_marginal(logits: torch.Tensor, prev: torch.Tensor,
                           tilt: float = 0.0) -> np.ndarray:
    """(N,10) decode-realistic duration marginal: held masked, joint renorm."""
    n = len(logits)
    l = logits.clone().reshape(n, 3, N_BUCKETS)
    if tilt:
        l = l + tilt * torch.arange(N_BUCKETS, dtype=l.dtype).reshape(1, 1, -1)
    has_prev = prev >= 0
    l[has_prev, prev[has_prev].long()] = -1e9
    p = torch.softmax(l.reshape(n, -1), dim=-1).reshape(n, 3, N_BUCKETS)
    return p.sum(dim=1).numpy()


def fit_tilt(logits: torch.Tensor, prev: torch.Tensor, target_mean: float) -> float:
    """Bisection on the scalar bucket tilt to match expected sampled duration."""
    def mean_at(t: float) -> float:
        pm = masked_bucket_marginal(logits, prev, tilt=t)
        return float((pm @ _MID).mean())
    lo, hi = -1.0, 1.0
    while mean_at(hi) < target_mean and hi < 4.0:
        hi *= 2.0
    while mean_at(lo) > target_mean and lo > -4.0:
        lo *= 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if mean_at(mid) < target_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@torch.inference_mode()
def fit_dur_tilt(run_dir: Path, cache_dir: Path | str,
                 out_path: Path | None = None) -> dict:
    """Fit the per-axis ``move.commit_dur_tilt`` for one checkpoint.

    Teacher-forced over the engaged val cache (CPU, ~10-15 min). Returns (and
    optionally writes) the calibration report; ``dur_tilt`` is the [fb, lr]
    pair ready for the decode config. ``cache_dir`` must be the run's OWN
    pinned corpus (decode-fit passes ctx.corpus_dir): teacher-forcing on any
    other collect is out-of-distribution AND may not even share the obs
    layout (an atlas model on the pre-atlas qwd cache has no spatial token).
    """
    from qnn.diag.loader import load_policy
    from qnn.bc.supervised_loop import make_resident_source_from_cache

    run_dir = Path(run_dir)
    policy, _probe = load_policy(run_dir, device="cpu")
    # same checkpoint-resolution order as load_policy (provenance stamp)
    cks = (sorted((run_dir / "checkpoints").glob("best_*.pth"))
           or sorted((run_dir / "checkpoints").glob("bc_best_model.pth")))
    source = make_resident_source_from_cache(
        Path(cache_dir) / "precomputed_val", torch.device("cpu"),
        segment_mask={"act.target": {"$ne": 0}})
    rows = collect_events(policy, source)

    report: dict = {"run_dir": str(run_dir), "bucket_mid": _MID.tolist(),
                    "checkpoint": str(cks[0]) if cks else ""}
    for ax in AXES:
        L = torch.cat(rows[ax]["logit"])
        T = torch.cat(rows[ax]["tgt"])
        P = torch.cat(rows[ax]["prev"])
        bk_t = (T % N_BUCKETS).numpy()
        emp = np.bincount(bk_t, minlength=N_BUCKETS).astype(np.float64)
        emp /= emp.sum()
        pred = masked_bucket_marginal(L, P).mean(axis=0)
        human_mean = float(emp @ _MID)
        pred_mean = float((masked_bucket_marginal(L, P) @ _MID).mean())
        tilt = fit_tilt(L, P, human_mean)
        report[ax] = {
            "n_events": int(len(T)),
            "human_bucket_hist": [round(x, 4) for x in emp],
            "pred_bucket_marginal": [round(float(x), 4) for x in pred],
            "human_mean_dur": round(human_mean, 3),
            "pred_mean_dur": round(pred_mean, 3),
            "fitted_tilt": round(tilt, 4),
        }
    report["dur_tilt"] = [report["fb"]["fitted_tilt"], report["lr"]["fitted_tilt"]]
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
