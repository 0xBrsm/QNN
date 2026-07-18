"""Training loop for the move labeler.

Reads native-rate slim collect output produced by qnn.labeler.collect,
chunks each episode into fixed-length windows for batching, and trains
fb/lr cross-entropy with per-axis class weighting under bf16 autocast.

Pipeline shape (avoiding the scripts/seq_labeler.py issues this rewrite
addresses):
  - DataLoader with num_workers + pin_memory + non_blocking H2D
  - mmap'd shard arrays; features built lazily per chunk
  - real batch size, no padded full episodes; chunk_len fixed at 1024
    native frames (~13s context vs +-1.8s receptive field)
  - autocast bf16 around forward + loss
  - per-axis class weights with capped 'none' weight (matches the
    bakeoff's heuristic that else 'none' dominates)

Usage:
    PYTHONPATH=src python -m qnn.labeler.train \\
        --data-dir artifacts/collect/qwd_labeler \\
        --output   runs/labeler/v1 \\
        --epochs 12 --batch-size 32 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from qnn.actions import (
    MOVE_CLASS_NEG,
    MOVE_CLASS_NONE,
    MOVE_CLASS_POS,
    decode_move_pressbyte,
)
from .data import (
    _EpisodeView,
    _load_split,
    matched_episode_strides,
    op_input_keep_mask,
)
from .seg_stats import (
    downsample_axis,
    segment_parity,
    window_all_true,
    window_ids,
)
from .model import (
    CORE_FEAT_DIM,
    FeatureSpec,
    MoveLabeler,
    N_CLASSES,
    VELOCITY_SCALE,
    build_features,
    decode_move_fb_lr,
    decode_move_fb_lr_ud,
)


# ── config ────────────────────────────────────────────────────────────────────

_AXIS_CHOICES = ("both", "fb", "lr")


def _accumulate_cm(cm: np.ndarray, true: torch.Tensor, pred: torch.Tensor) -> None:
    """Add a (B, T) batch's pairs into a 3x3 confusion matrix.  Ignores
    targets equal to -100 (chunk padding).  Mutates `cm` in place."""
    mask = true != -100
    t = true[mask].reshape(-1).cpu().numpy()
    p = pred[mask].reshape(-1).cpu().numpy()
    # np.add.at handles repeated indices correctly.
    np.add.at(cm, (t, p), 1)


def _acc_and_macro_f1(cm: np.ndarray) -> tuple[float, float]:
    """Return (accuracy as percent, macro-F1 over classes that have
    nonzero truth support).  macro-F1 ignores classes with zero truth
    so the score isn't pulled down by absent classes (e.g. ud=neg)."""
    total = cm.sum()
    if total == 0:
        return 0.0, 0.0
    acc = 100.0 * np.trace(cm) / total
    f1s = []
    for c in range(cm.shape[0]):
        true_c = cm[c, :].sum()
        if true_c == 0:
            continue
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = true_c - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(acc), float(np.mean(f1s)) if f1s else 0.0


@dataclass
class TrainConfig:
    data_dir: Path
    output:   Path
    feat_spec: FeatureSpec = field(default_factory=FeatureSpec)
    chunk_len:    int   = 1024
    epochs:       int   = 12
    batch_size:   int   = 32
    lr:           float = 1e-3
    weight_decay: float = 0.0
    grad_clip:    float = 1.0
    dtype:        str   = "bf16"        # "bf16" | "fp16" | "fp32"
    num_workers:  int   = 8
    pin_memory:   bool  = True
    none_weight:  float = 0.5           # cap on the 'none' class weight
    uniform_class_weights: bool = False # disable class reweighting entirely
    baseline_skip_init_scale: float = 3.0  # scale of identity init on the baseline skip path
    train_axis: str = "both"            # "both" | "fb" | "lr" — which axis to train
    channels:     int   = 64
    n_layers:     int   = 6
    dropout:      float = 0.1
    seed:         int   = 17
    resume:       Path | None = None    # checkpoint to resume from (e.g., latest.pt)
    smooth_target: int = 0              # majority-vote window over move target (0/1 = off)
    predict_ud:   bool = False          # add a third head for ud (jump press)
    # When set, per-tick targets are CE-masked using the operative-input
    # bitmask (`act/op_input`) emitted by the C worker.  Bit i of
    # the mask corresponds to axis i in {fb, lr, ud, fire, impulse};
    # a 0 bit means the player's input on that axis at that tick had no
    # observable engine effect (e.g., jump-while-airborne, fire-in-
    # cooldown) and the target is replaced with ignore_index=-100 so CE
    # skips it.  Silently no-op for shards that don't carry the sidecar
    # (older formats — falls through unmodified).
    sanitize_targets: bool = False
    # Cap training to at most this many frames (0 = all).  Episodes are shuffled
    # by `seed` and accumulated until the cap (cap-first, so smaller caps nest in
    # larger ones) — for data-scaling / learning-curve sweeps.
    max_train_frames: int = 0
    # Native collect rate; the val segment-parity gate downsamples pred/truth
    # to 20 Hz (stride = round(native_hz / 20)) before computing move_seg
    # onset/duration parity, matching the GBT relabel-quality table.
    native_hz: int = 77


# Episode loading (_EpisodeView, _load_split) lives in qnn.labeler.data and is
# imported above — shared with the GBT path's materialize_split.


# ── chunked dataset ───────────────────────────────────────────────────────────

def _smooth_move_target(move_packed: np.ndarray, window: int) -> np.ndarray:
    """Per-axis majority-vote smoothing of the packed press-byte move target.

    For each tick t, replace the per-axis class with the mode over
    [t - window//2 .. t + window//2], clamped to episode bounds.  Window
    sizes of 0 or 1 are a no-op (no smoothing).  Reduces frame-boundary
    transition noise (e.g., button-just-released ticks) at the cost of
    blurring labels near genuine press boundaries.

    Decodes/re-encodes in the press-byte layout so the result round-trips
    cleanly back through ``decode_move_pressbyte`` (the smoothed byte carries
    only the fb/lr/ud direction bits; attack/jump-button bits, which the
    labeler never reads from the target, are dropped).
    """
    if window <= 1:
        return move_packed
    half = window // 2
    arr = np.asarray(move_packed, dtype=np.uint8).reshape(-1)
    n = arr.shape[0]
    axes = decode_move_pressbyte(arr)          # (n, 3) {0=neg, 1=none, 2=pos}
    fb = axes[:, 0].astype(np.int32)
    lr = axes[:, 1].astype(np.int32)
    ud = axes[:, 2].astype(np.int32)
    # Press-byte bit positions for the negative/positive class of each axis.
    neg_bit = (1, 3, 5)
    pos_bit = (2, 4, 6)
    out = np.zeros(n, dtype=np.uint8)
    for t in range(n):
        lo = max(0, t - half)
        hi = min(n, t + half + 1)
        for axis, col in enumerate((fb, lr, ud)):
            cls = int(np.bincount(col[lo:hi], minlength=3).argmax())
            if cls == MOVE_CLASS_NEG:
                out[t] |= np.uint8(1 << neg_bit[axis])
            elif cls == MOVE_CLASS_POS:
                out[t] |= np.uint8(1 << pos_bit[axis])
    return out


class ChunkedDataset(Dataset):
    """Episodes chunked into fixed-length windows. Last partial chunk is
    zero-padded with a -100 ignore_index in the targets so CE skips it."""

    def __init__(self, episodes: list[_EpisodeView], chunk_len: int,
                 spec: FeatureSpec, smooth_target: int = 0,
                 with_ud: bool = False, sanitize_targets: bool = False) -> None:
        self.episodes = episodes
        self.chunk_len = chunk_len
        self.spec = spec
        self.smooth_target = int(smooth_target)
        self.with_ud = bool(with_ud)
        self.sanitize_targets = bool(sanitize_targets)

        # Precompute per-episode features + decoded/sanitized targets ONCE.
        # Previously build_features + decode + op_input sanitize ran inside
        # every __getitem__, i.e. re-featurizing the whole corpus on every
        # epoch across only num_workers processes — the GPU-starving
        # bottleneck.  Paying it a single time here makes __getitem__ a pure
        # slice+pad.  feats stored fp16 (~1.8 GB for 100M frames, cast back
        # to fp32 on copy), targets int8 (-100 ignore_index fits).  Built in
        # the main process before the loader forks; workers share via COW.
        t_pre = time.time()
        self._feats: list[np.ndarray] = []   # (T, F) fp16
        self._fb:    list[np.ndarray] = []   # (T,)   int8, -100 = ignore
        self._lr:    list[np.ndarray] = []
        self._ud:    list[np.ndarray] = []   # filled only if with_ud
        for ep in episodes:
            feats = build_features(
                ep.self_velocity, ep.self_movement_id, ep.look,
                weapon_id=ep.weapon_id if ep.weapon_id is not None else None,
                gbt_probs=ep.gbt_probs if ep.gbt_probs is not None else None,
                spec=spec,
            ).astype(np.float16, copy=False)
            move_src = (_smooth_move_target(np.asarray(ep.move), self.smooth_target)
                        if self.smooth_target > 1 else np.asarray(ep.move))
            decoded = (decode_move_fb_lr_ud(move_src) if self.with_ud
                       else decode_move_fb_lr(move_src))      # (T, 2|3)
            fb = decoded[:, 0].astype(np.int8)
            lr = decoded[:, 1].astype(np.int8)
            ud = decoded[:, 2].astype(np.int8) if self.with_ud else None
            if self.sanitize_targets and ep.op_input is not None:
                keep = op_input_keep_mask(decoded, np.asarray(ep.op_input),
                                          with_ud=self.with_ud)   # (T, n_axes)
                fb = np.where(keep[:, 0], fb, np.int8(-100)).astype(np.int8)
                lr = np.where(keep[:, 1], lr, np.int8(-100)).astype(np.int8)
                if self.with_ud:
                    ud = np.where(keep[:, 2], ud, np.int8(-100)).astype(np.int8)
            self._feats.append(feats)
            self._fb.append(fb)
            self._lr.append(lr)
            if self.with_ud:
                self._ud.append(ud)

        self.index: list[tuple[int, int, int]] = []   # (ep_idx, start, valid_len)
        for ep_idx, ep in enumerate(episodes):
            T = ep.n_frames
            if T == 0:
                continue
            n_full = T // chunk_len
            for c in range(n_full):
                self.index.append((ep_idx, c * chunk_len, chunk_len))
            tail = T - n_full * chunk_len
            if tail > 0:
                self.index.append((ep_idx, n_full * chunk_len, tail))
        print(f"  precomputed features for {len(episodes)} eps in "
              f"{time.time() - t_pre:.1f}s", flush=True)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        ep_idx, start, valid_len = self.index[idx]
        end = start + valid_len
        L = self.chunk_len
        F_dim = self.spec.dim

        # Pure slice + pad off the precomputed per-episode arrays.  Assigning
        # the fp16 feats / int8 targets into the fp32 / long destination
        # tensors casts on copy.
        feats = torch.zeros(L, F_dim, dtype=torch.float32)
        fb    = torch.full((L,), -100, dtype=torch.long)
        lr    = torch.full((L,), -100, dtype=torch.long)
        feats[:valid_len] = torch.from_numpy(self._feats[ep_idx][start:end])
        fb[:valid_len]    = torch.from_numpy(self._fb[ep_idx][start:end])
        lr[:valid_len]    = torch.from_numpy(self._lr[ep_idx][start:end])
        if self.with_ud:
            ud = torch.full((L,), -100, dtype=torch.long)
            ud[:valid_len] = torch.from_numpy(self._ud[ep_idx][start:end])
        else:
            ud = None

        if self.spec.use_baseline_skip:
            ep = self.episodes[ep_idx]
            # Baseline one-hot (fb, lr) computed from velocity, fed as a
            # separate input that bypasses the encoder.
            from .model import _baseline_classes_from_vel, BASELINE_EPS, BASELINE_DIM
            baseline_classes = _baseline_classes_from_vel(
                np.asarray(ep.self_velocity[start:end], dtype=np.float32),
                eps=BASELINE_EPS,
            )
            baseline_oh = np.zeros((valid_len, BASELINE_DIM), dtype=np.float32)
            for axis in range(2):
                for cls in range(N_CLASSES):
                    baseline_oh[:, axis * N_CLASSES + cls] = (
                        baseline_classes[:, axis] == cls).astype(np.float32)
            baseline = torch.zeros(L, BASELINE_DIM, dtype=torch.float32)
            baseline[:valid_len] = torch.from_numpy(baseline_oh)
            if self.with_ud:
                return feats, fb, lr, ud, baseline
            return feats, fb, lr, baseline
        if self.with_ud:
            return feats, fb, lr, ud
        return feats, fb, lr


# ── class weights ─────────────────────────────────────────────────────────────

def _class_weights(
    episodes: list[_EpisodeView], none_weight: float, uniform: bool = False,
    include_ud: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse-frequency weights per class (fb, lr, ud), with the 'none'
    weight capped at none_weight x median(neg, pos).  If ``uniform`` is
    set, all axes return all-ones.  ud weights are returned regardless of
    ``include_ud``; the caller decides whether to use them.

    ud distribution is heavily 'none'-skewed (~92-97% none in QW); the
    same cap rule applies, but the cap may bind harder on ud than on the
    horizontal axes.
    """
    if uniform or not episodes:
        eye = torch.ones(N_CLASSES, dtype=torch.float32)
        return eye, eye, eye

    all_moves = np.concatenate([np.asarray(ep.move) for ep in episodes]).astype(np.uint8)
    # move is the BC press-byte — decode to 3-class {0=neg,1=none,2=pos} per axis
    # (NOT the retired field3 2-bit fields, which read the wrong bits here and
    # yield a 4-class histogram → a 4-long weight tensor cross_entropy rejects).
    axes = decode_move_pressbyte(all_moves)   # (N, 3)
    fb, lr, ud = axes[:, 0], axes[:, 1], axes[:, 2]

    def _w(counts: np.ndarray) -> torch.Tensor:
        total = counts.sum()
        if total == 0:
            return torch.ones(N_CLASSES, dtype=torch.float32)
        w = total / (N_CLASSES * np.maximum(counts, 1))
        w[1] = min(w[1], none_weight * float(np.median(w[[0, 2]])))
        return torch.tensor(w, dtype=torch.float32)

    return (
        _w(np.bincount(fb, minlength=N_CLASSES)),
        _w(np.bincount(lr, minlength=N_CLASSES)),
        _w(np.bincount(ud, minlength=N_CLASSES)),
    )


# ── train / eval loops ────────────────────────────────────────────────────────

def _autocast_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.output.mkdir(parents=True, exist_ok=True)
    cfg_dict = {k: (str(v) if isinstance(v, Path) else asdict(v) if hasattr(v, "__dataclass_fields__") else v)
                for k, v in asdict(cfg).items()}
    (cfg.output / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    print(f"loading data from {cfg.data_dir}", flush=True)
    train_eps = _load_split(cfg.data_dir / "precomputed_train")
    val_eps   = _load_split(cfg.data_dir / "precomputed_val")
    if not train_eps:
        raise SystemExit(f"no training episodes under {cfg.data_dir}/precomputed_train")
    if cfg.max_train_frames > 0:
        # Cap-first episode subsample for data-scaling sweeps (shuffle by seed,
        # take whole episodes until the frame cap; nested across cap sizes).
        order = np.random.default_rng(cfg.seed).permutation(len(train_eps))
        kept, n = [], 0
        for i in order:
            kept.append(train_eps[i]); n += train_eps[i].n_frames
            if n >= cfg.max_train_frames:
                break
        train_eps = kept
        print(f"  [max_train_frames={cfg.max_train_frames:,}] capped to "
              f"{len(train_eps)} eps / {n:,} frames")
    train_frames = sum(ep.n_frames for ep in train_eps)
    val_frames   = sum(ep.n_frames for ep in val_eps)
    print(f"  train: {len(train_eps)} eps  {train_frames:,} frames")
    print(f"  val:   {len(val_eps)} eps  {val_frames:,} frames")

    train_ds = ChunkedDataset(train_eps, cfg.chunk_len, cfg.feat_spec,
                              smooth_target=cfg.smooth_target,
                              with_ud=cfg.predict_ud,
                              sanitize_targets=cfg.sanitize_targets)
    val_ds   = ChunkedDataset(val_eps,   cfg.chunk_len, cfg.feat_spec,
                              smooth_target=cfg.smooth_target,
                              with_ud=cfg.predict_ud,
                              sanitize_targets=cfg.sanitize_targets)
    print(f"  train chunks: {len(train_ds)}   val chunks: {len(val_ds)}")

    w_fb, w_lr, w_ud = _class_weights(train_eps, cfg.none_weight,
                                      uniform=cfg.uniform_class_weights)
    print(f"  class weights  fb={w_fb.numpy().round(2).tolist()}  "
          f"lr={w_lr.numpy().round(2).tolist()}"
          + (f"  ud={w_ud.numpy().round(2).tolist()}" if cfg.predict_ud else ""))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MoveLabeler(
        feat_dim=cfg.feat_spec.dim,
        channels=cfg.channels,
        n_layers=cfg.n_layers,
        p_drop=cfg.dropout,
        use_baseline_skip=cfg.feat_spec.use_baseline_skip,
        baseline_skip_init_scale=cfg.baseline_skip_init_scale,
        skip_fb=cfg.feat_spec.skip_fb,
        skip_lr=cfg.feat_spec.skip_lr,
        predict_ud=cfg.predict_ud,
    ).to(device)
    w_fb_d = w_fb.to(device)
    w_lr_d = w_lr.to(device)
    w_ud_d = w_ud.to(device)
    print(f"  device: {device}  feat_dim: {cfg.feat_spec.dim}  "
          f"params: {sum(p.numel() for p in model.parameters()):,}  "
          f"rf: +-{model.receptive_field}")

    opt   = torch.optim.AdamW(model.parameters(),
                              lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    use_autocast  = cfg.dtype != "fp32" and device.type == "cuda"
    autocast_dt   = _autocast_dtype(cfg.dtype)

    pin = cfg.pin_memory and device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin,
        persistent_workers=cfg.num_workers > 0, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin,
        persistent_workers=cfg.num_workers > 0,
    )

    skip    = cfg.feat_spec.use_baseline_skip
    with_ud = cfg.predict_ud

    # ── val segment-parity setup (fb/lr — what the a25 move_seg head trains
    # on).  Truth streams come from the dataset's precomputed per-episode
    # targets (post smooth/sanitize, i.e. exactly what CE supervises); -100
    # frames break segments via the valid mask.  Preds are gathered chunk by
    # chunk during validation — the val loader is shuffle=False, so chunk
    # order follows val_ds.index and reconstructs episodes exactly.
    val_ep_starts = np.zeros(len(val_eps) + 1, dtype=np.int64)
    val_ep_starts[1:] = np.cumsum([ep.n_frames for ep in val_eps])
    n_val_frames = int(val_ep_starts[-1])
    truth_streams = {
        "fb": np.concatenate(val_ds._fb) if val_ds._fb else np.zeros(0, np.int8),
        "lr": np.concatenate(val_ds._lr) if val_ds._lr else np.zeros(0, np.int8),
    }
    seg_stride: "int | np.ndarray" = max(1, round(cfg.native_hz / 20))
    ep_strides = matched_episode_strides(cfg.data_dir / "precomputed_val",
                                         int(seg_stride))
    if ep_strides is not None and ep_strides.shape[0] == len(val_eps):
        seg_stride = ep_strides       # matched corpus: real per-episode rates
    val_win_id, val_n_windows, val_win_starts = window_ids(val_ep_starts,
                                                           seg_stride)

    def _unpack(batch):
        # Layouts: (feats, fb, lr) | (feats, fb, lr, baseline)
        #          (feats, fb, lr, ud) | (feats, fb, lr, ud, baseline)
        if skip and with_ud:
            feats, fb, lr, ud, baseline = batch
        elif skip:
            feats, fb, lr, baseline = batch
            ud = None
        elif with_ud:
            feats, fb, lr, ud = batch
            baseline = None
        else:
            feats, fb, lr = batch
            ud = None
            baseline = None
        return (
            feats.to(device, non_blocking=True),
            fb.to(device, non_blocking=True),
            lr.to(device, non_blocking=True),
            ud.to(device, non_blocking=True) if ud is not None else None,
            baseline.to(device, non_blocking=True) if baseline is not None else None,
        )

    best_avg = 0.0
    start_epoch = 1
    if cfg.resume is not None:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        # Deliberately skip opt_state and sched_state.  The saved optimizer
        # state carries the *old* cosine LR (near-zero at end of original
        # schedule); loading it would override the fresh scheduler.  Re-init
        # both: fresh AdamW + fresh cosine with new T_max, advanced to the
        # resumed step so we're mid-curve at a useful LR.
        opt = torch.optim.AdamW(model.parameters(),
                                lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        saved_epoch = int(ckpt.get("epoch", 0))
        for _ in range(saved_epoch):
            sched.step()
        # Validate feature spec compatibility — mismatched architecture is a
        # foot-gun; better to fail loudly than to load weights into a
        # mismatched model.
        saved_spec = ckpt.get("feat_spec")
        if saved_spec is not None:
            cur_spec = asdict(cfg.feat_spec)
            for k in ("use_weapon_id",
                      "use_baseline", "use_baseline_skip", "use_gbt_stack",
                      "clip_velocity", "baseline_skip_axes"):
                if saved_spec.get(k) != cur_spec.get(k):
                    raise ValueError(
                        f"resume feat_spec mismatch on '{k}': "
                        f"saved={saved_spec.get(k)} vs current={cur_spec.get(k)}"
                    )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        # best.pt is selected on macro-F1 (`if avg_f1 > best_avg` below), so seed
        # best_avg from the saved F1 keys — NOT val_avg (accuracy 0..100, which
        # would exceed any F1 and freeze best.pt for the whole resumed run).
        best_avg = float(ckpt.get("best_avg_f1", ckpt.get("val_avg_f1", 0.0)))
        if start_epoch > cfg.epochs:
            raise ValueError(
                f"resume target already trained: saved epoch={start_epoch-1} >= "
                f"cfg.epochs={cfg.epochs}; bump --epochs to continue.")
        print(f"  resumed from {cfg.resume}: epoch {start_epoch-1}, "
              f"best_avg={best_avg:.2f}%, continuing through epoch {cfg.epochs}",
              flush=True)
    n_batches = len(train_loader)
    log_every = max(1, n_batches // 50)         # ~50 step lines per epoch
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        win_t, win_i = t0, 0
        loss_accum = torch.zeros((), device=device)   # on-GPU; no per-batch sync
        chunks = 0
        for i, batch in enumerate(train_loader, 1):
            feats, fb, lr, ud, baseline = _unpack(batch)

            with torch.autocast(device_type=device.type, dtype=autocast_dt,
                                enabled=use_autocast):
                out = model(feats, baseline=baseline) if skip else model(feats)
                loss_fb = F.cross_entropy(
                    out["fb"].reshape(-1, N_CLASSES), fb.reshape(-1),
                    weight=w_fb_d, ignore_index=-100,
                ) if cfg.train_axis in ("both", "fb") else torch.tensor(0.0, device=device)
                loss_lr = F.cross_entropy(
                    out["lr"].reshape(-1, N_CLASSES), lr.reshape(-1),
                    weight=w_lr_d, ignore_index=-100,
                ) if cfg.train_axis in ("both", "lr") else torch.tensor(0.0, device=device)
                if with_ud and cfg.train_axis in ("both", "ud"):
                    loss_ud = F.cross_entropy(
                        out["ud"].reshape(-1, N_CLASSES), ud.reshape(-1),
                        weight=w_ud_d, ignore_index=-100,
                    )
                else:
                    loss_ud = torch.tensor(0.0, device=device)
                loss = loss_fb + loss_lr + loss_ud

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            loss_accum += loss.detach()
            chunks   += 1

            if i % log_every == 0 or i == n_batches:
                now = time.time()
                it_s = (i - win_i) / max(now - win_t, 1e-9)
                fr_s = it_s * cfg.batch_size * cfg.chunk_len
                eta  = (n_batches - i) / max(it_s, 1e-9)
                avg  = (loss_accum / max(1, chunks)).item()   # sole sync point
                print(f"  e{epoch:2d} step {i:>5d}/{n_batches}  loss={avg:.4f}  "
                      f"{it_s:5.1f} it/s  {fr_s/1e3:6.0f}k fr/s  eta {eta:4.0f}s",
                      flush=True)
                win_t, win_i = now, i

        loss_sum = float(loss_accum.item())
        sched.step()

        # Validate.  Track per-axis 3x3 confusion matrices so we can
        # report macro-F1 (the right metric for class-imbalanced axes
        # like ud where "always-none" hits 93% accuracy trivially).
        model.eval()
        cm_fb = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        cm_lr = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        cm_ud = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        pred_streams = {"fb": np.zeros(n_val_frames, dtype=np.int8),
                        "lr": np.zeros(n_val_frames, dtype=np.int8)}
        chunk_ptr = 0
        with torch.no_grad():
            for batch in val_loader:
                feats, fb, lr, ud, baseline = _unpack(batch)
                with torch.autocast(device_type=device.type, dtype=autocast_dt,
                                    enabled=use_autocast):
                    out = model(feats, baseline=baseline) if skip else model(feats)
                pred_fb = out["fb"].argmax(-1)
                pred_lr = out["lr"].argmax(-1)
                _accumulate_cm(cm_fb, fb, pred_fb)
                _accumulate_cm(cm_lr, lr, pred_lr)
                if with_ud:
                    pred_ud = out["ud"].argmax(-1)
                    _accumulate_cm(cm_ud, ud, pred_ud)
                # scatter preds back into flat episode streams for the
                # segment-parity gate (chunk order == val_ds.index order)
                pf = pred_fb.cpu().numpy()
                pl = pred_lr.cpu().numpy()
                for b in range(pf.shape[0]):
                    ep_idx, start, vlen = val_ds.index[chunk_ptr + b]
                    off = int(val_ep_starts[ep_idx]) + start
                    pred_streams["fb"][off:off + vlen] = pf[b, :vlen]
                    pred_streams["lr"][off:off + vlen] = pl[b, :vlen]
                chunk_ptr += pf.shape[0]

        # ── 20 Hz segment parity (onset rate + duration-bucket law) ──
        seg20 = {}
        for axis in ("fb", "lr"):
            truth = truth_streams[axis].astype(np.int64)
            valid = truth != -100
            p20 = downsample_axis(pred_streams[axis].astype(np.int64),
                                  val_win_id, val_n_windows)
            t20 = downsample_axis(truth, val_win_id, val_n_windows)
            v20 = window_all_true(valid, val_win_id, val_n_windows)
            seg20[axis] = segment_parity(p20, t20, val_win_starts, valid=v20)

        acc_fb, f1_fb = _acc_and_macro_f1(cm_fb)
        acc_lr, f1_lr = _acc_and_macro_f1(cm_lr)
        if with_ud and cm_ud.sum() > 0:
            acc_ud, f1_ud = _acc_and_macro_f1(cm_ud)
        else:
            acc_ud = f1_ud = None
        avg_acc_axes = [acc_fb, acc_lr] + ([acc_ud] if acc_ud is not None else [])
        avg_f1_axes  = [f1_fb, f1_lr]  + ([f1_ud]  if f1_ud  is not None else [])
        avg_acc = sum(avg_acc_axes) / len(avg_acc_axes)
        avg_f1  = sum(avg_f1_axes)  / len(avg_f1_axes)
        msg = (f"epoch {epoch:2d}/{cfg.epochs}  "
               f"loss={loss_sum / max(1, chunks):.4f}  "
               f"fb=({acc_fb:.2f}/{f1_fb:.3f})  lr=({acc_lr:.2f}/{f1_lr:.3f})  ")
        if acc_ud is not None:
            msg += f"ud=({acc_ud:.2f}/{f1_ud:.3f})  "
        msg += (f"avg-acc={avg_acc:.2f}  avg-F1={avg_f1:.3f}  "
                f"{time.time() - t0:.1f}s")
        print(msg, flush=True)
        seg_bits = []
        for axis in ("fb", "lr"):
            s = seg20[axis]
            ratio = (f"x{s['onset_ratio']:.2f}" if s["onset_ratio"] is not None
                     else "n/a")
            seg_bits.append(f"{axis} onset {ratio} durTV {s['dur_tv']:.3f}")
        print(f"  seg20: {'  |  '.join(seg_bits)}", flush=True)

        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "opt_state":  opt.state_dict(),
            "sched_state": sched.state_dict(),
            "feat_spec": asdict(cfg.feat_spec),
            "feat_dim": cfg.feat_spec.dim,
            "val_fb": acc_fb, "val_lr": acc_lr, "val_avg": avg_acc,
            "val_ud": acc_ud,
            "f1_fb": f1_fb, "f1_lr": f1_lr, "f1_ud": f1_ud,
            "val_avg_f1": avg_f1,
            "seg20": seg20,
            "best_avg_f1": max(best_avg, avg_f1),
            "channels": cfg.channels,
            "n_layers": cfg.n_layers,
            "predict_ud": cfg.predict_ud,
        }
        torch.save(ckpt, cfg.output / "latest.pt")
        if avg_f1 > best_avg:
            best_avg = avg_f1
            torch.save(ckpt, cfg.output / "best.pt")
            print(f"  -> saved best (macro-F1={avg_f1:.3f})")

    print(f"best val macro-F1: {best_avg:.3f}  ->  {cfg.output / 'best.pt'}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir",   type=Path, required=True)
    ap.add_argument("--output",     type=Path, required=True)
    ap.add_argument("--epochs",     type=int,   default=12)
    ap.add_argument("--batch-size", type=int,   default=32)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--chunk-len",  type=int,   default=1024)
    ap.add_argument("--num-workers", type=int,  default=8)
    ap.add_argument("--dtype",      choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--seed",       type=int,   default=17)
    ap.add_argument("--train-axis", default="both",
                    choices=["both", "fb", "lr", "ud"],
                    help="Which axis to train.  'both' = fb+lr (and ud if "
                         "--predict-ud).  'fb'/'lr'/'ud' = single-axis loss "
                         "for separate-encoder runs.")
    ap.add_argument("--clip-velocity", action="store_true",
                    help="Clip body-frame velocity to ±1.0 (suppress server-"
                         "snapshot spike outliers that produce normalized vel "
                         "magnitudes >>1 on rare frames).")
    ap.add_argument("--use-gbt-stack", action="store_true",
                    help="Read per-frame GBT softmax probabilities from "
                         "`<shard>_obs_gbt_probs.npy` and concat as 6 extra "
                         "input features (3 fb + 3 lr).  Requires running "
                         "scripts/gbt_oof_stack.py to generate the shards.")
    ap.add_argument("--channels", type=int, default=64,
                    help="TCN channel count (default 64).  Larger = more "
                         "capacity, slower training.  Try 128 for big corpus.")
    ap.add_argument("--n-layers", type=int, default=6,
                    help="TCN dilated-conv layers (default 6 → ±126 RF). "
                         "7 → ±254 RF, 8 → ±510 RF.")
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="TCN per-layer dropout (default 0.1).")
    ap.add_argument("--resume", type=Path, default=None,
                    help="Path to a checkpoint (.pt) to resume from. Restores "
                         "model + optimizer + scheduler state and continues "
                         "from saved epoch + 1.  Architecture flags must match "
                         "the saved run; --epochs sets the new ceiling.")
    ap.add_argument("--smooth-target", type=int, default=0,
                    help="Width of a per-axis majority-vote window applied to "
                         "the move target on train AND val (centered, edges "
                         "padded with the original value).  0/1 = off; 3 or 5 "
                         "smooths frame-boundary press-transition noise.")
    ap.add_argument("--predict-ud", action="store_true",
                    help="Add a third head for the jump-press axis (ud).  "
                         "ud truth = decode_move_pressbyte(move)[...,2] (press-byte "
                         "ud bits + jump).  Loss for ud is included whenever "
                         "train-axis is 'both' or 'ud'.")
    ap.add_argument("--use-baseline", action="store_true",
                    help="Feed the per-frame sign(velocity) baseline as an "
                         "additional input feature so the model starts from "
                         "the deterministic rule and learns to refine it.")
    ap.add_argument("--uniform-class-weights", action="store_true",
                    help="Disable inverse-frequency class reweighting (use "
                         "all-ones).  Useful to test whether the class-weight "
                         "scheme is suppressing the model's natural fit.")
    ap.add_argument("--use-baseline-skip", action="store_true",
                    help="Feed the per-frame sign(velocity) baseline as a "
                         "skip-connection at the output head (logits = "
                         "encoder + identity*baseline_one_hot) instead of "
                         "concatenating it into the encoder input.  Encoder "
                         "sees only the 9 core features and learns the "
                         "residual from the baseline.")
    ap.add_argument("--baseline-skip-init-scale", type=float, default=3.0,
                    help="Scale of the identity init on the baseline skip "
                         "projection.  Larger = stronger prior at init (slower "
                         "to escape); smaller = weaker prior, less warmup.")
    ap.add_argument("--baseline-skip-axes", default="fb_lr",
                    choices=["fb_lr", "fb", "lr"],
                    help="Which axes get the baseline skip-connection (only "
                         "applies when --use-baseline-skip is on).  'fb_lr' "
                         "(default): both axes.  'lr': lr only (fb head is "
                         "encoder-only).  'fb': fb only.")
    ap.add_argument("--use-weapon-id", action="store_true",
                    help="One-hot the server-held weapon_id (9 dims: 0=none, "
                         "1..8 = axe..LG) and append to the encoder input.")
    ap.add_argument("--sanitize-targets", action="store_true",
                    help="Replace per-tick targets with ignore_index=-100 on "
                         "ticks where the player's input on that axis was "
                         "a no-op (engine-rejected jump-while-airborne, "
                         "fire-in-cooldown, etc.).  Uses the per-axis "
                         "`act/op_input` bitmask emitted by the C worker "
                         "(bit0=fb, bit1=lr, bit2=ud; 1=keep, 0=drop). "
                         "No-op for shards that don't carry the sidecar.")
    ap.add_argument("--max-train-frames", type=int, default=0,
                    help="Cap training to at most N frames (whole episodes, "
                         "shuffled by --seed, cap-first). 0 = all. For "
                         "data-scaling / learning-curve sweeps.")
    ap.add_argument("--native-hz", type=int, default=77,
                    help="Native collect rate; the val segment-parity gate "
                         "downsamples to 20 Hz with stride = round(native_hz/20) "
                         "(default 77 → stride 4), matching the GBT report.")
    args = ap.parse_args()

    spec = FeatureSpec(
        use_weapon_id=args.use_weapon_id,
        use_baseline=args.use_baseline,
        use_baseline_skip=args.use_baseline_skip,
        baseline_skip_axes=args.baseline_skip_axes,
        clip_velocity=args.clip_velocity,
        use_gbt_stack=args.use_gbt_stack,
    )
    cfg = TrainConfig(
        data_dir   =args.data_dir,
        output     =args.output,
        feat_spec  =spec,
        chunk_len  =args.chunk_len,
        epochs     =args.epochs,
        batch_size =args.batch_size,
        lr         =args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        dtype      =args.dtype,
        seed       =args.seed,
        uniform_class_weights=args.uniform_class_weights,
        baseline_skip_init_scale=args.baseline_skip_init_scale,
        train_axis=args.train_axis,
        channels   =args.channels,
        n_layers   =args.n_layers,
        dropout    =args.dropout,
        resume     =args.resume,
        smooth_target=args.smooth_target,
        predict_ud =args.predict_ud,
        sanitize_targets=args.sanitize_targets,
        max_train_frames=args.max_train_frames,
        native_hz  =args.native_hz,
    )
    train(cfg)


if __name__ == "__main__":
    main()
