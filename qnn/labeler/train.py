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

from .model import (
    CORE_FEAT_DIM,
    FeatureSpec,
    MoveLabeler,
    N_CLASSES,
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
    # When set, per-tick targets are CE-masked using the engine-effective
    # bitmask sidecar (`obs/target_valid_mask`) emitted by the C worker.
    # Bit i of the mask corresponds to axis i in {fb, lr, ud, fire,
    # weapon_switch}; a 0 bit means the press at that tick had no
    # observable engine effect (e.g., jump-while-airborne, fire-in-
    # cooldown) and the target is replaced with ignore_index=-100 so CE
    # skips it.  Silently no-op for shards collected before
    # LOBS_FRAME_SIZE = 31 (no mask available — falls through unmodified).
    sanitize_targets: bool = False


# ── episode loading (mmap-backed, lazy) ───────────────────────────────────────

@dataclass
class _EpisodeView:
    """mmap-backed slices into shard arrays for one episode."""
    self_velocity:    np.ndarray
    self_movement_id: np.ndarray
    look:             np.ndarray
    move:             np.ndarray
    c_rule_fire:      np.ndarray | None
    c_rule_jump:      np.ndarray | None
    gbt_probs:        np.ndarray | None
    # Per-tick engine-effective bitmask emitted by the C worker (bit0=fb,
    # bit1=lr, bit2=ud, bit3=fire, bit4=weapon_switch).  Optional; older
    # shards collected with LABELER_FRAME_SIZE=30 won't have this array
    # and the trainer falls back to no masking even with --sanitize-targets.
    target_valid_mask: np.ndarray | None
    n_frames:         int


def _load_split(split_dir: Path) -> list[_EpisodeView]:
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    episodes: list[_EpisodeView] = []

    for shard in manifest["shards"]:
        obs = shard["obs"]
        acts = shard["actions"]

        vel  = np.load(split_dir / obs["self_velocity"],    mmap_mode="r")
        mid  = np.load(split_dir / obs["self_movement_id"], mmap_mode="r")
        look = np.load(split_dir / obs["look"],             mmap_mode="r")
        move = np.load(split_dir / acts["move"])  # uint8, small — load fully
        fire = (np.load(split_dir / obs["c_rule_fire"], mmap_mode="r")
                if "c_rule_fire" in obs else None)
        jump = (np.load(split_dir / obs["c_rule_jump"], mmap_mode="r")
                if "c_rule_jump" in obs else None)
        tvm  = (np.load(split_dir / obs["target_valid_mask"], mmap_mode="r")
                if "target_valid_mask" in obs else None)
        # GBT stacking probs are an optional sidecar written by
        # scripts/gbt_oof_stack.py.  Path convention: same prefix as the
        # other obs arrays, suffix `_obs_gbt_probs.npy`.
        gbt_rel = obs.get("gbt_probs")
        if gbt_rel is None:
            # Backward-compat: derive from shard arrays' prefix.
            vel_rel = obs["self_velocity"]
            guess = vel_rel.replace("_obs_self_velocity", "_obs_gbt_probs")
            if (split_dir / guess).exists():
                gbt_rel = guess
        gbt  = (np.load(split_dir / gbt_rel, mmap_mode="r")
                if gbt_rel is not None else None)

        start = 0
        for length in shard["episode_lengths"]:
            stop = start + int(length)
            episodes.append(_EpisodeView(
                self_velocity   =vel[start:stop],
                self_movement_id=mid[start:stop],
                look            =look[start:stop],
                move            =move[start:stop],
                c_rule_fire     =fire[start:stop] if fire is not None else None,
                c_rule_jump     =jump[start:stop] if jump is not None else None,
                gbt_probs       =gbt[start:stop]  if gbt  is not None else None,
                target_valid_mask=tvm[start:stop] if tvm  is not None else None,
                n_frames        =stop - start,
            ))
            start = stop
    return episodes


# ── chunked dataset ───────────────────────────────────────────────────────────

def _smooth_move_target(move_packed: np.ndarray, window: int) -> np.ndarray:
    """Per-axis majority-vote smoothing of the packed move target.

    For each tick t, replace the per-axis class with the mode over
    [t - window//2 .. t + window//2], clamped to episode bounds.  Window
    sizes of 0 or 1 are a no-op (no smoothing).  Reduces frame-boundary
    transition noise (e.g., button-just-released ticks) at the cost of
    blurring labels near genuine press boundaries.
    """
    if window <= 1:
        return move_packed
    half = window // 2
    arr = np.asarray(move_packed, dtype=np.uint8).reshape(-1)
    n = arr.shape[0]
    fb = (arr & 0x3).astype(np.int32)
    lr = ((arr >> 2) & 0x3).astype(np.int32)
    ud = ((arr >> 4) & 0x3).astype(np.int32)
    out = arr.copy()
    for t in range(n):
        lo = max(0, t - half)
        hi = min(n, t + half + 1)
        # Mode via bincount per axis.
        new_fb = int(np.bincount(fb[lo:hi], minlength=3).argmax())
        new_lr = int(np.bincount(lr[lo:hi], minlength=3).argmax())
        new_ud = int(np.bincount(ud[lo:hi], minlength=3).argmax())
        out[t] = (new_fb & 0x3) | ((new_lr & 0x3) << 2) | ((new_ud & 0x3) << 4)
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
        # Pre-smooth per-episode targets so __getitem__ stays cheap.
        if self.smooth_target > 1:
            self._smoothed_moves = [
                _smooth_move_target(np.asarray(ep.move), self.smooth_target)
                for ep in episodes
            ]
        else:
            self._smoothed_moves = None
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

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        ep_idx, start, valid_len = self.index[idx]
        ep = self.episodes[ep_idx]
        end = start + valid_len
        L = self.chunk_len
        F_dim = self.spec.dim

        feats_np = build_features(
            ep.self_velocity[start:end],
            ep.self_movement_id[start:end],
            ep.look[start:end],
            c_rule_fire=ep.c_rule_fire[start:end] if ep.c_rule_fire is not None else None,
            c_rule_jump=ep.c_rule_jump[start:end] if ep.c_rule_jump is not None else None,
            gbt_probs  =ep.gbt_probs[start:end]   if ep.gbt_probs   is not None else None,
            spec=self.spec,
        )
        move_src = (self._smoothed_moves[ep_idx] if self._smoothed_moves is not None
                    else np.asarray(ep.move))
        if self.with_ud:
            decoded = decode_move_fb_lr_ud(move_src[start:end])
        else:
            decoded = decode_move_fb_lr(move_src[start:end])

        feats = torch.zeros(L, F_dim, dtype=torch.float32)
        fb    = torch.full((L,), -100, dtype=torch.long)
        lr    = torch.full((L,), -100, dtype=torch.long)
        feats[:valid_len] = torch.from_numpy(feats_np)
        fb[:valid_len]    = torch.from_numpy(decoded[:, 0])
        lr[:valid_len]    = torch.from_numpy(decoded[:, 1])
        if self.with_ud:
            ud = torch.full((L,), -100, dtype=torch.long)
            ud[:valid_len] = torch.from_numpy(decoded[:, 2])
        else:
            ud = None

        # Engine-effectiveness mask: per-tick bits emitted by the C
        # worker.  When --sanitize-targets is set and the sidecar exists,
        # replace ineffective-press targets with -100 so CE skips them.
        # The labeler isn't penalized for failing to predict presses
        # that have no observable kinematic effect (engine-rejected jumps,
        # in-cooldown fires).  When the sidecar is missing this is a no-op.
        if self.sanitize_targets and ep.target_valid_mask is not None:
            mask = np.asarray(ep.target_valid_mask[start:end], dtype=np.uint8)
            fb_ok = torch.from_numpy(((mask & 0x01) != 0).astype(np.bool_))
            lr_ok = torch.from_numpy(((mask & 0x02) != 0).astype(np.bool_))
            fb[:valid_len] = torch.where(fb_ok, fb[:valid_len],
                                         torch.full_like(fb[:valid_len], -100))
            lr[:valid_len] = torch.where(lr_ok, lr[:valid_len],
                                         torch.full_like(lr[:valid_len], -100))
            if self.with_ud:
                ud_ok = torch.from_numpy(((mask & 0x04) != 0).astype(np.bool_))
                ud[:valid_len] = torch.where(ud_ok, ud[:valid_len],
                                             torch.full_like(ud[:valid_len], -100))

        if self.spec.use_baseline_skip:
            # Baseline one-hot (fb, lr) computed from velocity, fed as a
            # separate input that bypasses the trunk.
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

    all_moves = np.concatenate([np.asarray(ep.move) for ep in episodes])
    fb = all_moves & 0x3
    lr = (all_moves >> 2) & 0x3
    ud = (all_moves >> 4) & 0x3

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
            for k in ("is_firing", "is_jumping", "use_baseline",
                      "use_baseline_skip", "use_gbt_stack",
                      "clip_velocity", "baseline_skip_axes"):
                if saved_spec.get(k) != cur_spec.get(k):
                    raise ValueError(
                        f"resume feat_spec mismatch on '{k}': "
                        f"saved={saved_spec.get(k)} vs current={cur_spec.get(k)}"
                    )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_avg = float(ckpt.get("best_avg", ckpt.get("val_avg", 0.0)))
        if start_epoch > cfg.epochs:
            raise ValueError(
                f"resume target already trained: saved epoch={start_epoch-1} >= "
                f"cfg.epochs={cfg.epochs}; bump --epochs to continue.")
        print(f"  resumed from {cfg.resume}: epoch {start_epoch-1}, "
              f"best_avg={best_avg:.2f}%, continuing through epoch {cfg.epochs}",
              flush=True)
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = chunks = 0
        for batch in train_loader:
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
            loss_sum += float(loss.item())
            chunks   += 1

        sched.step()

        # Validate.  Track per-axis 3x3 confusion matrices so we can
        # report macro-F1 (the right metric for class-imbalanced axes
        # like ud where "always-none" hits 93% accuracy trivially).
        model.eval()
        cm_fb = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        cm_lr = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        cm_ud = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
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
                         "for separate-trunk runs.")
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
                         "ud truth = (move_packed >> 4) & 0x3.  Loss for ud "
                         "is included whenever train-axis is 'both' or 'ud'.")
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
                         "trunk + identity*baseline_one_hot) instead of "
                         "concatenating it into the trunk input.  Trunk "
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
                         "trunk-only).  'fb': fb only.")
    ap.add_argument("--use-fire",   action="store_true",
                    help="Include c_rule_fire as input feature (requires it in collect output)")
    ap.add_argument("--use-jump",   action="store_true",
                    help="Include c_rule_jump as input feature (requires it in collect output)")
    ap.add_argument("--sanitize-targets", action="store_true",
                    help="Replace per-tick targets with ignore_index=-100 on "
                         "ticks where the press had no observable engine "
                         "effect (engine-rejected jump-while-airborne, "
                         "fire-in-cooldown, etc.).  Uses the per-axis "
                         "`obs/target_valid_mask` bitmask emitted by the C "
                         "worker (LOBS bit0=fb, bit1=lr, bit2=ud).  No-op "
                         "for shards collected before LABELER_FRAME_SIZE=31.")
    args = ap.parse_args()

    spec = FeatureSpec(
        is_firing=args.use_fire,
        is_jumping=args.use_jump,
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
    )
    train(cfg)


if __name__ == "__main__":
    main()
