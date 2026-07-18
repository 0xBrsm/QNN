"""Shared, torch-free data layer for the move labeler.

This module owns the *numpy* half of the labeler pipeline — the parts that
both the torch TCN trainer (``qnn.labeler.train``) and the lightgbm GBT
trainer (``qnn.labeler.gbt``) need:

  * the feature spec + ``build_features`` (vel / movement_id / look →
    ``(T, F)`` float32 matrix),
  * the press-byte → per-axis-class decode wrappers,
  * mmap-backed episode loading from a labeler corpus (``_load_split``),
  * ``materialize_split`` — the flat ``(N, F)`` / ``(N, 3)`` / ``(N,)``
    feature / label / keep-mask construction, the same featurize + decode +
    op_input keep-mask logic the ``ChunkedDataset`` applies per chunk, but
    materialized once over the whole split.

Keeping this torch-free lets the GBT path stay light and fast (no torch
import). ``qnn.labeler.model`` re-exports the feature/spec/decode names from
here so existing call sites (and the torch ``MoveLabeler``) are unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qnn.actions import (
    MOVE_CLASS_NONE,
    decode_move_pressbyte,
)

# ── feature layout (was qnn.labeler.model) ─────────────────────────────────────
#
# Storage (per-frame npy cache, ~14 B/frame core):
#   self_velocity     fp16/i16 (T, 3)  body-frame, pre-normalized by C worker
#   self_movement_id  uint8 (T,)       0=ground, 1=air, 2+=water
#   look              fp16 (T, 3)      per-emit view delta (forward dot anchor_basis)
#   act_move (target) uint8 (T,)       packed (fb | lr<<2 | ud<<4) press byte

VELOCITY_SCALE = 2000.0    # informational; matches QNN_VELOCITY_SCALE
CORE_FEAT_DIM  = 9         # vel(3) + mid_oh(3) + look(3)
BASELINE_DIM   = 6         # per-axis one-hot {neg, none, pos} for fb + lr
BASELINE_EPS   = 0.01      # normalized 20 u/s — same threshold as the bakeoff B variant
GBT_STACK_DIM  = 6         # per-axis softmax probs from GBT (fb 3 + lr 3) as TCN inputs
WEAPON_OH_DIM  = 9         # one-hot of weapon_id ∈ {0..8} (0=none, 1=axe..8=LG)
VEL_CLIP       = 1.0       # clip body-frame normalized vel to [-1, 1] (matches QW max ~700 u/s)

N_CLASSES = 3              # {0: neg, 1: none, 2: pos}


@dataclass(frozen=True)
class FeatureSpec:
    use_weapon_id:     bool = False   # one-hot of server-held weapon (9 dims)
    use_baseline:      bool = False   # per-frame sign(velocity) baseline concatenated into input
    use_baseline_skip: bool = False   # baseline bypasses encoder and adds directly to output logits
    # When use_baseline_skip is set, restrict the skip to specific axes.
    # Empty means both fb and lr.  Values: "fb", "lr", "fb_lr".
    baseline_skip_axes: str = "fb_lr"
    clip_velocity:     bool = False   # clip body-frame normalized vel to ±VEL_CLIP
    use_gbt_stack:     bool = False   # add 6 GBT softmax probs as input features

    @property
    def dim(self) -> int:
        return (CORE_FEAT_DIM
            + (WEAPON_OH_DIM if self.use_weapon_id else 0)
            + (BASELINE_DIM if self.use_baseline else 0)
            + (GBT_STACK_DIM if self.use_gbt_stack else 0))

    @property
    def skip_fb(self) -> bool:
        return self.use_baseline_skip and "fb" in self.baseline_skip_axes

    @property
    def skip_lr(self) -> bool:
        return self.use_baseline_skip and "lr" in self.baseline_skip_axes


# ── feature build ─────────────────────────────────────────────────────────────

def build_features(
    self_velocity: np.ndarray,                      # (T, 3)
    self_movement_id: np.ndarray,                   # (T,)
    look: np.ndarray,                               # (T, 3)
    *,
    weapon_id: np.ndarray | None = None,            # (T,) uint8 ∈ {0..8}
    gbt_probs: np.ndarray | None = None,            # (T, 6) softmax probs (fb 3 + lr 3)
    spec: FeatureSpec | None = None,
) -> np.ndarray:
    """(T, spec.dim) float32 input matrix.

    All inputs are already in the right scale: self_velocity is body-frame
    pre-normalized by the C worker; look is unit-norm by construction.
    """
    if spec is None:
        spec = FeatureSpec(
            use_weapon_id=weapon_id is not None,
            use_gbt_stack=gbt_probs is not None,
        )

    vel = np.asarray(self_velocity, dtype=np.float32)
    if spec.clip_velocity:
        vel = np.clip(vel, -VEL_CLIP, VEL_CLIP)
    mid = np.asarray(self_movement_id, dtype=np.int32).reshape(-1)
    mid_oh = np.stack([
        (mid == 0).astype(np.float32),
        (mid == 1).astype(np.float32),
        (mid >= 2).astype(np.float32),
    ], axis=1)
    lk = np.asarray(look, dtype=np.float32)

    parts: list[np.ndarray] = [vel, mid_oh, lk]

    if spec.use_weapon_id:
        if weapon_id is None:
            raise ValueError("spec.use_weapon_id set but weapon_id is None")
        wid = np.asarray(weapon_id, dtype=np.int32).reshape(-1)
        wid = np.clip(wid, 0, WEAPON_OH_DIM - 1)
        weapon_oh = np.zeros((wid.shape[0], WEAPON_OH_DIM), dtype=np.float32)
        weapon_oh[np.arange(wid.shape[0]), wid] = 1.0
        parts.append(weapon_oh)
    if spec.use_baseline:
        # Per-axis sign-of-velocity baseline (fb, lr) as 2x one-hot {neg, none, pos}.
        baseline = _baseline_classes_from_vel(vel, eps=BASELINE_EPS)  # (T, 2) {0,1,2}
        baseline_oh = np.zeros((vel.shape[0], BASELINE_DIM), dtype=np.float32)
        for axis in range(2):
            for cls in range(3):
                baseline_oh[:, axis * 3 + cls] = (baseline[:, axis] == cls).astype(np.float32)
        parts.append(baseline_oh)
    if spec.use_gbt_stack:
        if gbt_probs is None:
            raise ValueError("spec.use_gbt_stack set but gbt_probs is None")
        parts.append(np.asarray(gbt_probs, dtype=np.float32).reshape(-1, GBT_STACK_DIM))

    feats = np.concatenate(parts, axis=1).astype(np.float32)
    assert feats.shape[1] == spec.dim, f"feat dim {feats.shape[1]} != spec.dim {spec.dim}"
    return feats


def _baseline_classes_from_vel(vel: np.ndarray, eps: float = BASELINE_EPS) -> np.ndarray:
    """Per-frame sign-of-velocity baseline: (T, 3) -> (T, 2) {0=neg, 1=none, 2=pos}.

    Only fb (axis 0) and lr (axis 1) — same axes the labeler predicts.  ud is
    handled by the C-rule jump detector.
    """
    v = np.asarray(vel, dtype=np.float32)
    fb = np.where(v[:, 0] >  eps, 2, np.where(v[:, 0] < -eps, 0, 1))
    lr = np.where(v[:, 1] >  eps, 2, np.where(v[:, 1] < -eps, 0, 1))
    return np.stack([fb, lr], axis=1).astype(np.int64)


# ── target decode ─────────────────────────────────────────────────────────────

def decode_move_fb_lr(packed: np.ndarray) -> np.ndarray:
    """uint8 press-byte -> (N, 2) int64 [fb, lr] in {0=neg, 1=none, 2=pos}."""
    return decode_move_pressbyte(packed)[:, :2].astype(np.int64)


def decode_move_fb_lr_ud(packed: np.ndarray) -> np.ndarray:
    """uint8 press-byte -> (N, 3) int64 [fb, lr, ud] in {0=neg, 1=none, 2=pos}."""
    return decode_move_pressbyte(packed)[:, :3].astype(np.int64)


# ── episode loading (mmap-backed, lazy) ───────────────────────────────────────

@dataclass
class _EpisodeView:
    """mmap-backed slices into shard arrays for one episode."""
    self_velocity:    np.ndarray
    self_movement_id: np.ndarray
    look:             np.ndarray
    move:             np.ndarray
    weapon_id:        np.ndarray | None
    gbt_probs:        np.ndarray | None
    # Per-tick operative-input bitmask emitted by the C worker (bit0=fb,
    # bit1=lr, bit2=ud, bit3=fire, bit4=impulse).  1 = engine acted on this
    # axis's input this tick (keep in loss); 0 = no-op (drop).  Optional;
    # older shards won't carry the sidecar and the keep-mask falls back to
    # all-keep.
    op_input: np.ndarray | None
    n_frames:         int


def _load_split(split_dir: Path) -> list[_EpisodeView]:
    """Load mmap-backed episode views from a labeler corpus.

    The labeler corpus is a field-selected QOBS collect (see
    qnn.labeler.collect), so the on-disk field names are the QOBS-native
    ones written by qnn.bc.collect's ShardWriter:
      obs  : vel (i16 ×3), self_movement_id (u8), self_weapon_id (u8)
      act  : move (u8 press byte), look (f16 ×3), op_input (u8)
    """
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    episodes: list[_EpisodeView] = []

    for shard in manifest["shards"]:
        obs = shard["obs"]
        acts = shard["actions"]

        # vel is i16 view-frame raw Quake units (QNN_VELOCITY_SCALE-scaled
        # model-side); the labeler feature builder pre-normalizes, so divide
        # to body-frame normalized units here.
        vel  = (np.load(split_dir / obs["vel"], mmap_mode="r").astype(np.float32)
                / VELOCITY_SCALE)
        mid  = np.load(split_dir / obs["self_movement_id"], mmap_mode="r")
        look = np.load(split_dir / acts["look"], mmap_mode="r")
        move = np.load(split_dir / acts["move"])  # uint8, small — load fully
        wid  = (np.load(split_dir / obs["self_weapon_id"], mmap_mode="r")
                if "self_weapon_id" in obs else None)
        opi  = (np.load(split_dir / acts["op_input"], mmap_mode="r")
                if "op_input" in acts else None)
        # GBT stacking probs are an optional sidecar written by
        # scripts/gbt_oof_stack.py.  Path convention: same prefix as the
        # other obs arrays, suffix `_obs_gbt_probs.npy`.
        gbt_rel = obs.get("gbt_probs")
        if gbt_rel is None:
            vel_rel = obs["vel"]
            guess = vel_rel.replace("_obs_vel", "_obs_gbt_probs")
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
                weapon_id       =wid[start:stop]  if wid  is not None else None,
                gbt_probs       =gbt[start:stop]  if gbt  is not None else None,
                op_input        =opi[start:stop] if opi  is not None else None,
                n_frames        =stop - start,
            ))
            start = stop
    return episodes


# ── matched-corpus per-episode 20 Hz strides ──────────────────────────────────

def matched_episode_strides(
    slim_split_dir: Path,
    default_stride: int,
) -> np.ndarray | None:
    """Per-episode 20 Hz window strides for a matched-collect slim split.

    Demos record at their native client rate (77 Hz and 60 Hz are both
    common), so a single global ``round(native_hz / 20)`` stride puts half a
    mixed corpus in the wrong duration units.  The matched collect's qobs
    twin carries ``native_index`` per 20 Hz frame — its per-episode median
    delta IS that episode's native-frames-per-model-frame.

    Joins slim episodes to qobs episodes via the manifests' ``demo_idxs``
    and returns an ``(E,)`` int64 array aligned with ``_load_split`` /
    ``materialize_split`` episode order.  Returns None when there is no
    paired qobs split (plain labeler corpora); demos missing from the qobs
    manifest fall back to ``default_stride``.
    """
    slim_split_dir = Path(slim_split_dir)
    if slim_split_dir.parent.name != "slim":
        return None
    qobs_split = slim_split_dir.parent.parent / "qobs" / slim_split_dir.name
    slim_manifest = slim_split_dir / "manifest.json"
    qobs_manifest = qobs_split / "manifest.json"
    if not (slim_manifest.exists() and qobs_manifest.exists()):
        return None

    # demo_idx -> per-episode median native_index deltas across qobs episodes
    per_demo: dict[int, list[float]] = {}
    qman = json.loads(qobs_manifest.read_text())
    for shard in qman["shards"]:
        ni_rel = (shard.get("obs") or {}).get("native_index")
        if ni_rel is None:
            return None
        ni = np.load(qobs_split / ni_rel, mmap_mode="r")
        start = 0
        for length, demo_idx in zip(shard["episode_lengths"], shard["demo_idxs"]):
            stop = start + int(length)
            d = np.diff(ni[start:stop].astype(np.int64))
            d = d[d > 0]
            if d.size:
                per_demo.setdefault(int(demo_idx), []).append(float(np.median(d)))
            start = stop

    sman = json.loads(slim_manifest.read_text())
    strides: list[int] = []
    for shard in sman["shards"]:
        for demo_idx in shard["demo_idxs"]:
            eps = per_demo.get(int(demo_idx))
            s = float(np.median(eps)) if eps else float(default_stride)
            strides.append(max(1, int(round(s))))
    return np.asarray(strides, dtype=np.int64)


# ── op_input keep-mask ─────────────────────────────────────────────────────────

def op_input_keep_mask(decoded: np.ndarray, op_input: np.ndarray | None,
                       with_ud: bool = True) -> np.ndarray:
    """Per-axis keep mask matching ``ChunkedDataset``'s sanitize logic.

    Returns ``(T, n_axes)`` bool where n_axes is 3 if ``with_ud`` else 2.
    A frame/axis is KEPT when the player did not press that axis (the "none"
    class, always legitimate training data) OR the C worker's op_input bit
    for that axis is set (the press had an observable engine effect).

    ``op_input`` bit layout: bit0=fb, bit1=lr, bit2=ud.  When ``op_input`` is
    None (older shards lack the sidecar) every frame/axis is kept.
    """
    n_axes = 3 if with_ud else 2
    T = decoded.shape[0]
    if op_input is None:
        return np.ones((T, n_axes), dtype=np.bool_)
    mask = np.asarray(op_input, dtype=np.uint8).reshape(-1)
    bits = (0x01, 0x02, 0x04)
    keep = np.empty((T, n_axes), dtype=np.bool_)
    for axis in range(n_axes):
        no_press = decoded[:, axis] == MOVE_CLASS_NONE
        keep[:, axis] = no_press | ((mask & bits[axis]) != 0)
    return keep


# ── flat materialization (shared by GBT path) ──────────────────────────────────

@dataclass
class MaterializedSplit:
    """Flat, episode-concatenated feature / label / mask matrices.

    X            : (N, F) float32  — build_features output, episode-concatenated
    Y            : (N, 3) uint8    — per-axis class {0=neg,1=none,2=pos} (fb,lr,ud)
    mask         : (N, 3) bool     — op_input keep mask per axis (True = keep)
    episode_index: (N,)   int32    — episode id per row (for boundary-aware
                                     downsampling); contiguous 0..E-1
    episode_starts: (E+1,) int64   — row offsets so episode e is
                                     rows[episode_starts[e]:episode_starts[e+1]]
    """
    X: np.ndarray
    Y: np.ndarray
    mask: np.ndarray
    episode_index: np.ndarray
    episode_starts: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_episodes(self) -> int:
        return int(self.episode_starts.shape[0] - 1)


def materialize_split(
    split_dir: Path,
    spec: FeatureSpec,
    *,
    episodes: list[_EpisodeView] | None = None,
) -> MaterializedSplit:
    """Materialize a labeler split into flat (N, F) features / (N, 3) labels /
    (N, 3) keep-mask, with an episode index for boundary-aware downsampling.

    Goes through the same ``build_features`` + ``decode_move_fb_lr_ud`` +
    op_input keep-mask code the TCN ``ChunkedDataset`` uses, so the GBT and
    TCN paths share one featurize/decode/mask implementation.

    Pass ``episodes`` to reuse an already-loaded split; otherwise it is loaded
    via ``_load_split(split_dir)``.
    """
    if episodes is None:
        episodes = _load_split(split_dir)
    episodes = [ep for ep in episodes if ep.n_frames > 0]
    if not episodes:
        empty_f = np.zeros((0, spec.dim), dtype=np.float32)
        return MaterializedSplit(
            X=empty_f,
            Y=np.zeros((0, 3), dtype=np.uint8),
            mask=np.zeros((0, 3), dtype=np.bool_),
            episode_index=np.zeros((0,), dtype=np.int32),
            episode_starts=np.zeros((1,), dtype=np.int64),
        )

    X_parts: list[np.ndarray] = []
    Y_parts: list[np.ndarray] = []
    M_parts: list[np.ndarray] = []
    idx_parts: list[np.ndarray] = []
    starts = [0]

    for ep_idx, ep in enumerate(episodes):
        feats = build_features(
            ep.self_velocity,
            ep.self_movement_id,
            ep.look,
            weapon_id=ep.weapon_id if ep.weapon_id is not None else None,
            gbt_probs=ep.gbt_probs if ep.gbt_probs is not None else None,
            spec=spec,
        )
        decoded = decode_move_fb_lr_ud(np.asarray(ep.move))  # (T, 3) int64
        keep = op_input_keep_mask(decoded, ep.op_input, with_ud=True)  # (T, 3) bool
        X_parts.append(feats)
        Y_parts.append(decoded.astype(np.uint8))
        M_parts.append(keep)
        idx_parts.append(np.full(ep.n_frames, ep_idx, dtype=np.int32))
        starts.append(starts[-1] + ep.n_frames)

    return MaterializedSplit(
        X=np.concatenate(X_parts, axis=0),
        Y=np.concatenate(Y_parts, axis=0),
        mask=np.concatenate(M_parts, axis=0),
        episode_index=np.concatenate(idx_parts, axis=0),
        episode_starts=np.asarray(starts, dtype=np.int64),
    )
