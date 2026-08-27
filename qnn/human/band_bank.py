"""Human-band corpus WINDOW-FEATURE BANK — the model-agnostic per-collect baseline
the human-band membership test scores against.

This is the corpus-derived half of the human-band machinery: from a collect's demo
episodes it builds a bank of per-window behavior-feature vectors (look / move / attack /
weapon), tagged by demo so the scorer can build demo-level split nulls. It depends only
on the raw collect corpus + the torch-free feature primitives in
``qnn.eval.humanlikeness.core`` — a corpus baseline, computed once per collect, cached
beside the other ``qnn.human`` baselines under ``<collect>/human_baseline/``.

The MMD²/null/subject-scoring (model-specific, eval-time) lives in
``qnn.eval.humanlikeness.human_band``; it imports the bank from here. See
``research/human-band.md`` for the methodology.

v5 (harness revision; the MMD²/null statistic is unchanged):

* ENGAGEMENT CONDITIONING (Axiom 2 — style is judged GIVEN context): windows are
  still cut on 15 s wall-clock, but every channel's features are computed on
  keep ∧ engaged frames, where engaged = ≥1 LOS actor this frame
  (entity_types == TOKEN_ACTOR & recency == 0 — the LOS half of
  ``qnn.human.op_attack``'s engaged-LOS). A window enters the bank only if its
  engaged mass ≥ max(25% of the window, 2 s) — the v4 keep rule's shape, moved
  onto the conditioned mask (measured survival ≈ 2.7k windows on the qwd val
  split; comfortably above starvation). The null and the anchor are built from
  the same conditioning.

* ATTACK ON DISCHARGES (Axiom 1 — perfect imitation must score human): the
  attack channel is computed from ENGINE-VISIBLE discharge events, never
  button/press semantics. Human discharge = (move bit0) & (input_mask bit0)
  — the op-attack decision frame, cooldown-stamped by the engine model; model
  discharge = attack action ∧ cooldown-ready (see human_band.load_rc_episodes).
  Features per window, on keep ∧ engaged frames:
    - disch_per_engaged_s : discharges / engaged-second
    - gap_excess_med_s / gap_excess_p90_s : median / p90 of
      (gap_i − COOLDOWN_SEC[weapon_i]) pooled over gaps between consecutive
      discharges within unbroken (keep ∧ engaged ∧ same-held-weapon) runs —
      the discharging weapon's engine cooldown floor is removed, so the
      features are weapon-mix-robust and commensurate across held triggers
      (human) and single-tick decisions (model). Right-censored at the window
      length when no gap sample exists (<2 discharges, or discharges split
      across runs).

* SHUFFLED-HUMAN ANCHOR (Axiom 3 — effect-size decisions): the bank artifact
  carries a frame-shuffled held-out-human window bank (fixed 40-demo holdout,
  seed 17) under the ``_anchor`` key. The scorer turns it into a per-channel
  MMD² denominator so verdicts are anchored ratios, not significance bars.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from qnn.eval.aim_kernel import action_attack_context, iter_shard_episodes
from qnn.eval.humanlikeness.core import dwell_times, switch_rate
from qnn.eval.humanlikeness.human_reference import _turn_deg, _unpack_move
from qnn.vocab import TOKEN_ACTOR
from qnn.weapons import COOLDOWN_SEC

BANK_VERSION = 5
HUMAN_HZ = 20.0
FLICK_DEG = 15.0
ZERO_TURN_DEG = 1e-3  # float32 arccos noise floor; human exact-holds land here

# Engaged-mass admission rule for a window: ≥ max(ENGAGED_MIN_FRAC · window,
# ENGAGED_MIN_SEC) frames of keep ∧ engaged. Same shape as the v4 25%-keep
# rule, applied to the conditioned mask (Axiom 2).
ENGAGED_MIN_FRAC = 0.25
ENGAGED_MIN_SEC = 2.0

# Shuffled-human anchor (cached in the bank artifact; Axiom 3).
ANCHOR_HOLDOUT_DEMOS = 40
ANCHOR_SEED = 17

CHANNELS = ("look", "move", "attack", "weapon")
FEATURE_NAMES = {
    "look": ("turn_mean_dps", "turn_p90_dps", "zero_frac", "flick_per_s",
             "zero_run_med_s", "turn_lag1_ac"),
    "move": ("sw_fb_ps", "sw_lr_ps", "occ_fb", "occ_lr",
             "dwell_med_fb_s", "dwell_med_lr_s"),
    "attack": ("disch_per_engaged_s", "gap_excess_med_s", "gap_excess_p90_s"),
    "weapon": ("sw_ps", "occ_entropy"),
}

# Episode dict schema (all per-frame, equal length):
#   fb/lr      int8  move class {0,1,2}
#   attack     int8  raw attack button/head bit (NOT a feature input — kept for
#                    the calibration perturbations / label re-encoding)
#   weapon     int8  weapon label / command byte (impulse space)
#   wimp       int8  HELD weapon impulse 0..8 (engine state at each frame)
#   discharge  bool  engine-visible discharge event (op-attack decision frame)
#   turn       f8    per-frame turn magnitude, degrees
#   keep       bool  validity mask (in-distribution / actor-visible)
#   engaged    bool  ≥1 LOS actor this frame


# ---------------------------------------------------------------------------
# Episode loading (collect cache)
# ---------------------------------------------------------------------------
def _los_actor_per_frame(cnt: np.ndarray, typ: np.ndarray, rec: np.ndarray) -> np.ndarray:
    """(T,) bool: any entity with type==TOKEN_ACTOR & recency==0 this frame."""
    flag = ((np.asarray(typ) == TOKEN_ACTOR)
            & (np.asarray(rec, dtype=np.float64) <= 0.0)).astype(np.int64)
    off = np.concatenate([[0], np.asarray(cnt, dtype=np.int64).cumsum()])
    cs = np.concatenate([[0], flag.cumsum()])
    return (cs[off[1:]] - cs[off[:-1]]) > 0


def load_human_episodes(root: Path, split: str) -> list[tuple[int, dict]]:
    """All (demo_idx, episode) pairs from the collect cache, v5 schema."""
    root = Path(root)
    dd = root / split
    manifest = json.loads((dd / "manifest.json").read_text())
    out: list[tuple[int, dict]] = []
    for sh in manifest["shards"]:
        for _ei, dmi, fsl, esl, arr in iter_shard_episodes(
                sh, str(dd),
                obs=("entity_types", "entity_recency"),
                acts=("move", "attack", "target_probs", "look")):
            packed = np.asarray(arr["move"][fsl], dtype=np.uint8).reshape(-1)
            move = _unpack_move(packed)
            tp = np.asarray(arr["target_probs"][fsl], dtype=np.float32)
            attack = np.asarray(arr["attack"][fsl], dtype=np.int64).reshape(-1)
            weapon_context = action_attack_context(attack)
            out.append((int(dmi), {
                "fb": move[:, 0].astype(np.int8),
                "lr": move[:, 1].astype(np.int8),
                "attack": attack.astype(np.int8),
                "weapon": weapon_context,
                "wimp": weapon_context,
                "discharge": (attack > 0),
                "turn": _turn_deg(np.asarray(arr["look"][fsl])).astype(np.float64),
                "keep": (1.0 - tp[:, 0]) != 0.0,
                "engaged": _los_actor_per_frame(
                    np.asarray(arr["entity_count"][fsl], dtype=np.int64),
                    arr["entity_types"][esl], arr["entity_recency"][esl]),
            }))
    return out


def decimate2(ep: dict) -> dict:
    """20 Hz episode -> the 10 Hz view of the same behavior.

    turn: sum of the two sub-frame magnitudes (small-angle additive);
    attack/discharge/keep/engaged: block-any (min engine cooldown 0.2 s means
    two discharges can never share a 0.1 s block, so block-any is lossless for
    discharge counts); fb/lr/weapon/wimp: block-last (state carried into the
    next tick).
    """
    t = (len(ep["turn"]) // 2) * 2
    return {
        "fb": ep["fb"][1:t:2],
        "lr": ep["lr"][1:t:2],
        "attack": ep["attack"][:t].reshape(-1, 2).max(axis=1),
        "weapon": ep["weapon"][1:t:2],
        "wimp": ep["wimp"][1:t:2],
        "discharge": ep["discharge"][:t].reshape(-1, 2).any(axis=1),
        "turn": ep["turn"][:t].reshape(-1, 2).sum(axis=1),
        "keep": ep["keep"][:t].reshape(-1, 2).any(axis=1),
        "engaged": ep["engaged"][:t].reshape(-1, 2).any(axis=1),
    }


# ---------------------------------------------------------------------------
# Window featurization
# ---------------------------------------------------------------------------
def _entropy(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def _gap_excess(disch: np.ndarray, wimp: np.ndarray, mask: np.ndarray,
                hz: float) -> np.ndarray:
    """Cooldown-excess gaps between consecutive discharges within unbroken
    (mask ∧ same-held-weapon) runs — op_attack's run convention. ``disch`` is
    already mask-conditioned by the caller."""
    idx = np.nonzero(mask)[0]
    if idx.size < 2:
        return np.empty(0, dtype=np.float64)
    run_break = np.nonzero(np.diff(idx) != 1)[0] + 1
    wpn_break = np.nonzero(np.diff(wimp[idx].astype(np.int64)) != 0)[0] + 1
    starts = np.unique(np.concatenate([[0], run_break, wpn_break]))
    bounds = np.append(starts, idx.size)
    out: list[np.ndarray] = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        seg = idx[s:e]
        ft = seg[disch[seg]]
        if ft.size >= 2:
            out.append(np.diff(ft) / hz - COOLDOWN_SEC[int(wimp[seg[0]])])
    return np.concatenate(out) if out else np.empty(0, dtype=np.float64)


def window_features(ep: dict, s: int, e: int, hz: float) -> dict[str, np.ndarray | None]:
    """Per-channel feature vector for frames [s, e); None = window unusable.

    All channels condition on keep ∧ engaged (Axiom 2); the window is admitted
    only when the conditioned mass clears the engaged-mass rule.
    """
    w = e - s
    mask = ep["keep"][s:e] & ep["engaged"][s:e]
    n_mask = int(mask.sum())
    out: dict[str, np.ndarray | None] = {ch: None for ch in CHANNELS}
    if n_mask < max(int(ENGAGED_MIN_FRAC * w), int(ENGAGED_MIN_SEC * hz)):
        return out
    eng_sec = n_mask / hz

    # look — engaged-frame turn dynamics. zero_run/lag1_ac are the TEMPORAL
    # features: without them the channel is permutation-invariant and the
    # frame-shuffle control sails through (v1 calibration finding).
    turn_full = ep["turn"][s:e]
    turn = turn_full[mask]
    zero_runs = dwell_times((turn_full < ZERO_TURN_DEG).astype(np.int64),
                            mask, only_value=1)
    if turn.size >= 2 and turn.std() > 0:
        lag1 = float(np.corrcoef(turn[:-1], turn[1:])[0, 1])
    else:
        lag1 = 0.0
    out["look"] = np.array([
        float(turn.mean()) * hz,
        float(np.percentile(turn, 90)) * hz,
        float((turn < ZERO_TURN_DEG).mean()),
        float((turn >= FLICK_DEG).sum()) / eng_sec,
        float(np.median(zero_runs)) / hz if zero_runs.size else 0.0,
        lag1,
    ])

    # move — fb/lr hold-switch dynamics
    vals = []
    ok = True
    for ax in ("fb", "lr"):
        lab = ep[ax][s:e]
        _, ns, nt = switch_rate(lab, mask)
        if nt == 0:
            ok = False
            break
        vals.append(ns / (nt / hz))
    if ok:
        occ = [float((ep[ax][s:e][mask] != 1).mean()) for ax in ("fb", "lr")]
        dw = [float(np.median(dwell_times(ep[ax][s:e], mask))) / hz
              for ax in ("fb", "lr")]
        out["move"] = np.array([vals[0], vals[1], occ[0], occ[1], dw[0], dw[1]])

    # attack — engine-visible discharge events on engaged frames: rate per
    # engaged-second + cooldown-excess gap timing (right-censored at the
    # window length when no gap sample exists).
    disch = ep["discharge"][s:e] & mask
    win_sec = w / hz
    excess = _gap_excess(disch, ep["wimp"][s:e], mask, hz)
    if excess.size:
        med, p90 = float(np.median(excess)), float(np.percentile(excess, 90))
    else:
        med = p90 = win_sec
    out["attack"] = np.array([float(disch.sum()) / eng_sec, med, p90])

    # weapon — commitment dynamics (label-invariant)
    wpn = ep["weapon"][s:e]
    _, ns, nt = switch_rate(wpn, mask)
    if nt > 0:
        out["weapon"] = np.array([ns / (nt / hz), _entropy(wpn[mask])])

    for ch, v in out.items():
        if v is not None and not np.all(np.isfinite(v)):
            out[ch] = None
    return out


def featurize(eps: list, hz: float, window_sec: float,
              demos: list | None = None) -> dict:
    """Episodes -> bank {channel: {"X": (n,d), "demo": (n,)}}.

    ``eps`` is a list of episode dicts; ``demos`` the parallel demo-idx list
    (model subjects pass None -> episode index is used as the cluster id).
    """
    w = int(round(window_sec * hz))
    acc = {ch: {"X": [], "demo": []} for ch in CHANNELS}
    for i, ep in enumerate(eps):
        did = demos[i] if demos is not None else i
        t = len(ep["turn"])
        for s in range(0, t - w + 1, w):
            feats = window_features(ep, s, s + w, hz)
            for ch, v in feats.items():
                if v is not None:
                    acc[ch]["X"].append(v)
                    acc[ch]["demo"].append(did)
    return {
        ch: {
            "X": (np.stack(a["X"]) if a["X"]
                  else np.empty((0, len(FEATURE_NAMES[ch])))),
            "demo": np.asarray(a["demo"], dtype=np.int64),
        }
        for ch, a in acc.items()
    }


# ---------------------------------------------------------------------------
# Shuffled-human anchor (Axiom 3 denominator; cached in the bank artifact)
# ---------------------------------------------------------------------------
def shuffle_episode(ep: dict, rng: np.random.Generator) -> dict:
    """Frame shuffle within episode: marginals intact, dynamics destroyed."""
    idx = rng.permutation(len(ep["turn"]))
    return {k: v[idx] for k, v in ep.items()}


def _build_anchor(human_eps: list[tuple[int, dict]], hz: float,
                  window_sec: float) -> tuple[dict, np.ndarray]:
    """Frame-shuffled held-out-human window bank + the held-out demo ids."""
    rng = np.random.default_rng(ANCHOR_SEED)
    demos = np.unique([d for d, _ in human_eps])
    hold = rng.choice(demos, min(ANCHOR_HOLDOUT_DEMOS, demos.size),
                      replace=False)
    hold_set = set(int(d) for d in hold)
    eps = [(d, ep) for d, ep in human_eps if d in hold_set]
    shuffled = [shuffle_episode(ep, rng) for _, ep in eps]
    anchor = featurize(shuffled, hz, window_sec, [d for d, _ in eps])
    return anchor, np.sort(hold.astype(np.int64))


# ---------------------------------------------------------------------------
# Bank cache — per-collect, beside the other qnn.human baselines
# ---------------------------------------------------------------------------
def bank_cache_path(root: str | Path, hz: float, window_sec: float) -> Path:
    """Per-collect bank cache under the collect's ``human_baseline/`` dir, so it
    travels with the corpus and can never be a stale cross-corpus hit (the old
    global ``runs/head_probe/`` path silently reused one corpus's bank for another)."""
    from qnn.human import BASELINE_SUBDIR
    return (Path(root) / BASELINE_SUBDIR
            / f"_human_band_bank_hz{int(hz)}_w{int(window_sec)}.npz")


def load_or_build_bank(root: Path, split: str, hz: float, window_sec: float,
                       human_eps: list | None = None) -> tuple[dict, list | None]:
    """Cached human window bank at ``hz``; returns (bank, human_eps or None).

    The bank dict carries the corpus windows under the CHANNELS keys plus the
    shuffled-human anchor under ``_anchor`` / ``_anchor_demos`` (scoring
    denominators; Axiom 3). ``human_eps`` may be passed in to avoid a second
    corpus scan when both rates are built in one process; it is loaded lazily
    on cache miss and returned so the caller can reuse it.
    """
    root = Path(root)
    path = bank_cache_path(root, hz, window_sec)
    if path.exists():
        z = np.load(path)
        if int(z["version"][0]) == BANK_VERSION and str(z["split"]) == split:
            bank = {ch: {"X": z[f"{ch}_X"], "demo": z[f"{ch}_demo"]}
                    for ch in CHANNELS}
            bank["_anchor"] = {ch: {"X": z[f"anchor_{ch}_X"],
                                    "demo": z[f"anchor_{ch}_demo"]}
                               for ch in CHANNELS}
            bank["_anchor_demos"] = z["anchor_demos"]
            return bank, human_eps
    if human_eps is None:
        human_eps = load_human_episodes(root, split)
    demos = [d for d, _ in human_eps]
    eps = [ep for _, ep in human_eps]
    if hz != HUMAN_HZ:
        if not math.isclose(hz * 2, HUMAN_HZ):
            raise ValueError(f"only {HUMAN_HZ}->{HUMAN_HZ/2} decimation supported, got hz={hz}")
        eps = [decimate2(ep) for ep in eps]
    bank = featurize(eps, hz, window_sec, demos)
    anchor, anchor_demos = _build_anchor(list(zip(demos, eps)), hz, window_sec)
    bank["_anchor"] = anchor
    bank["_anchor_demos"] = anchor_demos
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"version": np.array([BANK_VERSION]), "split": np.array(split),
              "anchor_demos": anchor_demos}
    for ch in CHANNELS:
        arrays[f"{ch}_X"] = bank[ch]["X"]
        arrays[f"{ch}_demo"] = bank[ch]["demo"]
        arrays[f"anchor_{ch}_X"] = anchor[ch]["X"]
        arrays[f"anchor_{ch}_demo"] = anchor[ch]["demo"]
    np.savez_compressed(path, **arrays)
    return bank, human_eps


def bank_filter(bank: dict, demos: set, *, exclude: bool) -> dict:
    """Filter the CHANNELS window banks by demo id; ``_anchor*`` keys (frozen
    denominators) pass through untouched."""
    out = {}
    for ch in CHANNELS:
        b = bank[ch]
        m = np.isin(b["demo"], list(demos))
        if exclude:
            m = ~m
        out[ch] = {"X": b["X"][m], "demo": b["demo"][m]}
    for k in ("_anchor", "_anchor_demos"):
        if k in bank:
            out[k] = bank[k]
    return out
