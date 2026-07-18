"""Weapon-head analysis functions — importable core compute for the weapon slice.

Each public function corresponds to one or more canonical analysis scripts:

    corpus_stats              ← _weapon_corpus_stats.py
    intent_decompose          ← _weapon_intent_decompose.py
    intent_psth               ← _weapon_intent_psth.py  (npz-input)
    gate_sweep                ← _weapon_switch_gate_sweep.py
    switch_gated              ← _weapon_switch_gated.py
    anticip_roc               ← _weapon_switch_anticip_roc.py  (npz-input)
    decode_sweep              ← _weapon_decode_sweep.py  (npz-input)
    switch_window_roc         ← _weapon_switch_window_roc.py  (needs forward pass)
    switch_decompose          ← _weapon_switch_decompose.py
    switch_timing_detail      ← _weapon_switch_timing_detail.py
    switch_leadtime           ← _weapon_switch_leadtime.py
    switch_leak_test          ← _weapon_switch_leak_test.py
    switch_vs_token           ← _weapon_switch_vs_token.py
    switchframe_decomp        ← _weapon_switchframe_decomp.py  (needs forward pass)
    when_switch_detect        ← _weapon_when_switch_detect.py

``collect_frames(policy, source)`` is the shared (policy, source) forward pass —
produces ``probs`` (N,8) and ``label`` (impulse 1..8, 0=none).  It is the canonical
entry used by gate_sweep, switch_gated, switch_decompose, switch_timing_detail,
switch_leadtime, switch_leak_test, switch_vs_token, and when_switch_detect.

``_committed_stream`` in ALL scripts now delegates to
``qnn.bc.decode_fit._committed_stream`` — that is the ONE canonical implementation.

OPERATIVE filter for the weapon slice: weapon-present = label != 0.
Do NOT apply input_mask & 1 to weapon (bit 0 is the attack bit; weapon is NOT
input_mask-gated). Switch-leak-free metric: pred != self_weapon_id at attack frames.

Shared loaders:
  * ``qnn.diag.loader.load_policy`` — canonical checkpoint loader.
  * ``qnn.bc.decode_fit._committed_stream`` — canonical carry-forward stream.
  * ``scripts/analysis/_weapon_metrics`` — canonical switch/dwell definitions.

``analyze(policy, source)`` runs the (policy, source)-compatible subset —
corpus_stats, gate_sweep, and switch_gated — and returns a per-head dict.
Functions that require additional inputs (npz caches, forward passes different
from collect_frames) are standalone named functions with their own signatures.

Usage (thin-wrapper pattern)::

    from qnn.diag.weapon import (
        collect_frames,
        corpus_stats,
        intent_decompose,
        intent_psth,
        gate_sweep,
        switch_gated,
        anticip_roc,
        decode_sweep,
        switch_decompose,
        switch_timing_detail,
        switch_leadtime,
        switch_leak_test,
        switch_vs_token,
        switchframe_decomp,
        when_switch_detect,
        analyze,
    )

    # Or for thin wrappers:
    from qnn.diag.weapon import collect_frames, grade, carry_forward_sweep
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qnn.bc.supervised_loop import make_resident_source_from_cache
from qnn.bc.decode_fit import (
    _committed_stream,  # canonical — one copy, imported here and re-exported
    _to_np,
)
from qnn.diag.loader import load_policy
from qnn.schema import WEAPON_HEAD_SIZE
from qnn.vocab import self_weapon_id_to_impulse


@torch.inference_mode()
def _forward_weapon_logits(policy, source) -> torch.Tensor:
    """Per-frame 8-way weapon logits (N, 8) — the split-weapon-head forward.

    CANONICAL HOME (moved from qnn.bc.decode_fit when the a24 fits were retired;
    the 8-way head is an a24-era analysis subject, but this diag surface still
    reads retained runs). Forward each episode as a (1, T) sequence through the
    plain network — NO bench side-channel, matching the live act() path."""
    import numpy as _np
    from qnn.model.network import WEAPON_HEAD as _WH
    offsets = _np.asarray(source.episode_offsets, dtype=_np.int64)
    out = torch.empty((int(offsets[-1]), WEAPON_HEAD_SIZE), dtype=torch.float32)
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e <= s:
            continue
        T = e - s
        idx = torch.arange(s, e, dtype=torch.int64, device=policy.device)
        obs_seq = {k: v.index_select(0, idx).reshape((1, T) + tuple(v.shape[1:]))
                   for k, v in source.obs.items()}
        _f, logits, _v, _nh, _tl = policy.model(obs_seq, hidden=None, reset_mask=None)
        out[s:e] = logits[_WH].reshape(T, WEAPON_HEAD_SIZE).float().cpu()
    return out

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

WEAPON_NAMES = ["axe", "shotgun", "super_shotgun", "nailgun",
                "super_nailgun", "grenade", "rocket", "lightning"]
WEAPON_NAMES_ABBR = ["axe", "sg", "ssg", "ng", "sng", "gl", "rl", "lg"]
WEAPON_HEAD_SIZE = 8

#: Default segment mask used by all weapon analysis (combat / target-present).
SEGMENT_MASK = {"act.target": {"$ne": 0}}


# ---------------------------------------------------------------------------
# Shared helpers (re-exported for backward-compat with thin wrappers)
# ---------------------------------------------------------------------------

# NOTE: _committed_stream is imported from decode_fit and is directly accessible
# as qnn.diag.weapon._committed_stream.  Scripts that used to inline their own
# copy should import this instead.


def _tok_impulse(source) -> np.ndarray:
    """Extract self_weapon_id as impulse 1..8 from a resident source."""
    widr = source.obs["self_weapon_id"]
    return _to_np(
        self_weapon_id_to_impulse(
            torch.as_tensor(_to_np(widr).reshape(-1)).long()
        )
    ).reshape(-1)


def _weapon_probs_conf_margin(probs: np.ndarray):
    """Return (pred_imp 1..8, conf, margin) from an (N,8) probs array."""
    pred_imp = probs.argmax(1) + 1
    s = np.sort(probs, axis=1)
    conf = s[:, -1]
    margin = s[:, -1] - s[:, -2]
    return pred_imp, conf, margin


def _macro_f1(pred, label, valid) -> float:
    """Macro-F1 over weapon classes 1..WEAPON_HEAD_SIZE on ``valid`` frames."""
    f1s = []
    for c in range(1, WEAPON_HEAD_SIZE + 1):
        pc = valid & (pred == c)
        gc = valid & (label == c)
        if not gc.any():
            continue
        tp = int((pc & gc).sum()); fp = int((pc & ~gc).sum()); fn = int((gc & ~pc).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def _per_class_f1(pred, label) -> dict[str, dict]:
    """Per-class F1/precision/recall/support for a committed stream vs label."""
    valid = label != 0
    out = {}
    for c in range(1, WEAPON_HEAD_SIZE + 1):
        pc = valid & (pred == c); gc = valid & (label == c)
        tp = int((pc & gc).sum()); fp = int((pc & ~gc).sum()); fn = int((gc & ~pc).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[WEAPON_NAMES[c - 1]] = {"f1": f1, "precision": prec, "recall": rec,
                                     "support": int(gc.sum())}
    return out


def _dilate_within_episodes(flag: np.ndarray, offsets: np.ndarray,
                             lo: int, hi: int) -> np.ndarray:
    """Mark frame f if any flag in [f+lo, f+hi] within same episode.

    lo < 0 means look backward; hi > 0 means look forward.
    Symmetric ±k: lo=-k, hi=k.
    """
    out = np.zeros_like(flag)
    offs = np.asarray(offsets, dtype=np.int64)
    for i in range(len(offs) - 1):
        s, e = int(offs[i]), int(offs[i + 1])
        if e <= s:
            continue
        seg = flag[s:e]
        d = np.zeros_like(seg)
        for sh in range(lo, hi + 1):
            if sh == 0:
                d |= seg
            elif sh > 0:
                d[:-sh or None] |= seg[sh:]
            else:
                d[-sh:] |= seg[:sh]
        out[s:e] = d
    return out


# ---------------------------------------------------------------------------
# collect_frames  — canonical (policy, source) forward pass
# ---------------------------------------------------------------------------

@torch.inference_mode()
def collect_frames(policy, source) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame probs (N,8) and GT label (impulse 1..8, 0=none).

    This is the SHARED forward pass for all functions that accept a
    ``(policy, source)`` signature.  It matches the original
    ``_weapon_switch_threshold_eval.collect_frames`` exactly — the bench
    side-channel is wired via ``bench_side_channel_scope`` so that bench
    heads that depend on it (e.g. target supervision) produce correct logits.
    """
    from qnn.model.bench.side_channels import bench_side_channel_scope
    from qnn.model.network import WEAPON_HEAD

    offsets = np.asarray(source.episode_offsets, dtype=np.int64)
    probs_all, label_all = [], []
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e <= s:
            continue
        obs = {k: v[s:e] for k, v in source.obs.items()}
        act = {k: v[s:e] for k, v in source.actions.items()}
        obs_t = {k: torch.as_tensor(v).unsqueeze(1).to(policy.device)
                 for k, v in obs.items()}
        act_t = {k: torch.as_tensor(v).to(policy.device)
                 for k, v in act.items()}
        with bench_side_channel_scope(act_t, None):
            _, logits, _, _, _ = policy._forward_tensors(obs_t, hidden=None, masks=None)
        wl = logits[WEAPON_HEAD].reshape(-1, WEAPON_HEAD_SIZE)
        probs_all.append(torch.softmax(wl, dim=-1).float().cpu().numpy())
        lab = act.get("weapon")
        if lab is None:
            lab = act_t["weapon"].cpu()
        lab = (lab.cpu().numpy() if torch.is_tensor(lab) else np.asarray(lab)).reshape(-1)
        label_all.append(lab.astype(np.int64))
    return np.concatenate(probs_all), np.concatenate(label_all)


# ---------------------------------------------------------------------------
# grade  — head-level grading (re-export from te, now canonical here)
# ---------------------------------------------------------------------------

def grade(probs: np.ndarray, label: np.ndarray) -> dict:
    """Head-level grade: macro-F1, micro-acc, NLL, per-class (impulse 1..8)."""
    valid = label != 0
    p = probs[valid]
    gt = label[valid] - 1
    pred = p.argmax(1)
    nll = -np.log(np.clip(p[np.arange(len(gt)), gt], 1e-12, 1.0))
    per_class = {}
    for c in range(WEAPON_HEAD_SIZE):
        gtc, prc = (gt == c), (pred == c)
        tp = int((gtc & prc).sum()); fp = int((~gtc & prc).sum()); fn = int((gtc & ~prc).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        sup = int(gtc.sum())
        per_class[WEAPON_NAMES[c]] = {
            "f1": f1, "precision": prec, "recall": rec,
            "nll": float(nll[gt == c].mean()) if sup else float("nan"),
            "support": sup,
        }
    return {
        "n_frames": int(valid.sum()),
        "macro_f1": _macro_f1(pred + 1, label, valid),
        "micro_acc": float((pred == gt).mean()),
        "nll_mean": float(nll.mean()),
        "per_class": per_class,
    }


def carry_forward_sweep(
    probs: np.ndarray, label: np.ndarray, offsets, confs, margins
) -> tuple[list[dict], dict]:
    """Sweep (conf, margin) → committed-stream macro-F1 vs GT.

    Uses the canonical ``_committed_stream`` from ``decode_fit``; seeds each
    episode from the GT label (label != 0 → the true held weapon; 0 → prev
    held).  Returns (rows, always_commit_result).
    """
    # The canonical _committed_stream seeds from ``seed`` (impulse array);
    # for the threshold-optimization sweep the seed is the GT label itself
    # (so episode-start frames initialize to the true held weapon, matching
    # the original te._committed_stream behavior).
    valid = label != 0
    rows = []
    for C in confs:
        for M in margins:
            com, above = _committed_stream(probs, label, offsets, float(C), float(M))
            rows.append({
                "conf": round(float(C), 3), "margin": round(float(M), 3),
                "macro_f1": _macro_f1(com, label, valid),
                "acc": float((com[valid] == label[valid]).mean()),
                "commit_rate": float(above[valid].mean()),
            })
    pred_imp = probs.argmax(1) + 1
    always = {
        "macro_f1": _macro_f1(pred_imp, label, valid),
        "acc": float((pred_imp[valid] == label[valid]).mean()),
    }
    return rows, always


# ---------------------------------------------------------------------------
# corpus_stats  (_weapon_corpus_stats.py core)
# ---------------------------------------------------------------------------

def corpus_stats(data_dir: Path, *, max_shards=None) -> dict:
    """Model-free corpus weapon statistics: persistence + target-segment determinants.

    Section A: persistence (full val weapon stream) — switch rate, per-weapon
    dwell medians, and H(weapon) vs H(weapon|prev_weapon).

    Section B: target-segment determinants (resident source, segment
    act.target!=0) — H(weapon|X) for X in {owned_set, n_owned,
    target_distance, health}.

    Parameters
    ----------
    data_dir:
        Base corpus directory containing ``precomputed_val/``.
    max_shards:
        Optional shard cap for load_val_episodes.

    Returns
    -------
    dict with keys: ``persistence``, ``target_segment``.
    """
    from qnn.diag.data import load_val_episodes

    val = data_dir / "precomputed_val"
    SEG = SEGMENT_MASK
    REL = 3  # entity_scalars_raw col index for relative position start

    def npy(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    def H(y):
        _, c = np.unique(y, return_counts=True)
        p = c / c.sum()
        return float(-(p * np.log2(p)).sum())

    def H_cond(y, x):
        o = np.argsort(x, kind="stable"); xs, ys = x[o], y[o]
        bnd = np.where(np.diff(xs) != 0)[0] + 1
        n = len(ys)
        return float(sum((len(g) / n) * H(g) for g in np.split(ys, bnd)))

    def qbucket(v, nb=10):
        qs = np.quantile(v, np.linspace(0, 1, nb + 1)[1:-1])
        return np.digitize(v, qs)

    # --- A: persistence ---
    sw_c = sw_f = 0
    dwell: dict[int, list] = {i: [] for i in range(8)}
    Wf_list, PV_list = [], []
    for ep in load_val_episodes(data_dir, split="val", max_shards=max_shards):
        w = npy(ep["actions"]["weapon"]).reshape(-1).astype(int)
        T = len(w)
        if T < 2:
            continue
        sw_f += T - 1
        sw_c += int((w[1:] != w[:-1]).sum())
        for run in np.split(w, np.where(np.diff(w) != 0)[0] + 1):
            if 1 <= run[0] <= 8:
                dwell[run[0] - 1].append(len(run))
        Wf_list.append(w[1:]); PV_list.append(w[:-1])
    Wf = np.concatenate(Wf_list); PV = np.concatenate(PV_list)
    mwf = (Wf >= 1) & (Wf <= 8); Wf, PV = Wf[mwf], PV[mwf]
    switch_rate_pct = 100 * sw_c / sw_f if sw_f else 0.0
    hw = H(Wf); hw_cond = H_cond(Wf, PV)
    per_weapon_dwell = {}
    for i in range(8):
        d = dwell[i]
        if d:
            per_weapon_dwell[WEAPON_NAMES[i]] = {
                "median": int(np.median(d)),
                "mean": float(np.mean(d)),
                "n_runs": len(d),
            }

    persistence = {
        "n_frames": int(len(Wf)),
        "switch_rate_pct": round(switch_rate_pct, 3),
        "H_weapon": round(hw, 4),
        "H_weapon_given_prev": round(hw_cond, 4),
        "delta_H": round(hw - hw_cond, 4),
        "per_weapon_dwell": per_weapon_dwell,
    }

    # --- B: target-segment determinants ---
    src = make_resident_source_from_cache(val, torch.device("cpu"), segment_mask=SEG)
    w = npy(src.actions["weapon"]).reshape(-1).astype(int)
    tp = npy(src.actions["target_probs"])
    full = tp.argmax(1).astype(int); slot = np.clip(full - 1, 0, 15)
    ent = npy(src.obs["entity_scalars_raw"])
    rel = ent[np.arange(len(w)), slot, REL:REL + 3]
    dist = np.linalg.norm(rel.astype(np.float32), axis=1)
    rdy = npy(src.obs["self_weapon_readiness"]).reshape(len(w), -1)
    owned = (rdy > 0)
    okey = (owned * (1 << np.arange(owned.shape[1]))).sum(1)
    nown = owned.sum(1)
    hp = npy(src.obs["self_state_scalars"]).reshape(len(w), -1)[:, 0]
    m = (w >= 1) & (w <= 8)
    w, dist, okey, nown, hp = w[m], dist[m], okey[m], nown[m], hp[m]
    Hw = H(w)
    determinants = {}
    for name, x in [("owned_set", okey), ("n_owned", nown),
                    ("target_distance", qbucket(dist)), ("health", qbucket(hp))]:
        hc = H_cond(w, x)
        determinants[name] = {
            "H_given_x": round(hc, 4),
            "delta_H": round(Hw - hc, 4),
            "delta_pct": round(100 * (Hw - hc) / Hw, 2),
            "n_x": int(len(np.unique(x))),
        }
    per_weapon_dist = {}
    for i in range(8):
        mm = w == (i + 1)
        if mm.sum() > 50:
            d = dist[mm]
            per_weapon_dist[WEAPON_NAMES[i]] = {
                "n": int(mm.sum()),
                "median_dist": float(np.median(d)),
                "q25": float(np.quantile(d, 0.25)),
                "q75": float(np.quantile(d, 0.75)),
            }

    return {
        "persistence": persistence,
        "target_segment": {
            "n_frames": int(len(w)),
            "H_weapon": round(Hw, 4),
            "determinants": determinants,
            "per_weapon_target_dist": per_weapon_dist,
        },
    }


# ---------------------------------------------------------------------------
# intent_decompose  (_weapon_intent_decompose.py core)
# ---------------------------------------------------------------------------

AMMO = {2: "ammo_shells", 3: "ammo_shells", 4: "ammo_nails", 5: "ammo_nails",
        6: "ammo_rockets", 7: "ammo_rockets", 8: "ammo_cells"}  # 1=axe: none
ITBIT = {2: 1, 3: 2, 4: 4, 5: 8, 6: 16, 7: 32, 8: 64}          # impulse -> IT_ weapon flag


def intent_decompose(source) -> dict:
    """Decompose human switch onsets: INTENT vs ENGINE-FORCED vs UNEXPLAINED.

    onset f: act.weapon[f] != act.weapon[f-1] != 0, per episode.
    lead (INTENT): self_imp[f] != w_new  (recovered impulse leads the equip).
    no-lead: classified by cause (respawn / ammo_out / pickup / unexplained).

    Parameters
    ----------
    source:
        Resident source with segment_mask=SEGMENT_MASK.  Must have obs keys:
        self_weapon_id, health, self_items (optional), ammo_*.

    Returns
    -------
    dict with keys: total, intent, nolead, cause.
    """
    def _np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    offs = np.asarray(source.episode_offsets, dtype=np.int64)
    weapon = _np(source.actions["weapon"]).reshape(-1).astype(int)
    si = _to_np(
        self_weapon_id_to_impulse(
            torch.as_tensor(_np(source.obs["self_weapon_id"]).reshape(-1)).long()
        )
    ).reshape(-1).astype(int)
    health = _np(source.obs["health"]).reshape(-1).astype(float)
    items = _np(source.obs.get("self_items",
                               np.zeros_like(weapon))).reshape(-1).astype(int)
    ammo = {k: _np(source.obs[k]).reshape(-1).astype(float)
            for k in ("ammo_shells", "ammo_nails", "ammo_rockets", "ammo_cells")
            if k in source.obs}

    n_intent = n_nolead = 0
    cause: dict[str, int] = {"respawn": 0, "ammo_out": 0, "pickup": 0, "unexplained": 0}
    for i in range(len(offs) - 1):
        s, e = int(offs[i]), int(offs[i + 1])
        w = weapon[s:e]
        chg = np.flatnonzero((w[1:] != w[:-1]) & (w[1:] != 0)) + 1
        for f in chg:
            g = s + f
            w_old, w_new = int(w[f - 1]), int(w[f])
            if si[g] != w_new:
                n_intent += 1          # impulse-led => genuine intent
                continue
            n_nolead += 1
            lo = max(s, g - 4)
            if health[g] - health[g - 1] >= 50 or (health[lo:g] <= 0).any():
                cause["respawn"] += 1
            elif (w_old in AMMO and AMMO[w_old] in ammo
                  and ammo[AMMO[w_old]][g - 1] <= 0):
                cause["ammo_out"] += 1
            elif (w_new in ITBIT and (items[g] & ITBIT[w_new])
                  and not (items[g - 1] & ITBIT[w_new])):
                cause["pickup"] += 1
            else:
                cause["unexplained"] += 1

    tot = n_intent + n_nolead
    return {"total": tot, "intent": n_intent, "nolead": n_nolead, "cause": cause}


# ---------------------------------------------------------------------------
# intent_psth  (_weapon_intent_psth.py core — npz-input)
# ---------------------------------------------------------------------------

def intent_psth(npz: Path) -> dict:
    """Lead-time PSTH on INTENT switches only (impulse-led).

    Aligned to the intent onset (tau=0 = the impulse/label-lead frame).
    Returns per-tau rows and per-switch lead-vs-obs statistics.

    Parameters
    ----------
    npz:
        Path to a probs npz written by _weapon_switch_window_roc (or similar)
        with arrays: probs (N,8 float32), weapon (N int), self_imp (N int),
        offsets (n+1 int), hz (scalar).
    """
    d = np.load(npz)
    weapon = d["weapon"].astype(int); si = d["self_imp"].astype(int)
    offs = d["offsets"].astype(int); probs = d["probs"].astype(np.float32)
    pred = probs.argmax(1) + 1; hz = float(d["hz"])

    F, WN, LO, HI = [], [], [], []
    for i in range(len(offs) - 1):
        s, e = int(offs[i]), int(offs[i + 1])
        w = weapon[s:e]
        for f in (np.flatnonzero((w[1:] != w[:-1]) & (w[1:] != 0)) + 1):
            g = s + f; wn = int(w[f])
            if si[g] != wn:          # lead fired => genuine intent
                F.append(g); WN.append(wn); LO.append(s); HI.append(e)
    F = np.array(F); WN = np.array(WN); LO = np.array(LO); HI = np.array(HI)
    n = len(F)

    psth_rows = []
    for tau in range(int(round(-1.0 * hz)), int(round(0.5 * hz)) + 1):
        idx = F + tau; ok = (idx >= LO) & (idx < HI)
        iv = idx[ok]; wv = WN[ok]
        o = float((si[iv] == wv).mean()) if ok.any() else float("nan")
        m = float((pred[iv] == wv).mean()) if ok.any() else float("nan")
        pw = float(probs[iv, wv - 1].mean()) if ok.any() else float("nan")
        psth_rows.append({"tau_s": round(tau / hz, 3), "obs_eq_wnew": o,
                          "mdl_eq_wnew": m, "p_wnew": pw, "n": int(ok.sum())})

    # per-switch lead of model commit over obs equip
    W = int(round(1.0 * hz))
    lead_m = same = lag = tot = 0; leads = []
    for g, wn, lo, hi in zip(F, WN, LO, HI):
        tm = to = None
        for tau in range(-W, 2 * W + 1):
            j = g + tau
            if not (lo <= j < hi):
                continue
            if tm is None and pred[j] == wn:
                tm = tau
            if to is None and si[j] == wn:
                to = tau
        if tm is None or to is None:
            continue
        tot += 1
        if tm < to:
            lead_m += 1; leads.append(to - tm)
        elif tm > to:
            lag += 1
        else:
            same += 1; leads.append(0)
    la = np.array(leads) if leads else np.array([], dtype=float)

    return {
        "n_intent_switches": n,
        "hz": hz,
        "psth": psth_rows,
        "lead_stats": {
            "n_matched": tot,
            "model_leads_pct": round(lead_m / tot * 100, 2) if tot else float("nan"),
            "same_frame_pct": round(same / tot * 100, 2) if tot else float("nan"),
            "lags_pct": round(lag / tot * 100, 2) if tot else float("nan"),
            "lead_median_frames": float(np.median(la)) if len(la) else float("nan"),
            "lead_mean_frames": float(la.mean()) if len(la) else float("nan"),
            "lead_p90_frames": float(np.percentile(la, 90)) if len(la) else float("nan"),
        },
    }


# ---------------------------------------------------------------------------
# gate_sweep  (_weapon_switch_gate_sweep.py core)
# ---------------------------------------------------------------------------

def gate_sweep(
    policy,
    source,
    *,
    confs=None,
    margins=None,
) -> dict:
    """Confidence/margin gate sweep: switch-event rate + macro-F1 per (conf, margin).

    Uses the canonical _committed_stream (seeded from self_weapon_id).
    assert_is_event_rate is called on the human rate to guard mislabeling.

    Parameters
    ----------
    policy:
        Loaded QNNPolicy (eval mode).
    source:
        Resident source with segment_mask=SEGMENT_MASK.
    confs, margins:
        Grid values.  Defaults to a standard sweep.
    """
    from qnn.diag import weapon_metrics as wm  # canonical switch/dwell definitions

    if confs is None:
        confs = [0.0, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8]
    if margins is None:
        margins = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

    probs, label = collect_frames(policy, source)
    tok = _tok_impulse(source)
    offs = np.asarray(source.episode_offsets, np.int64)
    valid = label != 0

    pred_imp, conf, marg = _weapon_probs_conf_margin(probs)
    human_rate = wm.switch_rate(label, offs)   # act.weapon transitions ≈ 4%
    wm.assert_is_event_rate(human_rate, "human switch-event rate")

    def _macro(pred):
        f1s = []
        for c in range(1, 9):
            tp = ((pred == c) & (label == c) & valid).sum()
            fp = ((pred == c) & (label != c) & valid).sum()
            fn = ((pred != c) & (label == c) & valid).sum()
            d = 2 * tp + fp + fn
            f1s.append(2 * tp / d if d else 0.0)
        return float(np.mean(f1s))

    rows = []
    for c in confs:
        for mg in margins:
            ss = (pred_imp != tok) & (conf >= c) & (marg >= mg)
            chosen = np.where(ss, pred_imp, tok)   # teacher-forced gated
            ev = wm.switch_rate(chosen, offs)
            rows.append({
                "conf": c, "margin": mg,
                "event_rate": ev,
                "macro_f1": _macro(chosen),
                "acc": float((chosen[valid] == label[valid]).mean()),
                "rate_gap": abs(ev - human_rate),
            })

    best_f1 = max(rows, key=lambda r: r["macro_f1"])
    best_rate = min(rows, key=lambda r: r["rate_gap"])

    return {
        "human_switch_event_rate": human_rate,
        "n_valid": int(valid.sum()),
        "rows": rows,
        "best_by_macro_f1": best_f1,
        "best_by_rate_match": best_rate,
    }


# ---------------------------------------------------------------------------
# switch_gated  (_weapon_switch_gated.py core)
# ---------------------------------------------------------------------------

def switch_gated(
    policy,
    source,
    *,
    conf: float = 0.65,
    margin: float = 0.15,
) -> dict:
    """Switch-event rate + dwell + intent!=state decision skill at a fixed gate.

    Returns structured dict.  Prints nothing — the thin wrapper handles printing.
    """
    from qnn.diag import weapon_metrics as wm

    probs, label = collect_frames(policy, source)
    tok = _tok_impulse(source)
    offs = np.asarray(source.episode_offsets, np.int64)

    pred_imp, top1, top12 = _weapon_probs_conf_margin(probs)
    marg = top12
    should_switch = (pred_imp != tok) & (top1 >= conf) & (marg >= margin)
    gated_chosen = np.where(should_switch, pred_imp, tok)

    human_rate = wm.switch_rate(label, offs)
    model_ungated_rate = wm.switch_rate(pred_imp, offs)
    commit_rate = float(should_switch.mean())
    wm.assert_is_event_rate(human_rate, "human switch-event rate")

    hd = wm.dwell_times(label, offs)
    md = wm.dwell_times(pred_imp, offs)

    mism = wm.intent_state_mismatch(label, tok)
    v = label != 0
    acc_un = (pred_imp == label)

    return {
        "conf": conf, "margin": margin,
        "n_frames": int(len(label)),
        "switch_event": {
            "human_rate": human_rate,
            "model_ungated_rate": model_ungated_rate,
            "commit_rate": commit_rate,
            "self_weapon_id_rate": wm.switch_rate(tok, offs),
        },
        "dwell": {
            "human_median": float(np.median(hd)),
            "human_mean": float(hd.mean()),
            "model_median": float(np.median(md)),
            "model_mean": float(md.mean()),
        },
        "intent_state_skill": {
            "mismatch_frac": float(mism.sum() / v.sum()) if v.any() else float("nan"),
            "acc_on_mismatch": float(acc_un[mism].mean()) if mism.any() else float("nan"),
            "acc_on_hold": float(acc_un[v & ~mism].mean()) if (v & ~mism).any() else float("nan"),
        },
    }


# ---------------------------------------------------------------------------
# anticip_roc  (_weapon_switch_anticip_roc.py core — npz-input)
# ---------------------------------------------------------------------------

_ANTICIP_C_GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.55, 0.57, 0.6, 0.65,
                   0.7, 0.75, 0.8, 0.82, 0.85, 0.9]
_ANTICIP_M_GRID = [0.0, 0.1]


def _anticip_events(stream: np.ndarray, offsets: np.ndarray,
                    nonzero: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(stream)
    chg = np.zeros(n, dtype=bool); chg[1:] = stream[1:] != stream[:-1]
    ep_start = np.zeros(n, dtype=bool)
    ep_start[offsets[:-1][offsets[:-1] < n]] = True
    chg &= ~ep_start
    if nonzero:
        chg &= stream != 0
    idx = np.flatnonzero(chg)
    ep = np.searchsorted(offsets, idx, side="right") - 1
    return idx, stream[idx].astype(np.int64), ep


def _anticip_group(ev_ep):
    g: dict[int, list] = defaultdict(list)
    for k, e in enumerate(ev_ep):
        g[int(e)].append(k)
    return g


def _match_anticip(h_fr, h_w, h_ep, m_fr, m_w, mby, W):
    """Greedy 1:1, model on-frame-or-early within W."""
    pairs = []
    for hi in range(len(h_fr)):
        e = int(h_ep[hi]); hf = int(h_fr[hi]); hw = int(h_w[hi])
        for mi in mby.get(e, ()):
            if int(m_w[mi]) != hw:
                continue
            dt = int(m_fr[mi]) - hf
            if -W <= dt <= 0:
                pairs.append((-dt, dt, hi, mi))
    pairs.sort()
    h_used: set = set(); m_used: set = set(); lat = []
    for _adt, dt, hi, mi in pairs:
        if hi in h_used or mi in m_used:
            continue
        h_used.add(hi); m_used.add(mi); lat.append(dt)
    return len(h_used), lat, h_used


def _reactive_late(h_fr, h_w, h_ep, m_fr, m_w, mby, W, matched_h):
    late = 0
    for hi in range(len(h_fr)):
        if hi in matched_h:
            continue
        e = int(h_ep[hi]); hf = int(h_fr[hi]); hw = int(h_w[hi])
        for mi in mby.get(e, ()):
            if int(m_w[mi]) != hw:
                continue
            dt = int(m_fr[mi]) - hf
            if 0 < dt <= W:
                late += 1; break
    return late


def anticip_roc(npz: Path, *, windows=(0.0, 0.5, 1.0)) -> dict:
    """Anticipatory (backward-window) switch-event ROC from a probs npz.

    NOTE: FORCED-INCLUSIVE — scores recall/precision against ALL human switches
    (intent + engine-forced). For the weapon-switch ARBITER use ``switch_window_f1``
    (intent-only); this is kept as a forced-inclusive diagnostic only.

    Parameters
    ----------
    npz:
        Probs npz written by _weapon_switch_window_roc or similar.
    windows:
        Anticipatory windows in seconds.

    Returns
    -------
    dict with per-window ROC tables, F1-vs-lead-window, PSTH.
    """
    d = np.load(npz)
    probs = d["probs"].astype(np.float32)
    weapon = d["weapon"].astype(np.int64)
    self_imp = d["self_imp"].astype(np.int64)
    offsets = d["offsets"].astype(np.int64)
    hz = float(d["hz"])
    n_fr = int(offsets[-1]); sim_min = n_fr / hz / 60.0

    pred_imp = probs.argmax(1) + 1
    s = np.sort(probs, axis=1)
    conf = s[:, -1]; margin = s[:, -1] - s[:, -2]

    h_fr, h_w, h_ep = _anticip_events(weapon, offsets, nonzero=True)
    n_human = len(h_fr)

    result: dict[str, Any] = {
        "hz": hz, "sim_min": sim_min, "n_human": n_human,
        "human_per_min": round(n_human / sim_min, 3),
        "windows": {},
    }

    all_cache = []
    for Wsec in windows:
        W = int(round(Wsec * hz))
        rows = []
        best_recall = 0.0
        for M in _ANTICIP_M_GRID:
            for C in _ANTICIP_C_GRID:
                com = _committed_stream(probs, self_imp, offsets, C, M)[0]
                m_fr, m_w, m_ep = _anticip_events(com, offsets, nonzero=False)
                mby = _anticip_group(m_ep)
                n_model = len(m_fr)
                matched, lat, mh = _match_anticip(h_fr, h_w, h_ep, m_fr, m_w, mby, W)
                react = _reactive_late(h_fr, h_w, h_ep, m_fr, m_w, mby, W, mh)
                recall = matched / n_human if n_human else 0.0
                prec = matched / n_model if n_model else float("nan")
                f1 = (2 * prec * recall / (prec + recall)
                      if recall and prec and prec + recall > 0 else 0.0)
                rows.append({
                    "C": C, "M": M, "recall": recall, "prec": prec, "f1": f1,
                    "mdl_per_min": n_model / sim_min,
                    "fp_per_min": (n_model - matched) / sim_min,
                    "react_late": react / n_human if n_human else float("nan"),
                    "lat_med": float(np.median(lat)) if lat else float("nan"),
                })
                best_recall = max(best_recall, recall)
                all_cache.append((C, M, recall, prec, f1,
                                  n_model / sim_min, (n_model - matched) / sim_min,
                                  react / n_human if n_human else 0.0, lat))
        result["windows"][f"{Wsec:g}s"] = {"rows": rows, "best_recall": best_recall}

    # F1 peak vs lead-window at the F1-optimal gate from all windows combined
    if all_cache:
        bc, bm = max(all_cache, key=lambda r: r[4])[:2]
        com = _committed_stream(probs, self_imp, offsets, bc, bm)[0]
        m_fr, m_w, m_ep = _anticip_events(com, offsets, nonzero=False)
        mby = _anticip_group(m_ep)
        n_model = len(m_fr)
        fw_rows = []
        for W in range(0, int(round(1.5 * hz)) + 1):
            matched, lat, _ = _match_anticip(h_fr, h_w, h_ep, m_fr, m_w, mby, W)
            recall = matched / n_human if n_human else 0.0
            prec = matched / n_model if n_model else float("nan")
            f1 = (2 * prec * recall / (prec + recall)
                  if recall and prec and prec + recall > 0 else 0.0)
            fw_rows.append({
                "lead_s": round(W / hz, 3), "recall": recall, "prec": prec, "f1": f1,
                "fp_per_min": (n_model - matched) / sim_min,
                "lat_med": float(np.median(lat)) if lat else float("nan"),
            })
        result["f1_vs_lead_window"] = {"C": bc, "M": bm, "rows": fw_rows}

    return result


# ---------------------------------------------------------------------------
# switch_window_f1 — INTENT-ONLY windowed F1 (the weapon-switch ARBITER)
# ---------------------------------------------------------------------------

def _intent_window_score(m_fr, m_w, m_ep, hi, hf, W, n_intent):
    """Backward-only intent match → (recall, precision, f1, n_model, FP).

    Recall/precision scored against human INTENT switches only. Matching is
    BACKWARD-ONLY (model in [h-W, h]) — a forward window would credit the held
    weapon catching up to the intent target (~1 frame late) = copycat. Model
    switches coincident (+/-W) with a FORCED switch are excluded from precision
    (forced-coincident: neither TP nor FP — the engine owns forced switches).
    """
    mby: dict = defaultdict(list)
    for j in range(len(m_fr)):
        mby[(int(m_ep[j]), int(m_w[j]))].append(j)
    hi_fr, hi_w, hi_ep = hi
    pairs = []
    for i in range(len(hi_fr)):
        for j in mby.get((int(hi_ep[i]), int(hi_w[i])), ()):
            dt = int(m_fr[j]) - int(hi_fr[i])
            if -W <= dt <= 0:
                pairs.append((-dt, i, j))
    pairs.sort()
    hu: set = set(); mu: set = set()
    for _a, i, j in pairs:
        if i in hu or j in mu:
            continue
        hu.add(i); mu.add(j)
    TP = len(mu)
    recall = TP / n_intent if n_intent else 0.0
    fby: dict = defaultdict(list)
    hf_fr, hf_w, hf_ep = hf
    for k in range(len(hf_fr)):
        fby[(int(hf_ep[k]), int(hf_w[k]))].append(int(hf_fr[k]))
    nM = len(m_fr); fc = 0
    for j in range(nM):
        if j in mu:
            continue
        if any(abs(int(m_fr[j]) - ff) <= W for ff in fby.get((int(m_ep[j]), int(m_w[j])), ())):
            fc += 1
    FP = nM - TP - fc
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    return recall, prec, f1, nM, FP


def switch_window_f1(npz: Path, *, windows=(0.0, 0.5, 1.0)) -> dict:
    """INTENT-ONLY windowed switch F1 (recall + precision) — the weapon arbiter.

    Forced switches (ammo-out / pickup / respawn) are the ENGINE's job, so
    recall/precision are scored against the human's INTENT switches only
    (self_imp != new weapon). Matching is BACKWARD-ONLY (see _intent_window_score):
    a pure copycat scores F1~0.01 here, vs a bogus ~0.9 under a symmetric window
    (the held-weapon catch-up). This is the precision-aware, timing-tolerant
    successor to decode_sweep's recall-only intent_recall and the forced-inclusive
    anticip_roc. Per (C,M,W): recall, precision, f1, committed/min, spurious/min.
    """
    d = np.load(npz)
    probs = d["probs"].astype(np.float32)
    weapon = d["weapon"].astype(np.int64); self_imp = d["self_imp"].astype(np.int64)
    offsets = d["offsets"].astype(np.int64); hz = float(d["hz"])
    sim_min = int(offsets[-1]) / hz / 60.0
    h_fr, h_w, h_ep = _anticip_events(weapon, offsets, nonzero=True)
    is_int = self_imp[h_fr] != h_w
    hi = (h_fr[is_int], h_w[is_int], h_ep[is_int])
    hf = (h_fr[~is_int], h_w[~is_int], h_ep[~is_int])
    n_intent = int(len(hi[0]))
    result: dict[str, Any] = {
        "hz": hz, "sim_min": sim_min, "n_intent": n_intent,
        "intent_per_min": round(n_intent / sim_min, 3), "windows": {},
    }
    for Wsec in windows:
        W = int(round(Wsec * hz))
        rows = []
        for M in _DS_M_GRID:
            for C in _DS_C_GRID:
                com = _committed_stream(probs, self_imp, offsets, C, M)[0]
                m_fr, m_w, m_ep = _anticip_events(com, offsets, nonzero=False)
                rec, prec, f1, nM, FP = _intent_window_score(
                    m_fr, m_w, m_ep, hi, hf, W, n_intent)
                rows.append({"C": C, "M": M, "recall": rec, "prec": prec, "f1": f1,
                             "committed_per_min": nM / sim_min, "spur_per_min": FP / sim_min})
        best = max(rows, key=lambda r: r["f1"]) if rows else None
        result["windows"][f"{Wsec:g}s"] = {"rows": rows, "best_f1": best}
    return result


# ---------------------------------------------------------------------------
# decode_sweep  (_weapon_decode_sweep.py core — npz-input)
# ---------------------------------------------------------------------------

_DS_C_GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.85, 0.9]
_DS_M_GRID = [0.0, 0.1]


def _ds_events(stream, offs, nonzero):
    n = len(stream); chg = np.zeros(n, bool); chg[1:] = stream[1:] != stream[:-1]
    ep = np.zeros(n, bool); ep[offs[:-1][offs[:-1] < n]] = True
    chg &= ~ep
    if nonzero:
        chg &= stream != 0
    idx = np.flatnonzero(chg)
    return idx, stream[idx].astype(int), (np.searchsorted(offs, idx, "right") - 1)


def _ds_standing_intent(stream, offs):
    s = stream.copy()
    for i in range(len(offs) - 1):
        a, b = int(offs[i]), int(offs[i + 1]); seg = s[a:b]; nz = seg != 0
        if not nz.any():
            continue
        idx = np.maximum.accumulate(np.where(nz, np.arange(b - a), 0))
        f = seg[idx]; f[: int(np.argmax(nz))] = 0; s[a:b] = f
    return s


def _ds_dwell(stream, offs):
    out = []
    for i in range(len(offs) - 1):
        seg = stream[int(offs[i]):int(offs[i + 1])]
        if len(seg) == 0:
            continue
        b = np.flatnonzero(np.diff(seg) != 0) + 1
        out.extend(np.diff(np.concatenate([[0], b, [len(seg)]])))
    return np.asarray(out, float)


def _ds_emd(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    a = np.sort(a); b = np.sort(b); g = np.union1d(a, b)
    ca = np.searchsorted(a, g, "right") / len(a)
    cb = np.searchsorted(b, g, "right") / len(b)
    w = np.diff(np.concatenate([g, g[-1:]]))
    return float(np.sum(np.abs(ca - cb)[:-1] * w[:-1]))


def _matched_frac(h_fr, h_w, h_ep, m_by, W, backward_only):
    if len(h_fr) == 0:
        return 0.0
    hit = 0
    for k in range(len(h_fr)):
        e = int(h_ep[k]); f = int(h_fr[k]); w = int(h_w[k])
        for mi in m_by.get(e, ()):
            dt = mi[0] - f
            ok = (-W <= dt <= 0) if backward_only else (abs(dt) <= W)
            if mi[1] == w and ok:
                hit += 1; break
    return hit / len(h_fr)


def decode_sweep(npz: Path, *, well_timed: float = 0.5,
                 onset_classes: Path | None = None) -> dict:
    """Intent-aware (C,M) decode sweep from a probs npz.

    Uses the canonical _committed_stream (seeded from self_imp = self_weapon_id
    impulses). Reports intent_recall, forced_recall, spurious/min, committed/min,
    dwell_emd per (C,M).

    ARBITER IS INTENT-ONLY. Forced switches (ammo-out / pickup / respawn) are the
    ENGINE's job, not the model's, so the operating-point recommendation is gated
    on intent_recall + spurious only. ``dwell_emd`` is computed against the
    forced-INCLUSIVE occupancy (``_ds_standing_intent`` forward-fills the full
    weapon stream), so it REWARDS echoing the engine's forced churn (= copycat).
    It is kept as a DIAGNOSTIC column only and MUST NOT gate the recommendation.
    (See the retired "dwell-EMD gate principle".)

    Parameters
    ----------
    npz:
        Probs npz with arrays: probs, weapon, self_imp, offsets, hz.
    well_timed:
        Match window in seconds.
    onset_classes:
        Optional npz from ``_weapon_onset_classes.py`` (script-aware onset
        classification, weapon-head.md §10). When given, the INTENT reference
        is ``class == deliberate`` — script cycles (dump/return) and forced
        switches are all excluded from the reference AND from the spurious
        budget rate. Without it the legacy impulse-lead test is used, which
        §10 showed is cooldown-gated detectability, not deliberateness.
    """
    d = np.load(npz)
    weapon = d["weapon"].astype(int); si = d["self_imp"].astype(int)
    offs = d["offsets"].astype(int); probs = d["probs"].astype(np.float32)
    pred = probs.argmax(1) + 1
    s = np.sort(probs, 1); conf = s[:, -1]; margin = s[:, -1] - s[:, -2]
    hz = float(d["hz"]); sim_min = int(offs[-1]) / hz / 60.0
    W = int(round(well_timed * hz))

    h_idx, h_w, h_ep = _ds_events(weapon, offs, True)
    class_counts = None
    if onset_classes is not None:
        oc = np.load(onset_classes)
        if not np.array_equal(oc["onset_idx"].astype(int), h_idx):
            raise ValueError(
                f"{onset_classes}: onset population mismatch — the classes npz "
                "was built for a different cache/segment than this probs dump")
        names = [str(x) for x in oc["class_names"]]
        cls = oc["classes"].astype(int)
        is_intent = cls == names.index("deliberate")
        class_counts = {n: int((cls == i).sum()) for i, n in enumerate(names)}
    else:
        is_intent = si[h_idx] != h_w
    hi_fr, hi_w, hi_ep = h_idx[is_intent], h_w[is_intent], h_ep[is_intent]
    hf_fr, hf_w, hf_ep = h_idx[~is_intent], h_w[~is_intent], h_ep[~is_intent]
    intent_rate = len(hi_fr) / sim_min
    human_all = set(zip(h_ep.tolist(), h_idx.tolist(), h_w.tolist()))

    sintent = _ds_standing_intent(weapon, offs); hd = _ds_dwell(sintent, offs)
    rows = []
    for M in _DS_M_GRID:
        for C in _DS_C_GRID:
            com = _committed_stream(probs, si, offs, C, M)[0]
            m_idx, m_w2, m_ep = _ds_events(com, offs, False)
            m_by: dict = defaultdict(list)
            for f, w, e in zip(m_idx, m_w2, m_ep):
                m_by[int(e)].append((int(f), int(w)))
            ir = _matched_frac(hi_fr, hi_w, hi_ep, m_by, W, backward_only=True)
            fr = _matched_frac(hf_fr, hf_w, hf_ep, m_by, W, backward_only=False)
            h_by: dict = defaultdict(list)
            for e, f, w in human_all:
                h_by[e].append((f, w))
            spur = sum(
                1 for f, w, e in zip(m_idx, m_w2, m_ep)
                if not any(ww == w and abs(ff - f) <= W
                           for ff, ww in h_by.get(int(e), ()))
            )
            de = _ds_emd(_ds_dwell(com, offs), hd)
            rows.append({
                "C": C, "M": M,
                "intent_recall": ir, "forced_recall": fr,
                "spur_per_min": spur / sim_min,
                "committed_per_min": len(m_idx) / sim_min,
                "dwell_emd": de,
            })

    # INTENT-ONLY operating point. Forced switches are the engine's job, so the
    # arbiter is intent_recall + spurious — NOT dwell_emd (forced-inclusive, rewards
    # copycat). Pick the highest intent_recall whose spurious rate stays within the
    # human intent budget; tie-break on higher C (stickier = less chatter).
    recommendations = []
    for M in _DS_M_GRID:
        cand = [r for r in rows if r["M"] == M and r["spur_per_min"] <= intent_rate]
        if cand:
            r = max(cand, key=lambda r: (r["intent_recall"], r["C"]))
            recommendations.append({"M": M, "C": r["C"],
                                     "intent_recall": r["intent_recall"],
                                     "spur_per_min": r["spur_per_min"],
                                     "committed_per_min": r["committed_per_min"],
                                     "dwell_emd": r["dwell_emd"]})  # diagnostic only
        else:
            recommendations.append({"M": M, "C": None,
                                     "note": "no C keeps spurious <= human intent rate"})

    return {
        "hz": hz, "sim_min": sim_min,
        "human_intent_rate": intent_rate,
        "intent_switch_count": int(len(hi_fr)),
        "forced_switch_count": int(len(hf_fr)),
        "onset_class_counts": class_counts,
        "rows": rows,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# switch_decompose  (_weapon_switch_decompose.py core)
# ---------------------------------------------------------------------------

def switch_decompose(policy, source) -> dict:
    """Decompose argmax stream switches: recall of true switches vs chatter.

    Returns structured dict with peakedness, switch rates, tolerance-windowed
    recall/precision.
    """
    probs, label = collect_frames(policy, source)
    offs = np.asarray(source.episode_offsets, dtype=np.int64)
    pred = probs.argmax(1) + 1
    n = len(label)
    valid = label != 0
    p_inc = probs[np.arange(n), label - 1]

    has_prev = np.ones(n, dtype=bool)
    has_prev[offs[:-1][offs[:-1] < n]] = False
    true_sw = np.zeros(n, dtype=bool); true_sw[1:] = label[1:] != label[:-1]
    pred_sw = np.zeros(n, dtype=bool); pred_sw[1:] = pred[1:] != pred[:-1]
    m = has_prev & valid
    th = m & true_sw   # frames where demonstrator switched
    ho = m & ~true_sw  # frames where demonstrator held

    tol_rows = []
    for k in (0, 1, 2, 3, 5):
        pd = _dilate_within_episodes(pred_sw, offs, -k, k)
        td = _dilate_within_episodes(true_sw, offs, -k, k)
        recall = float(pd[th].mean()) if th.sum() else float("nan")
        prec = float(td[m & pred_sw].mean()) if (m & pred_sw).sum() else float("nan")
        tol_rows.append({"k": k, "recall": recall, "precision": prec})

    return {
        "n_frames": n,
        "mean_p_incumbent": float(p_inc[valid].mean()),
        "argmax_eq_incumbent_rate": float((pred[valid] == label[valid]).mean()),
        "true_switch_rate_pct": round(true_sw[m].mean() * 100, 3),
        "pred_switch_rate_pct": round(pred_sw[m].mean() * 100, 3),
        "hold_chatter_pct": round(pred_sw[ho].mean() * 100, 3),
        "tolerance_windowed": tol_rows,
    }


# ---------------------------------------------------------------------------
# switch_timing_detail  (_weapon_switch_timing_detail.py core)
# ---------------------------------------------------------------------------

_STD_KS = [0, 1, 2, 3, 5, 8, 12, 16, 20, 30, 40]


def switch_timing_detail(policy, source) -> dict:
    """Detailed switch timing: symmetric recall/precision + directional analysis."""
    probs, label = collect_frames(policy, source)
    offs = np.asarray(source.episode_offsets, dtype=np.int64)
    pred = probs.argmax(1) + 1
    n = len(label)
    start = np.zeros(n, bool); start[offs[:-1][offs[:-1] < n]] = True
    true_sw = np.zeros(n, bool); true_sw[1:] = label[1:] != label[:-1]; true_sw &= ~start
    pred_sw = np.zeros(n, bool); pred_sw[1:] = pred[1:] != pred[:-1]; pred_sw &= ~start

    # signed nearest offset per true switch
    offs_signed = []; miss = 0
    for i in range(len(offs) - 1):
        s, e = int(offs[i]), int(offs[i + 1])
        if e <= s:
            continue
        ti = np.flatnonzero(true_sw[s:e]); pi = np.flatnonzero(pred_sw[s:e])
        if len(ti) == 0:
            continue
        if len(pi) == 0:
            miss += len(ti); continue
        idx = np.searchsorted(pi, ti)
        for j, t in enumerate(ti):
            cand = []
            if idx[j] < len(pi): cand.append(pi[idx[j]] - t)
            if idx[j] > 0: cand.append(pi[idx[j] - 1] - t)
            d = min(cand, key=abs)
            if abs(d) <= 40: offs_signed.append(int(d))
            else: miss += 1
    so = np.array(offs_signed)
    nt = int(true_sw.sum())

    sym_rows = []
    for k in _STD_KS:
        pd = _dilate_within_episodes(pred_sw, offs, -k, k)
        td = _dilate_within_episodes(true_sw, offs, -k, k)
        rec = float(pd[true_sw].mean() * 100) if true_sw.any() else float("nan")
        prec = float(td[pred_sw].mean() * 100) if pred_sw.any() else float("nan")
        sym_rows.append({"k": k, "recall_pct": rec, "precision_pct": prec})

    directional: dict[str, Any] = {"n_matched": len(so), "n_total": nt, "n_missed": miss}
    if len(so):
        directional.update({
            "lead_pct": float((so < 0).mean() * 100),
            "exact_pct": float((so == 0).mean() * 100),
            "lag_pct": float((so > 0).mean() * 100),
            "median_offset": float(np.median(so)),
            "mean_offset": float(so.mean()),
            "p10": float(np.percentile(so, 10)),
            "p90": float(np.percentile(so, 90)),
        })
    one_sided = []
    for k in [1, 2, 5, 10, 20, 40]:
        lead = float(_dilate_within_episodes(pred_sw, offs, -k, 0)[true_sw].mean() * 100
                     if true_sw.any() else float("nan"))
        lag = float(_dilate_within_episodes(pred_sw, offs, 0, k)[true_sw].mean() * 100
                    if true_sw.any() else float("nan"))
        one_sided.append({"k": k, "lead_recall_pct": lead, "lag_recall_pct": lag})

    # attack-frame break-down
    attack_stats: dict | None = None
    atk = source.actions.get("attack")
    if atk is not None:
        atk = (np.asarray(atk.cpu() if hasattr(atk, "cpu") else atk).reshape(-1) > 0.5)
        valid = label != 0
        correct = pred == label
        af = atk & valid; na = (~atk) & valid
        f1s = []
        for c in range(1, 9):
            gc = af & (label == c); pc = af & (pred == c)
            tp = int((gc & pc).sum()); fp = int((pc & ~gc).sum()); fn = int((gc & ~pc).sum())
            pr = tp / (tp + fp) if tp + fp else 0.0
            re = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * pr * re / (pr + re) if pr + re else 0.0)
        post_sw = []
        for k in (3, 5, 10, 20):
            m2 = af & _dilate_within_episodes(true_sw, offs, -k, 0)
            if m2.sum():
                post_sw.append({"k": k, "acc_pct": float(correct[m2].mean() * 100),
                                 "n": int(m2.sum())})
        attack_stats = {
            "attack_frame_pct": float(af.mean() * 100),
            "acc_at_attack": float(correct[af].mean() * 100) if af.any() else float("nan"),
            "acc_at_non_attack": float(correct[na].mean() * 100) if na.any() else float("nan"),
            "macro_f1_at_attack": float(np.mean(f1s)),
            "post_switch_acc": post_sw,
        }

    return {
        "n_true_switches": nt,
        "n_pred_switches": int(pred_sw.sum()),
        "symmetric_tol": sym_rows,
        "directional": directional,
        "one_sided_cumulative": one_sided,
        "attack_frames": attack_stats,
    }


# ---------------------------------------------------------------------------
# switch_leadtime  (_weapon_switch_leadtime.py core)
# ---------------------------------------------------------------------------

def switch_leadtime(policy, source) -> dict:
    """Of switch frames (target!=equipped): intent-leads-engine vs stale label."""
    probs, label = collect_frames(policy, source)
    offs = np.asarray(source.episode_offsets, np.int64)
    tok = _tok_impulse(source)
    pred = probs.argmax(1) + 1
    n = len(label)
    valid = label != 0; correct = pred == label
    div = valid & (label != tok)  # genuine switch command

    W = 40
    equip_lead = np.full(n, -1, int)
    for i in range(len(offs) - 1):
        s, e = int(offs[i]), int(offs[i + 1])
        for t in range(s, e):
            if not div[t]:
                continue
            tgt = label[t]
            for k in range(1, W + 1):
                if t + k >= e:
                    break
                if tok[t + k] == tgt:
                    equip_lead[t] = k; break

    ant = div & (equip_lead > 0)   # in-flight: engine equips target soon
    stale = div & (equip_lead < 0)  # engine never equips target within 40f
    lt = equip_lead[ant]

    return {
        "n_switch_frames": int(div.sum()),
        "intent_leads_engine": {
            "count": int(ant.sum()),
            "pct": round(ant.sum() / div.sum() * 100, 2) if div.any() else float("nan"),
            "head_acc_pct": round(float(correct[ant].mean() * 100), 2) if ant.any() else float("nan"),
        },
        "stale_label": {
            "count": int(stale.sum()),
            "pct": round(stale.sum() / div.sum() * 100, 2) if div.any() else float("nan"),
            "head_acc_pct": round(float(correct[stale].mean() * 100), 2) if stale.any() else float("nan"),
        },
        "lead_time": {
            "median_frames": float(np.median(lt)) if len(lt) else float("nan"),
            "mean_frames": float(lt.mean()) if len(lt) else float("nan"),
            "p90_frames": float(np.percentile(lt, 90)) if len(lt) else float("nan"),
            "within_1f_pct": float((lt <= 1).mean() * 100) if len(lt) else float("nan"),
            "within_3f_pct": float((lt <= 3).mean() * 100) if len(lt) else float("nan"),
            "within_5f_pct": float((lt <= 5).mean() * 100) if len(lt) else float("nan"),
            "within_10f_pct": float((lt <= 10).mean() * 100) if len(lt) else float("nan"),
        },
        "overall_switch_acc_pct": round(float(correct[div].mean() * 100), 2) if div.any() else float("nan"),
    }


# ---------------------------------------------------------------------------
# switch_leak_test  (_weapon_switch_leak_test.py core)
# ---------------------------------------------------------------------------

def switch_leak_test(policy, source) -> dict:
    """Leak test: model vs input-token as switch detectors, token-echo analysis."""
    probs, label = collect_frames(policy, source)
    offs = np.asarray(source.episode_offsets, np.int64)
    tok = _tok_impulse(source)
    pred = probs.argmax(1) + 1
    n = len(label)
    start = np.zeros(n, bool); start[offs[:-1][offs[:-1] < n]] = True

    def sw(a):
        s = np.zeros(n, bool); s[1:] = a[1:] != a[:-1]; s &= ~start; return s

    true_sw, pred_sw, tok_sw = sw(label), sw(pred), sw(tok)

    sym_rows = []
    for k in (0, 1, 2, 3, 5, 10, 20, 40):
        pd = _dilate_within_episodes(pred_sw, offs, -k, k)
        td = _dilate_within_episodes(true_sw, offs, -k, k)
        kd = _dilate_within_episodes(tok_sw, offs, -k, k)
        mr = float(pd[true_sw].mean() * 100) if true_sw.any() else float("nan")
        mp = float(td[pred_sw].mean() * 100) if pred_sw.any() else float("nan")
        tr = float(kd[true_sw].mean() * 100) if true_sw.any() else float("nan")
        tp = float(td[tok_sw].mean() * 100) if tok_sw.any() else float("nan")
        sym_rows.append({"k": k, "model_recall_pct": mr, "model_prec_pct": mp,
                         "token_recall_pct": tr, "token_prec_pct": tp})

    prevlab = np.empty(n, int); prevlab[1:] = label[:-1]; prevlab[0] = 0
    ts = true_sw
    tok_new = (tok == label) & ts
    tok_old = (tok == prevlab) & ts

    so = []
    for i in range(len(offs) - 1):
        s, e = int(offs[i]), int(offs[i + 1])
        pi = np.flatnonzero(pred_sw[s:e]); ki = np.flatnonzero(tok_sw[s:e])
        if len(pi) == 0 or len(ki) == 0:
            continue
        idx = np.searchsorted(ki, pi)
        for j, p in enumerate(pi):
            c = []
            if idx[j] < len(ki): c.append(ki[idx[j]] - p)
            if idx[j] > 0: c.append(ki[idx[j] - 1] - p)
            d = min(c, key=abs)
            if abs(d) <= 40: so.append(int(d))
    so_arr = np.array(so)

    timing: dict[str, Any] = {"n_matched": len(so_arr)}
    if len(so_arr):
        timing.update({
            "model_leads_token_pct": float((so_arr > 0).mean() * 100),
            "same_frame_pct": float((so_arr == 0).mean() * 100),
            "model_lags_token_pct": float((so_arr < 0).mean() * 100),
            "median": float(np.median(so_arr)),
            "mean": float(so_arr.mean()),
        })

    return {
        "n_true_switches": int(ts.sum()),
        "detection_by_window": sym_rows,
        "token_leak_at_switch": {
            "leaked_pct": round(tok_new.sum() / ts.sum() * 100, 2) if ts.any() else float("nan"),
            "genuine_pct": round(tok_old.sum() / ts.sum() * 100, 2) if ts.any() else float("nan"),
            "third_weapon_pct": round(
                (ts.sum() - tok_new.sum() - tok_old.sum()) / ts.sum() * 100, 2
            ) if ts.any() else float("nan"),
        },
        "model_vs_token_timing": timing,
    }


# ---------------------------------------------------------------------------
# switch_vs_token  (_weapon_switch_vs_token.py core)
# ---------------------------------------------------------------------------

def switch_vs_token(policy, source) -> dict:
    """Leak-free decision-skill metric: accuracy on switch frames (target!=token)."""
    probs, label = collect_frames(policy, source)
    tok = _tok_impulse(source)
    atk_raw = source.actions.get("attack")
    atk = (np.asarray(atk_raw.cpu() if hasattr(atk_raw, "cpu") else atk_raw).reshape(-1) > 0.5
           if atk_raw is not None else None)

    pred = probs.argmax(1) + 1
    valid = label != 0; correct = pred == label
    hold = valid & (label == tok)   # target == equipped -> hold
    div = valid & (label != tok)    # target != equipped -> switch command

    result = {
        "n_valid": int(valid.sum()),
        "hold_frames": {
            "pct": round(hold.sum() / valid.sum() * 100, 2) if valid.any() else float("nan"),
            "acc_pct": round(float(correct[hold].mean() * 100), 2) if hold.any() else float("nan"),
        },
        "switch_frames": {
            "pct": round(div.sum() / valid.sum() * 100, 2) if valid.any() else float("nan"),
            "acc_pct": round(float(correct[div].mean() * 100), 2) if div.any() else float("nan"),
            "eq_target_pct": round(float(correct[div].mean() * 100), 2) if div.any() else float("nan"),
            "eq_token_pct": round(float((pred[div] == tok[div]).mean() * 100), 2) if div.any() else float("nan"),
            "eq_other_pct": round(
                float(((pred[div] != label[div]) & (pred[div] != tok[div])).mean() * 100), 2
            ) if div.any() else float("nan"),
        },
        "overall_acc_pct": round(float(correct[valid].mean() * 100), 2) if valid.any() else float("nan"),
    }
    if atk is not None:
        result["attack_breakdown"] = {
            "switch_at_attack": {
                "n": int((div & atk).sum()),
                "acc_pct": round(float(correct[div & atk].mean() * 100), 2) if (div & atk).any() else float("nan"),
            },
            "hold_at_attack_acc_pct": round(float(correct[hold & atk].mean() * 100), 2) if (hold & atk).any() else float("nan"),
        }
    return result


# ---------------------------------------------------------------------------
# switchframe_decomp  (_weapon_switchframe_decomp.py core — needs forward pass)
# ---------------------------------------------------------------------------

_SFD_C_GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.57, 0.6, 0.7, 0.8, 0.82, 0.9]
_SFD_M_GRID = [0.0, 0.1, 0.2]


def switchframe_decomp(policy, source, *, dump: Path | None = None) -> dict:
    """Switch-frame decomposition: HIT/STUCK/OTHER + (C,M) tradeoff table.

    Uses the deployment-faithful _forward_weapon_logits (no bench side-channel).
    """
    logits = _forward_weapon_logits(policy, source)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    weapon = _to_np(source.actions["weapon"]).reshape(-1).astype(np.int64)
    wid_raw = torch.as_tensor(_to_np(source.obs["self_weapon_id"]).reshape(-1)).long()
    self_imp = _to_np(self_weapon_id_to_impulse(wid_raw)).reshape(-1).astype(np.int64)
    attack = _to_np(source.actions["attack"]).reshape(-1).astype(bool)

    offsets = np.asarray(source.episode_offsets, dtype=np.int64)
    n_ep = len(offsets) - 1; n_fr = int(offsets[-1])

    argmax = probs.argmax(1) + 1
    conf = probs.max(1)
    top2 = np.sort(probs, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]

    sw_idx, sw_wnew, sw_wprev = [], [], []
    for i in range(n_ep):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e - s < 2:
            continue
        w = weapon[s:e]
        chg = np.flatnonzero((w[1:] != w[:-1]) & (w[1:] != 0)) + 1
        for f in chg:
            sw_idx.append(s + f); sw_wnew.append(int(w[f])); sw_wprev.append(int(w[f - 1]))
    sw_idx = np.asarray(sw_idx, dtype=np.int64)
    sw_wnew = np.asarray(sw_wnew, dtype=np.int64)
    sw_wprev = np.asarray(sw_wprev, dtype=np.int64)
    n = len(sw_idx)

    if n == 0:
        return {"n_switch_frames": 0}

    g_argmax = argmax[sw_idx]; g_conf = conf[sw_idx]; g_margin = margin[sw_idx]
    g_held = self_imp[sw_idx]; g_pwnew = probs[sw_idx, sw_wnew - 1]; g_pargmax = g_conf

    hit = g_argmax == sw_wnew
    stuck = (~hit) & (g_argmax == g_held)
    other = (~hit) & (~stuck)

    def _bucket(mask):
        k = int(mask.sum())
        if k == 0:
            return {"n": 0, "pct": 0.0, "mean_p_argmax": float("nan"), "mean_p_wnew": float("nan")}
        return {"n": k, "pct": round(k / n * 100, 2),
                "mean_p_argmax": float(g_pargmax[mask].mean()),
                "mean_p_wnew": float(g_pwnew[mask].mean())}

    is_sw = np.zeros(n_fr, dtype=bool); is_sw[sw_idx] = True
    hold_mask = attack & (self_imp > 0) & (~is_sw)
    h_argmax = argmax[hold_mask]; h_conf = conf[hold_mask]
    h_margin = margin[hold_mask]; h_held = self_imp[hold_mask]
    n_hold = int(hold_mask.sum())

    sweep_rows = []
    for M in _SFD_M_GRID:
        for C in _SFD_C_GRID:
            gate_sw = (g_conf >= C) & (g_margin >= M)
            tr = float((hit & gate_sw).mean())
            ar = float(((g_argmax != g_held) & gate_sw).mean())
            fs = float(((h_argmax != h_held) & (h_conf >= C) & (h_margin >= M)).mean()) if n_hold else float("nan")
            sweep_rows.append({"C": C, "M": M,
                               "target_recall": tr, "anyswitch_recall": ar,
                               "false_switch": fs})

    result = {
        "n_switch_frames": n,
        "n_hold_frames": n_hold,
        "ceiling_target_recall": float(hit.mean()),
        "buckets": {
            "hit": _bucket(hit),
            "stuck": _bucket(stuck),
            "other": _bucket(other),
        },
        "low_p_wnew_tail": {},
        "sweep": sweep_rows,
    }
    for thr in (0.10, 0.30):
        lo = g_pwnew < thr
        klo = int(lo.sum())
        if klo:
            h2 = int((hit & lo).sum()); s2 = int((stuck & lo).sum()); o2 = int((other & lo).sum())
            result["low_p_wnew_tail"][f"p_wnew_lt_{thr:.2f}"] = {
                "n": klo, "pct_of_switches": round(klo / n * 100, 2),
                "hit_pct": round(h2 / klo * 100, 2),
                "stuck_pct": round(s2 / klo * 100, 2),
                "other_pct": round(o2 / klo * 100, 2),
                "mean_p_argmax": float(g_pargmax[lo].mean()),
            }

    if dump is not None:
        dump.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dump, idx=sw_idx, w_new=sw_wnew, w_prev=sw_wprev,
                            held=g_held, argmax=g_argmax, conf=g_conf, margin=g_margin,
                            p_wnew=g_pwnew)

    return result


# ---------------------------------------------------------------------------
# when_switch_detect  (_weapon_when_switch_detect.py core)
# ---------------------------------------------------------------------------

def _rank_auc(score, y):
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def when_switch_detect(policy, source) -> dict:
    """WHEN-head switch detection: hazard AUC + windowed recall/precision."""
    head = policy.model.weapon_head
    dev = policy.device
    with torch.inference_mode():
        di = torch.arange(head._n_dwell, device=dev)
        h = torch.sigmoid(head.when_mlp(head.dwell_embed(di)).reshape(-1)).float().cpu().numpy()

    offs = np.asarray(source.episode_offsets, dtype=np.int64)
    w_raw = source.actions["weapon"]
    w = (w_raw.cpu().numpy() if torch.is_tensor(w_raw) else np.asarray(w_raw)).reshape(-1).astype(np.int64)
    n = len(w)

    dwell = np.zeros(n, dtype=np.int64)
    switch_next = np.zeros(n, dtype=bool)
    has_next = np.zeros(n, dtype=bool)
    for i in range(len(offs) - 1):
        s, e = int(offs[i]), int(offs[i + 1])
        if e <= s:
            continue
        seg = w[s:e]; d = 0
        for t in range(len(seg)):
            dwell[s + t] = d
            if t + 1 < len(seg):
                has_next[s + t] = True
                if seg[t + 1] != seg[t]:
                    switch_next[s + t] = True; d = 0
                else:
                    d += 1

    hazard = h[np.clip(dwell, 0, len(h) - 1)]
    m = has_next
    true_rate = float(switch_next[m].mean())
    auc = _rank_auc(hazard[m], switch_next[m])

    midx = np.flatnonzero(m)
    k_alarm = int(round(true_rate * len(midx)))
    order = np.argsort(hazard[midx], kind="mergesort")[::-1][:k_alarm]
    pred = np.zeros(n, dtype=bool); pred[midx[order]] = True

    th = m & switch_next; pp = m & pred
    tol_rows = []
    for k in (0, 1, 2, 3, 5):
        pdil = _dilate_within_episodes(pred, offs, -k, k)
        tdil = _dilate_within_episodes(switch_next, offs, -k, k)
        rec = float(pdil[th].mean()) if th.sum() else float("nan")
        prec = float(tdil[pp].mean()) if pp.sum() else float("nan")
        tol_rows.append({"k": k, "recall": rec, "precision": prec})

    return {
        "hazard_auc": auc,
        "true_switch_rate": true_rate,
        "pred_rate_top_k": float(pred[m].mean()),
        "tolerance_windowed": tol_rows,
    }


# ---------------------------------------------------------------------------
# analyze  — (policy, source)-compatible entry point
# ---------------------------------------------------------------------------

def analyze(
    policy,
    source,
    *,
    segment: str = "engaged",
) -> dict:
    """Run the (policy, source)-compatible weapon-head analysis functions.

    Calls ``gate_sweep``, ``switch_gated``, ``switch_decompose``, and
    ``switch_vs_token`` and returns a per-head dict.

    Functions that require additional inputs — npz caches
    (``anticip_roc``, ``decode_sweep``, ``intent_psth``), a different forward
    path (``switchframe_decomp``, ``switch_window_roc``), a WHEN-head
    (``when_switch_detect``), or a corpus-level loader
    (``corpus_stats``, ``intent_decompose``) — are standalone named functions
    with their own signatures.

    Parameters
    ----------
    policy:
        Loaded QNNPolicy (eval mode).
    source:
        Resident source with ``segment_mask=SEGMENT_MASK``.
    segment:
        Metadata tag (e.g. ``"engaged"``).

    Returns
    -------
    dict with keys matching the Phase-2 per-head schema.
    """
    # The (policy, source) pass is shared across these four tools.
    probs, label = collect_frames(policy, source)
    tok = _tok_impulse(source)
    offs = np.asarray(source.episode_offsets, np.int64)
    valid = label != 0

    # grade
    g = grade(probs, label)

    # switch_gated at default gate (0.65/0.15)
    gated = switch_gated(policy, source)

    # switch_decompose
    decompose = switch_decompose(policy, source)

    # switch_vs_token
    svt = switch_vs_token(policy, source)

    return {
        "segment": segment,
        "grade": g,
        "switch_gated": gated,
        "switch_decompose": decompose,
        "switch_vs_token": svt,
    }
