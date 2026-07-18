"""Canonical weapon-switch metric definitions — the single source of truth so the
two distinct "switch" quantities are never conflated again (they were, once:
a per-frame intent!=state fraction got mislabeled as the switch rate).

TWO DIFFERENT THINGS — do not mix them:

1. SWITCH EVENT  (this is "the switch rate")
   A weapon-CHANGE event: the weapon transitions, w[t] != w[t-1], within an episode.
   THE CANONICAL SEQUENCE IS ``act.weapon`` (the per-frame weapon LABEL = what the
   head predicts), matching the doc's detector (_weapon_corpus_stats.py uses
   actions["weapon"]). Human rate ≈ **4.03% of combat frames** (weapon-head.md §3).
   This is the human-likeness target — reproduce the switch-event RATE and the
   dwell-time distribution (weapon-head.md §4, EMD/MMD). The model's analogue is its
   predicted-weapon (argmax/gated) transition rate. Use `switch_events()` /
   `switch_rate()` / `dwell_times()` on the act.weapon (label) / predicted sequence.

   DO NOT use self_weapon_id (engine *equipped* state) for the switch rate — its
   transition rate is a DIFFERENT, lower quantity (~2.4%; equipped lags/differs from
   the intent label, the auto-switch gap). The doc's 4.03% is act.weapon.

2. INTENT-STATE MISMATCH  (NOT a switch rate)
   Frames where the per-frame weapon LABEL (act.weapon, the intended weapon)
   differs from the equipped weapon (self_weapon_id). ≈ **25%** of frames, and
   per memory `weapon_label_vs_state` that gap is largely the *known auto-switch
   labeling artifact* (e.g. rl→sg on ammo-out where intent lags engine state) —
   NOT player switch decisions. The `_weapon_switch_vs_token` "leak-free decision
   skill" metric scores prediction on these frames; that's legitimate, but its
   ~25% rate is NOT the switch rate. Use `intent_state_mismatch()` and label it
   as such.

Rule of thumb / sanity guard: anything you call a "switch rate" should be ~4%
(combat). If it's ~20-25% you are measuring intent!=state, not switch events —
`assert_is_event_rate()` enforces this.
"""
from __future__ import annotations
import numpy as np

HUMAN_COMBAT_SWITCH_RATE = 0.0403   # weapon-head.md §3 (combat = argmax(target_probs)!=0)
INTENT_STATE_MISMATCH_RATE = 0.257  # act.weapon != self_weapon_id; ~25% auto-switch gap


def switch_events(weapon_seq, episode_offsets):
    """Per-frame bool mask of weapon-CHANGE events (w[t] != w[t-1], within episode).

    ``weapon_seq``: int array of the held/chosen weapon (impulse 1..8 or class id),
    one entry per frame, concatenated across episodes. ``episode_offsets``: the
    episode boundary indices (len = n_episodes + 1). First frame of each episode is
    False (no prior frame). This is THE switch rate (human ≈ 4% combat).
    """
    w = np.asarray(weapon_seq).reshape(-1)
    offs = np.asarray(episode_offsets, np.int64)
    mask = np.zeros(len(w), bool)
    for i in range(len(offs) - 1):
        s, e = int(offs[i]), int(offs[i + 1])
        if e - s > 1:
            mask[s + 1:e] = w[s + 1:e] != w[s:e - 1]
    return mask


def switch_rate(weapon_seq, episode_offsets):
    """Fraction of frames that are switch EVENTS (≈ 4% for human combat)."""
    return float(switch_events(weapon_seq, episode_offsets).mean())


def dwell_times(weapon_seq, episode_offsets):
    """Run-lengths (frames) the weapon is held between switches, per episode.

    Returns a 1-D array of dwell lengths. Human combat dwell medians ≈ 3-21 frames
    (weapon-head.md §3). Compare distributions (EMD/median), not just the mean.
    """
    w = np.asarray(weapon_seq).reshape(-1)
    offs = np.asarray(episode_offsets, np.int64)
    out = []
    for i in range(len(offs) - 1):
        s, e = int(offs[i]), int(offs[i + 1])
        if e <= s:
            continue
        seg = w[s:e]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.concatenate(([0], change, [len(seg)]))
        out.extend(np.diff(bounds).tolist())
    return np.asarray(out, np.int64)


def intent_state_mismatch(act_weapon, self_weapon_id_impulse):
    """Frames where intended weapon (act.weapon) != equipped (self_weapon_id).

    NOT a switch rate (≈25%, largely the auto-switch labeling gap — see module
    docstring + memory weapon_label_vs_state). Both args impulse-coded 1..8.
    """
    a = np.asarray(act_weapon).reshape(-1)
    s = np.asarray(self_weapon_id_impulse).reshape(-1)
    return (a != s) & (a != 0)


def assert_is_event_rate(rate, name="switch rate", tol=0.10):
    """Guard: a real switch-event rate is ~4%. If >tol (default 10%) you've almost
    certainly measured intent!=state (~25%), not switch events. Raises to stop the
    mislabel at the source."""
    if rate > tol:
        raise AssertionError(
            f"{name}={rate:.3f} > {tol}: this is NOT a switch-event rate "
            f"(human ~{HUMAN_COMBAT_SWITCH_RATE:.3f}). You are likely measuring "
            f"intent!=state (~{INTENT_STATE_MISMATCH_RATE:.2f}); see _weapon_metrics docstring.")
