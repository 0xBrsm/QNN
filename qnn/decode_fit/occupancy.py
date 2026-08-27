"""a26rc1b active-fire OCCUPANCY — the estimand and its two bias controllers.

The estimand (:func:`selection_profile`) is the rc1b repair's preference
ruler: operative discharges weighted by each weapon's refire duration. Human
weapon scripts make equipped-weapon frame mass an invalid preference ruler, so
dwell is estimated as ``discharge count x refire seconds`` instead of counted
off equipped-weapon frames.

One closed-loop controller steps a selection bias until that occupancy tracks
a named target profile:

* :func:`calibrate_occupancy_to_human` — the anti-camp pass. qnn_arena8 is
  full-loadout (no pickups) so weapon share is pure preference; pull the
  profile toward the human corpus.

The companion "jitter-fix" pass (occupancy -> the model's own native profile)
was DELETED 2026-08-26 with weapon.switch_margin: it existed only to re-aim
the sticky attractor the held-anchor hysteresis created, and with no anchor
there is no attractor to re-aim.

Both were run by hand to produce the deployed a26rc1b decode config;
``scripts/analysis/_occ_calib_a26rc1b.py`` and ``_bias_to_native_a26rc1b.py``
are thin drivers over them. For NEW fits prefer ``gates.attack_trim``, the
reconciled controller that fits fire cadence, selection, switch rate and
threat-break jointly on ``weapon.preference_bias_vec``; these two stay
importable and versioned because a shipped artifact was produced by exactly
this logic and must remain reproducible.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from qnn.decode_fit.context import IMPULSE_NAME, read_json
from qnn.weapons import COOLDOWN_SEC

# Active-fire dwell weights in seconds, indexed by raw weapon id 1..8 (slot 0
# = "no weapon", unused) — nominal per-weapon refire durations, deliberately
# NOT inferred from equipped-weapon frames or weapon-script state. ``gates``
# re-exports this.
#
# SOURCE OF TRUTH is qnn.weapons.COOLDOWN_SEC, parsed from the engine header
# (src/engine/common/qnn_weapon.h). Until 2026-08-26 this was a hand-copied
# literal that had DRIFTED from it: NG/SNG/LG carried 0.1 (the literal
# W_Attack delay) where the engine table carries 0.2 (the OPERATIVE
# think-chain cadence). Since this weights OPERATIVE discharges, the literal
# halved every continuous weapon's dwell against RL/SG — and
# occupancy_bias_loop steers weapon.preference_bias_vec against these shares,
# so the error fed straight into the fitted vector. qnn.weapons' own docstring
# already said op-attack cadence math must use that table. Do not re-copy it.
REFIRE_SEC = COOLDOWN_SEC
RC1B_REFIRE_SEC = REFIRE_SEC        # legacy alias (gates re-exports this name)

# ── controller constants (rc1b values; both passes share these) ──────────────
OCC_DAMPING = 0.6                  # step = damping x log-ratio (anti-oscillation)
OCC_BIAS_CLAMP = (-8.0, 4.0)
OCC_MAX_ITERS = 8

# ── pass 1: occupancy → HUMAN corpus profile ─────────────────────────────────
# The stored human active-fire share profile (SG 43.8 / RL 40.3 / ...) was
# DELETED 2026-08-26 and calibrate_occupancy_to_human now REQUIRES an explicit
# target_pct. Two reasons, in order of weight:
#
#  1. RAW SHARE IS THE WRONG RULER, and was already known to be. It was
#     superseded the same day it was wired into the fit report by pairwise
#     possession matching (human_occupancy_report / _pairwise_win_seconds
#     below), because a raw profile boosts whichever weapon is always
#     possessed: "SG's 43.8% raw share is map economy", and the pairwise view
#     INVERTS the raw-share story (the apparent 5x LG overuse was an
#     availability artifact). See agents/plans/decode-fit-pins-crest-
#     mismatch.md, which records the number and that reading.
#  2. Its weighting provenance is NOT RECOVERABLE. It entered as a bare
#     literal in scripts/analysis/_occ_calib_a26rc1b.py beside that script's
#     own copy of the refire table — the drifted one, NG/SNG/LG at 0.1s — with
#     no derivation, and the promotion into this package was value-identical.
#     So whether it is commensurate with the corrected estimand above is
#     unknown, not known-stale. That alone disqualifies it as a default.
#
# Re-measure against the current estimand if a raw-share target is wanted.
HUMAN_OCC_TOL = 0.30
HUMAN_OCC_SHARE_FLOOR_PCT = 0.05
# only weapons carrying real share on either side adjudicate convergence
HUMAN_OCC_CONVERGE_MIN_PCT = 3.0

# ``launch_eval(config_path, tag) -> npz path | None`` (injected free-play).
LaunchEval = Callable[[Path, str], Path | None]


def _log(msg: str) -> None:
    print(f"[occupancy] {msg}", flush=True)


# ══ 1. estimand ═══════════════════════════════════════════════════════════════

def selection_profile(npz: Path) -> dict[str, Any] | None:
    """Active-fire occupancy from refire-weighted operative discharges.

    Human weapon scripts make equipped-weapon frame mass an invalid preference
    ruler. The fixed rc1b repair instead estimated active-fire dwell as
    ``operative discharge count * weapon refire duration`` and preserved the
    zero-margin distribution of that quantity. Keep that exact estimand while
    using the reconciled selection-only preference vector as its control.
    """
    try:
        z = np.load(npz)
        discharge = np.asarray(z["discharge"]).reshape(-1).astype(bool)
        weapon = np.asarray(z["weapon_imp"]).reshape(-1).astype(np.int64)
    except Exception:
        return None
    if len(weapon) != len(discharge) or not discharge.any():
        return None
    counts = {name: int(((weapon == impulse) & discharge).sum())
              for impulse, name in IMPULSE_NAME.items()}
    weighted = {
        name: float(counts[name] * RC1B_REFIRE_SEC[impulse])
        for impulse, name in IMPULSE_NAME.items()
    }
    total = sum(weighted.values())
    if total <= 0.0:
        return None
    return {
        "metric": "refire-weighted operative-discharge share",
        "operative_discharges": sum(counts.values()),
        "counts": counts,
        "refire_seconds": {
            name: float(RC1B_REFIRE_SEC[impulse])
            for impulse, name in IMPULSE_NAME.items()
        },
        "weighted_seconds": weighted,
        "shares": {w: weighted[w] / total for w in weighted},
    }


def occupancy_share_pct(npz: Path) -> dict[int, float] | None:
    """:func:`selection_profile` shares as PERCENT keyed by raw weapon id.

    The controllers below think in the rc1b percent units their targets are
    written in; the gate thinks in fractions. One estimand, two renderings —
    never a second estimator.
    """
    profile = selection_profile(Path(npz))
    if profile is None:
        return None
    shares = profile["shares"]
    return {impulse: 100.0 * shares[name]
            for impulse, name in IMPULSE_NAME.items()}


# ── pairwise possession-matched preference vs the human corpus ───────────────
# Brian 2026-08-09: the human-balance yardstick is PAIRWISE MATCHING when
# both weapons are possessed — the raw corpus profile boosts SG because SG
# is the one weapon you always have. A pair only adjudicates where it
# carries this much refire-weighted mass on BOTH sides, and the row flags
# on a bot-vs-human odds ratio beyond 2× either way.
PAIRWISE_MIN_SEC = 10.0
PAIRWISE_FLAG_LOG_ODDS = float(np.log(2.0))


def _pairwise_win_seconds(weapons: np.ndarray, feas_bits: np.ndarray) -> np.ndarray:
    """(9, 9) matrix: W[a, b] = refire-weighted seconds firing ``a`` while
    ``b`` was ALSO in the feasibility menu — the possession-matched "a beat
    b" mass. ``weapons`` are discharge impulses (1..8); ``feas_bits`` the
    packed feasibility mask at each discharge tick."""
    w = np.asarray(weapons, dtype=np.int64).reshape(-1)
    m = np.asarray(feas_bits, dtype=np.int64).reshape(-1)
    dur = RC1B_REFIRE_SEC[w]
    out = np.zeros((9, 9), dtype=np.float64)
    for b in range(1, 9):
        sel = (w != b) & (((m >> (b - 1)) & 1) != 0)
        if sel.any():
            np.add.at(out[:, b], w[sel], dur[sel])
    return out


def pairwise_pref_matrix_npz(npz: Path) -> np.ndarray | None:
    """Possession-matched win matrix for a schema-6 wave: engaged
    ``weapon_pref`` discharge events × ``weapon_feas`` at those ticks."""
    try:
        z = np.load(Path(npz))
        wp = np.asarray(z["weapon_pref"]).reshape(-1)
        feas = np.asarray(z["weapon_feas"]).reshape(-1)
        keep = np.asarray(z["keep"]).reshape(-1).astype(bool)
    except Exception:
        return None
    ev = np.flatnonzero((wp >= 1) & (wp <= 8) & keep)
    if not ev.size:
        return None
    return _pairwise_win_seconds(wp[ev], feas[ev])


def pairwise_pref_matrix_human(corpus_dir: Path, split: str) -> np.ndarray | None:
    """Possession-matched win matrix over the corpus: engaged discharges
    (``attack_class``) × the per-tick feasibility mask (the same
    ``engine_norm.weapon_feasibility_bits`` derivation the eval writer
    uses). None when the cache lacks the feasibility streams."""
    from qnn.eval.humanlikeness.human_reference import _episodes
    from qnn.eval.humanlikeness.rc import _preference_events
    total = np.zeros((9, 9), dtype=np.float64)
    seen = False
    for ep in _episodes(Path(corpus_dir), split):
        streams = ep.get("pref_streams")
        if streams is None:
            continue
        ev, wlab = _preference_events(ep["attack_class"], ep["keep"])
        if not ev.size:
            continue
        seen = True
        total += _pairwise_win_seconds(wlab, np.asarray(streams["feas"])[ev])
    return total if seen else None


def human_occupancy_report(npz: Path, corpus_dir: Path | None = None,
                           split: str = "precomputed_val",
                           human_wins: np.ndarray | None = None,
                           ) -> dict[str, Any] | None:
    """PAIRWISE possession-matched weapon preference vs the human corpus —
    the balance row (Brian 2026-08-09: raw-profile comparison is wrong;
    "it boosts SG because that's the one weapon you always have").

    For each weapon pair (A, B): P(A over B) = refire-weighted active-fire
    seconds on A / (A + B), counting ONLY discharges where both weapons
    were in the feasibility menu (``weapon_feas`` — the model's own
    choice-set predicate) — so availability can never masquerade as
    preference on either side. Pairs adjudicate only with
    ``PAIRWISE_MIN_SEC`` mass on both sides; the row flags when any
    adjudicated pair's bot-vs-human odds ratio exceeds
    ``PAIRWISE_FLAG_LOG_ODDS``. FLAGGED, never gating: a genuine
    preference gap is a TRAINING target — the selection-only vector must
    not become a preference optimizer. Raw shares stay visible in the
    gated ``weapon_occupancy`` arm (self-reference) and this row's
    ``observed_pct``; ``HUMAN_OCC_SHARE_PCT`` remains only as
    :func:`calibrate_occupancy_to_human`'s historical target.

    ``human_wins`` short-circuits the corpus walk (the caller may cache
    it); otherwise ``corpus_dir`` is walked.
    """
    bot = pairwise_pref_matrix_npz(Path(npz))
    if bot is None:
        return None
    if human_wins is None:
        if corpus_dir is None:
            return None
        human_wins = pairwise_pref_matrix_human(Path(corpus_dir), split)
    if human_wins is None:
        return None
    pairs: dict[str, dict[str, Any]] = {}
    worst = 0.0
    for a in range(1, 9):
        for b in range(a + 1, 9):
            hm = float(human_wins[a, b] + human_wins[b, a])
            bm = float(bot[a, b] + bot[b, a])
            if hm < PAIRWISE_MIN_SEC or bm < PAIRWISE_MIN_SEC:
                continue
            ph = float(human_wins[a, b]) / hm
            pb = float(bot[a, b]) / bm
            eps = 1e-3
            lo = float(np.log((np.clip(pb, eps, 1 - eps) / (1 - np.clip(pb, eps, 1 - eps)))
                              / (np.clip(ph, eps, 1 - eps) / (1 - np.clip(ph, eps, 1 - eps)))))
            worst = max(worst, abs(lo))
            pairs[f"{IMPULSE_NAME[a]}>{IMPULSE_NAME[b]}"] = {
                "human": round(ph, 3), "bot": round(pb, 3),
                "log_odds_delta": round(lo, 3),
                "sec": {"human": round(hm, 1), "bot": round(bm, 1)},
            }
    obs_pct = occupancy_share_pct(Path(npz)) or {}
    return {
        "metric": ("pairwise possession-matched active-fire preference "
                   "(refire-weighted, both weapons in feasibility menu)"),
        "flagged": bool(worst > PAIRWISE_FLAG_LOG_ODDS),
        "pairs": pairs,
        "max_abs_log_odds_delta": round(worst, 3),
        "flag_log_odds": round(PAIRWISE_FLAG_LOG_ODDS, 3),
        "min_pair_sec": PAIRWISE_MIN_SEC,
        "observed_pct": {name: round(obs_pct.get(impulse, 0.0), 2)
                         for impulse, name in IMPULSE_NAME.items()},
    }


# ── direct human weapon-transition ruler ───────────────────────────────────
# A weapon decision is factored into the two questions the decoder actually
# has to answer:
#
#   1. continue the last discharge weapon, or leave it?  (transition diagonal)
#   2. after leaving it, which other feasible weapon wins? (choice matrix)
#
# This avoids the old pooled-switch cancellation: a28rc1b happened to match
# the aggregate human continuation rate while under-continuing SG/GL by ~2x
# and over-continuing LG.  It also avoids treating repeated same-weapon
# discharges as fresh preference votes.  Choice events are switch destinations;
# the old anchor is excluded because the continuation controller already owns
# that comparison.
CHOICE_MIN_EVENTS = 20
CONTINUE_MIN_PAIRS = 100
CHOICE_PROB_TOL = 0.10
CONTINUE_PROB_TOL = 0.05
TRANSITION_PSEUDOCOUNT = 0.5


def _add_transition_episode(
    transitions: np.ndarray,
    event_idx: np.ndarray,
    weapons: np.ndarray,
    invalid: np.ndarray | None,
    *,
    choices: np.ndarray | None = None,
    commands: np.ndarray | None = None,
) -> None:
    """Accumulate physical transitions and optional decoded leave requests.

    ``transitions`` is the physical-discharge sequence and owns continuation.
    When supplied, ``commands`` is the decoder's weapon request at those same
    discharge events; a command different from the physical anchor is the
    destination choice that initiated a leave.  Measuring that request avoids
    charging delayed/one-tick equip-stat changes to the wrong anchor.
    """
    ev = np.asarray(event_idx, dtype=np.int64).reshape(-1)
    w = np.asarray(weapons, dtype=np.int64).reshape(-1)
    if len(w) < 2:
        return
    good = (np.ones(len(w) - 1, dtype=bool) if invalid is None
            else ~np.asarray(invalid, dtype=bool).reshape(-1))
    src, dst = w[:-1][good], w[1:][good]
    np.add.at(transitions, (src, dst), 1)
    if choices is not None and commands is not None:
        cmd = np.asarray(commands, dtype=np.int64).reshape(-1)
        if len(cmd) != len(w):
            raise ValueError("weapon command/event lengths disagree")
        leave = good & (cmd[:-1] >= 1) & (cmd[:-1] <= 8) & (cmd[:-1] != w[:-1])
        np.add.at(choices, (w[:-1][leave], cmd[:-1][leave]), 1)


def human_weapon_behavior_reference(
    corpus_dir: Path,
    split: str = "precomputed_val",
) -> dict[str, Any] | None:
    """Human switch-choice counts + exact-weapon transition counts.

    Consecutive operative discharges count only when the feasibility menu is
    stationary and no death intervened — the same pair validity contract used
    by :mod:`qnn.eval.humanlikeness.rc` on the bot side.
    """
    from qnn.eval.humanlikeness.human_reference import _episodes
    from qnn.eval.humanlikeness.rc import _invalidated_pairs, _preference_events

    transitions = np.zeros((9, 9), dtype=np.int64)
    seen = False
    n_episodes = 0
    try:
        for ep in _episodes(Path(corpus_dir), split):
            streams = ep.get("pref_streams")
            if streams is None:
                continue
            event_idx, weapons = _preference_events(ep["attack_class"], ep["keep"])
            if not event_idx.size:
                continue
            seen = True
            n_episodes += 1
            _add_transition_episode(
                transitions, event_idx, weapons,
                _invalidated_pairs(event_idx, streams))
    except (OSError, KeyError, ValueError):
        return None
    if not seen:
        return None
    return {"transitions": transitions, "n_episodes": n_episodes,
            "split": split}


def bot_weapon_behavior(npz: Path) -> dict[str, Any] | None:
    """Bot-side twin of :func:`human_weapon_behavior_reference`."""
    from qnn.eval.humanlikeness.rc import _invalidated_pairs

    try:
        z = np.load(Path(npz))
        offsets = np.asarray(z["episode_offsets"], dtype=np.int64).reshape(-1)
        weapon = np.asarray(z["weapon_pref"], dtype=np.int64).reshape(-1)
        command = np.asarray(z["weapon"], dtype=np.int64).reshape(-1)
        discharge = np.asarray(z["discharge"]).reshape(-1).astype(bool)
        keep = np.asarray(z["keep"]).reshape(-1).astype(bool)
        feas = np.asarray(z["weapon_feas"], dtype=np.int64).reshape(-1)
        health = np.asarray(z["health"], dtype=np.int64).reshape(-1)
    except Exception:
        return None
    transitions = np.zeros((9, 9), dtype=np.int64)
    choices = np.zeros((9, 9), dtype=np.int64)
    for a, b in zip(offsets[:-1], offsets[1:]):
        a, b = int(a), int(b)
        event_idx = np.flatnonzero(discharge[a:b] & keep[a:b])
        weapons = weapon[a:b][event_idx]
        commands = command[a:b][event_idx]
        valid = (weapons >= 1) & (weapons <= 8)
        event_idx, weapons, commands = (
            event_idx[valid], weapons[valid], commands[valid])
        streams = {"feas": feas[a:b], "health": health[a:b]}
        _add_transition_episode(
            transitions, event_idx, weapons,
            _invalidated_pairs(event_idx, streams),
            choices=choices, commands=commands)
    if not transitions[1:, 1:].sum():
        return None
    return {"transitions": transitions, "choices": choices,
            "n_episodes": max(len(offsets) - 1, 0), "npz": str(npz)}


def _smoothed_probability(wins: float, total: float) -> float:
    a = float(TRANSITION_PSEUDOCOUNT)
    return (float(wins) + a) / (float(total) + 2.0 * a)


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


def transition_choice_prob_matrix(human_ref: dict[str, Any]) -> np.ndarray:
    """Return the direct human ``P(destination | anchor, leave)`` matrix.

    Rows and columns are impulse order 1..8.  The diagonal is exactly zero:
    :data:`weapon.continue_prob_vec` owns that decision.  Jeffreys mass keeps
    a corpus-zero destination reachable when it is feasible at runtime.
    """
    transitions = np.asarray(human_ref["transitions"], dtype=np.float64)
    out = np.zeros((8, 8), dtype=np.float64)
    for anchor in range(1, 9):
        row = transitions[anchor, 1:].copy() + TRANSITION_PSEUDOCOUNT
        row[anchor - 1] = 0.0
        out[anchor - 1] = row / max(float(row.sum()), 1.0)
    return out


def weapon_behavior_report(
    npz: Path,
    human_ref: dict[str, Any],
    *,
    choice_min_events: int = CHOICE_MIN_EVENTS,
    continue_min_pairs: int = CONTINUE_MIN_PAIRS,
    choice_prob_tol: float = CHOICE_PROB_TOL,
    continue_prob_tol: float = CONTINUE_PROB_TOL,
) -> dict[str, Any] | None:
    """Score bot weapon choice and per-weapon continuation against humans."""
    bot = bot_weapon_behavior(Path(npz))
    if bot is None:
        return None
    ht = np.asarray(human_ref["transitions"], dtype=np.float64)
    bt = np.asarray(bot["transitions"], dtype=np.float64)
    bc = np.asarray(bot["choices"], dtype=np.float64)

    choice_rows: dict[str, dict[str, Any]] = {}
    missing_choice: list[str] = []
    worst_choice = 0.0
    human_choice = transition_choice_prob_matrix(human_ref)
    for anchor in range(1, 9):
        hrow = ht[anchor, 1:].copy()
        brow = bc[anchor, 1:].copy()
        hrow[anchor - 1] = 0.0
        brow[anchor - 1] = 0.0
        hn, bn = float(hrow.sum()), float(brow.sum())
        if hn < choice_min_events:
            continue
        name = IMPULSE_NAME[anchor]
        card: dict[str, Any] = {
            "human": {IMPULSE_NAME[w]: round(float(human_choice[anchor - 1, w - 1]), 4)
                      for w in range(1, 9) if w != anchor},
            "events": {"human": int(hn), "bot": int(bn)},
        }
        if bn < choice_min_events:
            card.update(bot=None, max_abs_prob_delta=None,
                        total_variation=None, measurable=False)
            missing_choice.append(name)
        else:
            bot_prob = (brow + TRANSITION_PSEUDOCOUNT)
            bot_prob[anchor - 1] = 0.0
            bot_prob /= float(bot_prob.sum())
            delta = bot_prob - human_choice[anchor - 1]
            max_delta = float(np.max(np.abs(delta)))
            worst_choice = max(worst_choice, max_delta)
            card.update(
                bot={IMPULSE_NAME[w]: round(float(bot_prob[w - 1]), 4)
                     for w in range(1, 9) if w != anchor},
                max_abs_prob_delta=round(max_delta, 4),
                total_variation=round(float(0.5 * np.abs(delta).sum()), 4),
                measurable=True,
            )
        choice_rows[name] = card

    continuation: dict[str, dict[str, Any]] = {}
    missing_continue: list[str] = []
    worst_continue = 0.0
    for w in range(1, 9):
        hn = float(ht[w, 1:].sum())
        if hn < continue_min_pairs:
            continue
        bn = float(bt[w, 1:].sum())
        ph = _smoothed_probability(ht[w, w], hn)
        name = IMPULSE_NAME[w]
        card = {"human": round(ph, 4),
                "pairs": {"human": int(hn), "bot": int(bn)}}
        if bn < continue_min_pairs:
            card.update(bot=None, prob_delta=None,
                        log_odds_delta=None, measurable=False)
            missing_continue.append(name)
        else:
            pb = _smoothed_probability(bt[w, w], bn)
            prob_delta = pb - ph
            worst_continue = max(worst_continue, abs(prob_delta))
            card.update(bot=round(pb, 4), prob_delta=round(prob_delta, 4),
                        log_odds_delta=round(_logit(pb) - _logit(ph), 4),
                        measurable=True)
        continuation[name] = card

    choice_ok = bool(choice_rows and not missing_choice
                     and worst_choice <= choice_prob_tol)
    continue_ok = bool(continuation and not missing_continue
                       and worst_continue <= continue_prob_tol)
    return {
        "choice": {
            "ok": choice_ok,
            "metric": ("P(next discharge destination | previous exact weapon, "
                       "decoded leave request, stationary feasibility menu)"),
            "rows": choice_rows,
            "missing_weapons": missing_choice,
            "max_abs_prob_delta": round(worst_choice, 4),
            "tol": float(choice_prob_tol),
            "min_events": int(choice_min_events),
        },
        "continuation": {
            "ok": continue_ok,
            "metric": ("P(next discharge weapon = current | current exact "
                       "weapon, stationary feasibility menu)"),
            "weapons": continuation,
            "missing_weapons": missing_continue,
            "max_abs_prob_delta": round(worst_continue, 4),
            "tol": float(continue_prob_tol),
            "min_pairs": int(continue_min_pairs),
        },
        "bot_n_episodes": int(bot.get("n_episodes", 0)),
    }


# ══ 2. controller ═════════════════════════════════════════════════════════════

def occupancy_bias_loop(config_path: Path, launch_eval: LaunchEval, *,
                        target_pct: dict[int, float],
                        tol: float,
                        share_floor_pct: float,
                        converge_on: Callable[[int, float, float], bool],
                        step_on: Callable[[int, float, float], bool],
                        tag_prefix: str,
                        label: str,
                        max_iters: int = OCC_MAX_ITERS,
                        damping: float = OCC_DAMPING,
                        bias_clamp: tuple[float, float] = OCC_BIAS_CLAMP,
                        bias_key: str = "weapon.preference_bias_vec",
                        ) -> dict[str, Any]:
    """Damped log-ratio bias controller on the rc1b occupancy estimand.

    Each iteration writes ``config_path + '.occtrim.json'`` at the current
    bias, runs one free-play wave through ``launch_eval``, measures occupancy,
    and steps ``bias_key[k-1] += damping * log(target/observed)``
    (both sides floored at ``share_floor_pct`` so an empty weapon cannot
    produce an infinite step, result clamped to ``bias_clamp``).

    ``converge_on(impulse, target_pct, observed_pct)`` selects the weapons
    that adjudicate convergence — a profile is converged when every selected
    weapon's ``|log-ratio|`` is within ``tol``. ``step_on`` selects the
    weapons that get a correction; weapons too small on both sides are left
    alone rather than chased through noise.

    The bias frozen back into ``config_path`` is always the one that PRODUCED
    the last MEASURED occupancy, never the stepped-but-unevaluated vector.

    ``bias_key`` is the config param this loop steers — the rc1b presets both
    ride the legacy ``attack.bias_vec`` (their historical, still-reproducible
    control), but the SAME estimand/controller applies unchanged to
    ``weapon.preference_bias_vec`` (the weapon-switch-evidence phase-2 fit's
    occupancy re-fit at a new (λ,θ) operating point — see
    the removed weapon-switch phase-2 fit): only the key
    the bias is read from/written to differs, never the estimand or the
    stepping arithmetic."""
    config_path = Path(config_path)
    cfg = read_json(config_path)
    bias = [float(b) for b in (cfg["params"].get(bias_key)
                               or [0.0] * 8)]
    work = Path(str(config_path) + ".occtrim.json")

    history: list[dict[str, Any]] = []
    frozen = [round(b, 4) for b in bias]
    converged = False
    status = "MAX-ITERS"
    note: str | None = None
    for it in range(int(max_iters)):
        used = [round(b, 4) for b in bias]
        cfg["params"][bias_key] = used
        work.write_text(json.dumps(cfg, indent=2) + "\n")
        npz = launch_eval(work, f"{tag_prefix}{it}")
        share = occupancy_share_pct(Path(npz)) if npz is not None else None
        if share is None:
            status = "EVAL-FAILED"
            note = (f"iteration {it} produced no measurable occupancy — "
                    "loop aborted, last measured bias kept")
            _log(f"{label}: it{it} eval FAILED")
            break
        frozen = used
        worst = 0.0
        for impulse in sorted(target_pct):
            t_pct = float(target_pct[impulse])
            o_pct = float(share[impulse])
            log_ratio = math.log(max(t_pct, share_floor_pct)
                                 / max(o_pct, share_floor_pct))
            if converge_on(impulse, t_pct, o_pct):
                worst = max(worst, abs(log_ratio))
            if step_on(impulse, t_pct, o_pct):
                bias[impulse - 1] = float(np.clip(
                    bias[impulse - 1] + damping * log_ratio, *bias_clamp))
        history.append({
            "iter": it,
            "bias_used": used,
            "occupancy_pct": {IMPULSE_NAME[k]: round(share[k], 1)
                              for k in sorted(share)},
            "target_pct": {IMPULSE_NAME[k]: float(target_pct[k])
                           for k in sorted(target_pct)},
            "worst_abs_log_ratio": round(worst, 3),
        })
        _log(f"{label}: it{it} worst|log-ratio|={worst:.2f} "
             + " ".join(f"{IMPULSE_NAME[k]} {share[k]:.1f}/{target_pct[k]}"
                        for k in sorted(target_pct)
                        if converge_on(k, float(target_pct[k]),
                                       float(share[k]))))
        if worst <= tol:
            converged = True
            status = "CONVERGED"
            _log(f"{label}: CONVERGED at it{it}")
            break

    # freeze the bias that PRODUCED the last measured occupancy
    cfg["params"][bias_key] = frozen
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    work.unlink(missing_ok=True)
    report: dict[str, Any] = {
        "controller": label,
        "config": str(config_path),
        "bias_key": bias_key,
        "status": status,
        "converged": converged,
        "tol": float(tol),
        "max_iters": int(max_iters),
        "target_pct": {IMPULSE_NAME[k]: float(target_pct[k])
                       for k in sorted(target_pct)},
        "final_bias_vec": frozen,
        "history": history,
    }
    if note:
        report["note"] = note
    return report


def calibrate_occupancy_to_human(config_path: Path, launch_eval: LaunchEval, *,
                                 target_pct: dict[int, float],
                                 max_iters: int = OCC_MAX_ITERS,
                                 ) -> dict[str, Any]:
    """Anti-camp pass: occupancy → a measured human corpus profile.

    The preference bias is the only lever — the held-anchor hysteresis this
    pass used to hold fixed is retired. Every weapon is stepped; only weapons
    above 3% on either side adjudicate convergence.

    ``target_pct`` is REQUIRED: the old frozen default was measured under a
    refire table that has since been corrected, so there is no profile that
    can be trusted as a default. Re-measure against the current estimand.
    """
    return occupancy_bias_loop(
        config_path, launch_eval,
        target_pct=target_pct,
        tol=HUMAN_OCC_TOL,
        share_floor_pct=HUMAN_OCC_SHARE_FLOOR_PCT,
        converge_on=lambda _k, t, o: max(t, o) > HUMAN_OCC_CONVERGE_MIN_PCT,
        step_on=lambda _k, _t, _o: True,
        tag_prefix="occ",
        label="occupancy->human",
        max_iters=max_iters)
