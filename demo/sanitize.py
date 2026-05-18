"""Sanitize usercmd targets to engine-effective presses only.

For BC labeler training: filter targets to ticks where the press has an
observable kinematic effect.  Engine-rejected presses become
``ignore_index = -100`` masks so the labeler isn't penalized for failing
to predict press states with no downstream signal — those frames carry
the player's *intent* but not their *effect*, and the labeler operates on
effects (the MVD-recoverable obs stream).

The "human element" (timing, rhythm, held-duration, counter-strafe ticks,
strafejump cadence) lives in the *accepted* presses — those stay in the
target.  The C-side label emission engineering (back-shift ring,
chain-fill, log-normal tail) separately re-injects natural-looking held
durations into the C-rule MVD labels so downstream BC learns plausible
output shapes.  Different artifacts, different roles.

Rules sourced from the QW engine (vanilla, commit
``bf4ac424ce754894ac8f1dae6a3981954bc9852d``):

    forwardmove / sidemove  always integrated by pmove unless dead /
                            intermission
    upmove > 0  (jump)      grounded (onground != -1) OR in water
                            (waterlevel >= 2)
    upmove < 0  (swim-down) in water only (no-op on ground in vanilla QW
                            — no crouch animation, no hull-shrink)
    attack                  outside per-weapon ``attack_finished``
                            cooldown AND not dead
    weapon (held)           dense per-frame state — alive only, no
                            engine-rejection case to filter

See ``agents/plans/labeler-jump-fire-refinement.md`` and the agent audit
that produced this table.

Public API: per-axis ``effective_*_mask`` functions returning a bool
array of shape ``(T,)`` where ``True`` means "this tick's target is
kinematically effective; keep it in the loss".  Callers translate ``False``
to ``ignore_index = -100`` in their CE targets.
"""

from __future__ import annotations

import numpy as np


# ── Per-weapon attack_finished cooldowns ──────────────────────────────────
#
# Source: vanilla QW ``vendor/quake/QW/progs/weapons.qc`` ``attack_finished``
# delays.  Keys are QNN subject weapon IDs (matches obs/self_weapon_id;
# 3=Axe..10=LG).  The 20 Hz numbers come from
# ``scripts/compare_collects.py:FIRE_COOLDOWN_BY_WEAPON_20HZ``; the 77 Hz
# numbers are derived as ``round(delay_seconds * 77)``.

FIRE_COOLDOWN_NATIVE: dict[int, int] = {
    3:  38,  # Axe   0.5 s
    4:  38,  # SG    0.5 s
    5:  54,  # SSG   0.7 s
    6:  15,  # NG    0.2 s
    7:  15,  # SNG   0.2 s
    8:  46,  # GL    0.6 s
    9:  62,  # RL    0.8 s
    10:  8,  # LG    0.1 s
}


# ── Movement axes ─────────────────────────────────────────────────────────

def effective_move_mask(health: np.ndarray) -> np.ndarray:
    """fb / lr targets are accepted by pmove every tick — the engine
    integrates them unconditionally.  Filter only frames where the player
    is dead or otherwise frozen (intermission, signon).  ``health <= 0``
    is sufficient for our recordings (no intermission frames in trimmed
    play windows).

    Parameters
    ----------
    health : np.ndarray  shape (T,)
        ``snapshot.health`` per tick — the player's HP.  Negative values
        encode death.

    Returns
    -------
    mask : np.ndarray  shape (T,), dtype=bool
        ``True`` where the move target should contribute to loss.
    """
    return np.asarray(health, dtype=np.int32) > 0


# ── Jump (upmove > 0) ─────────────────────────────────────────────────────

def effective_jump_mask(
    ud_truth: np.ndarray,
    movement_id: np.ndarray,
    health: np.ndarray,
) -> np.ndarray:
    """ud=pos (jump press) accepted only when grounded or in water.
    Airborne presses are engine-rejected (the impulse never applies); the
    labeler can't predict these from observable obs because they leave no
    kinematic trace.

    Filters only the ``ud=pos`` ticks: ``ud=none`` ticks pass through
    unchanged (they're the easy "no press" majority class).  ``ud=neg``
    targets are unused in QW (vanilla has no separate swim-down/crouch
    encoding distinct from ``upmove < 0`` — that case is handled by
    :func:`effective_swim_down_mask` if you treat the axis 3-class).

    Parameters
    ----------
    ud_truth : np.ndarray  shape (T,)
        The ud class per tick: ``0=neg, 1=none, 2=pos``.
    movement_id : np.ndarray  shape (T,)
        ``0=ground, 1=air, 2..=water``.  Engine accepts a jump press when
        ``movement_id == 0`` (grounded) or ``movement_id >= 2`` (water).
    health : np.ndarray  shape (T,)
        HP per tick; dead frames are filtered out wholesale.
    """
    ud = np.asarray(ud_truth, dtype=np.int32)
    mid = np.asarray(movement_id, dtype=np.int32)
    alive = np.asarray(health, dtype=np.int32) > 0

    press_at_t = (ud == 2)
    accepted = (mid == 0) | (mid >= 2)
    # Filter: drop ticks where the player pressed jump but the engine refused.
    rejected_press = press_at_t & ~accepted
    return alive & ~rejected_press


# ── Swim-down / crouch (upmove < 0, ud=neg) ────────────────────────────────

def effective_swim_down_mask(
    ud_truth: np.ndarray,
    movement_id: np.ndarray,
    health: np.ndarray,
) -> np.ndarray:
    """``upmove < 0`` only produces an observable effect in water (swim
    down).  Vanilla QW has no crouch hull-shrink or animation — on land
    a downward upmove is a pure no-op.  Mask out land-side ud=neg
    targets.
    """
    ud = np.asarray(ud_truth, dtype=np.int32)
    mid = np.asarray(movement_id, dtype=np.int32)
    alive = np.asarray(health, dtype=np.int32) > 0

    press_at_t = (ud == 0)
    in_water = (mid >= 2)
    rejected_press = press_at_t & ~in_water
    return alive & ~rejected_press


# ── Fire (attack) ─────────────────────────────────────────────────────────

def effective_fire_mask(
    fire_truth: np.ndarray,
    weapon_id: np.ndarray,
    health: np.ndarray,
    cooldown_table: dict[int, int] = FIRE_COOLDOWN_NATIVE,
) -> np.ndarray:
    """attack accepted only when the held weapon's ``attack_finished``
    has elapsed.  Walks the fire-press events in time order, tracks the
    earliest acceptable next-fire tick per held weapon, and marks each
    press effective iff it lands at or after that tick.

    Note this collapses cooldown-internal presses to "no-op" — the
    usercmd records them (player intent), but the engine never produces
    a shot, so the labeler has no observable target to learn against.

    Parameters
    ----------
    fire_truth : np.ndarray  shape (T,)
        ``usercmd.attack`` per tick (0/1).
    weapon_id : np.ndarray  shape (T,)
        QNN subject weapon ID per tick (3..10 for Axe..LG; 0/1/2 for
        non-firing slots).  Cooldown is per-weapon: switching weapons
        does NOT reset the previous weapon's cooldown but the new
        weapon's cooldown starts fresh at switch (the engine treats
        each weapon's ``attack_finished`` independently).
    health : np.ndarray  shape (T,)
        HP per tick.  Dead frames are filtered out wholesale.
    cooldown_table : dict[int, int]
        Native-tick cooldown per weapon ID.  Default
        :data:`FIRE_COOLDOWN_NATIVE`.

    Returns
    -------
    mask : np.ndarray  shape (T,), dtype=bool
    """
    fire = np.asarray(fire_truth, dtype=np.uint8)
    wid  = np.asarray(weapon_id, dtype=np.int32)
    alive = np.asarray(health, dtype=np.int32) > 0
    T = fire.shape[0]

    out = np.ones(T, dtype=bool)
    # Per-weapon "next allowed fire tick".  Default 0 (immediately ready).
    next_ok: dict[int, int] = {}
    for t in range(T):
        if not alive[t]:
            out[t] = False
            continue
        if fire[t] == 0:
            # No press — pass through unmasked (the "none" class is fine
            # to learn from; only press-on-cooldown is the no-op).
            continue
        w = int(wid[t])
        cd = cooldown_table.get(w)
        if cd is None:
            # Unknown weapon (e.g., no weapon held): treat press as
            # ineffective.
            out[t] = False
            continue
        nxt = next_ok.get(w, 0)
        if t < nxt:
            # Press inside cooldown — ineffective.
            out[t] = False
        else:
            # Press accepted: schedule next cooldown.
            next_ok[w] = t + cd
    return out


# ── Weapon (held-weapon target) ───────────────────────────────────────────

def effective_weapon_mask(health: np.ndarray) -> np.ndarray:
    """Weapon target is the currently-held weapon byte (dense per-frame
    state), not a sparse switch event.  It's observable and
    unambiguous on every alive frame, so there is no engine-rejection
    case to filter.  Identical semantics to fb/lr: mask out only dead
    frames.

    Parameters
    ----------
    health : np.ndarray  shape (T,)
    """
    return np.asarray(health, dtype=np.int32) > 0


# ── Apply masks to packed targets ─────────────────────────────────────────

def apply_mask_to_packed_move(
    move_packed: np.ndarray,
    fb_mask: np.ndarray | None = None,
    lr_mask: np.ndarray | None = None,
    ud_mask: np.ndarray | None = None,
    ignore_value: int = -100,
) -> np.ndarray:
    """Translate per-axis bool masks into per-axis ``ignore_value`` for
    consumers that pass the packed move target through CE loss.

    Returns a ``(T, 3) int64`` array of [fb, lr, ud] classes where each
    masked-out tick gets ``ignore_value`` (default -100, the standard
    PyTorch CE ignore_index).
    """
    p = np.asarray(move_packed, dtype=np.uint8).reshape(-1)
    fb = (p & 0x3).astype(np.int64)
    lr = ((p >> 2) & 0x3).astype(np.int64)
    ud = ((p >> 4) & 0x3).astype(np.int64)
    if fb_mask is not None:
        fb = np.where(np.asarray(fb_mask, dtype=bool), fb, ignore_value)
    if lr_mask is not None:
        lr = np.where(np.asarray(lr_mask, dtype=bool), lr, ignore_value)
    if ud_mask is not None:
        ud = np.where(np.asarray(ud_mask, dtype=bool), ud, ignore_value)
    return np.stack([fb, lr, ud], axis=1)
