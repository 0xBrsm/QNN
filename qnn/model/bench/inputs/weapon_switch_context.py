"""Forward-scoped weapon switch/dwell supervision for the switch-process head.

Mirrors :mod:`target_supervision_context`: the bench side-channel derives, from
the BC weapon-label stream + reset_mask, the labels the switch-process weapon head
needs and enters this context before the model forward. Per the WHEN/WHAT framework
(see src/docs/persistence-and-changepoints.md) the WHEN hazard is trained on STATE
only (dwell-age + inventory), NOT the held-weapon identity — copycat-safe. The
``held_weapon``/``dwell_age`` here are *supervision/feature derivations*, and only
``dwell_age`` (a duration, not the action) is fed to the hazard.

Fields (all flattened (B*,) to match Network's per-step flatten):
  - dwell_age           : frames the current weapon has been held (0 at a change/reset)
  - switch_next         : {0,1} float — weapon changes at the NEXT frame (WHEN label)
  - new_weapon_target   : long — new weapon index 0..7 on switch-next frames, else -100
                          (WHAT label; ignore_index=-100 for held frames + last frame)
  - valid               : bool — frame has a defined next frame in this BPTT window
"""
from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Iterator

import torch

from qnn.schema import WEAPON_HEAD_SIZE


@dataclass(frozen=True, slots=True)
class WeaponSwitchContext:
    dwell_age: torch.Tensor | None = None          # (B*,) float — WHEN input (copycat-safe)
    held_weapon: torch.Tensor | None = None        # (B*,) long 1..8 (0 none) — from-weapon, WHAT input
    switch_next: torch.Tensor | None = None        # (B*,) float {0,1} — WHEN label
    new_weapon_target: torch.Tensor | None = None  # (B*,) long, -100 = ignore — WHAT label
    valid: torch.Tensor | None = None              # (B*,) bool


_CTX: "contextvars.ContextVar[WeaponSwitchContext | None]" = contextvars.ContextVar(
    "qnn_weapon_switch_ctx", default=None,
)


@contextlib.contextmanager
def weapon_switch_context(ctx: WeaponSwitchContext | None) -> Iterator[None]:
    token = _CTX.set(ctx)
    try:
        yield
    finally:
        _CTX.reset(token)


def current_weapon_switch_context() -> WeaponSwitchContext | None:
    return _CTX.get()


def derive_weapon_switch_labels(
    weapon: torch.Tensor,             # (T, B) long — held weapon impulse 1..8, 0 = none
    reset_mask: torch.Tensor | None,  # (T, B) bool — sequence start (no carry from t-1)
) -> WeaponSwitchContext:
    """Derive dwell-age + switch/WHAT labels along the TIME axis of the BPTT
    ``(T, B)`` batch, then flatten ``(T, B) -> reshape(-1)`` to match the model's
    frame order (canonical losses flatten actions the same way).

    CRITICAL: the BC pipeline delivers ``(T, B)`` batches where each COLUMN ``b``
    is a time-contiguous sequence; consecutive *timesteps* are ``weapon[t, b]`` and
    ``weapon[t+1, b]``. A naive ``reshape(-1)`` makes "adjacent" frames different
    lanes at the same ``t`` (weapons differ ~65% of the time on shuffled episodes)
    — scrambling every switch/dwell label. So we MUST compute along ``dim 0``
    (time), per lane, mirroring the target side-channel's prev-shift contract.

    Frame ``(t+1, b)`` is ``(t, b)``'s next UNLESS ``reset_mask[t+1, b]``.
    Fully vectorized (cummax along time for dwell-age) — no Python loop.

    dwell_age[t,b]      = timesteps the weapon at (t,b) has been held (0 at a change/reset)
    switch_next[t,b]    = weapon[t+1,b] != weapon[t,b] (and t+1 is same sequence)
    new_weapon_target   = weapon[t+1,b]-1 on real-weapon switches else -100 (ignore)
    valid[t,b]          = t+1 exists in-window and is the same sequence
    """
    weapon = weapon.long()
    if weapon.ndim == 1:                         # 1D fallback: one contiguous lane
        weapon = weapon.unsqueeze(1)             # (T, 1)
    elif weapon.ndim > 2:                        # (T, B, 1) -> (T, B)
        weapon = weapon.reshape(weapon.shape[0], weapon.shape[1])
    T, B = weapon.shape
    dev = weapon.device
    if reset_mask is not None:
        reset = reset_mask.bool().reshape(T, B)
    else:
        reset = torch.zeros((T, B), dtype=torch.bool, device=dev)

    pos = torch.arange(T, device=dev).unsqueeze(1).expand(T, B)
    change = torch.zeros((T, B), dtype=torch.bool, device=dev)
    if T >= 2:
        change[1:] = weapon[1:] != weapon[:-1]
    boundary = change | reset
    boundary[0] = True
    last_b = torch.where(boundary, pos, torch.full_like(pos, -1))
    last_b = torch.cummax(last_b, dim=0).values          # most-recent boundary (per lane)
    dwell = (pos - last_b).to(torch.float32)

    switch_next = torch.zeros((T, B), dtype=torch.float32, device=dev)
    new_tgt = torch.full((T, B), -100, dtype=torch.long, device=dev)
    valid = torch.zeros((T, B), dtype=torch.bool, device=dev)
    if T >= 2:
        nxt, cur, same = weapon[1:], weapon[:-1], ~reset[1:]
        changed = (nxt != cur) & same
        switch_next[:-1] = changed.float()
        valid[:-1] = same
        # WHAT target valid ONLY on a switch TO a real weapon (impulse
        # 1..WEAPON_HEAD_SIZE -> class 0..WEAPON_HEAD_SIZE-1). A switch to "none"
        # (nxt=0) or out-of-range id must be ignore_index, NOT nxt-1: a target of
        # -1 is an out-of-bounds gather in cross_entropy, which on ROCm wedges the
        # GPU queue asynchronously (bench sharp-edge #3) -> CPU spins forever.
        nxt_real = (nxt >= 1) & (nxt <= WEAPON_HEAD_SIZE)
        keep = changed & nxt_real
        new_tgt[:-1] = torch.where(keep, nxt - 1, torch.full_like(nxt, -100))

    return WeaponSwitchContext(
        dwell_age=dwell.reshape(-1), held_weapon=weapon.reshape(-1),
        switch_next=switch_next.reshape(-1), new_weapon_target=new_tgt.reshape(-1),
        valid=valid.reshape(-1),
    )
