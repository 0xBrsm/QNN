"""Bench Network wrapper for the weapon-aim ablation.

Pre-stashes ``entity_rel``, ``entity_vel``, impulse-indexed ``weapon_id``,
and ``noop`` (derived from obs) on the bench look + attack heads as side-channel
attributes, then delegates to canonical ``Network.forward``. The bench
heads themselves compute ``aim_vec`` and ``target_feat`` from their
canonical inputs + the stashed extras — no forward hooks, no plumbing
of new tensors through canonical inter-component contracts. Canonical
Network, LookHeadInput, AttackHeadInput stay untouched.

Why side-channel pre-stash + per-head compute (not a forward hook):
a previous version registered a forward hook on the canonical target
pointer to compute ``aim_vec`` once and stash it on both heads. That
worked in isolated reproductions but hung the BC training loop in
production — the hook's interaction with the autograd graph during
per-step forward+backward was the culprit. Computing aim_vec inside
each head's forward avoids that entirely (two cheap recomputes per
step instead of one shared one, but no hook).

noop derivation: currently from obs (``attack_finished > 1 tick`` OR
held-weapon ammo == 0 OR ``weapon_id == 0``). Engine-side
``input_mask``-driven noop is a follow-up.
"""

from __future__ import annotations

from typing import Dict

import torch

from qnn.bc.weapon_physics import (
    ACTOR_REL_OFFSET, ACTOR_VEL_OFFSET, build_model_weapon_scalars,
)
from qnn.model.bench.weapon_aim.context import (
    WeaponAimContext, weapon_aim_context,
)
from qnn.model.dequant import (
    IDX_AMMO_CELLS, IDX_AMMO_NAILS, IDX_AMMO_ROCKETS, IDX_AMMO_SHELLS,
    IDX_ATTACK_FINISHED,
)
from qnn.model.network import Network, _flatten_obs
from qnn.vocab import self_weapon_id_to_impulse


# Held-weapon impulse → ammo idx in self_scalars. -1 sentinel = no ammo check
# (axe / no-weapon path). Used as a lookup table to vectorize the per-weapon
# ammo gate that used to be a Python loop over 8 entries × 3 GPU ops each.
_WEAPON_AMMO_IDX = (
    -1,                # 0  no weapon (already caught by `no_weapon`)
    -1,                # 1  axe — no ammo
    IDX_AMMO_SHELLS,   # 2  SG
    IDX_AMMO_SHELLS,   # 3  SSG
    IDX_AMMO_NAILS,    # 4  NG
    IDX_AMMO_NAILS,    # 5  SNG
    IDX_AMMO_ROCKETS,  # 6  GL
    IDX_AMMO_ROCKETS,  # 7  RL
    IDX_AMMO_CELLS,    # 8  LG
)


def _derive_noop(
    self_scalars: torch.Tensor,
    weapon_impulse: torch.Tensor,
    ammo_idx_table: torch.Tensor,
) -> torch.Tensor:
    """Approximate engine noop from continuous obs. Returns (B,) {0,1}.

    Three OR'd predicates:
      * ``no_weapon``      — weapon_impulse == 0
      * ``cooldown_active`` — attack_finished > 0 (small eps for f16 noise)
      * ``ammo_low``       — held weapon's ammo column < 1e-3 (skipped via the
                              -1 sentinel for axe / no-weapon)

    Vectorized via a (9,) lookup table indexed by ``weapon_impulse``; no
    Python loop over weapons.
    """
    no_weapon = (weapon_impulse == 0)
    cooldown_active = (self_scalars[:, IDX_ATTACK_FINISHED] > 1e-4)
    # Sentinel-aware gather: clamp -1 → 0 so we can call gather, then mask
    # the result with `needs_ammo_check`. The clamped-to-0 lookup picks
    # health (column 0) which we discard via the mask.
    raw_ammo_idx = ammo_idx_table[weapon_impulse]
    needs_ammo_check = (raw_ammo_idx >= 0)
    safe_idx = raw_ammo_idx.clamp(min=0).unsqueeze(1)
    ammo = self_scalars.gather(1, safe_idx).squeeze(1)
    ammo_low = needs_ammo_check & (ammo < 1e-3)
    return (no_weapon | cooldown_active | ammo_low).to(self_scalars.dtype)


class WeaponAimNetwork(Network):
    """Network subclass that stashes obs-derived extras on bench heads.

    Pre-forward stash (from obs, before super().forward()):
      * ``entity_rel``, ``entity_vel``, ``weapon_id``  → both heads
      * ``noop`` (derived)                              → attack head only
      * ``weapon_static`` table reference               → both heads

    The bench head ``forward`` reads its own canonical LookHeadInput /
    AttackHeadInput + these stashed extras and computes ``aim_vec`` and
    ``target_feat`` itself (using the already-available
    ``target_logits``, ``entity_rel``, etc.).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Static weapon-trajectory table (9, 7).
        weapon_static = torch.from_numpy(build_model_weapon_scalars())
        self.register_buffer("_weapon_static", weapon_static, persistent=False)
        # (9,) lookup table — held weapon's ammo idx in self_scalars, or -1
        # for axe / no-weapon. Used by _derive_noop's vectorized gather.
        self.register_buffer(
            "_ammo_idx_table",
            torch.tensor(_WEAPON_AMMO_IDX, dtype=torch.long),
            persistent=False,
        )

    def _build_context(self, flat_obs: Dict[str, torch.Tensor]) -> "WeaponAimContext":
        entity_rel = flat_obs["entity_scalars_raw"][..., ACTOR_REL_OFFSET:ACTOR_REL_OFFSET + 3]
        entity_vel = flat_obs["entity_scalars_raw"][..., ACTOR_VEL_OFFSET:ACTOR_VEL_OFFSET + 3]
        # obs.self_weapon_id is ENTITY_IDS-encoded (axe..LG = 3..10), while
        # build_model_weapon_scalars() is impulse-indexed (0..8). ROCm
        # surfaces out-of-range GPU gathers late/asynchronously, so
        # normalize before any bench head can index the static table.
        weapon_id = self_weapon_id_to_impulse(
            flat_obs["self_weapon_id"].long().squeeze(-1)
        ).long()
        noop = _derive_noop(flat_obs["self_scalars"], weapon_id, self._ammo_idx_table)
        return WeaponAimContext(
            entity_rel=entity_rel,
            entity_vel=entity_vel,
            weapon_id=weapon_id,
            weapon_static=self._weapon_static,
            noop=noop,
        )

    def forward(  # type: ignore[override]
        self,
        obs: Dict[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
        target_gt: torch.Tensor | None = None,
        target_probs_idx: torch.Tensor | None = None,
        prev_target_probs: torch.Tensor | None = None,
    ):
        _, flat_obs = _flatten_obs(obs)
        ctx = self._build_context(flat_obs)
        with weapon_aim_context(ctx):
            return super().forward(
                obs=obs,
                hidden=hidden,
                reset_mask=reset_mask,
                target_gt=target_gt,
                target_probs_idx=target_probs_idx,
                prev_target_probs=prev_target_probs,
            )
