"""WeaponQueryTargetPointer — weapon-spec query against enemy tokens.

Drop-in for ``TargetPointer`` in Network's ``target_pointer`` slot that
replaces the cls/GRU query with one built purely from the held weapon's
physical specs + an impulse-keyed vocab embed:

    query = weapon_proj(weapon_static[impulse(weapon_id)])
          + weapon_embed(impulse(weapon_id)) * (impulse > 0)

No ``attack_finished``, no ammo, no kind tag — strictly the 7 static
``build_model_weapon_scalars`` columns (damage, cooldown, v_horiz,
v_vert_0, gravity, max_dist, radius) projected to ``d_model``, plus a
learned per-weapon embedding.

Logits are the canonical pointer dot product against post-encoder
``entity_outs``, with non-enemy entities masked to ``-1e9`` (the
labeler's ``target_probs`` mass is zero on non-enemy indices, so this
just removes pointless logit competition). ``enemy_mask`` arrives via
:class:`TargetPointerInput` directly.

The bench Network wrapper is responsible for stashing the per-frame
weapon impulse via :meth:`stash` before each forward.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from qnn.bc.weapon_physics import MODEL_TOKEN_SCALAR_DIM, build_model_weapon_scalars
from qnn.model.target import TargetPointerInput, TargetPointerOutput


_WEAPON_VOCAB = 9  # impulse 0=NONE, 1=AXE, ..., 8=LG


class WeaponQueryTargetPointer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        static_scalar_indices: Sequence[int] | None = None,
    ) -> None:
        """``static_scalar_indices`` selects a subset of the 7 columns of
        ``build_model_weapon_scalars`` to feed into ``weapon_proj``. Default
        (None) uses all 7. Subset ablations (e.g. ``[0, 6]`` for
        damage+radius after seeing dead/harmful inputs in the saliency
        report) feed only those columns to the Linear, keeping the
        ``weapon_embed`` path unchanged.
        """
        super().__init__()
        self.d_model = int(d_model)
        if static_scalar_indices is None:
            indices = tuple(range(MODEL_TOKEN_SCALAR_DIM))
        else:
            indices = tuple(int(i) for i in static_scalar_indices)
            if not indices:
                raise ValueError("static_scalar_indices must be non-empty")
            for i in indices:
                if not (0 <= i < MODEL_TOKEN_SCALAR_DIM):
                    raise ValueError(
                        f"static_scalar_indices entry {i} out of range "
                        f"[0, {MODEL_TOKEN_SCALAR_DIM})"
                    )
        self.static_scalar_indices = indices
        self.weapon_proj = nn.Linear(len(indices), self.d_model)
        self.weapon_embed = nn.Embedding(_WEAPON_VOCAB, self.d_model)
        nn.init.normal_(self.weapon_embed.weight, std=0.02)
        weapon_static = torch.from_numpy(build_model_weapon_scalars())  # (9, 7)
        self.register_buffer("_weapon_static", weapon_static, persistent=False)
        self.register_buffer(
            "_static_idx", torch.tensor(indices, dtype=torch.long), persistent=False,
        )
        self._self_weapon_impulse: torch.Tensor | None = None

    def stash(self, *, weapon_impulse: torch.Tensor) -> None:
        """Bench wrapper supplies the per-frame impulse (0..8) computed
        from ``obs['self_weapon_id']``."""
        self._self_weapon_impulse = weapon_impulse

    def forward(self, inp: TargetPointerInput) -> TargetPointerOutput:
        if self._self_weapon_impulse is None:
            raise RuntimeError(
                "WeaponQueryTargetPointer.forward called without stashed "
                "weapon_impulse — the bench Network wrapper must call "
                ".stash(weapon_impulse=...) before each forward."
            )
        wid = self._self_weapon_impulse.long().clamp(0, _WEAPON_VOCAB - 1)
        static = self._weapon_static[wid].to(inp.entity_outs.dtype)               # (B*, 7)
        static = static.index_select(dim=-1, index=self._static_idx)              # (B*, K)
        query = self.weapon_proj(static)                                          # (B*, D)
        has_weapon = (wid > 0).to(query.dtype).unsqueeze(-1)
        query = query + self.weapon_embed(wid) * has_weapon

        logits = (inp.entity_outs * query.unsqueeze(1)).sum(dim=-1)               # (B*, N)
        # Enemy-only mask AND-ed with the encoder's valid-entity mask so
        # padding still drops out.
        valid = inp.enemy_mask & inp.entity_mask
        valid_f = valid.to(logits.dtype)
        logits = logits.masked_fill(valid_f == 0, -1e9)

        # Zero target_feat for empty (no-enemy) scenes so a uniform softmax
        # over masked-out indices doesn't leak through downstream.
        has_any = (valid_f.sum(dim=-1, keepdim=True) > 0).to(logits.dtype)
        probs = torch.softmax(logits, dim=-1)
        target_feat = (probs.unsqueeze(-1) * inp.entity_outs).sum(dim=1) * has_any

        return TargetPointerOutput(target_logits=logits, target_feat=target_feat)
