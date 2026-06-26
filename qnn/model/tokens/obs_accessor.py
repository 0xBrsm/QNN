"""ObsAccessor — uniform read access to obs scalars + vocab ids.

A token builder gets one ``ObsAccessor`` and asks it for fields by name
(``obs_fields.SCALAR_FIELDS`` / ``VOCAB_FIELDS``)
instead of re-deriving slices, masks, weapon-impulse conversions, and
readiness gathers by hand. The accessor is a lightweight read view: it
holds **no learnable parameters** (so no device-drift concern) and runs the
stateless :class:`SelfDequantizer` idempotently on construction.

Scope plumbing:

* ``obs_accessor_scope(accessor)`` / ``current_obs_accessor()`` mirror the
  other forward-scoped bench contexts (``engagement_ema_context`` …).
* ``obs_accessor_scope_from_obs(obs)`` flattens + dequants raw obs and
  enters the scope — the single helper a bench ``Network`` wrapper calls.
Aux signals (``engagement``) are NOT in obs — they are built at preload from
labels and entered as their own contexts by the bench side-channel provider
*outside* this scope. ``ObsAccessor.aux``
delegates to those active contexts, so the head still only needs the one
accessor object.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Iterator, Mapping

import torch

from qnn.model.tokens.obs_fields import (
    SCALAR_FIELDS, VOCAB_FIELDS, WT_DAMAGE, WT_RADIUS,
)
from qnn.vocab import self_weapon_id_to_impulse


def collect_powerup_ids(obs_dict: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
    """Return the (B*, 5) powerup-id tensor when present.

    Prefer the single 5-wide ``self_powerup_ids`` field if the dataloader
    produces it; otherwise compose from the bench-schema split fields
    (``self_{state,arsenal,motion}_powerup_ids``). Returns None when neither
    is present so callers can no-op the powerup contribution.
    """
    pids = obs_dict.get("self_powerup_ids")
    if pids is None:
        parts = [
            obs_dict[k] for k in (
                "self_state_powerup_ids",
                "self_arsenal_powerup_ids",
                "self_motion_powerup_ids",
            ) if k in obs_dict
        ]
        if parts:
            pids = torch.cat(parts, dim=1)
    if pids is None:
        return None
    return pids.long()


class ObsAccessor:
    """Read view over a (dequanted, flattened) obs dict for token building."""

    def __init__(
        self,
        obs: Mapping[str, torch.Tensor],
        *,
        aux: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        # Idempotent, stateless dequant — passes through if already dequanted.
        from qnn.model.dequant import SelfDequantizer
        self.dq: dict[str, torch.Tensor] = SelfDequantizer()(obs)
        self._aux_override = dict(aux) if aux is not None else None
        ref = self.dq["self_motion_scalars"]
        self.batch = int(ref.shape[0])
        self.device = ref.device

    # ── scalars ──────────────────────────────────────────────────────
    def scalar(self, name: str) -> torch.Tensor:
        """Return the named scalar field as ``(B*, width)`` float."""
        spec = SCALAR_FIELDS[name]
        if spec.slice_key is not None:
            return self.dq[spec.slice_key][..., spec.start:spec.stop]
        return getattr(self, f"_compute_{spec.compute}")()

    # ── vocab ids ────────────────────────────────────────────────────
    def vocab_ids(self, name: str) -> torch.Tensor:
        """Return the raw (unclamped) long ids for a vocab field.

        Single-id fields come back ``(B*,)``; powerup bundles ``(B*, P)``.
        The TokenBuilder clamps to its table size and applies the mask.
        """
        spec = VOCAB_FIELDS[name]
        if name == "powerup_all":
            pids = collect_powerup_ids(self.dq)
            if pids is None:
                return torch.zeros(self.batch, 1, dtype=torch.long, device=self.device)
            return pids.long()
        t = self.dq[spec.obs_key].long()
        if spec.reduce == "sum":            # powerup bundle (B*, P)
            return t
        return t.squeeze(-1)                # single id (B*, 1) → (B*,)

    def readiness(self) -> torch.Tensor:
        """Per-weapon readiness ``(B*, 8)`` (axe-first) for the einsum source."""
        return self.dq["self_weapon_readiness"]

    def aux(self, name: str) -> torch.Tensor:
        """Return a non-obs aux signal, delegating to its active context."""
        if self._aux_override is not None and name in self._aux_override:
            return self._aux_override[name]
        if name == "engagement":
            from qnn.model.bench.inputs.engagement_ema_context import (
                current_engagement_ema_context,
            )
            return current_engagement_ema_context().engagement_ema
        raise KeyError(f"unknown aux signal {name!r}")

    # ── computed scalar fields ───────────────────────────────────────
    def _impulse(self) -> torch.Tensor:
        """ENTITY_IDS self_weapon_id → impulse byte (0..8), shape (B*,)."""
        return self_weapon_id_to_impulse(
            self.dq["self_weapon_id"].long().squeeze(-1)
        )

    def _weapon_static_table(self) -> torch.Tensor:
        return _weapon_static_cpu().to(self.device)

    def _compute_weapon_static(self) -> torch.Tensor:
        table = self._weapon_static_table()
        imp = self._impulse().clamp(0, table.shape[0] - 1)
        return table[imp]                                       # (B*, 7)

    def _compute_weapon_dmg_rad(self) -> torch.Tensor:
        table = self._weapon_static_table()
        imp = self._impulse().clamp(0, table.shape[0] - 1)
        return table[imp][:, (WT_DAMAGE, WT_RADIUS)]            # (B*, 2)

    def _compute_held_readiness(self) -> torch.Tensor:
        imp = self._impulse()
        readiness = self.readiness()                            # (B*, 8)
        held_idx = (imp - 1).clamp_min(0).unsqueeze(-1)         # (B*, 1)
        held = readiness.gather(1, held_idx)                    # (B*, 1)
        has_weapon = (imp >= 1).to(readiness.dtype).unsqueeze(-1)
        return held * has_weapon

    def _compute_engagement(self) -> torch.Tensor:
        e = self.aux("engagement")
        if e.dim() == 0:
            e = e.expand(self.batch)
        return e.unsqueeze(-1)                                  # (B*, 1)


# Weapon static table is pure data (no params); cache one cpu copy and move
# to the obs device on demand (9×7 — negligible).
_WEAPON_STATIC_CACHE: torch.Tensor | None = None


def _weapon_static_cpu() -> torch.Tensor:
    global _WEAPON_STATIC_CACHE
    if _WEAPON_STATIC_CACHE is None:
        from qnn.bc.weapon_physics import build_model_weapon_scalars
        _WEAPON_STATIC_CACHE = torch.from_numpy(build_model_weapon_scalars()).float()
    return _WEAPON_STATIC_CACHE


# ── forward-scoped accessor context ──────────────────────────────────

_CTX: "contextvars.ContextVar[ObsAccessor | None]" = contextvars.ContextVar(
    "qnn_obs_accessor_ctx", default=None,
)


@contextlib.contextmanager
def obs_accessor_scope(accessor: ObsAccessor | None) -> Iterator[None]:
    token = _CTX.set(accessor)
    try:
        yield
    finally:
        _CTX.reset(token)


def current_obs_accessor() -> ObsAccessor:
    acc = _CTX.get()
    if acc is None:
        raise RuntimeError(
            "ObsAccessor requested but no scope set — a bench Network wrapper "
            "(BenchObsNetwork / MoveAimNetwork) must enter the accessor scope "
            "before calling the head."
        )
    return acc


def flatten_obs(
    obs: Mapping[str, torch.Tensor],
) -> tuple[tuple[int, int] | None, dict[str, torch.Tensor]]:
    """Detect sequence-vs-flat obs and return ``(seq_shape, flat_obs)``."""
    obs_dict = dict(obs)
    sample = obs_dict.get("vel")
    if sample is None:
        sample = obs_dict["self_scalars"]
    if sample.ndim == 3:
        seq_len = int(sample.shape[0])
        batch_size = int(sample.shape[1])
        flat = {
            key: value.reshape(seq_len * batch_size, *value.shape[2:])
            for key, value in obs_dict.items()
        }
        return (seq_len, batch_size), flat
    return None, obs_dict


def obs_accessor_scope_from_obs(obs: Mapping[str, torch.Tensor]):
    """Flatten raw obs, build an ObsAccessor, and return its scope manager."""
    _, flat = flatten_obs(obs)
    return obs_accessor_scope(ObsAccessor(flat))
