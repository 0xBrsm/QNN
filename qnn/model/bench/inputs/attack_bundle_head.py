"""Attack head with composable input bundles.

Consumes the GT-pooled ``target_feat`` (the second half of
``AttackHeadInput.features``; ``features = cat(self_readout, target_feat)``
under the head_probe layout, and the self_readout half is discarded by
design) plus an optional set of bundles. Each bundle is a
:class:`TokenBuilder` over named ``obs_fields``, projected to ``d_model`` and
concatenated alongside ``target_feat`` into the head MLP:

* ``"engaged_ema"`` — ``Linear(1, d_model)`` over the engagement_ema scalar.
* ``"weapon_embed"`` — pure tied ``entity_embed(self_weapon_id)`` lookup.
* ``"weapon"``      — ``Linear(3, d_model)`` over
                      ``[damage, radius, attack_finished]`` plus
                      ``entity_embed(self_weapon_id)``.
* ``"motion"``      — ``Linear(7, d_model)`` over ``[vel(3), view_pitch(1),
                      look_delta(3)]`` plus ``movement_embed`` (ground stage)
                      plus the SUIT powerup embed.

All obs scalars (including ``look_delta``, now a first-class self-motion field)
and vocab ids come from the forward-scoped :class:`ObsAccessor` (entered by
``BenchObsNetwork``); ``engagement_ema`` arrives through the accessor's aux
delegation to its side-channel context. ``bundles`` is a tuple of names; the empty tuple
(or the ``"none"`` sentinel) is the target_feat-only baseline. Bundle order
is fixed by ``_VALID_BUNDLES`` so probe.json key order never changes the MLP
weight layout.

Attack-head MLP: ``Linear(d_in, d_attack) → GELU → Linear(d_attack, 1)``,
with ``d_in == d_model * (1 + len(bundles))``.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from qnn.model._mlp import make_head_mlp
from qnn.model.attack_head import AttackHeadInput, AttackHeadOutput
from qnn.model.tokens.obs_accessor import current_obs_accessor
from qnn.model.tokens.obs_fields import (
    MOTION_FIELDS, ScalarGroup, VocabEmbed,
)
from qnn.model.tokens.token_builder import TokenBuilder


_VALID_BUNDLES = ("engaged_ema", "weapon_embed", "weapon", "motion")

# Each bundle's token = sum of these FieldSources (one Linear per ScalarGroup).
_BUNDLE_FIELDS: dict[str, list] = {
    "engaged_ema": [ScalarGroup(["engagement"])],
    "weapon_embed": [VocabEmbed("weapon_id")],
    "weapon": [ScalarGroup(["weapon_dmg_rad", "attack_finished"]), VocabEmbed("weapon_id")],
    "motion": list(MOTION_FIELDS),
}


def _normalize_bundles(raw: Iterable[str] | str | None) -> tuple[str, ...]:
    """Probe.json → canonical tuple of bundle names.

    Accepts a single string (legacy ``bundle`` knob) or an iterable of
    strings (``bundles`` knob). ``"none"`` and empty iterables both collapse
    to the empty tuple. The returned order is the canonical one defined by
    ``_VALID_BUNDLES`` so probe.json key ordering does not affect the
    parameter layout / checkpoint shape.
    """
    if raw is None:
        return ()
    if isinstance(raw, str):
        if raw == "none":
            return ()
        items = [raw]
    else:
        items = [str(x) for x in raw]
    deduped: set[str] = set()
    for item in items:
        if item == "none":
            continue
        if item not in _VALID_BUNDLES:
            raise ValueError(
                f"bundle name must be one of {_VALID_BUNDLES + ('none',)}, "
                f"got {item!r}"
            )
        deduped.add(item)
    return tuple(name for name in _VALID_BUNDLES if name in deduped)


class AttackBundleHead(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_attack: int,
        bundles: Iterable[str] | str | None,
        entity_embed: nn.Embedding,
        movement_embed: nn.Embedding,
    ) -> None:
        """Tied embeds: both ``entity_embed`` and ``movement_embed`` are the
        encoder's (held-weapon rocket and entity-token rocket are the same
        concept; same for movement IDs). The factory in ``attack_bundle.py``
        wires both from ObsEmbedding."""
        super().__init__()
        self.d_model = int(d_model)
        self.d_attack = int(d_attack)
        self.bundles = _normalize_bundles(bundles)

        # One TokenBuilder per active bundle, inserted in canonical order.
        self.bundle_builders = nn.ModuleDict()
        for name in self.bundles:
            self.bundle_builders[name] = TokenBuilder(
                self.d_model, _BUNDLE_FIELDS[name],
                entity_embed=entity_embed, movement_embed=movement_embed,
            )

        d_in = self.d_model * (1 + len(self.bundles))
        self.mlp = make_head_mlp(d_in, 1, self.d_attack, "gelu")

    def forward(self, inp: AttackHeadInput) -> AttackHeadOutput:
        target_feat = inp.features[..., self.d_model:]
        parts: list[torch.Tensor] = [target_feat]
        if self.bundles:
            acc = current_obs_accessor()
            # Canonical order — matches _VALID_BUNDLES so the MLP weight
            # layout is independent of probe.json bundle ordering.
            for name in _VALID_BUNDLES:
                if name in self.bundle_builders:
                    parts.append(
                        self.bundle_builders[name](acc, dtype=inp.features.dtype)
                    )
        mlp_in = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        delta = self.mlp(mlp_in)
        zeros = torch.zeros_like(delta)
        return AttackHeadOutput(
            attack_logit=delta,
            prior_logit=zeros,
            delta_attack=delta,
        )
