"""attack_geom_bundle — no-CLS attack head fed EXPLICIT soft-pooled (rel, dist).

The gap this fills: every prior attack probe got target geometry either as the
pooled GT ``target_feat`` token (redundant given CLS) or — in the attack_prior
heads — as an alignment *prior-logit* (which didn't help). Nobody fed the
explicit ``(rel, dist)`` to the attack MLP as INPUT FEATURES on a no-CLS scaffold.
This head does exactly that, to measure how far explicit target geometry alone
drives attack (the prior-offloading question).

Scaffold (identical to attack_bundle, NO CLS / NO GRU):
    ObsEmbedding(monolithic, include_spatial=False)
      → PreAttnEncoder (passthrough; no attention)
      → GTTargetPointer (oracle pool by labeler GT target_probs)
      → AttackGeomBundleHead

The head ignores the pooled ``target_feat`` entirely. It computes
    soft_target_rel = Σ softmax(target_logits) · entity_rel   (the GT target's rel)
    dist            = ‖soft_target_rel‖
    rel_dir         = soft_target_rel / dist                  (unit direction)
and feeds [rel_dir(3), dist(1)] → Linear(4, d_model) as the geometry block, plus
optional ``attack_finished`` (cooldown) and ``vel`` blocks. Same soft-pool as
qnn.model.bench.attack_prior.engaged_geom_weapon_embed_attack_head, but as
features not a prior-logit.

The geometry is fed as (unit DIRECTION, magnitude), not raw ``rel`` — the
forward-alignment cosine ``rel_dir[..., 0]`` is what predicts attack and it is
only linearly available after normalization (raw ``soft_rel[0]`` scales with
range; alignment is the ratio ``rel_x/dist``). Feeding raw rel collapses the
head to the all-negative base rate (tp=fp=0). This matches the normalization in
the attack_prior heads' ``look_prior``.

probe.json: head=attack_geom_bundle, d_model, d_attack, use_cooldown (bool),
use_vel (bool). self_weapon_embed_in_self is forced False (irrelevant here).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from qnn.bc.weapon_physics import ACTOR_REL_OFFSET
from qnn.model._mlp import make_head_mlp
from qnn.model.attack_head import AttackHeadInput, AttackHeadOutput
from qnn.model.bench.attack import attack_bce_loss, attack_metrics
from qnn.model.bench.inputs.gt_target_pointer import GTTargetPointer
from qnn.model.bench.inputs.obs_network import BenchObsNetwork
from qnn.model.bench.inputs.preattn_encoder import PreAttnEncoder
from qnn.model.bench.spec import HeadBuildResult, HeadLossSpec, HeadSpec, neutral_model_config
from qnn.model.network import Off
from qnn.model.transformer import ObsEmbedding

# self_scalars (17-dim monolithic) layout — see qnn.model.dequant / schema.
_SS_VEL = slice(13, 16)            # view-relative velocity
_SS_ATTACK_FINISHED = slice(16, 17)
_REL = slice(ACTOR_REL_OFFSET, ACTOR_REL_OFFSET + 3)


class AttackGeomBundleHead(nn.Module):
    """Explicit soft-pooled (rel, dist) → attack logit; optional cooldown / vel."""

    def __init__(self, *, d_model: int, d_attack: int, use_cooldown: bool, use_vel: bool) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.use_cooldown = bool(use_cooldown)
        self.use_vel = bool(use_vel)
        self.geom_proj = nn.Linear(4, d_model)            # [rel(3), dist(1)]
        if self.use_cooldown:
            self.cooldown_proj = nn.Linear(1, d_model)    # attack_finished
        if self.use_vel:
            self.vel_proj = nn.Linear(3, d_model)         # velocity
        d_in = d_model * (1 + int(self.use_cooldown) + int(self.use_vel))
        self.mlp = make_head_mlp(d_in, 1, d_attack, "gelu")

    def forward(self, inp: AttackHeadInput) -> AttackHeadOutput:
        assert inp.target_logits is not None and inp.entity_scalars is not None
        assert inp.actor_mask is not None and inp.self_scalars is not None
        dtype = inp.features.dtype

        entity_rel = inp.entity_scalars[..., _REL].to(dtype)              # (B*, N, 3)
        probs = F.softmax(inp.target_logits, dim=-1).to(dtype)           # (B*, N)
        soft_rel = (probs.unsqueeze(-1) * entity_rel).sum(dim=-2)        # (B*, 3)
        has_actor = inp.actor_mask.any(dim=-1, keepdim=True).to(dtype)
        soft_rel = soft_rel * has_actor
        dist = torch.linalg.vector_norm(soft_rel, dim=-1, keepdim=True)  # (B*, 1)
        # Feed the unit DIRECTION (not raw rel) so the forward-alignment
        # cosine — rel_dir[..., 0] — is a *linear* input feature. The
        # attack signal lives in that normalized cosine (attack rate
        # 4%→24% across its quartiles, corr≈0.15), NOT in raw soft_rel[0]
        # (corr≈0.03): raw rel scales with range, so the alignment is the
        # nonlinear ratio rel_x/dist that a from-scratch MLP fails to
        # recover, collapsing to the all-negative base rate. This mirrors
        # qnn.model.bench.attack_prior's look_prior = normalize(soft_rel).
        # has_actor=0 → soft_rel=0 → dir=0 (stays zero); seg-masked frames
        # always have a target so dist>0 in training.
        rel_dir = soft_rel / dist.clamp(min=1e-6)                        # (B*, 3) unit dir
        geom = self.geom_proj(torch.cat([rel_dir, dist], dim=-1))        # (B*, d_model)

        parts = [geom]
        if self.use_cooldown:
            parts.append(self.cooldown_proj(inp.self_scalars[..., _SS_ATTACK_FINISHED].to(dtype)))
        if self.use_vel:
            parts.append(self.vel_proj(inp.self_scalars[..., _SS_VEL].to(dtype)))

        mlp_in = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        delta = self.mlp(mlp_in)
        zeros = torch.zeros_like(delta)
        return AttackHeadOutput(attack_logit=delta, prior_logit=zeros, delta_attack=delta)


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(f"probe.json must define {key!r} for head=attack_geom_bundle")
    return probe[key]


def _build_attack_geom_bundle(probe: Mapping[str, Any]) -> HeadBuildResult:
    d_model      = int(_required(probe, "d_model"))
    d_attack     = int(_required(probe, "d_attack"))
    use_cooldown = bool(probe.get("use_cooldown", False))
    use_vel      = bool(probe.get("use_vel", False))

    model_config = neutral_model_config(d_model=d_model, self_weapon_embed_in_self=False)

    def factory(obs_dim: int, model_cfg) -> BenchObsNetwork:
        return BenchObsNetwork(
            obs_dim=obs_dim,
            model=model_cfg,
            obs_embedding=ObsEmbedding(
                d_model=d_model, self_weapon_embed_in_self=False, include_spatial=False,
            ),
            encoder=PreAttnEncoder(),
            target_pointer=GTTargetPointer(d_model=d_model),
            temporal=Off,
            move_head=Off,
            look_head=Off,
            weapon_head=Off,
            attack_head=AttackGeomBundleHead(
                d_model=d_model, d_attack=d_attack,
                use_cooldown=use_cooldown, use_vel=use_vel,
            ),
        )

    return model_config, factory


ATTACK_GEOM_BUNDLE = HeadSpec(
    name="attack_geom_bundle",
    loss=HeadLossSpec(
        loss_fn=attack_bce_loss,
        metrics_fn=attack_metrics,
        label_key="attack",
        output_dim=1,
        selection_metric="attack_skill",
        selection_lower_is_better=False,
    ),
    build=_build_attack_geom_bundle,
)
