"""weapon_switch — switch-process weapon head: WHAT (new weapon) + WHEN (hazard).

The corpus says weapon is a hazard-driven switch process (src/docs/weapon-head.md):
~85% combat persistence, ~4% switch rate, WHEN driven by dwell-age, WHAT by
inventory + from-weapon (geometry is noise). This head models that directly:

  WHAT logits (8-way, the WEAPON_HEAD slot) = MLP([arsenal inventory, from-weapon
       embed]) — the new weapon to switch to. Trained ONLY on switch frames via
       new_weapon_target (ignore_index=-100 elsewhere).
  WHEN logit (loss-only _weapon_switch_when) = MLP([dwell_age]) — switch hazard.
       Copycat-safe: dwell-age only, NOT the held-weapon identity.

Loss lives entirely in this head (policy dispatches to weapon_loss, mirroring
look_head.look_loss): CE(WHAT | switch) + BCE(WHEN, switch_next, pos_weight).
Labels from the weapon_switch_context side-channel. No incumbent in WHEN, no
distance anywhere. Eval distributionally (hazard calibration), not per-frame.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.bench.inputs.obs_network import BenchObsNetwork
from qnn.model.bench.inputs.preattn_encoder import PreAttnEncoder
from qnn.model.bench.inputs.weapon_switch_context import current_weapon_switch_context
from qnn.model.bench.spec import HeadBuildResult, HeadLossSpec, HeadSpec, neutral_model_config
from qnn.model.network import WEAPON_HEAD, Off
from qnn.model.tokens.obs_accessor import current_obs_accessor
from qnn.model.tokens.obs_fields import VocabSum, WeaponReadiness
from qnn.model.tokens.token_builder import TokenBuilder
from qnn.model.transformer import ObsEmbedding
from qnn.model.weapon_head import WeaponHeadInput, WeaponHeadOutput
from qnn.schema import WEAPON_HEAD_SIZE


class WeaponSwitchHead(nn.Module):
    def __init__(self, *, d_model: int, d_hidden: int,
                 entity_embed: nn.Embedding, movement_embed: nn.Embedding) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.arsenal_builder = TokenBuilder(
            self.d_model, [WeaponReadiness(), VocabSum("powerup_arsenal")],
            entity_embed=entity_embed, movement_embed=movement_embed,
        )
        self.from_embed = nn.Embedding(WEAPON_HEAD_SIZE + 1, self.d_model)  # held 0..8
        self.what_mlp = make_head_mlp(2 * self.d_model, WEAPON_HEAD_SIZE, d_hidden, "gelu")
        # WHEN hazard from dwell-age. Represent dwell as a d_model-wide BUCKET
        # EMBEDDING (not a Linear(1,·) — 1-wide inputs hit slow/broken MIOpen
        # paths on the Strix Halo iGPU per the bench skill) — also a richer,
        # nonlinear duration representation matching the decreasing hazard.
        self._n_dwell = 64
        self.dwell_embed = nn.Embedding(self._n_dwell, self.d_model)
        self.when_mlp = make_head_mlp(self.d_model, 1, d_hidden, "gelu")    # canonical-width input
        self.embed = nn.Embedding(WEAPON_HEAD_SIZE, self.d_model)           # motor-head context (unused)
        # Calibrated hazard: pos_weight=1.0 (plain BCE). A class-balancing
        # pos_weight=(1-p)/p~=24 floods predictions high (mean hazard ~0.6 vs
        # ~0.04 actual) and DESTROYS calibration — but a hazard's value IS its
        # calibration (P(switch|dwell-age) near base rate, modulated by dwell).
        # We want the MLE-calibrated probability, not balanced recall.
        self.when_pos_weight = 1.0

    def forward(self, inp: WeaponHeadInput) -> WeaponHeadOutput:
        dtype = inp.selector.dtype
        acc = current_obs_accessor()
        ctx = current_weapon_switch_context()
        arsenal = self.arsenal_builder(acc, dtype=dtype)                    # (B*, d_model)
        B = arsenal.shape[0]
        if ctx is not None and ctx.held_weapon is not None:
            from_w = ctx.held_weapon.clamp(0, WEAPON_HEAD_SIZE).long()
            dwell_b = ctx.dwell_age.clamp(0, self._n_dwell - 1).long().reshape(-1)
        else:  # inference / no side-channel: neutral
            from_w = torch.zeros(B, dtype=torch.long, device=arsenal.device)
            dwell_b = torch.zeros(B, dtype=torch.long, device=arsenal.device)
        what = self.what_mlp(torch.cat([arsenal, self.from_embed(from_w)], dim=-1))
        when = self.when_mlp(self.dwell_embed(dwell_b))                      # (B*, 1)
        context = F.softmax(what, dim=-1) @ self.embed.weight
        return WeaponHeadOutput(logits=what, context=context, when_logit=when)

    def weapon_loss(self, logits, actions, valid_flat, compute_metrics):
        ctx = current_weapon_switch_context()
        what = logits[WEAPON_HEAD].reshape(-1, WEAPON_HEAD_SIZE)
        when = logits["_weapon_switch_when"].reshape(-1)
        what_tgt = ctx.new_weapon_target.reshape(-1)
        sw = ctx.switch_next.reshape(-1)
        valid = ctx.valid.reshape(-1).to(when.dtype)
        # SYNC-FREE (no .any()/.item()/bool-index — those stall the ROCm dispatch
        # queue, as the canonical weapon path warns): masked sum / count on-GPU.
        # WHAT: CE over switch frames only (others carry -100 -> ignored).
        ce_sum = F.cross_entropy(what, what_tgt, ignore_index=-100, reduction="sum")
        what_cnt = (what_tgt >= 0).sum().clamp(min=1)
        what_loss = ce_sum / what_cnt
        # WHEN: BCE over frames with a defined next, fixed pos_weight for the ~4% rate.
        pw = torch.as_tensor(self.when_pos_weight, device=when.device, dtype=when.dtype)
        bce = F.binary_cross_entropy_with_logits(when, sw, pos_weight=pw, reduction="none")
        valid_cnt = valid.sum().clamp(min=1)
        when_loss = (bce * valid).sum() / valid_cnt
        loss = what_loss + when_loss
        metrics = {}
        if compute_metrics:
            metrics["loss_weapon"] = loss.detach()
            metrics["loss_weapon_what"] = what_loss.detach()
            metrics["loss_weapon_when"] = when_loss.detach()
            with torch.no_grad():  # sync-free masked means
                # NOTE: keys MUST start with an _AVERAGED_METRIC_PREFIXES entry
                # (acc_/pred_rate_/pos_rate_) or supervised_loop drops them from
                # epoch history (it only reduces sum-prefixed or avg-prefixed keys).
                p = torch.sigmoid(when)
                metrics["pred_rate_weapon_when"] = ((p * valid).sum() / valid_cnt).detach()   # model hazard rate
                metrics["pos_rate_weapon_when"] = ((sw * valid).sum() / valid_cnt).detach()    # actual switch rate
                msk = (what_tgt >= 0).to(what.dtype)
                correct = (what.argmax(-1) == what_tgt).to(what.dtype) * msk
                metrics["acc_weapon_what"] = (correct.sum() / msk.sum().clamp(min=1)).detach()
        return loss, metrics


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(f"probe.json must define {key!r} for head=weapon_switch")
    return probe[key]


def _build_weapon_switch(probe: Mapping[str, Any]) -> HeadBuildResult:
    d_model  = int(_required(probe, "d_model"))
    d_hidden = int(_required(probe, "d_hidden"))
    model_config = neutral_model_config(d_model=d_model, self_weapon_embed_in_self=False)

    def factory(obs_dim: int, model_cfg) -> BenchObsNetwork:
        obs_embedding = ObsEmbedding(
            d_model=d_model, self_weapon_embed_in_self=False, include_spatial=False,
        )
        return BenchObsNetwork(
            obs_dim=obs_dim, model=model_cfg, obs_embedding=obs_embedding,
            encoder=PreAttnEncoder(), target_pointer=Off, temporal=Off,
            move_head=Off, look_head=Off, attack_head=Off,
            weapon_head=WeaponSwitchHead(
                d_model=d_model, d_hidden=d_hidden,
                entity_embed=obs_embedding.entity_embed,
                movement_embed=obs_embedding.movement_embed,
            ),
        )
    return model_config, factory


def _stub(*_a, **_k):
    return torch.zeros(())


WEAPON_SWITCH = HeadSpec(
    name="weapon_switch",
    loss=HeadLossSpec(
        loss_fn=_stub, metrics_fn=lambda *_a, **_k: {},
        label_key="weapon", output_dim=WEAPON_HEAD_SIZE,
        selection_metric="loss_weapon", selection_lower_is_better=True,
    ),
    build=_build_weapon_switch,
)
