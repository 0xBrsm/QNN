"""full_multitrunk — the trunk-split progression of full_4head.

Same four bench heads (move_cls / look_polar / attack_cls / weapon_cls), same
shared transformer encoder, but the single CLS→GRU trunk is split into N
independent trunks. Each trunk is its own learnable CLS query token (one shared
encoder pools the token stream into all of them) feeding its own GRU; heads are
routed to trunks by ``probe.json["trunks"]``:

    trunks = [["move","look"], ["attack","weapon"]]      # 2 trunks (run 2)
    trunks = [["move"],["look"],["attack"],["weapon"]]   # 4 trunks (run 3)

Tests the capacity-contention hypothesis: CCA showed the per-head CLS tokens
share only ~6/64 dims, so a single shared GRU may force the heads to compete.
Splitting the trunk gives each head-group its own pooled view + recurrence while
keeping the encoder shared (the representation is learned once; only the pooling
queries and the temporal integration are per-trunk).

No target pointer, no cross-trunk weapon_context — each head reads ONLY its
trunk's GRU readout (in_dim = d_gru). Decoupling the trunks is the whole point;
re-coupling them through weapon_context would defeat it. "If target is off,
it's off" (see the network target-padding fix) — target_feat is never fed.

Hidden-state contract: the policy sizes the recurrent hidden from
``ModelConfig.d_gru`` and threads it opaquely. We expose d_gru = N * d_gru_trunk
and pack/split the N GRU states along the last dim, so the canonical recurrent
loop drives N trunks unchanged. Per-head losses are dispatched by
QNNPolicy._compute_head_losses_and_metrics (move CE / attack BCE / weapon CE;
look carries PurePolarLookHead.look_loss). HeadLossSpec.loss_fn is a no-op stub.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from qnn.model.bench.spec import HeadBuildResult, HeadLossSpec, HeadSpec, neutral_model_config
from qnn.model.bench.inputs.held_weapon_split_obs_embedding import HeldWeaponSplitObsEmbedding
from qnn.model.bench.move_cls_transformer import CLSMoveHead
from qnn.model.bench.attack_cls_transformer import CLSAttackHead
from qnn.model.bench.weapon_cls_transformer import CLSWeaponHead
from qnn.model.bench.look_head_polar import PurePolarLookHead
from qnn.model.network import (
    ModelConfig, MOVE_HEAD, LOOK_HEAD, ATTACK_HEAD, WEAPON_HEAD,
    _flatten_obs, _restore_outputs,
)
from qnn.model.transformer import TransformerEncoder
from qnn.model.temporal import Temporal, TemporalInput
from qnn.model.move_head import MoveHeadInput
from qnn.model.look_head import LookHeadInput
from qnn.model.attack_head import AttackHeadInput
from qnn.model.weapon_head import WeaponHeadInput

_ALL_HEADS = (MOVE_HEAD, LOOK_HEAD, ATTACK_HEAD, WEAPON_HEAD)


class MultiTrunkNetwork(nn.Module):
    """N-trunk variant: shared encoder, N CLS queries → N GRUs → routed heads."""

    def __init__(
        self, *, d_model: int, n_heads: int, n_layers: int, d_ffn: int,
        d_gru: int, trunks: "Sequence[Sequence[str]]",
        d_move: int, d_look: int, d_attack: int, d_weapon: int,
        activation: str, attn_dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.trunks = [list(g) for g in trunks]
        self.n_cls = len(self.trunks)
        self.d_gru_trunk = int(d_gru)
        # Width exposed to the policy's recurrent loop: N packed GRU states.
        self.d_gru = self.n_cls * self.d_gru_trunk
        self.use_gru = True

        # head -> trunk index. Validate the routing covers each head once.
        self.head_trunk: dict[str, int] = {}
        for i, group in enumerate(self.trunks):
            for h in group:
                if h not in _ALL_HEADS:
                    raise RuntimeError(f"full_multitrunk: unknown head {h!r} in trunks")
                if h in self.head_trunk:
                    raise RuntimeError(f"full_multitrunk: head {h!r} assigned to >1 trunk")
                self.head_trunk[h] = i
        missing = [h for h in _ALL_HEADS if h not in self.head_trunk]
        if missing:
            raise RuntimeError(f"full_multitrunk: heads not routed to any trunk: {missing}")

        self.obs_embedding = HeldWeaponSplitObsEmbedding(
            d_model=self.d_model, self_weapon_embed_in_self=False, include_spatial=True,
        )
        self.encoder = TransformerEncoder(
            d_model=self.d_model, n_heads=int(n_heads), n_layers=int(n_layers),
            d_ffn=int(d_ffn), dropout=float(attn_dropout),
        )
        # The obs embedding already emits one CLS at slot 0; add N-1 more query
        # tokens. Small-normal init breaks the inter-trunk symmetry (identical
        # query vectors would pool identically with no positional encoding).
        self.extra_cls = nn.ParameterList(
            [nn.Parameter(torch.randn(self.d_model) * 0.02) for _ in range(self.n_cls - 1)]
        )
        self.temporals = nn.ModuleList(
            [Temporal(self.d_model, self.d_gru_trunk) for _ in range(self.n_cls)]
        )

        self.move_head = CLSMoveHead(in_dim=self.d_gru_trunk, d_move=int(d_move), activation=activation)
        self.look_head = PurePolarLookHead(self.d_gru_trunk, int(d_look), activation)
        self.attack_head = CLSAttackHead(in_dim=self.d_gru_trunk, d_attack=int(d_attack), activation=activation)
        self.weapon_head = CLSWeaponHead(
            in_dim=self.d_gru_trunk, d_model=self.d_model, d_weapon=int(d_weapon), activation=activation,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs, hidden=None, reset_mask=None):
        seq_shape, flat_obs = _flatten_obs(obs)
        enc_in = self.obs_embedding(flat_obs)
        tokens = enc_in.tokens
        batch_flat = int(tokens.shape[0])
        kpm = enc_in.key_padding_mask

        # Prepend the extra CLS query tokens → N CLS at slots 0..N-1.
        if self.n_cls > 1:
            extras = [c.view(1, 1, -1).expand(batch_flat, 1, -1).to(tokens.dtype)
                      for c in self.extra_cls]
            tokens = torch.cat([*extras, tokens], dim=1)
            if kpm is not None:
                pad = torch.zeros((batch_flat, self.n_cls - 1), dtype=kpm.dtype, device=kpm.device)
                kpm = torch.cat([pad, kpm], dim=1)

        for block in self.encoder.blocks:
            tokens = block(tokens, key_padding_mask=kpm)
        tokens = self.encoder.final_ln(tokens)
        readouts = [tokens[:, i, :] for i in range(self.n_cls)]   # per-trunk pooled CLS

        # Per-trunk GRU. Split the packed hidden into N trunk states.
        if hidden is not None:
            hsplit = torch.split(hidden, self.d_gru_trunk, dim=-1)
        else:
            hsplit = [None] * self.n_cls
        gru_outs: list[torch.Tensor] = []
        next_h: list[torch.Tensor] = []
        for i, temporal in enumerate(self.temporals):
            t = temporal(TemporalInput(
                flat_pool=readouts[i], hidden=hsplit[i],
                reset_mask=reset_mask, seq_shape=seq_shape,
            ))
            gru_outs.append(t.flat_out)
            next_h.append(t.next_hidden)
        next_hidden = torch.cat(next_h, dim=-1)

        def feat(head: str) -> torch.Tensor:
            return gru_outs[self.head_trunk[head]]

        zero_kw = {"dtype": gru_outs[0].dtype, "device": gru_outs[0].device}
        logits_flat: dict[str, torch.Tensor] = {}

        logits_flat[MOVE_HEAD] = self.move_head(MoveHeadInput(features=feat(MOVE_HEAD))).logits

        lo = self.look_head(LookHeadInput(
            features=feat(LOOK_HEAD),
            target_logits=torch.zeros((batch_flat, 1), **zero_kw),
            entity_rel=torch.zeros((batch_flat, 1, 3), **zero_kw),
            actor_mask=torch.zeros((batch_flat, 1), dtype=torch.bool, device=gru_outs[0].device),
        ))
        logits_flat[LOOK_HEAD] = lo.look_predict
        logits_flat["_look_prior"] = lo.look_prior
        logits_flat["_look_delta"] = lo.look_delta
        logits_flat["_look_mag_logits"] = lo.look_mag_logits
        logits_flat["_look_dir_logits"] = lo.look_dir_logits

        ao = self.attack_head(AttackHeadInput(features=feat(ATTACK_HEAD)))
        logits_flat[ATTACK_HEAD] = ao.attack_logit
        logits_flat["_attack_prior"] = ao.prior_logit
        logits_flat["_attack_delta"] = ao.delta_attack

        wo = self.weapon_head(WeaponHeadInput(selector=feat(WEAPON_HEAD), obs_weapon_id=None))
        logits_flat[WEAPON_HEAD] = wo.logits

        features_flat = gru_outs[0]
        values_flat = torch.zeros((batch_flat,), **zero_kw)
        target_logits = torch.zeros((batch_flat, 1), **zero_kw)   # pointer off

        if seq_shape is None:
            return features_flat, logits_flat, values_flat, next_hidden, target_logits
        features, logits, values, target_logits = _restore_outputs(
            features_flat, logits_flat, values_flat, target_logits, seq_shape,
        )
        return features, logits, values, next_hidden, target_logits


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(f"probe.json must define {key!r} for head=full_multitrunk")
    return probe[key]


def _build_full_multitrunk(probe: Mapping[str, Any]) -> HeadBuildResult:
    d_model    = int(_required(probe, "d_model"))
    n_heads    = int(_required(probe, "n_heads"))
    n_layers   = int(_required(probe, "n_layers"))
    d_ffn      = int(_required(probe, "d_ffn"))
    d_gru      = int(_required(probe, "d_gru"))       # per-trunk GRU width
    trunks     = _required(probe, "trunks")           # list[list[str]]
    if not isinstance(trunks, (list, tuple)) or not all(isinstance(g, (list, tuple)) for g in trunks):
        raise RuntimeError("probe.json 'trunks' must be a list of head-name lists")
    d_move     = int(probe.get("d_move", 64))
    d_look     = int(probe.get("d_look", 64))
    d_attack   = int(probe.get("d_attack", 64))
    d_weapon   = int(probe.get("d_weapon", 64))
    activation = str(probe.get("activation", "gelu"))
    attn_dropout = float(probe.get("attn_dropout", 0.0))
    n_cls = len(trunks)

    # Policy-layer flags only (model_factory bypasses canonical Network build).
    # d_gru = N packed trunk states so the policy sizes/threads hidden correctly.
    model_config = dataclasses.replace(
        neutral_model_config(d_model=d_model, self_weapon_embed_in_self=False),
        n_heads=n_heads, n_layers=n_layers, d_ffn=d_ffn, attn_dropout=attn_dropout,
        use_gru=True, d_gru=n_cls * d_gru, look_bypass_gru=False,
        use_weapon_head=True, weapon_sources=("gru",),
        d_move=d_move, d_look=d_look, d_attack=d_attack, d_weapon=d_weapon,
    )

    def factory(obs_dim: int, model_cfg: ModelConfig) -> nn.Module:
        return MultiTrunkNetwork(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ffn=d_ffn,
            d_gru=d_gru, trunks=trunks,
            d_move=d_move, d_look=d_look, d_attack=d_attack, d_weapon=d_weapon,
            activation=activation, attn_dropout=attn_dropout,
        )

    return model_config, factory


def _stub(*_a: Any, **_k: Any) -> torch.Tensor:
    return torch.zeros(())


FULL_MULTITRUNK = HeadSpec(
    name="full_multitrunk",
    loss=HeadLossSpec(
        loss_fn=_stub,
        metrics_fn=lambda *_a, **_k: {},
        label_key="look",
        output_dim=0,
        selection_metric="loss",
        selection_lower_is_better=True,
    ),
    build=_build_full_multitrunk,
)
