"""Attack-head ablation probe over 4 input bundles.

Goal: isolate which signals (beyond the GT target feature) the attack
head actually needs. All four variants share the same scaffold:

  ObsEmbedding(monolithic, include_spatial=False)
    → PreAttnEncoder (passthrough; no attention)
    → GTTargetPointer (oracle pool of entity tokens by labeler GT probs)
    → AttackBundleHead(bundle=…)

The head receives ``target_feat`` from the GT pointer plus an optional
bundle whose composition is selected by the ``bundle`` knob:

* ``none``         — target_feat only.
* ``engaged_ema``  — target_feat + engagement_ema-projection.
* ``weapon``       — target_feat + weapon token (damage/radius/
                     attack_finished + vocab weapon embed).
* ``motion``       — target_feat + motion token (vel/pitch/look_delta +
                     movement_embed + motion-powerup embed).

Look/move/weapon/temporal heads are Off — the only training signal is
attack BCE. The encoder is absent (PreAttnEncoder), so the only place
the attack signal can shape the entity tokens is via ObsEmbedding's
per-type projections and ID-embed tables, all of which receive
gradient through target_feat.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qnn.model.bench.attack import attack_bce_loss, attack_metrics
from qnn.model.bench.inputs.attack_bundle_head import AttackBundleHead
from qnn.model.bench.inputs.obs_network import BenchObsNetwork
from qnn.model.bench.inputs.gt_target_pointer import GTTargetPointer
from qnn.model.bench.inputs.preattn_encoder import PreAttnEncoder
from qnn.model.bench.spec import (
    HeadBuildResult,
    HeadLossSpec,
    HeadSpec,
    neutral_model_config,
)
from qnn.model.network import Off
from qnn.model.transformer import ObsEmbedding


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(
            f"probe.json must define {key!r} for head=attack_bundle "
            "(no Python-level defaults — see qnn.model.bench.templates)."
        )
    return probe[key]


def _build_attack_bundle(probe: Mapping[str, Any]) -> HeadBuildResult:
    """Required keys in probe.json:

      d_model  (int)         — token width.
      d_attack (int)         — attack-head MLP hidden width.
      bundles  (list[str])   — subset of {"engaged_ema", "weapon", "motion"};
                               empty list = target_feat only.

    Backward compatibility: a single-string ``bundle`` key (legacy
    format from the first wave of attack_bundle runs) is accepted and
    folded into ``bundles = [bundle]``.
    """
    d_model = int(_required(probe, "d_model"))
    d_attack = int(_required(probe, "d_attack"))
    if "bundles" in probe:
        bundles: Any = probe["bundles"]
    elif "bundle" in probe:
        bundles = probe["bundle"]
    else:
        raise RuntimeError(
            "probe.json must define either 'bundles' (list) or 'bundle' (str) "
            "for head=attack_bundle."
        )

    model_config = neutral_model_config(
        d_model=d_model, self_weapon_embed_in_self=False,
    )

    def factory(obs_dim: int, model_cfg) -> BenchObsNetwork:
        # Build the encoder first so the attack head can tie its
        # entity_embed to the encoder's. Same vocab, same params — held-
        # weapon rocket and entity-token rocket are the same concept.
        obs_embedding = ObsEmbedding(
            d_model=d_model,
            self_weapon_embed_in_self=False,
            include_spatial=False,
        )
        return BenchObsNetwork(
            obs_dim=obs_dim,
            model=model_cfg,
            obs_embedding=obs_embedding,
            encoder=PreAttnEncoder(),
            target_pointer=GTTargetPointer(d_model=d_model),
            temporal=Off,
            move_head=Off,
            look_head=Off,
            weapon_head=Off,
            attack_head=AttackBundleHead(
                d_model=d_model, d_attack=d_attack, bundles=bundles,
                entity_embed=obs_embedding.entity_embed,
                movement_embed=obs_embedding.movement_embed,
            ),
        )

    return model_config, factory


ATTACK_BUNDLE = HeadSpec(
    name="attack_bundle",
    loss=HeadLossSpec(
        loss_fn=attack_bce_loss,
        metrics_fn=attack_metrics,
        label_key="attack",
        output_dim=1,
        selection_metric="attack_skill",
        selection_lower_is_better=False,
    ),
    build=_build_attack_bundle,
)
