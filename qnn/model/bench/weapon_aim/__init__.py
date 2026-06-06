"""Weapon-aim ablation: joint look + attack heads fed by (aim_vec, target_feat).

Look head's aim and attack head's fire decision share a single
weapon-aware geometric prior: ``aim_vec``, the soft-pooled
lead-corrected hit-zone direction for the held weapon. Both heads
also see ``target_feat`` (canonical soft-pooled entity feature). The
attack head additionally consumes ``noop`` (binary fire-feasibility
gate, currently derived from obs; engine-side ``input_mask`` source is
a follow-up).

Two probe variants in probe.json (``variant`` field):

  ``canonical``   — canonical LookHead + canonical AttackHead. Baseline.
  ``weapon_aim``  — WeaponAimLookHead + canonical AttackHead. The look
                    head uses aim_vec as its geometric prior; attack is
                    intentionally out of scope for this pass.

Hypothesis: if aim_vec is the right inductive bias, the weapon_aim
variant should converge to lower look loss and higher attack F1 with
less MLP capacity, because lead time / ballistic / hitscan corrections
are baked into the prior instead of being relearned by each head.
"""

from __future__ import annotations

from qnn.model.bench.weapon_aim.spec import WEAPON_AIM

__all__ = ["WEAPON_AIM"]
