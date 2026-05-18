"""Move-axis labeler — bidirectional inverse dynamics for QWD/MVD.

  - model.py     architecture + features
  - collect.py   labeler-only native-rate collect (CLI: python -m qnn.labeler.collect)
  - train.py     training loop                    (CLI: python -m qnn.labeler.train)

Apply (relabel a forced-MVD collect) is intentionally deferred. It needs
the native-rate forced-MVD collect path and tightened C-rule fire/jump.

See:
  - project_seq_labeler_axes        why the labeler isn't a mini-policy
  - project_qwd_rate_distribution   corpus design (bc_included + trick at >=70 Hz)
  - scripts/move_inference_bakeoff.md   measured baselines we beat
"""

from .model import (
    CORE_FEAT_DIM,
    FeatureSpec,
    MoveLabeler,
    PREDICTED_AXES,
    VELOCITY_SCALE,
    build_features,
    decode_move_fb_lr,
)

__all__ = [
    "CORE_FEAT_DIM",
    "FeatureSpec",
    "MoveLabeler",
    "PREDICTED_AXES",
    "VELOCITY_SCALE",
    "build_features",
    "decode_move_fb_lr",
]
