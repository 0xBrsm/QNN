# a24

The a24-generation overrides of the promoted `qnn.model` base. Each head module
mirrors the cross-generation-stable base module it overrides and self-registers
into the model-graph node registry (`qnn.model.node_registry`); the shared decode
modules implement the a24 decode contract. A winning override is promoted up
into `qnn.model`.

The a24 model is assembled declaratively from the committed base graphs
(`qnn.model.graph.bases/full_{4,5}head.json`) plus per-run `probe.json`
overrides — there are no hand-wired `full_*.py` assembly classes (retired; the
graph is the single build path).

## Layout

| module | overrides | graph type |
|---|---|---|
| `move_head.py` (`CLSMoveHead`) | `qnn.model.move_head` | move `"cls"` |
| `attack_head.py` (`CLSAttackHead`) | `qnn.model.attack_head` | attack `"cls"` |
| `weapon_head.py` (`CLSWeaponHead`) | `qnn.model.weapon_head` | weapon `"cls"` |
| `look_head.py` (`PurePolarLookHead`) | `qnn.model.look_head` | look `"polar"` |
| `decode.py` | `qnn.model.decode` (pure readout) | — (shared decode) |
| `rc1.py` | — (release layer on `decode.py`) | — |
| `lead_aim.py` | — (a24-only aim primitive) | — |

The rc1/2/3 lineage (`full_5head` + weapon `gru`/`target.feat`) lives in
`runs/bc/bench/head_probe_full_5head_weapon_target_*` as self-describing graph
checkpoints. Concluded ablations are archived under `runs/_archive/bc/a24/`.

`decode.py` owns the a24 decode (move hazard machine, polar look, sticky weapon,
aim-prior) shared by Python eval and ONNX export. `rc1.py` is the release-
candidate decode layer on top of it (the attack splash guard). Generic
policy/export code calls into these rather than duplicating a24 rules.

Release-candidate variant labels include the rc prefix, e.g. `a24rc1b`,
`a24rc1h`, rather than plain `a24b`, once they are part of this project.
