# Arch axis — model internals / weight layout

The third contract axis: the model's **internal architecture** — subtoken
splits, head structure, encoder/GRU shapes, embedding tables. Distinct from wire
and semantics:

- It is **checkpoint-side only.** The **live bin ignores it** — an exported
  `.onnx` is self-contained (graph + weights), so arch never affects whether a
  model *loads* in the bin. It only matters when loading an old **checkpoint**
  (`.pth`) into current model code for retraining/eval.
- Its source of truth and migration engine is
  [`qnn/utils/checkpoint_converter.py`](../../../qnn/utils/checkpoint_converter.py),
  which recognizes and migrates prior checkpoint schemas. That converter — not
  this doc — is authoritative for which arch generations can be loaded.

Because arch is checkpoint-side, it is **not stamped as a selection key** in the
exported model; it is stamped only as **provenance** (`arch` metadata key, e.g.
`full_4head/gru64`).

## Generations (descriptive)
A given wire contract can be produced by successive arch generations without a
wire bump (e.g. the v14 self→state/arsenal/motion subtoken split changed the
arch but not `wire.9`). Rough lineage, newest first:

| Arch gen | Era | Notes | Wire |
|----------|-----|-------|------|
| `full_movearch` | A27 HEAD | Split self/combat tokens → CLS encoder → GRU → {move_seg, jump, look(polar), attack(9-way)} + TargetPointer | `wire.13` |
| `full_4head` | a24 | HeldWeaponSplitObsEmbedding → CLS encoder → GRU → {move, look(polar), attack, weapon} + TargetPointer | `wire.11` |
| native-split / subtoken | 0.20–0.21 | self → state/arsenal/motion subtokens + CLS; fire→attack; `qnn_action_t` 32→16 B | `wire.8`/`.9` |
| v11 packed | 0.17 | 8-class weapon, move 3×3 categorical, TargetPointer; packed wire | `wire.7` |
| v10 / v9 / v8 / cluster / flat | ≤0.15 | see [`wire/pre-v11.md`](../wire/pre-v11.md) + archived bundled docs | `wire.1`–`.6` |

This table is a map, not the authority — when adding an arch generation, extend
the `checkpoint_converter` migration chain and reference it here. Formal
`arch.N` numbering can be introduced if/when the converter's schema versions need
to be cited independently of the descriptive names.
