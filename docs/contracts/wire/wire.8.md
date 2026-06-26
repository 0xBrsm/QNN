# Wire contract `wire.8` — native split, 43 inputs (notional/reconstructed)

The native-split wire at release 0.21.0: `wire.9` **minus** `look_delta`
(43 obs inputs + `hidden`). **Band A**, but it has **no surviving artifact** —
the native ONNX exporter (`NATIVE_INPUTS`) postdates the `look_delta` commit, so
**no 43-input graph was ever exported**. `wire.8` describes the engine/Python
wire that existed at 0.21.0, reconstructed; it is documented for lineage
continuity, not built.

- **Definition:** [`wire.9`](wire.9.md) without input #13 `look_delta`. Every
  other input and all outputs are byte-identical to `wire.9`.
- **Semantics:** [`semantics.1`](../semantics/semantics.1.md). **Arch:** native split (v23-era).
- **Codec:** not built (no model to run). If a 43-input model ever surfaces,
  promote this to a full doc and add the codec.
- **Provenance:** engine wire + `engine_norm.py` at tag `0.21.0`
  (`SELF_BLOCK_BYTES = 21`, no `look_delta`); the native exporter arrived later
  with `look_delta` already present.
