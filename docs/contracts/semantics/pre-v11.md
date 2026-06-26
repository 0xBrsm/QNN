# Semantics contracts `semantics.0a`–`semantics.0d` — pre-v11

The semantics epochs before [`semantics.1`](semantics.1.md). Documented for
lineage; **none are built** (they pair only with the unbuilt pre-v11 wire
contracts in [`wire/pre-v11.md`](../wire/pre-v11.md)). The last semantics break
is the v11 entity-vocab 42→44 + weapon impulse renumber; from there to HEAD the
meaning is stable (`semantics.1`).

| Semantics | Wire span | What defines it |
|-----------|-----------|-----------------|
| `semantics.0a` | `wire.1`–`.2` (flat, 0.1–0.3) | World-frame scales (origin /2048 or /4096, vel /320), identity vocab — an entirely different normalization basis |
| `semantics.0b` | `wire.3` (v5–v7, 0.4–~0.9) | health /250, armor /200, cluster-feature scales; 5-class weapon vocab; canonical DIST scale not yet adopted |
| `semantics.0c` | `wire.4` (v8, 0.10) | Canonical DIST/VEL/TIME scales appear (1000/2000/60); health renormalized /100, effective_armor; subject vocab armor-before-health renumber |
| `semantics.0d` | `wire.5`–`.6` (v9–v10, 0.11–0.15) | `entity_embed=42`, pre-impulse weapon order (AXE=3, SG=4, NG=5, …); modality vocab 5→4 |

**Note on reconstruction:** even where a pre-v11 *wire* might be approximated by
zero-filling dead fields, its semantics are **not** automatically recoverable —
`semantics.0a–0d` use different scales/vocab than `semantics.1`, and the flat era
(`0a`) applied some scaling C-side. Any pre-v11 codec would need its own
semantics id wired in, not a reuse of `semantics.1`. Full per-epoch detail is in
the archived bundled docs (`docs/archive/token-spec-v*`, `obs-spec-v*`,
`vocab-v*`).
