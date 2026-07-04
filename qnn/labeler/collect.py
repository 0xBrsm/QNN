"""The force_mvd LOBS training collect (single-purpose).

Collects the LOBS move-labeler corpus = **MVD-domain features + usercmd-TRUTH
move**, at native rate, with the slim field selection.  This is the one and
only thing this tool does — there are no behavior toggles.  It runs through
the unified BC collect (:func:`qnn.bc.collect.run_collect`, the single real
pipeline) with the agnostic field selection (refactor 9626c5de):

  * ``force_mvd_emit`` is always ON → the obs features are MVD-domain
    (origin-delta velocity, scrubbed playerstate), matching the apply-time
    distribution the labeler will be applied against; but
  * the ``move`` action stays **usercmd-TRUTH** via the internal
    ``usercmd_move`` op field (QWD demos still carry usercmd_t, so the
    recorded press byte is recoverable even under force_mvd).  This decouples
    "MVD features" from "MVD-inferred move": the worker takes ``move`` from
    the QWD usercmd decoder instead of ``QNN_MvdInferEmitMove`` while the obs
    distribution is MVD-ified.  (Mirrors the matched-emit MLOB co-emission.)

  obs  keep = {vel, self_movement_id, self_weapon_id}
  act  keep = {move, look, op_input}
  spatial + entity blocks: COMPUTE-GATED OFF (skip_spatial / skip_entities
                           derived from the obs keep set — the worker skips
                           the spatial raycast + entity oracle/pathfinding
                           entirely, the real cost win of the old slim path)

On-disk field names (QOBS-native, via the BC ShardWriter):
  obs/vel               i16  (T, 3)   view-frame velocity (MVD-domain
                                      origin-delta), raw Quake units
  obs/self_movement_id  u8   (T,)     0=ground 1=air 2..4=water
  obs/self_weapon_id    u8   (T,)     subject-form weapon id (obs boundary)
  act/move              u8   (T,)     usercmd-truth press byte
                                      (QNN_PackInputMask layout):
                                      bit0=attack, bits1-6 = fb/lr/ud neg/pos,
                                      bit7 = jump.  The move-labeler decodes
                                      its fb/lr/ud 3-class targets from this.
  act/look              f16  (T, 3)   per-emit view delta (fwd·anchor basis)
  act/op_input          u8   (T,)     strict per-axis operativeness mask
                                      (bit0=fb bit1=lr bit2=ud bit3=fire
                                      bit4=impulse) — 1 = press AND engine
                                      acted on it this tick.

Demo-level filtering goes through ``--filter-config`` (same JSON schema as
qnn.bc.collect: nested ``demos`` / ``segments`` / ``tokens`` / ``actions``
axes with ``keep`` / ``drop`` sub-keys).  Pointing it at
``artifacts/collect/qwd/filter.json`` selects the same demo set the BC
corpus uses.  ``segments.drop`` intervals (signon / dead / intermission)
carve sub-episodes out of each demo — same semantics as bc.collect.

ALL fps are included — there is no native-rate gate.  The apply target
(real MVD demos) is itself mixed-fps, so training across the full rate
range matches the deployment distribution.

Usage:
    PYTHONPATH=src python -m qnn.labeler.collect \\
        --demo-dir       artifacts/corpus/qwd \\
        --manifest       artifacts/corpus/qwd_manifest.ndjson \\
        --filter-config  artifacts/collect/qwd/filter.json \\
        --output         artifacts/collect/qwd_labeler \\
        --workers        30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qnn.bc.collect import (
    _collect_one_demo,
    _default_demo_worker,
    _game_dir_for_demo_dir,
    _load_and_pin_filter,
    _run_per_demo_collect,
    _unpack_episode,
    run_collect,
)


# Field selection the move labeler consumes.  Self subset only (the
# expensive spatial/entity blocks are compute-gated OFF because none of
# their fields appear here — see _compute_gate_from_tokens in
# qnn.bc.collect).
_LABELER_TOKENS_KEEP = ("vel", "self_movement_id", "self_weapon_id")
_LABELER_ACTIONS_KEEP = ("move", "look", "op_input")


# ── per-demo collect: force_mvd features + usercmd-truth move ──────────────────

def _collect_demo_labeler(args: tuple) -> dict:
    """Pool entry: force_mvd LOBS collect for one demo.

    Same work-tuple shape qnn.bc.collect._collect_demo unpacks, but the
    worker op carries both ``force_mvd_emit=1`` (MVD-domain features) AND the
    internal ``usercmd_move=1`` (move from the QWD usercmd decoder, not MVD
    physics).  build_work_args sets force_mvd_emit in the tuple; usercmd_move
    is fixed here — it is NOT a user-facing flag."""
    (demo_name, force_mvd_emit, combat_only, labels,
     total_frames, drop_label_names, sight_only, target_probs_cache) = args
    return _run_per_demo_collect(
        demo_name,
        collect_fn=lambda proc: _collect_one_demo(
            proc, demo_name, force_mvd_emit=force_mvd_emit,
            usercmd_move=True),
        unpack_fn=lambda ticks: _unpack_episode(
            ticks, combat_only=combat_only, labels=labels,
            drop_label_names=drop_label_names, total_frames=total_frames,
            sight_only=sight_only,
            target_probs_cache=target_probs_cache),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo-dir",   type=Path, required=True)
    ap.add_argument("--manifest",   type=Path, default=None,
                    help="Corpus manifest .ndjson (default: <demo-dir>/../<demo-dir.name>_manifest.ndjson)")
    ap.add_argument("--output",     type=Path, required=True)
    ap.add_argument("--demo-worker", type=Path, default=None,
                    help="qw_demo_worker binary (default: assets/bin/qw_demo_worker)")
    ap.add_argument("--workers",    type=int, default=30)
    ap.add_argument("--filter-config", type=Path, default=None,
                    help="Path to a JSON filter config (same schema as "
                         "qnn.bc.collect --filter-config).  demos.keep / "
                         "demos.drop predicates gate inclusion; "
                         "segments.drop carves sub-episodes out of each "
                         "kept demo.  Point at artifacts/collect/qwd/"
                         "filter.json to mirror the BC corpus.")
    args = ap.parse_args()

    # Operational args only; everything else is fixed by this tool's purpose.
    asset_root = Path("assets").resolve()
    demo_dir = args.demo_dir.resolve()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    # Pin the filter to <output>/filter.json so the spec used to produce
    # this cache always travels with it.  Only the demos/segments axes are
    # honoured here; the tokens/actions selection is fixed by the labeler
    # (the slim subset above), not by the filter-config.
    filter_spec = _load_and_pin_filter(
        output, Path(args.filter_config) if args.filter_config else None)
    demos_block    = filter_spec.get("demos") or {}
    segments_block = filter_spec.get("segments") or {}
    keep_pred = demos_block.get("keep") or {}
    drop_pred = demos_block.get("drop") or {}
    drop_label_names = tuple(segments_block.get("drop") or ())

    manifest_path = (args.manifest if args.manifest
                     else demo_dir.parent / f"{demo_dir.name}_manifest.ndjson")
    if not manifest_path.exists():
        sys.exit(f"manifest not found: {manifest_path}")

    demo_worker = args.demo_worker or Path(_default_demo_worker("qwd"))
    if not Path(demo_worker).is_absolute():
        demo_worker = Path.cwd() / demo_worker
    if not demo_worker.exists():
        sys.exit(f"demo worker not found: {demo_worker}")
    game_dir = _game_dir_for_demo_dir(demo_dir, asset_root)

    # ALL fps included — no native-rate gate (the real-MVD apply target is
    # itself mixed-fps, so training across the full rate range matches it).
    # Per-demo work tuple matches the shape qnn.bc.collect._collect_demo
    # unpacks: (demo_name, force_mvd_emit, combat_only, labels,
    # total_frames, drop_label_names, sight_only, target_probs_cache).
    # force_mvd_emit is ON (MVD-domain features); the usercmd-truth move is
    # supplied by the internal usercmd_move op field set in
    # _collect_demo_labeler.  No combat gate, and NO target_probs labeler
    # (the entity oracle is gated off — the labeler view can't be densified
    # and isn't needed).
    def build_work_args(entry, labels, total_frames, drop_labels):
        return (entry["file"], True,
                False, labels, total_frames, drop_labels,
                False, False)

    run_collect(
        output=output,
        demo_dir=demo_dir,
        manifest_path=manifest_path,
        asset_root=asset_root,
        demo_worker=str(demo_worker),
        game_dir=game_dir,
        # tick_hz=0 → engine resolves to detected native rate per demo
        # (see qnn_collect_main.c).
        tick_hz=0,
        workers=args.workers,
        # At native ~77Hz, 200k rows ~= 43 minutes of demo time.
        shard_rows=200_000,
        train_ratio=0.9,
        seed=17,
        keep_pred=keep_pred,
        drop_pred=drop_pred,
        drop_label_names=drop_label_names,
        per_demo_fn=_collect_demo_labeler,
        build_work_args=build_work_args,
        shard_kind="labeler",
        extra_metadata={
            "format": "labeler_v11",
            "force_mvd_emit": True,
            "usercmd_move": True,
            "all_fps": True,
            "segments_drop": list(drop_label_names),
        },
        filter_path=output / "filter.json",
        # Field selection: drives both the Python projection AND the C
        # compute-gate (spatial/entity skipped because no field of either
        # block is in tokens_keep).
        tokens_keep=_LABELER_TOKENS_KEEP,
        actions_keep=_LABELER_ACTIONS_KEEP,
    )


if __name__ == "__main__":
    main()
