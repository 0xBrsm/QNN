#!/usr/bin/env python3
"""Canonical attack + jump label validator for a BC collect.

Compares the collected NPY attack / jump labels against each demo's OWN
``svc_sound`` byte truth (``qnn.bc.demo_truth``) and FAILS if the aggregate
deviation drifts outside an expected band (the established ~-6.7% normal —
EXPECTED_DEV_BAND). This is the single trustworthy
"are the attack/jump labels poisoned?" check that runs after every collect —
subsuming the old loose ``scripts/audit_fire*.py`` one-offs.

PATH SEMANTICS (see src/docs/mvd-attack-audit.md §Round 4 / §Round 5)
------------------------------------------------------------------
The collect's emit path determines which demo reference to compare against,
read deterministically from ``collect_metadata.json`` ``force_mvd_emit``:

  * QWD path (force_mvd_emit = false, the production default) — op-fire counts
    W_Attack TRIGGER PULLS (~5/sec for continuous weapons), so compare the
    collected attack events to the demo's TRIGGER events (continuous-weapon
    sound streams collapsed at MERGE_GAP).

  * force-MVD path (force_mvd_emit = true) — one collected event per fire
    SOUND (projectiles, ~10/sec for continuous weapons), so compare against
    the demo's raw SOUND count.

GROUND TRUTH (per demo, self / view entity)
  attack: self weapon-fire sounds classified by the fire vocab, collapsed
          into triggers for the QWD path.
  jump:   self ``player/plyrjmp8.wav`` sounds (one per jump). NOTE: the jump
          sound rides an unreliable PHS datagram, so the collected count runs
          a few % under — folded into the same expected band.

COLLECTED SIDE (shard NPYs + manifest, no env dumps)
  attack EVENTS = rising edges of ``act_move & 0x01``.
  jump   EVENTS = rising edges of ``act_move & 0x80`` (bit 7).
  per-weapon attack via the stored categorical ``act_attack`` label (1..8).

Usage:
  PYTHONPATH=src python -m qnn.bc.validate_labels <collect-dir> \\
      --demo-dir artifacts/corpus/qwd \\
      --manifest artifacts/corpus/qwd_manifest.ndjson
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from qnn.bc.demo_truth import MERGE_GAP, demo_attack_jump_truth
from qnn.bc.attack_vocab import WNAME


def _demo_ref_worker(args):
    """Pool worker: byte-parse one demo for its attack/jump reference. Top-level
    so it is picklable. The parse is the validator's hot path, so this is mapped
    across a process pool."""
    di, fname, path, collapse_gap, total_frames, drop_intervals = args
    attack_per_w, jumps, _ve = demo_attack_jump_truth(
        path, collapse_gap, drop_intervals=drop_intervals, total_frames=total_frames)
    return di, fname, attack_per_w, jumps


# The demo byte-truth reference is a pure, COLLECT-INDEPENDENT function of the
# demo bytes, so it is cached per-demo (keyed on the manifest sha256) and reused
# across every validation of every collect on the same corpus — only changed
# demos are re-parsed. Bump _REF_CACHE_VERSION whenever the parse / vocab /
# trigger-collapse logic changes, to invalidate stale entries.
# v2: reference excluded dropped-segment sounds (REVERTED — see below).
# v3: back to WHOLE-demo reference (the v2 segment filter's approximate
#     time->frame map falsely dropped ~5% of kept attacks; QWD ground truth
#     caught it at +5% vs -0.07% whole-demo).
# v4: LG reference reconstructed from the lstart+lhit heartbeat
#     (demo_truth.lg_op_attack_count) instead of trigger_events(lstart+lhit).
_REF_CACHE_VERSION = 4


def _ref_cache_path(manifest_path: Path) -> Path:
    return manifest_path.parent / f".{manifest_path.stem}_attackref_cache.json"


def _load_ref_cache(manifest_path: Path, collapse_gap: float) -> dict:
    p = _ref_cache_path(manifest_path)
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except (ValueError, OSError):
        return {}
    if (d.get("version") != _REF_CACHE_VERSION
            or abs(float(d.get("collapse_gap", -1.0)) - collapse_gap) > 1e-9):
        return {}  # logic or cadence changed -> ignore stale cache
    return d.get("demos", {})  # sha256 -> {"attack": {clsStr: count}, "jump": n}


def _save_ref_cache(manifest_path: Path, collapse_gap: float, demos: dict) -> None:
    p = _ref_cache_path(manifest_path)
    tmp = p.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps({"version": _REF_CACHE_VERSION,
                                   "collapse_gap": collapse_gap, "demos": demos}))
        tmp.replace(p)  # atomic
    except OSError:
        pass  # cache is an optimization; never fail the validation over it


def _load_manifest_sha(manifest_path: Path) -> dict:
    """demo_idx (manifest position) -> sha256, for cache keying."""
    out: dict[int, str] = {}
    for idx, line in enumerate(manifest_path.read_text().splitlines()):
        if not line.strip():
            continue
        sha = json.loads(line).get("sha256")
        if sha:
            out[idx] = sha
    return out


# Gate: BOTH paths score the OPERATIVE attack/jump (act_move&1 & input_mask&1 —
# no-op held frames excluded, the only inputs the model trains on) against the
# WHOLE-demo trigger-pull reference (sounds collapsed at MERGE_GAP, LG from the
# lstart+lhit heartbeat; see qnn.bc.demo_truth).
#
# The operative attack and jump counts both run a STABLE few-% UNDER their
# demo-sound references — an investigated-but-unexplained gap (resample
# rate, weapon attribution, and feasibility false-negatives all ruled out; see
# the NOTE below). Since that under-count is the established normal, the gate is
# a BAND around it, NOT a check against zero: it passes the known under-count and
# flags any DRIFT out of the band — either a regression that widens the gap or a
# change that closes it. This turns the validator into a label-regression
# detector. Override per run on the CLI with --attack-band / --jump-band.
#
# Recalibrated 2026-07-02: the original band was (-0.07, -0.06) around the
# ~-6.7% normal established on the pre-op_input worker. Worker fixes since
# (usercmd-truth op_input b299269c, hull-derived onground/waterlevel, weapon
# min-dwell 2f119172) legitimately closed part of the gap — measured on the
# current worker: QWD eval subset attack -3.4%, MVD forced jump -4.0%,
# full-corpus MVD jump -4.8%, MVD attack (post LG fill) -6.x%. The band now
# admits that spread while still catching real poisoning (the pre-LG-fill
# MVD attack read -12.6% and must keep failing).
EXPECTED_DEV_BAND = (-0.08, -0.02)   # (low, high) signed col-vs-ref deviation


@dataclass
class DemoReport:
    demo_idx: int
    demo_file: str
    # reference (from demo bytes)
    ref_attack: int = 0            # triggers (QWD) or sounds (MVD)
    ref_jump: int = 0
    ref_attack_per_w: dict[int, int] = field(default_factory=dict)
    # collected (from NPYs)
    col_attack: int = 0
    col_jump: int = 0
    col_attack_per_w: dict[int, int] = field(default_factory=dict)


@dataclass
class Aggregate:
    ref_attack: int = 0
    ref_jump: int = 0
    col_attack: int = 0
    col_jump: int = 0
    ref_attack_per_w: dict[int, int] = field(default_factory=lambda: {c: 0 for c in WNAME})
    col_attack_per_w: dict[int, int] = field(default_factory=lambda: {c: 0 for c in WNAME})


def _rising_edges(bits: np.ndarray) -> int:
    """Count 0->1 rising edges in a per-frame boolean stream (one per episode
    slice — the caller passes a single episode so no edge leaks across the
    boundary)."""
    if bits.size == 0:
        return 0
    b = bits.astype(np.int8)
    d = np.diff(np.concatenate([[0], b, [0]]))
    return int((d == 1).sum())


def _detect_force_mvd_emit(collect_dir: Path) -> bool:
    """Read the emit path from collect_metadata.json. Fails loud if the flag
    is missing so the validator never silently picks the wrong reference."""
    meta_path = collect_dir / "collect_metadata.json"
    if not meta_path.exists():
        raise SystemExit(
            f"{meta_path} not found — cannot determine the emit path "
            f"(QWD vs force-MVD) to pick the demo reference. Re-collect with "
            f"a current qnn.bc.collect (it records 'force_mvd_emit')."
        )
    meta = json.loads(meta_path.read_text())
    if "force_mvd_emit" not in meta:
        raise SystemExit(
            f"{meta_path} has no 'force_mvd_emit' field — this collect predates "
            f"emit-path recording. Re-collect so the validator can pick the "
            f"right demo reference (QWD triggers vs MVD sounds) deterministically."
        )
    return bool(meta["force_mvd_emit"])


def _iter_collected(split_dir: Path, force_mvd_emit: bool):
    """Yield (demo_idx, attack_events, jump_events, attack_per_weapon) per
    episode across all shards in a split. attack_per_weapon is a Counter-like
    dict keyed by attack class 1..8 at each effective attack.

    BOTH paths score the OPERATIVE attack — ``(act_move & 0x01) & (act_input_mask
    & 0x01)`` — i.e. only the un-masked inputs the model actually trains on. The
    input_mask zeroes no-op frames (a trigger held during cooldown, auto-fire
    continuations between shots); we do NOT count those (see the
    noop-holds-are-masked convention). Scoring the RAW ``act_move & 0x01`` instead
    counts those masked no-op frames and manufactures spurious deviation,
    especially on the force-MVD path. ``force_mvd_emit`` no longer changes the
    collected count — both paths are operative — but is kept for call-site
    stability."""
    _ = force_mvd_emit
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    for sh in manifest.get("shards", []):
        move_name = sh["actions"]["move"]
        base = move_name[: -len("_act_move.npy")]
        mv = np.load(split_dir / f"{base}_act_move.npy")
        attack_labels = np.load(split_dir / f"{base}_act_attack.npy")
        im = np.load(split_dir / f"{base}_act_input_mask.npy")
        off = 0
        for ln, di in zip(sh["episode_lengths"], sh["demo_idxs"]):
            ep_mv = mv[off:off + ln]
            ep_attack_labels = attack_labels[off:off + ln]
            # OPERATIVE only (un-masked); no-op held frames excluded. Attack =
            # input_mask bit 0, jump = input_mask bit 7 (same bit layout as move).
            ep_im = im[off:off + ln]
            attack = ((ep_mv & 0x01) & (ep_im & 0x01)).astype(np.int8)
            jump = (((ep_mv >> 7) & 0x01) & ((ep_im >> 7) & 0x01)).astype(np.int8)
            # Effective attack events and their directly encoded impulse class.
            edges = np.flatnonzero(ep_attack_labels > 0)
            per_w: dict[int, int] = {}
            for s in edges:
                cls = int(ep_attack_labels[s])
                if 1 <= cls <= 8:
                    per_w[cls] = per_w.get(cls, 0) + 1
            yield int(di), len(edges), _rising_edges(jump), per_w
            off += ln


def _load_manifest_files(manifest_path: Path) -> dict[int, str]:
    """demo_idx (manifest position) -> file name."""
    out: dict[int, str] = {}
    for idx, line in enumerate(manifest_path.read_text().splitlines()):
        if not line.strip():
            continue
        out[idx] = json.loads(line)["file"]
    return out


def validate_labels(
    collect_dir: Path,
    *,
    demo_dir: Path,
    manifest_path: Path,
    attack_band: tuple[float, float] | None = None,
    jump_band: tuple[float, float] | None = None,
    force_mvd_emit: bool | None = None,
    warn_only: bool = False,
    verbose: bool = True,
    workers: int | None = None,
    use_cache: bool = True,
) -> int:
    """Validate attack + jump labels for one collect against demo byte truth.

    Returns 0 on pass, 1 on failure (unless ``warn_only``). When
    ``force_mvd_emit`` is None it is detected from collect_metadata.json.
    ``attack_band`` / ``jump_band`` are the (low, high) signed-deviation bands
    the collected-vs-reference deviation must fall within (default
    ``EXPECTED_DEV_BAND`` — the established ~-6.7% normal). The reference is
    demo sounds collapsed into trigger pulls at MERGE_GAP."""
    if force_mvd_emit is None:
        force_mvd_emit = _detect_force_mvd_emit(collect_dir)
    if attack_band is None:
        attack_band = EXPECTED_DEV_BAND
    if jump_band is None:
        jump_band = EXPECTED_DEV_BAND
    # Both paths carry a trigger-pull label, so collapse demo sounds at
    # MERGE_GAP (the ~0.1s think-chain cadence) for an apples-to-apples
    # reference (see the gate-default note above).
    collapse_gap = MERGE_GAP
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    idx2file = _load_manifest_files(manifest_path)

    # Sum collected events per demo_idx across train + val splits.
    col_attack: dict[int, int] = {}
    col_jump: dict[int, int] = {}
    col_per_w: dict[int, dict[int, int]] = {}
    present_idxs: set[int] = set()
    for split in ("precomputed_train", "precomputed_val"):
        for di, atk, jmp, per_w in _iter_collected(collect_dir / split, force_mvd_emit):
            present_idxs.add(di)
            col_attack[di] = col_attack.get(di, 0) + atk
            col_jump[di] = col_jump.get(di, 0) + jmp
            d = col_per_w.setdefault(di, {})
            for c, v in per_w.items():
                d[c] = d.get(c, 0) + v

    if not present_idxs:
        raise SystemExit(
            f"{collect_dir}: no collected episodes found in "
            f"precomputed_train / precomputed_val — nothing to validate."
        )

    ref_kind = ("emit-collapsed SND (force-MVD)" if force_mvd_emit
                else "TRIG (QWD)")

    # Resolve + check every demo path up front (fail loud), then build the
    # reference by byte-parsing each demo. The parse is the hot path, so map it
    # across a process pool — one independent parse per demo.
    work = []
    for di in sorted(present_idxs):
        fname = idx2file.get(di)
        if fname is None:
            raise SystemExit(
                f"{collect_dir}: collected demo_idx {di} not in manifest "
                f"{manifest_path} (idx out of range)."
            )
        demo_path = demo_dir / fname
        if not demo_path.exists():
            raise SystemExit(f"demo file missing: {demo_path}")
        # WHOLE-demo reference (no segment filtering): pass (0, []). The earlier
        # segment-drop filter used an approximate time->frame map that falsely
        # dropped ~5% of KEPT attacks — caught because QWD (ground truth) read
        # +5% against it but -0.07% against whole-demo. Dropped segments hold ~no
        # attacks anyway; jump warmup-drops want an EXACT kept-span map, deferred.
        work.append((di, fname, str(demo_path), collapse_gap, 0, []))

    # Split into cache hits vs demos that must be re-parsed (changed/new sha256).
    idx2sha = _load_manifest_sha(manifest_path) if use_cache else {}
    cache = _load_ref_cache(manifest_path, collapse_gap) if use_cache else {}
    cached_truths = []
    to_parse = []  # (di, fname, path, gap, total_frames, drop_intervals, sha)
    for (di, fname, path, gap, tf, drops) in work:
        sha = idx2sha.get(di)
        ent = cache.get(sha) if sha else None
        if ent is not None:
            attack = {int(k): v for k, v in ent["attack"].items()}
            cached_truths.append((di, fname, attack, int(ent["jump"])))
        else:
            to_parse.append((di, fname, path, gap, tf, drops, sha))

    if workers is None:
        # The per-demo byte-parse is CPU-bound (~1s on large MVD demos), so use
        # all but two cores. Cache hits skip the parse entirely.
        workers = max(1, (os.cpu_count() or 2) - 2)
    parse_args = [w[:6] for w in to_parse]  # drop the trailing sha
    if workers > 1 and len(parse_args) > 1:
        with Pool(workers) as pool:
            parsed = pool.map(_demo_ref_worker, parse_args, chunksize=8)
    else:
        parsed = [_demo_ref_worker(w) for w in parse_args]

    if use_cache and to_parse:
        for (di, fn, attack, jumps), tp in zip(parsed, to_parse):
            sha = tp[-1]
            if sha:
                cache[sha] = {"attack": {str(k): v for k, v in attack.items()},
                              "jump": jumps}
        _save_ref_cache(manifest_path, collapse_gap, cache)

    truths = cached_truths + list(parsed)
    if verbose:
        print(f"reference: {len(cached_truths)} cached, {len(parse_args)} parsed "
              f"({workers} workers)")

    reports: list[DemoReport] = []
    agg = Aggregate()
    for di, fname, ref_per_w, jumps in truths:
        rep = DemoReport(
            demo_idx=di, demo_file=fname,
            ref_attack=sum(ref_per_w.values()),
            ref_jump=jumps,
            ref_attack_per_w=dict(ref_per_w),
            col_attack=col_attack.get(di, 0),
            col_jump=col_jump.get(di, 0),
            col_attack_per_w=col_per_w.get(di, {}),
        )
        reports.append(rep)
        agg.ref_attack += rep.ref_attack
        agg.ref_jump += rep.ref_jump
        agg.col_attack += rep.col_attack
        agg.col_jump += rep.col_jump
        for c in WNAME:
            agg.ref_attack_per_w[c] += ref_per_w.get(c, 0)
            agg.col_attack_per_w[c] += rep.col_attack_per_w.get(c, 0)

    if verbose:
        _print_report(collect_dir, ref_kind, reports, agg)

    def _dev(col: int, ref: int) -> float:
        # SIGNED, collected-relative-to-reference: +X% = the collect labeled X%
        # MORE events than the demo (over-count); -X% = fewer (under-count).
        # Checked against the expected band below.
        if ref == 0:
            return 0.0 if col == 0 else 1.0
        return (col - ref) / ref

    # Single aggregate attack number over ALL weapons. Every weapon's reference
    # is a per-trigger-pull fire-sound count (single-shot = 1 sound/shot;
    # continuous NG/SNG collapsed at MERGE_GAP; LG reconstructed from the
    # lstart+lhit heartbeat) — so all weapons are on the same footing and fold
    # into one number.
    attack_dev = _dev(agg.col_attack, agg.ref_attack)
    jump_dev = _dev(agg.col_jump, agg.ref_jump)

    def _band_str(b: tuple[float, float]) -> str:
        return f"[{b[0]:+.0%}, {b[1]:+.0%}]"

    failures: list[str] = []
    if not (attack_band[0] <= attack_dev <= attack_band[1]):
        failures.append(
            f"attack events: collected {agg.col_attack} vs demo "
            f"{ref_kind} {agg.ref_attack} → deviation {attack_dev:+.1%} "
            f"outside expected band {_band_str(attack_band)}"
        )
    if not (jump_band[0] <= jump_dev <= jump_band[1]):
        failures.append(
            f"jump events: collected {agg.col_jump} vs demo SND {agg.ref_jump} "
            f"→ deviation {jump_dev:+.1%} outside expected band {_band_str(jump_band)}"
        )

    if verbose:
        print()
        print(f"attack deviation: {attack_dev:+.1%}  "
              f"(+over / -under vs demo;  expected band {_band_str(attack_band)})")
        print(f"jump   deviation: {jump_dev:+.1%}  (expected band {_band_str(jump_band)})")

        # Known unexplained discrepancy: attack and jump both under-count their
        # demo-sound references by a similar ~few-percent margin. Ruled out:
        # resample rate (the gap is hz-stable, same at native tick), weapon
        # attribution (act_attack == QC-fired impulse — same deviation either
        # way), and attack feasibility false-negatives (~0.6% at fire frames).
        # Leading remaining suspect is cmd-window button0 press detection /
        # high-ping press->sound lead, but it's not been pinned down.
        if attack_dev < -0.02 and jump_dev < -0.02:
            print()
            print("NOTE: attack and jump both under-count vs demo sounds by a "
                  "similar margin — cause not determined")
            print("      (ruled out: resample rate, weapon attribution, "
                  "feasibility false-negatives).")

    if failures:
        banner = "LABEL VALIDATION WARNING" if warn_only else "LABEL VALIDATION FAILED"
        print(f"\n{banner}:")
        for f in failures:
            print(f"  - {f}")
        return 0 if warn_only else 1

    print("\nLabel validation OK")
    return 0


def _print_report(collect_dir: Path, ref_kind: str,
                  reports: list[DemoReport], agg: Aggregate) -> None:
    print("=" * 88)
    print(f"ATTACK/JUMP label validation — {collect_dir}")
    print(f"attack reference = demo {ref_kind};  jump reference = demo SND")
    print("=" * 88)
    print(f"{'demo':52} {'A_ref':>6} {'A_col':>6} {'J_ref':>6} {'J_col':>6}")
    print("-" * 88)
    for r in reports:
        print(f"{r.demo_file[:52]:52} {r.ref_attack:>6} {r.col_attack:>6} "
              f"{r.ref_jump:>6} {r.col_jump:>6}")
    print("-" * 88)
    print(f"{'TOTAL':52} {agg.ref_attack:>6} {agg.col_attack:>6} "
          f"{agg.ref_jump:>6} {agg.col_jump:>6}")

    cols = " ".join(f"{WNAME[c]:>5}" for c in WNAME)
    print("\n" + "=" * 88)
    print("AGGREGATE per-weapon attack (collected attribution by stored act_attack)")
    print("=" * 88)
    print(f"{'':14} {cols}  {'TOT':>6}")
    rrow = " ".join(f"{agg.ref_attack_per_w[c]:>5}" for c in WNAME)
    crow = " ".join(f"{agg.col_attack_per_w[c]:>5}" for c in WNAME)
    print(f"{'demo ' + ref_kind.split()[0]:14} {rrow}  {sum(agg.ref_attack_per_w.values()):>6}")
    print(f"{'collected':14} {crow}  {sum(agg.col_attack_per_w.values()):>6}")
    print("ratio col/ref:",
          {WNAME[c]: round(agg.col_attack_per_w[c] / agg.ref_attack_per_w[c], 2)
           for c in WNAME if agg.ref_attack_per_w[c]})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate collected attack/jump labels against demo byte truth")
    parser.add_argument("collect_dir", help="Collect output dir (has collect_metadata.json)")
    parser.add_argument("--demo-dir", required=True, help="Directory of source demos")
    parser.add_argument("--manifest", required=True, help="Corpus manifest.ndjson")
    parser.add_argument("--attack-band", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        default=None,
                        help="Expected attack-deviation band (default %s)" % str(EXPECTED_DEV_BAND))
    parser.add_argument("--jump-band", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        default=None,
                        help="Expected jump-deviation band (default %s)" % str(EXPECTED_DEV_BAND))
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel demo-parse workers (default cpu-2; 1 = serial)")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false",
                        help="Ignore the per-demo reference cache (force re-parse)")
    parser.add_argument("--force-mvd-emit", dest="force_mvd_emit", action="store_true",
                        default=None,
                        help="Override emit-path detection: compare to demo SOUNDS")
    parser.add_argument("--qwd-emit", dest="force_mvd_emit", action="store_false",
                        help="Override emit-path detection: compare to demo TRIGGERS")
    parser.add_argument("--warn-only", action="store_true",
                        help="Downgrade an out-of-band deviation to a warning (exit 0)")
    args = parser.parse_args()

    raise SystemExit(validate_labels(
        Path(args.collect_dir),
        demo_dir=Path(args.demo_dir),
        manifest_path=Path(args.manifest),
        attack_band=tuple(args.attack_band) if args.attack_band else None,
        jump_band=tuple(args.jump_band) if args.jump_band else None,
        force_mvd_emit=args.force_mvd_emit,
        warn_only=args.warn_only,
        workers=args.workers,
        use_cache=args.use_cache,
    ))


if __name__ == "__main__":
    main()
