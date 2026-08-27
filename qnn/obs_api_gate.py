"""Gate 2 — the served-what-you-trained-on gate (obs_api v1, WS4).

Per agents/plans/obs-api.md §gates: per model, replay corpus frames
through the model's declared plan + generic codec and require equality
with the training-time parse. This gate replaces wire stamps as the
deploy precondition once integration (WS5) lands.

── WHAT THIS GATE PROVES (and precisely what it does NOT) ──────────

The BC corpus caches store the OUTPUT of the training-time parse
(per-field NPY arrays written by qnn.bc.collect via the era's
``unpack_obs_buffer_native``), NOT raw engine frames — so a direct
raw-frame replay is impossible. The strongest available equivalence,
implemented here, is a re-pack + re-parse round trip:

    stored fields ──pack_frame(declared layout)──► synthetic frame
                  ──wire.unpack_frame(declared layout)──► fields'
    require fields' == stored fields, bitwise, per field.

This PROVES, over the sampled frames:
  * every stored field the training parse consumed is representable
    in the model's declared layout (state-field coverage, atlas
    parameterization/width, entity policy row coverage — token tags,
    per-type scalar columns, event windows), at the exact wire dtype
    and per-row shape (checked field-wide up front);
  * the stored content lies exactly in the IMAGE of the declared
    parse: cells the declared walk never serializes (event-pair
    tails beyond event_count, player ids on non-actor rows, scalar
    columns a token's type does not carry) must be zero, or the
    round trip cannot reproduce them and the field fails;
  * pack→unpack through the generic layout-driven codec
    (qnn.wire.unpack_frame) is the identity on that content — the
    packer is the registry-mirror inverse, so a disagreement here is
    a walk/offset bug in the layout math or codec.

It does NOT prove:
  * byte-level parity with live engine emit — no raw frames are
    retained in the caches, so engine-side emit bugs (gate 1 / gate 3
    territory) are out of reach here; corruption of a stored value
    that stays representable is likewise invisible (the round trip
    reproduces whatever is stored);
  * anything about fields the cache never stored: ``look_delta`` is
    dropped at cache-write time (wire-only inference field) and is
    reported SKIPPED, packed as zeros and excluded from equality;
  * anything beyond the sampled frames/shard.

Stored fields the declared layout CANNOT carry (e.g. full-stream
entity columns under a pure-combat policy) fail the gate when any
sampled value is nonzero — the training parse consumed content the
declared plan would not serve. Token tags outside the declared policy
make the whole frame's entity stream unrepresentable and are counted
(with a tag histogram) as hard failures; state/atlas comparison still
runs for those frames.

CLI:
    python -m qnn.obs_api_gate RUN_DIR [CORPUS_DIR] \
        [--split precomputed_train] [--sample 512] [--seed 0] \
        [--entities-policy vN]

``--entities-policy`` is a DIAGNOSTIC override of the declared entity
policy (e.g. to demonstrate that a v3-vs-v1 failure is policy-naming
only); the verdict it produces is not the model's gate verdict.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from qnn.obs_api import (
    ATLAS_OUT_KEY,
    ENTITY_POLICIES,
    Declaration,
    Layout,
    UnrepresentableTokenError,
    compile_layout,
    declaration_for_run,
    pack_frame,
    training_corpus_for_run,
)
from qnn.wire import unpack_frame

_SHARD = "shard000000"
_OBS_PREFIX = f"{_SHARD}_obs_"

# Field-report statuses.
MATCH = "MATCH"
MISMATCH = "MISMATCH"
SKIPPED = "SKIPPED (not stored by the training cache)"
NOT_DECLARED_ZERO = "NOT DECLARED (zero-only in sample — vacuously served)"
NOT_DECLARED_NONZERO = "NOT DECLARED (NONZERO in sample — unrepresentable)"
UNPACKABLE = "UNREPRESENTABLE (token tags outside the declared policy)"

_PASS_STATUSES = {MATCH, SKIPPED, NOT_DECLARED_ZERO}


@dataclass
class FieldReport:
    name: str
    status: str
    frames: int = 0          # frames compared (or inspected)
    mismatches: int = 0
    note: str = ""


@dataclass
class Gate2Report:
    run_dir: str
    corpus: str
    split: str
    declaration: str          # canonical JSON
    sampled: int
    total_rows: int
    unpackable_frames: int
    unpackable_tags: dict[int, int]
    fields: list[FieldReport]
    pack_errors: int = 0
    pack_error_notes: list[str] | None = None

    @property
    def passed(self) -> bool:
        return (self.unpackable_frames == 0 and self.pack_errors == 0
                and all(f.status in _PASS_STATUSES for f in self.fields))

    def format(self) -> str:
        lines = [
            "gate 2 — served-what-you-trained-on",
            f"  run:         {self.run_dir}",
            f"  corpus:      {self.corpus} ({self.split}, {_SHARD})",
            f"  declaration: {self.declaration}",
            f"  sample:      {self.sampled} of {self.total_rows} frames",
        ]
        if self.pack_errors:
            lines.append(
                f"  pack errors: {self.pack_errors}/{self.sampled} frames could "
                f"not be re-packed at all — first: {self.pack_error_notes}"
            )
        if self.unpackable_frames:
            lines.append(
                f"  entity stream: {self.unpackable_frames}/{self.sampled} frames "
                f"carry token tags outside the declared policy "
                f"(tag histogram over sampled tokens: {self.unpackable_tags})"
            )
        width = max(len(f.name) for f in self.fields)
        for f in sorted(self.fields, key=lambda f: f.name):
            detail = f.status
            if f.status == MISMATCH:
                detail += f" ({f.mismatches}/{f.frames} frames)"
            if f.note:
                detail += f" — {f.note}"
            lines.append(f"    {f.name:<{width}}  {detail}")
        lines.append(f"  verdict: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def _load_split(split_dir: Path) -> dict[str, np.ndarray]:
    """Mmap every stored obs field of shard 0 of a cache split."""
    fields = {
        p.name[len(_OBS_PREFIX):-len(".npy")]: np.load(p, mmap_mode="r")
        for p in sorted(split_dir.glob(f"{_OBS_PREFIX}*.npy"))
    }
    if not fields:
        raise ValueError(f"gate 2: no {_OBS_PREFIX}*.npy fields under {split_dir}")
    if "entity_count" not in fields:
        raise ValueError(f"gate 2: {split_dir} stores no entity_count field")
    return fields


def _bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def run_gate2(run_dir: Path | str | None,
              corpus_dir: Path | str | None = None,
              declaration: Declaration | None = None,
              split: str = "precomputed_train",
              sample: int = 512,
              seed: int = 0) -> Gate2Report:
    """Run the gate for one run dir + its corpus cache. See module doc."""
    if declaration is None:
        if run_dir is None:
            raise ValueError("gate 2: need a run_dir or an explicit declaration")
        declaration = declaration_for_run(run_dir, corpus_dir)
    if run_dir is not None or corpus_dir is None:
        corpus = training_corpus_for_run(run_dir, corpus_dir)
    else:
        corpus = Path(corpus_dir)
    layout: Layout = compile_layout(declaration)
    split_dir = corpus / split
    stored = _load_split(split_dir)

    counts = np.asarray(stored["entity_count"]).astype(np.int64)
    rows = int(counts.shape[0])
    tok_off = np.concatenate([[0], np.cumsum(counts)])
    rng = np.random.default_rng(seed)
    n_sample = min(int(sample), rows)
    picks = np.sort(rng.choice(rows, size=n_sample, replace=False))

    state_fields = [f for f in layout.fields if f.kind == "state"]
    sensor_fields = [f for f in layout.fields if f.kind == "sensor"]
    percept = next((f for f in layout.fields if f.kind == "percept"), None)
    policy = ENTITY_POLICIES[percept.params["policy"]] if percept else None
    policy_keys = [name for name, _, _, _ in policy.fields] if policy else []

    # Partition stored keys: per-frame (rows-long) vs per-token arrays.
    tokens_total = int(tok_off[-1])
    frame_keys, token_keys = set(), set()
    for key, arr in stored.items():
        if int(arr.shape[0]) == rows:
            frame_keys.add(key)
        elif int(arr.shape[0]) == tokens_total:
            token_keys.add(key)
        else:
            raise ValueError(
                f"gate 2: stored field {key!r} has leading dim {arr.shape[0]}, "
                f"matching neither rows ({rows}) nor tokens ({tokens_total})"
            )

    declared_frame_keys = ({f.name for f in state_fields}
                           | ({ATLAS_OUT_KEY} if sensor_fields else set())
                           | ({"entity_count"} if percept else set()))
    declared_token_keys = set(policy_keys)

    reports: dict[str, FieldReport] = {}

    def report(name: str) -> FieldReport:
        if name not in reports:
            reports[name] = FieldReport(name=name, status=MATCH)
        return reports[name]

    # Declared-but-not-stored fields: pack zeros, exclude from equality.
    for f in state_fields:
        if f.name not in stored:
            reports[f.name] = FieldReport(
                name=f.name, status=SKIPPED,
                note="packed as zeros; equality unverifiable")
    if sensor_fields and ATLAS_OUT_KEY not in stored:
        raise ValueError(
            "gate 2: declaration requests the atlas but the cache stores no "
            f"{ATLAS_OUT_KEY} — wrong corpus for this run?")

    # Dtype/shape pre-pass: a stored field whose dtype or per-row shape
    # disagrees with the declared wire block can never round-trip — one
    # MISMATCH report, excluded from packing and per-value compare.
    incompatible: set[str] = set()

    def _check_block(name: str, arr: np.ndarray, dtype: np.dtype,
                     tail: tuple[int, ...]) -> None:
        got = (arr.dtype, tuple(arr.shape[1:]))
        if got != (dtype, tuple(tail)):
            incompatible.add(name)
            reports[name] = FieldReport(
                name=name, status=MISMATCH, frames=n_sample,
                mismatches=n_sample,
                note=(f"stored dtype/shape {got[0]}/{got[1]} vs wire "
                      f"{dtype}/{tuple(tail)}"))

    for f in state_fields:
        if f.name in stored:
            _check_block(f.name, stored[f.name], f.np_dtype(), f.shape)
    if sensor_fields:
        _check_block(ATLAS_OUT_KEY, stored[ATLAS_OUT_KEY],
                     sensor_fields[0].np_dtype(), sensor_fields[0].shape)
    if policy is not None:
        for key, dtype, tail, _fill in policy.fields:
            if key in stored:
                _check_block(key, stored[key], np.dtype(dtype), tail)

    # Stored-but-not-declared fields: nonzero content is unservable.
    for key in sorted((frame_keys | token_keys)
                      - declared_frame_keys - declared_token_keys):
        arr = stored[key]
        if key in frame_keys:
            sampled_view = arr[picks]
        else:
            sampled_view = np.concatenate(
                [arr[tok_off[i]:tok_off[i + 1]] for i in picks]) \
                if n_sample else arr[:0]
        nonzero = int(np.count_nonzero(np.asarray(sampled_view)))
        reports[key] = FieldReport(
            name=key,
            status=NOT_DECLARED_NONZERO if nonzero else NOT_DECLARED_ZERO,
            frames=n_sample,
            note=(f"{nonzero} nonzero cells in sample" if nonzero else ""),
        )

    unpackable_frames = 0
    unpackable_tags: dict[int, int] = {}
    pack_errors = 0
    pack_error_notes: list[str] = []

    def compare(name: str, got, want) -> None:
        r = report(name)
        r.frames += 1
        if not _bitwise_equal(got, want):
            r.mismatches += 1
            r.status = MISMATCH
            if not r.note:
                got_a, want_a = np.asarray(got), np.asarray(want)
                r.note = (f"first divergence: dtype {got_a.dtype} vs "
                          f"{want_a.dtype}, shape {got_a.shape} vs {want_a.shape}"
                          if (got_a.dtype != want_a.dtype
                              or got_a.shape != want_a.shape)
                          else "values differ")

    for i in picks:
        i = int(i)
        n_tok = int(counts[i])
        tok = slice(int(tok_off[i]), int(tok_off[i + 1]))
        pack_fields: dict[str, np.ndarray] = {}
        for f in state_fields:
            if f.name in stored and f.name not in incompatible:
                pack_fields[f.name] = np.asarray(stored[f.name][i])
            else:
                pack_fields[f.name] = np.zeros(f.shape, dtype=f.np_dtype()) \
                    if f.shape else np.zeros((), dtype=f.np_dtype())
        for f in sensor_fields:
            pack_fields[ATLAS_OUT_KEY] = (
                np.asarray(stored[ATLAS_OUT_KEY][i])
                if ATLAS_OUT_KEY not in incompatible
                else np.zeros(f.shape, dtype=f.np_dtype()))
        entity_ok = percept is not None
        if percept is not None:
            if "entity_types" not in stored:
                raise ValueError("gate 2: cache stores no entity_types")
            # Policy fields the cache never stored (or stored at the wrong
            # dtype/shape) pack as zeros; everything else packs verbatim.
            for key, dtype, tail, _fill in policy.fields:
                if key in stored and key not in incompatible:
                    pack_fields[key] = np.asarray(stored[key][tok])
                else:
                    pack_fields[key] = np.zeros((n_tok, *tail), dtype=dtype)
            pack_fields["entity_count"] = np.array(n_tok, dtype=np.uint8)
        try:
            raw = pack_frame(pack_fields, layout)
        except UnrepresentableTokenError:
            unpackable_frames += 1
            entity_ok = False
            for tag in np.asarray(stored["entity_types"][tok]).tolist():
                unpackable_tags[int(tag)] = unpackable_tags.get(int(tag), 0) + 1
            # Re-pack with an empty stream so state/atlas still get checked.
            for key in list(pack_fields):
                if key in declared_token_keys:
                    pack_fields[key] = pack_fields[key][:0]
            pack_fields["entity_count"] = np.array(0, dtype=np.uint8)
            raw = pack_frame(pack_fields, layout)
        except ValueError as exc:
            # Content pack_frame refuses outright (oversized token/event
            # counts, ...) — a hard failure, reported rather than raised so
            # the rest of the sample still gets summarized.
            pack_errors += 1
            if len(pack_error_notes) < 3:
                pack_error_notes.append(f"frame {i}: {exc}")
            continue
        unpacked = unpack_frame(raw, layout)

        for f in state_fields:
            if f.name in stored and f.name not in incompatible:
                compare(f.name, unpacked[f.name], stored[f.name][i])
        for f in sensor_fields:
            if ATLAS_OUT_KEY not in incompatible:
                compare(ATLAS_OUT_KEY, unpacked[ATLAS_OUT_KEY],
                        stored[ATLAS_OUT_KEY][i])
        if percept is not None and entity_ok:
            compare("entity_count", unpacked["entity_count"],
                    np.array(n_tok, dtype=np.uint8))
            for key in policy_keys:
                if key in stored and key not in incompatible:
                    compare(key, unpacked[key], stored[key][tok])
                elif key not in reports:
                    reports[key] = FieldReport(
                        name=key, status=SKIPPED,
                        note="declared by the policy but never stored")

    if unpackable_frames and percept is not None:
        for key in policy_keys:
            r = report(key)
            if r.status == MATCH and r.frames < n_sample:
                r.status = UNPACKABLE
                r.note = (f"compared on {r.frames}/{n_sample} frames; the rest "
                          "carry tags outside the declared policy")

    return Gate2Report(
        run_dir=str(run_dir) if run_dir is not None else "(explicit declaration)",
        corpus=str(corpus),
        split=split,
        declaration=declaration.to_json(),
        sampled=n_sample,
        total_rows=rows,
        unpackable_frames=unpackable_frames,
        unpackable_tags=dict(sorted(unpackable_tags.items())),
        fields=list(reports.values()),
        pack_errors=pack_errors,
        pack_error_notes=pack_error_notes,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="training run dir (runs/<mode>/<name>)")
    ap.add_argument("corpus_dir", nargs="?", default=None,
                    help="corpus cache dir override (default: the run's own "
                         "bc_manifest bc_data_dir)")
    ap.add_argument("--split", default="precomputed_train")
    ap.add_argument("--sample", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--entities-policy", default=None,
                    help="DIAGNOSTIC: override the declared entity policy "
                         "(the verdict is then not the model's gate verdict)")
    args = ap.parse_args(argv)

    declaration = declaration_for_run(args.run_dir, args.corpus_dir)
    if args.entities_policy:
        if declaration.entities is None:
            raise SystemExit("--entities-policy: the declaration requests no entities")
        declaration = replace(declaration, entities={
            **declaration.entities, "policy": args.entities_policy})
        # Re-validate through the registry (fail loud on unknown policy).
        declaration = Declaration.from_dict(declaration.to_dict())
        print(f"DIAGNOSTIC entity-policy override: {args.entities_policy} "
              "(not the declared policy — verdict is informational)")
    result = run_gate2(args.run_dir, args.corpus_dir, declaration=declaration,
                       split=args.split, sample=args.sample, seed=args.seed)
    print(result.format())
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
