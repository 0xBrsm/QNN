"""Encounter slicing — the ONE encounter definition shared by the corpus
encounter-stats CLI (``scripts/corpus_encounter_analysis.py``) and the v6
encounter-sliced band-scoring path (``qnn.human.band_bank.featurize_encounters``).

Promoted (not duplicated — see ``feedback_no_legacy_paths_without_request``)
from ``scripts/corpus_encounter_analysis.py``'s ``_find_pid_runs`` /
``_split_run_at_boundaries``, which remain the authoritative rule; the script
now imports this module instead of carrying its own copy.

Corpus side (``corpus_pid_encounter_spans``)
---------------------------------------------
An "encounter" is a contiguous same-opponent (``pid``) stored run, optionally
split into sub-encounters at a recency-boundary crossing: a frame where the
target's recency reaches the SIGHT ceiling (about to drop out of obs — the
forward-extension label tail exhausted its window) immediately followed by
the SAME pid at recency ~0 (a fresh re-acquisition, "A->A"). A different-pid
run in between ("A->B->A") is always a separate encounter (``find_pid_runs``
alone already draws that boundary; no recency needed).

**Degenerate-recency corpora.** Post-a27 collects carry no graduated
``entity_recency`` at all — ``qnn.eval.aim_kernel.iter_shard_episodes``
synthesizes a BINARY stand-in (0 = in LOS this frame, else the float16 max
sentinel) from ``entity_modality_id`` ("A27 has no recency feature"). Fed
into the boundary rule verbatim, EVERY occlusion tick instantly reads as
"at the ceiling" regardless of true duration, so the A->A split degenerates
to "split on every 1-tick LOS flicker" — measured on the pinned bank corpus
``qwd_v4d_v3vis``: median sub-encounter collapses from ~2.1 s (continuous
recency, e.g. ``qwd_v4``) to ~0.8 s. ``has_graduated_recency`` detects this
and ``corpus_pid_encounter_spans`` auto-degrades to the unsplit pid-run
(A->B->A boundaries only) when it fires — the honest encounter definition
this corpus format actually supports.

Subject side (``engaged_encounter_spans``)
---------------------------------------------
Arena npz gate streams (``qnn.eval.humanlikeness.human_band.load_rc_episodes``)
are 1v1 — single opponent, no pid column, no recency channel, just a binary
``engaged`` bit per tick. The corresponding encounter definition: runs of
engaged frames, BRIDGING (not splitting on) gaps shorter than the SIGHT
ceiling used above. Correspondence: on the corpus side, an occlusion gap
shorter than the ceiling never breaks the same-pid run at all (the target
label forward-extends through it) — the gap is bridged into one encounter by
construction; only a gap at/past the ceiling, or a different opponent in
between, starts a new encounter. The subject stream has no recency to
consult, so it reuses the identical wall-clock threshold as its gap-bridge
rule — the closest available proxy for "the corpus wouldn't have split here
either."
"""
from __future__ import annotations

import numpy as np

# Entity persists in obs (forward-extended target label) while occlusion age
# <= this many seconds — the SIGHT ceiling both sides key off of.
QNN_RECENCY_MAX_SIGHT_S = 2.0


# ---------------------------------------------------------------------------
# Corpus side: same-opponent pid runs (scripts/corpus_encounter_analysis.py's
# _find_pid_runs / _split_run_at_boundaries, ported verbatim)
# ---------------------------------------------------------------------------
def find_pid_runs(pid_seq: np.ndarray) -> list[tuple[int, int, int]]:
    """(start, end_excl, player_id) for contiguous same-pid runs (pid > 0)."""
    T = len(pid_seq)
    if T == 0:
        return []
    padded = np.empty(T + 2, dtype=np.int64)
    padded[0] = 0
    padded[1:-1] = pid_seq
    padded[-1] = 0
    change = padded[1:] != padded[:-1]
    starts = np.flatnonzero(change[:-1] & (pid_seq > 0))
    ends = np.flatnonzero(change[1:] & (pid_seq > 0)) + 1
    return [(int(s), int(e), int(pid_seq[s])) for s, e in zip(starts, ends)]


def split_run_at_boundaries(
    run_start: int,
    run_end: int,
    recency: np.ndarray,
    *,
    boundary_recency: float,
    near_zero_recency: float,
) -> list[tuple[int, int]]:
    """Split a same-pid stored run into sub-encounters at recency boundaries.

    A boundary crossing is a consecutive pair (t, t+1) within the run where
    ``recency[t] >= boundary_recency`` (near/at the SIGHT ceiling) and
    ``recency[t+1] < near_zero_recency`` (fresh re-acquisition). The run is
    split after frame t.
    """
    rec = np.asarray(recency)[run_start:run_end]
    L = len(rec)
    if L < 2:
        return [(run_start, run_end)]
    at_ceil = rec[:-1] >= boundary_recency
    back_to_0 = rec[1:] < near_zero_recency
    split_local = np.flatnonzero(at_ceil & back_to_0)
    if len(split_local) == 0:
        return [(run_start, run_end)]
    spans: list[tuple[int, int]] = []
    cur = run_start
    for li in split_local:
        end = run_start + int(li) + 1
        spans.append((cur, end))
        cur = end
    spans.append((cur, run_end))
    return spans


def has_graduated_recency(
    recency: np.ndarray, *, near_zero_recency: float, boundary_recency: float
) -> bool:
    """True iff ``recency`` shows genuine intermediate values between the
    near-zero and boundary thresholds (a continuous decay signal). False for
    a binary in-LOS/occluded corpus (the a27+ ``entity_modality_id``
    fallback) — see the module docstring for why the boundary split
    degenerates in that case."""
    r = np.asarray(recency)
    mid = r[(r > near_zero_recency) & (r < boundary_recency)]
    return bool(mid.size > 0)


def pid_recency_from_tokens(
    entity_count: np.ndarray,
    target_probs: np.ndarray,
    entity_player_id: np.ndarray,
    entity_recency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (pid, recency) from the corpus's soft target label.

    ``target_probs`` is (T, 17): slot 0 = no-target, slot j = entity index
    j-1. Entities are ragged, indexed by the cumulative ``entity_count``
    local to this episode/shard slice (all four arrays must share the same
    local indexing convention — e.g. one episode's slices from
    ``qnn.eval.aim_kernel.iter_shard_episodes``).
    """
    tp = np.asarray(target_probs)
    cnt = np.asarray(entity_count, dtype=np.int64)
    T = tp.shape[0]
    off = np.concatenate([[0], cnt.cumsum()])
    tgt = tp.argmax(1) - 1  # -1 => no target, else entity index
    engaged = tgt >= 0
    gidx = off[:T] + np.clip(tgt, 0, None)
    pid_seq = np.zeros(T, dtype=np.int64)
    rec_seq = np.zeros(T, dtype=np.float64)
    pid_tok = np.asarray(entity_player_id)
    rec_tok = np.asarray(entity_recency, dtype=np.float64)
    pid_seq[engaged] = pid_tok[gidx[engaged]]
    rec_seq[engaged] = rec_tok[gidx[engaged]]
    pid_seq[pid_seq < 0] = 0
    return pid_seq, rec_seq


def corpus_pid_encounter_spans(
    pid_seq: np.ndarray, recency: np.ndarray, hz: float
) -> list[tuple[int, int]]:
    """The band-bank combinator: same-pid runs, split at recency boundaries
    UNLESS ``recency`` is degenerate-binary on this corpus (auto-detected),
    in which case the unsplit pid run (A->B->A boundaries only) is used.
    Spans are sorted by construction (``find_pid_runs`` walks left to right).
    """
    boundary = QNN_RECENCY_MAX_SIGHT_S - 1.0 / hz
    near_zero = 1.0 / hz
    graduated = has_graduated_recency(
        recency, near_zero_recency=near_zero, boundary_recency=boundary)
    spans: list[tuple[int, int]] = []
    for s, e, _pid in find_pid_runs(pid_seq):
        if graduated:
            spans.extend(split_run_at_boundaries(
                s, e, recency, boundary_recency=boundary,
                near_zero_recency=near_zero))
        else:
            spans.append((s, e))
    return spans


# ---------------------------------------------------------------------------
# Subject side: engaged-span runs with gap-bridging (no pid channel)
# ---------------------------------------------------------------------------
def engaged_encounter_spans(
    engaged: np.ndarray, hz: float, *, gap_bridge_s: float | None = None
) -> list[tuple[int, int]]:
    """Runs of ``engaged`` frames, bridging gaps shorter than
    ``gap_bridge_s`` (default: the same SIGHT-ceiling wall-clock threshold
    the corpus side uses — see the module docstring's "Subject side"
    section for the correspondence)."""
    if gap_bridge_s is None:
        gap_bridge_s = QNN_RECENCY_MAX_SIGHT_S - 1.0 / hz
    gap_bridge = int(round(gap_bridge_s * hz))
    e = np.asarray(engaged, dtype=bool)
    if e.size == 0:
        return []
    idx = np.flatnonzero(e)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks] + 1, [idx[-1] + 1]])
    spans: list[tuple[int, int]] = []
    cur_s, cur_e = int(starts[0]), int(ends[0])
    for s, en in zip(starts[1:], ends[1:]):
        if int(s) - cur_e <= gap_bridge:
            cur_e = int(en)
        else:
            spans.append((cur_s, cur_e))
            cur_s, cur_e = int(s), int(en)
    spans.append((cur_s, cur_e))
    return spans
