"""Within-encounter weapon-switch DECISION curve (BOT, a28rc1b) — the
h2h-stream counterpart of ``qnn.human.weapon_switch_curve``.

Ruler (identical to the human side — see that module's docstring, including
its "WHY THIS IS NOT A GAP-BINNED CURVE" respec note): within an encounter, do
consecutive firing DECISIONS pick the same weapon or a different one, as a
smooth function of the refire opportunities skipped past the deciding weapon's
first decidable one. This module owns ONLY the extraction of encounter-scoped
records from h2h eval stream npz files (``side_a28rc1b_streams.npz``); the
onset gate, the pairing-record CONTRACT, the covariate, the curve fit, the
cluster bootstrap and the mid-train statistic are the human module's —
imported, not reimplemented (feedback_reuse_existing_tooling,
feedback_no_legacy_paths_without_request).

The onset gate matters MOST here, not on the human side: this stream's
``discharge`` column resolves the NG/SNG/LG 0.1s think-chain at full rate
while the human corpus resolves it at 0.20s, so un-gated raw discharges made
84% of this side's pairs hold-train continuations against the human side's
37% — the two curves were not measuring the same event at all. Gating both
sides to decisions is what makes them comparable.

A second, independent gate is needed for the same reason (``world_shot_mask``).
``discharge`` is defined as ``attack AND (obs attack_finished expired)``
(``qnn.eval.h2h``) — the model's decision to select-and-fire, NOT confirmation
that a shot left the engine. The engine can refuse: ``W_CheckNoAmmo`` aborts
``W_Attack`` when the HELD weapon is dry (it swaps to ``W_BestWeapon`` and
returns FALSE), setting no cooldown, so the obs stays ready and the next tick
attacks again — a dry-weapon spin that emits a burst of records one tick apart.
On ammo-constrained venues that is 29-46% of all records (RL 46% on the box
arena, SG 39% / GL 45% on the real maps) against 2.4% on the decode-fit
freeplay sample. Left in, those bursts manufacture exactly the short-gap
"switches" this ruler exists to measure. The human side never needed this gate:
its label is the impulse class QC ``W_Attack`` actually advanced
(``qnn_qwd_collect.c``), so a refused attack is never an event there. This is
the requested-vs-forced decomposition the project already names
(feedback_weapon_switch_is_engine_forced) applied at the event boundary.

Encounter unit DIFFERS from the human side by necessity: the human corpus
scopes encounters via the v6 same-opponent pid-run slicer (entity-recency
tokens aren't in the h2h stream schema); h2h streams instead carry a direct
``engaged`` predicate (enemy-visible, per-tick bool) as the task spec
requires — an encounter here is a maximal contiguous ``engaged == True`` run,
never crossing a round boundary (``episode_offsets``). Both are "the ticks
during which this agent is in a fight with a specific opponent, gap-clustered
losses of sight aside"; the difference is the corpus-fingerprint reason
(feedback_collect_perception_regime_not_fingerprinted) recency tokens exist
for human demos and not for h2h dumps, not a different concept of encounter.

Forced-exclusion resolution (stated per task instructions): the task's plain
-English gloss ("drop a pair when the FIRST discharge's weapon's
weapon_feas bit is 0 at any tick between the two discharges") is a narrower
paraphrase of the actual reused predicate,
``qnn.eval.humanlikeness.rc._invalidated_pairs``, which the human module also
calls: that predicate drops a pair when the WHOLE feasibility bitmask changes
at all (any weapon entering/leaving the menu) OR a death tick occurs in
(t_i, t_i+1]. Per the DESIGN CONSTRAINT (one statistic, one implementation),
this module uses the exact same predicate the human baseline used rather
than a bespoke narrower one — a menu change unrelated to wk's own bit is
still evidence the "free choice between two stationary menus" precondition
failed (pickup, other weapon's ammo dry-out changing the salient menu,
respawn reset), and re-deriving a narrower rule here would make the two
curves not-directly-comparable, which is the one thing the task rules out.

The human baseline artifact is read for its schema tag and its own curve (for
the side-by-side table), never for bin edges — there are none:
    artifacts/collect/qwd_v4d_v3vis/human_baseline/_weapon_switch_transition_curve.json

Venue pooling: box-map (qnn_arena8, h2h_a28rc1b_vs_a28rc1c flat/tier seeds
43/44/45, free-choice arms only — pinlg_*/pinrl_* excluded, weapon pinned,
no switch decision) and real-map (bias_opt_ra2 vs_rc1b_a28rc1{c,e}_
{arendm1a,barena1,uarena1}_s60{1,2}_{fwd,rev}, 24 dirs) are reported
SEPARATELY as well as pooled — venue verdicts differ in this project
(feedback_operative_filter_all_comparisons); never merge silently.

Usage:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=src \\
    python -m qnn.eval.humanlikeness.weapon_switch_curve_bot \\
      --runs-root runs/eval \\
      --baseline artifacts/collect/qwd_v4d_v3vis/human_baseline/_weapon_switch_transition_curve.json \\
      --out artifacts/eval_analysis/weapon_switch_curve/a28rc1b_weapon_switch_transition_curve.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from qnn.human.weapon_switch_curve import (SCHEMA, atom_by_menu,
                                            covariate_distribution,
                                            decision_events, event_stats,
                                            fit_curve, gap_distribution,
                                            mid_train_switch, pair_accounting,
                                            pairs_from_decisions)

HZ = 20.0  # h2h stream tick_hz is verified 20.0 for every run dir this module
           # reads; qnn.human.weapon_switch_curve's covariate/gap_distribution
           # convert gap_frames -> seconds via this SAME module-level constant
           # (not a parameter), so reuse is only valid while every input file
           # matches it -- stream_pairs asserts this.

STREAM_NAME = "side_a28rc1b_streams.npz"

# ---------------------------------------------------------------------------
# Run-dir sets (task spec, restated explicitly rather than glob-discovered,
# so an unrelated future dir under the same roots can't silently enter the
# pool)
# ---------------------------------------------------------------------------
BOX_RUN_DIRS: tuple[str, ...] = tuple(
    f"h2h_a28rc1b_vs_a28rc1c/{arm}_s{seed}"
    for arm, seed in itertools.product(("flat", "tier"), (43, 44, 45))
)  # pinlg_*/pinrl_* deliberately excluded -- weapon pinned, no switch decision

REAL_RUN_DIRS: tuple[str, ...] = tuple(
    f"bias_opt_ra2/vs_rc1b_a28rc1{opp}_{arena}_s60{seed}_{direction}"
    for opp, arena, seed, direction in itertools.product(
        ("c", "e"), ("arendm1a", "barena1", "uarena1"), (1, 2), ("fwd", "rev"))
)


# ---------------------------------------------------------------------------
# World-shot gate (this module's own job — see the module docstring)
# ---------------------------------------------------------------------------
def world_shot_mask(discharge_ticks: np.ndarray, op_ready: np.ndarray,
                    ) -> np.ndarray:
    """Which discharge records were shots that actually LEFT the engine?

    ``op_ready`` is the obs's own ``attack_finished``-expired predicate, and
    every weapon's cooldown is at least 2 ticks at 20 Hz (LG, the shortest, is
    0.1s), so a real shot ALWAYS leaves the next tick not-ready. A record still
    ready on the very next tick set no cooldown, which means ``W_Attack``
    returned without firing — the dry-held-weapon path. The test needs no
    per-weapon cooldown table and no duration matching (unreliable at LG's
    2-tick resolution): it only asks whether a cooldown started at all.

    A record on the segment's LAST tick has no following tick to read and is
    KEPT (conservative — it is never silently dropped for want of evidence)."""
    op_ready = np.asarray(op_ready, dtype=bool).reshape(-1)
    n = op_ready.shape[0]
    idx = np.asarray(discharge_ticks, dtype=np.int64).reshape(-1)
    keep = np.ones(idx.shape, dtype=bool)
    has_next = idx + 1 < n
    keep[has_next] = ~op_ready[idx[has_next] + 1]
    return keep


# ---------------------------------------------------------------------------
# Encounter-scoped pairing (this module's own job)
# ---------------------------------------------------------------------------
def _engaged_runs(engaged: np.ndarray) -> list[tuple[int, int]]:
    """Maximal contiguous True runs in a 1-D bool array -> [(start, end), ...]
    (end exclusive), never crossing the caller's own segment boundary
    (caller passes one round's slice already)."""
    n = engaged.shape[0]
    if n == 0:
        return []
    e = engaged.astype(np.int8)
    d = np.diff(e)
    starts = list((np.flatnonzero(d == 1) + 1).tolist())
    if e[0]:
        starts = [0] + starts
    ends = list((np.flatnonzero(d == -1) + 1).tolist())
    if e[-1]:
        ends = ends + [n]
    return list(zip(starts, ends))


def episode_encounter_pairs(
    weapon_pref: np.ndarray, engaged: np.ndarray,
    weapon_feas: np.ndarray, health: np.ndarray, op_ready: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    """One episode (already sliced to one round via ``episode_offsets``) ->
    a list of per-encounter DECISION-pair records, same contract as
    ``qnn.human.weapon_switch_curve.episode_encounter_pairs``: ``gap_frames``,
    ``switch``, ``forced``, ``wk``, ``wk1``, ``x``, ``in_domain`` — all (n,).
    Encounter = maximal contiguous ``engaged`` run.

    Records the engine refused (no cooldown started — ``world_shot_mask``) are
    dropped BEFORE the onset gate: they are not shots, so they can neither be a
    decision nor extend a hold train.

    Like the human side, the onset gate runs over the WHOLE episode's
    discharges before the encounter spans are cut, so a lost-sight boundary
    landing mid-hold-train cannot manufacture a spurious decision."""
    wp = np.asarray(weapon_pref).reshape(-1)
    idx = np.flatnonzero(wp > 0)
    idx = idx[world_shot_mask(idx, op_ready)]
    wlab = wp[idx].astype(np.int8)
    onset_t, onset_w = decision_events(idx, wlab)
    streams = {"feas": weapon_feas, "health": health}
    out: list[dict[str, np.ndarray]] = []
    for s, e in _engaged_runs(np.asarray(engaged, dtype=bool)):
        m = (onset_t >= s) & (onset_t < e)
        rec = pairs_from_decisions(idx, wlab, onset_t[m], onset_w[m], streams, HZ)
        if rec is not None:
            out.append(rec)
    return out


def episode_event_counts(weapon_pref: np.ndarray, op_ready: np.ndarray,
                         ) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Per weapon: ``([n_discharges, n_decisions], n_refused)`` for one episode
    — the input to ``event_stats``'s collapse-ratio table (the cross-side
    comparability check) plus the count of records the engine refused."""
    wp = np.asarray(weapon_pref).reshape(-1)
    raw = np.flatnonzero(wp > 0)
    keep = world_shot_mask(raw, op_ready)
    idx = raw[keep]
    wraw = wp[raw].astype(np.int8)
    wlab = wp[idx].astype(np.int8)
    _, onset_w = decision_events(idx, wlab)
    return ({w: [int((wlab == w).sum()), int((onset_w == w).sum())]
             for w in range(1, 9)},
            {w: int((wraw[~keep] == w).sum()) for w in range(1, 9)})


def stream_pairs_with_events(
    npz_path: Path,
) -> tuple[list[dict[str, np.ndarray]], dict[int, list[int]], dict[int, int]]:
    """Extract encounter-scoped DECISION-pair records, the per-weapon
    world-shot/decision counts, and the per-weapon count of engine-REFUSED
    attack records, from one h2h ``side_a28rc1b_streams.npz``,
    segmented first by ``episode_offsets`` (round boundaries -- pairs never
    cross a round) and then by ``engaged`` runs within each round."""
    z = np.load(npz_path)
    hz = float(np.asarray(z["tick_hz"]).reshape(-1)[0])
    if hz != HZ:
        raise ValueError(
            f"{npz_path}: tick_hz={hz} != {HZ} -- the reused human-core "
            "covariate/gap_distribution hardcode HZ=20.0 for the "
            "frame->seconds conversion; a differently-clocked stream would "
            "silently mis-scale gaps (and the onset gate's tick threshold) "
            "if pooled here.")
    offs = np.asarray(z["episode_offsets"], dtype=np.int64)
    weapon_pref = np.asarray(z["weapon_pref"])
    engaged = np.asarray(z["engaged"], dtype=bool)
    weapon_feas = np.asarray(z["weapon_feas"])
    health = np.asarray(z["health"], dtype=np.int64)
    if "op_ready" not in z.files:
        raise ValueError(
            f"{npz_path}: no op_ready column — the world-shot gate cannot run, "
            "and without it a refused attack (dry held weapon) counts as a "
            "discharge and manufactures short-gap switches. Re-run the eval "
            "with a stream schema that emits op_ready rather than scoring this "
            "file.")
    op_ready = np.asarray(z["op_ready"], dtype=bool)
    out: list[dict[str, np.ndarray]] = []
    ev: dict[int, list[int]] = {w: [0, 0] for w in range(1, 9)}
    refused: dict[int, int] = {w: 0 for w in range(1, 9)}
    for i in range(len(offs) - 1):
        s, e = int(offs[i]), int(offs[i + 1])
        out.extend(episode_encounter_pairs(
            weapon_pref[s:e], engaged[s:e], weapon_feas[s:e], health[s:e],
            op_ready[s:e]))
        counts, ref = episode_event_counts(weapon_pref[s:e], op_ready[s:e])
        for w, (nd, no) in counts.items():
            ev[w][0] += nd
            ev[w][1] += no
        for w, n in ref.items():
            refused[w] += n
    return out, ev, refused


def stream_pairs(npz_path: Path) -> list[dict[str, np.ndarray]]:
    """``stream_pairs_with_events``'s pair records only — the form the
    decode-fit candidate loop consumes."""
    return stream_pairs_with_events(npz_path)[0]


# ---------------------------------------------------------------------------
# Venue pooling + corpus pass
# ---------------------------------------------------------------------------
def collect_venue(
    runs_root: Path, run_dirs: tuple[str, ...], stream_name: str = STREAM_NAME,
) -> tuple[list[dict[str, np.ndarray]], dict[int, list[int]],
           dict[int, int], list[str]]:
    """Pool encounter records + per-weapon world-shot/decision counts + refused
    -attack counts over a venue's run dirs. Returns ``(encounters,
    event_counts, refused, missing_paths)`` -- missing files are reported, not
    silently skipped-and-forgotten."""
    encounters: list[dict[str, np.ndarray]] = []
    ev: dict[int, list[int]] = {w: [0, 0] for w in range(1, 9)}
    refused: dict[int, int] = {w: 0 for w in range(1, 9)}
    missing: list[str] = []
    for rd in run_dirs:
        p = runs_root / rd / stream_name
        if not p.exists():
            missing.append(str(p))
            continue
        enc, e, ref = stream_pairs_with_events(p)
        encounters.extend(enc)
        for w, (nd, no) in e.items():
            ev[w][0] += nd
            ev[w][1] += no
        for w, n in ref.items():
            refused[w] += n
    return encounters, ev, refused, missing


def load_human_curve(baseline_path: Path) -> dict[str, Any]:
    """Read the human baseline's own fitted curve, refusing a stale artifact.

    The pre-respec gap-binned baseline has the SAME filename and a ``curve``
    key of an entirely different shape; scoring against it silently would
    reproduce exactly the failure the respec removes, so the schema tag is
    checked here and nowhere else needs to (feedback_no_legacy_paths_
    without_request: fail loud, no compat branch)."""
    data = json.loads(Path(baseline_path).read_text())
    schema = data.get("_meta", {}).get("schema")
    if schema != SCHEMA:
        raise ValueError(
            f"{baseline_path}: human baseline schema {schema!r} != {SCHEMA!r} "
            "— this is the pre-respec gap-binned artifact (or an unknown one). "
            "Rebuild it: PYTHONPATH=src python -m qnn.human.weapon_switch_curve "
            "--collect-dir <collect>")
    return data["curve"]


def _refused_report(ev: dict[int, list[int]], refused: dict[int, int],
                    ) -> dict[str, Any]:
    """Per weapon: how many attack records the engine REFUSED (no cooldown
    started -> no shot left), against the world shots that did. A large share
    is the dry-held-weapon spin described in the module docstring, and it must
    be visible: it is the difference between measuring selection behaviour and
    measuring an ammo-starved bot re-pressing attack."""
    rows = {}
    tot_shot = tot_ref = 0
    for w in range(1, 9):
        shots, ref = int(ev[w][0]), int(refused[w])
        tot_shot += shots
        tot_ref += ref
        if shots or ref:
            rows[str(w)] = {
                "n_world_shots": shots, "n_refused": ref,
                "refused_frac": (round(ref / (shots + ref), 4)
                                 if (shots + ref) else None)}
    return {
        "by_weapon_impulse": rows,
        "n_world_shots": tot_shot, "n_refused": tot_ref,
        "refused_frac": (round(tot_ref / (tot_shot + tot_ref), 4)
                         if (tot_shot + tot_ref) else None),
        "criterion": ("a discharge record whose obs cooldown never started —"
                      " W_Attack returned without firing (dry held weapon)."
                      " Dropped before the onset gate; see world_shot_mask"),
    }


def _venue_result(
    encounters: list[dict[str, np.ndarray]], ev: dict[int, list[int]],
    refused: dict[int, int],
) -> dict[str, Any]:
    return {
        "event_stats": event_stats(ev),
        "refused_attacks": _refused_report(ev, refused),
        "pair_accounting": pair_accounting(encounters),
        "gap_distribution": gap_distribution(encounters),
        "covariate_distribution": covariate_distribution(encounters),
        "curve": fit_curve(encounters),
        "atom_by_menu": atom_by_menu(encounters),
        "mid_train_switch": mid_train_switch(encounters),
        "n_runs_pooled_encounters": len(encounters),
    }


def run(
    runs_root: Path, baseline_path: Path, out_path: Path,
) -> dict[str, Any]:
    human_curve = load_human_curve(baseline_path)

    box_enc, box_ev, box_ref, box_missing = collect_venue(runs_root, BOX_RUN_DIRS)
    real_enc, real_ev, real_ref, real_missing = collect_venue(runs_root, REAL_RUN_DIRS)
    all_enc = box_enc + real_enc
    all_ev = {w: [box_ev[w][0] + real_ev[w][0], box_ev[w][1] + real_ev[w][1]]
              for w in range(1, 9)}
    all_ref = {w: box_ref[w] + real_ref[w] for w in range(1, 9)}

    out = {
        "_meta": {
            "schema": SCHEMA,
            "contract": ("within-encounter consecutive-DECISION transition "
                         "P(w_k+1 != w_k) as a smooth function of refire "
                         "opportunities skipped, for a28rc1b h2h eval "
                         "streams. Encounter = maximal contiguous "
                         "engaged==True run within one round "
                         "(episode_offsets). Discharge events = weapon_pref>0 "
                         "ticks that the engine's own cooldown clock confirms "
                         "left the engine (world_shot_mask — a refused attack "
                         "on a dry held weapon is not a discharge), gated to "
                         "hold-train ONSETS by the SAME rule the human "
                         "baseline uses (qnn.ppo.crest_reward). "
                         "Forced (menu-changed / death) pairs excluded via "
                         "qnn.eval.humanlikeness.rc._invalidated_pairs -- "
                         "SAME predicate as the human baseline -- and dropped, "
                         "counted separately. Covariate, curve fit, CIs and "
                         "mid-train statistic computed by "
                         "qnn.human.weapon_switch_curve, unmodified."),
            "runs_root": str(runs_root),
            "baseline_path": str(baseline_path),
            "hz": HZ,
            "box_run_dirs": list(BOX_RUN_DIRS),
            "real_run_dirs": list(REAL_RUN_DIRS),
            "box_missing_files": box_missing,
            "real_missing_files": real_missing,
        },
        "human_curve": human_curve,
        "box": _venue_result(box_enc, box_ev, box_ref),
        "real": _venue_result(real_enc, real_ev, real_ref),
        "all": _venue_result(all_enc, all_ev, all_ref),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    for venue in ("box", "real", "all"):
        acct = out[venue]["pair_accounting"]
        curve = out[venue]["curve"]
        p0 = "  ".join(f"{f}={v['p']}" for f, v in
                       curve.get("p_at_zero", {}).items())
        ref = out[venue]["refused_attacks"]
        print(f"{venue}: refused_attacks={ref['n_refused']} "
              f"({ref['refused_frac']}) world_shots={ref['n_world_shots']}")
        print(f"{venue}: pairs={acct['n_pairs_total']} "
              f"curve_domain={acct['n_curve_domain']} "
              f"forced={acct['n_dropped_forced']} "
              f"mid_train={acct['n_mid_train']} "
              f"slope={curve.get('slope')} | P(switch|x=0) {p0}")
    if box_missing or real_missing:
        print(f"MISSING: {len(box_missing)} box, {len(real_missing)} real "
              "stream files (see _meta.*_missing_files)")
    print(f"Written -> {out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=Path("runs/eval"))
    ap.add_argument("--baseline", type=Path,
                    default=Path("artifacts/collect/qwd_v4d_v3vis/human_baseline/"
                                 "_weapon_switch_transition_curve.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/eval_analysis/weapon_switch_curve/"
                                 "a28rc1b_weapon_switch_transition_curve.json"))
    args = ap.parse_args()
    run(args.runs_root, args.baseline, args.out)


if __name__ == "__main__":
    main()
