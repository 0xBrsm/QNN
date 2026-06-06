"""GBT probe for target slot prediction.

Two modes:

  --mode default   Standard slot ordering preserved. Direct comparison to
                   the TCN probe. Tests whether a non-temporal flat-feature
                   GBT can match the TCN's per-frame signal (since the TCN
                   ablation showed temporal context contributes little).

  --mode randomize Per-frame random permutation of the 16 slot positions
                   (target re-mapped). Breaks the engine's slot-0 prior so
                   the only available signal is per-slot features. Measures
                   how much of the "97.5% slot-0 baseline" was the engine's
                   ordering vs. the per-slot feature signal itself.

Per-frame features (flat):
  16 slots × 19 scalars + 16 type ids (one-hot) + 16 enemy flags
    + 16 self scalars + 3 movement one-hot + 16 weapon one-hot
    + 3 look + 1 fire

Multiclass LightGBM with class weighting (slot 0 down-weighted).

Usage:
    PYTHONPATH=src python -m qnn.labeler.probes.target_head_gbt \
        --data-dir artifacts/collect/qwd \
        --output runs/probe/gbt_default \
        --mode default --n-train 500000 --n-estimators 300
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qnn.vocab import MAX_TOKEN_OBJECTS, TOKEN_ACTOR
from qnn.labeler.probes.target_head_probe import (
    N_SLOTS, N_SLOT_SCALARS, N_SELF_SCALARS, N_TYPE_VOCAB, N_WEAPON_VOCAB,
    _Shard, _load_split, _enemy_flag,
)

_ACTOR_TEAM_OFFSET = 16


@dataclass
class FeatureMatrix:
    X: np.ndarray            # (N, F) float32
    y: np.ndarray            # (N,)   int   target slot (0..15)
    target_was_zero: np.ndarray  # (N,) bool — was the *original* (pre-permute) target slot 0
    feature_names: list[str]


_TEAM_SCALAR_OFFSET = 16  # team scalar index inside the 19-d per-slot vector


def _flat_feature_names(drop_masks: bool = False) -> list[str]:
    names: list[str] = []
    n_slot_scal = N_SLOT_SCALARS - (1 if drop_masks else 0)
    for s in range(N_SLOTS):
        for k in range(N_SLOT_SCALARS):
            if drop_masks and k == _TEAM_SCALAR_OFFSET:
                continue
            names.append(f"slot{s:02d}_scal{k:02d}")
    if not drop_masks:
        for s in range(N_SLOTS):
            names.append(f"slot{s:02d}_enemy")
        for s in range(N_SLOTS):
            for t in range(N_TYPE_VOCAB):
                names.append(f"slot{s:02d}_type{t:02d}")
    for k in range(N_SELF_SCALARS):
        names.append(f"self_scal{k:02d}")
    for t in range(3):
        names.append(f"self_mvmt{t}")
    for w in range(N_WEAPON_VOCAB):
        names.append(f"self_weapon{w:02d}")
    for k in range(3):
        names.append(f"look{k}")
    names.append("fire")
    return names


def _build_flat_features(
    shards: list[_Shard],
    rng: np.random.Generator,
    randomize: bool,
    max_per_shard: int | None = None,
    only_labeled: bool = True,
    drop_masks: bool = False,
) -> FeatureMatrix:
    """Build a flat feature matrix from a list of shards.

    If randomize=True, per-frame random permutation of the 16 slot positions
    is applied to both features and target slot.

    If drop_masks=True, removes all features that explicitly identify a slot
    as enemy or actor:
      - per-slot enemy flag
      - per-slot type one-hot
      - per-slot team scalar (offset 16 in entity_scalars_raw)
    """
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    twz: list[np.ndarray] = []
    names = _flat_feature_names(drop_masks=drop_masks)
    F = len(names)

    for sh_idx, sh in enumerate(shards):
        T = sh.target.shape[0]
        tgt = np.asarray(sh.target, dtype=np.int64)
        if only_labeled:
            valid = np.flatnonzero(tgt != -100)
        else:
            valid = np.arange(T)
        if max_per_shard is not None and len(valid) > max_per_shard:
            valid = rng.choice(valid, size=max_per_shard, replace=False)
            valid.sort()
        if len(valid) == 0:
            continue
        n = len(valid)

        # Pull and convert.
        es = np.asarray(sh.entity_scalars[valid], dtype=np.float32)   # (n,16,19)
        et_raw = np.asarray(sh.entity_types[valid], dtype=np.int64)    # (n,16)

        # Apply token_mask if present on this shard: zero scalars, set
        # types to -1, and skip rows whose target slot got masked out.
        if sh.token_keep is not None:
            keep_chunk = np.asarray(sh.token_keep[valid], dtype=bool)
            drop = ~keep_chunk
            if drop.any():
                es = es.copy(); es[drop] = 0.0
                et_raw = et_raw.copy(); et_raw[drop] = -1
                rows = np.arange(et_raw.shape[0])
                slot_idx = np.clip(tgt[valid], 0, N_SLOTS - 1)
                still_valid = (tgt[valid] == -100) | (et_raw[rows, slot_idx] != -1)
                if not still_valid.all():
                    es = es[still_valid]
                    et_raw = et_raw[still_valid]
                    rows_keep = np.flatnonzero(still_valid)
                    valid = valid[rows_keep]
                    n = len(valid)
                    if n == 0:
                        continue

        en = _enemy_flag(et_raw.astype(np.int8), es).astype(np.float32)   # (n,16)
        et = np.clip(et_raw, 0, N_TYPE_VOCAB - 1)
        ss = np.asarray(sh.self_scalars[valid], dtype=np.float32)      # (n,16)
        mi = np.asarray(sh.movement_id[valid], dtype=np.int64).reshape(-1)
        wi = np.asarray(sh.weapon_id[valid], dtype=np.int64).reshape(-1).clip(0, N_WEAPON_VOCAB - 1)
        lk = np.asarray(sh.look[valid], dtype=np.float32)              # (n,3)
        fr = np.asarray(sh.fire[valid], dtype=np.float32)              # (n,)
        y  = tgt[valid]
        was_zero = (y == 0)

        if randomize:
            perm = np.argsort(rng.random((n, N_SLOTS)), axis=1)  # (n,16) permutations
            # Apply to per-slot arrays.
            row_idx = np.arange(n)[:, None]
            es = es[row_idx, perm]
            et = et[row_idx, perm]
            en = en[row_idx, perm]
            # Map target slot through the permutation: new_slot[old_slot] = pos
            # of old_slot in perm. perm[i, new_slot] = old_slot, so we need the
            # inverse permutation.
            inv = np.argsort(perm, axis=1)
            y = inv[row_idx[:, 0], y]

        # One-hot for movement (3-way) and weapon (16-way).
        movement_oh = np.zeros((n, 3), dtype=np.float32)
        movement_oh[np.arange(n), 0] = (mi == 0).astype(np.float32)
        movement_oh[np.arange(n), 1] = (mi == 1).astype(np.float32)
        movement_oh[np.arange(n), 2] = (mi >= 2).astype(np.float32)
        weapon_oh = np.zeros((n, N_WEAPON_VOCAB), dtype=np.float32)
        weapon_oh[np.arange(n), wi] = 1.0

        if drop_masks:
            # Drop team scalar from per-slot scalars; drop enemy flag and type one-hot.
            es_kept = np.delete(es, _TEAM_SCALAR_OFFSET, axis=-1)  # (n, 16, 18)
            feat = np.concatenate([
                es_kept.reshape(n, -1),  # (n, 16*18)
                ss,                       # (n, 16)
                movement_oh,             # (n, 3)
                weapon_oh,               # (n, 16)
                lk,                       # (n, 3)
                fr[:, None],             # (n, 1)
            ], axis=1).astype(np.float32)
        else:
            # One-hot for types (16-way per slot).
            type_oh = np.zeros((n, N_SLOTS, N_TYPE_VOCAB), dtype=np.float32)
            type_oh[np.arange(n)[:, None], np.arange(N_SLOTS)[None, :], et] = 1.0
            feat = np.concatenate([
                es.reshape(n, -1),       # (n, 16*19)
                en,                       # (n, 16)
                type_oh.reshape(n, -1),  # (n, 16*16)
                ss,                       # (n, 16)
                movement_oh,             # (n, 3)
                weapon_oh,               # (n, 16)
                lk,                       # (n, 3)
                fr[:, None],             # (n, 1)
            ], axis=1).astype(np.float32)
        assert feat.shape[1] == F, f"feature dim {feat.shape[1]} != {F}"

        Xs.append(feat)
        ys.append(y.astype(np.int64))
        twz.append(was_zero)

    if not Xs:
        return FeatureMatrix(X=np.zeros((0, F)), y=np.zeros((0,), dtype=np.int64),
                              target_was_zero=np.zeros((0,), dtype=bool),
                              feature_names=names)
    return FeatureMatrix(
        X=np.concatenate(Xs, axis=0),
        y=np.concatenate(ys, axis=0),
        target_was_zero=np.concatenate(twz, axis=0),
        feature_names=names,
    )


def _eval(
    label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    target_was_zero: np.ndarray,
) -> dict:
    """Bucket eval based on the ORIGINAL (pre-randomization) target slot.

      bucket A: original target was slot 0 (engine and labeler agreed before
                 randomization). These are the trivial/easy cases.
      bucket B: original target was slot != 0 (engine and labeler disagreed
                 before randomization).
    """
    n = y_true.shape[0]
    correct = y_true == y_pred

    is_a = target_was_zero
    is_b = ~target_was_zero

    overall = 100 * correct.sum() / max(n, 1)
    a_acc = 100 * correct[is_a].sum() / max(is_a.sum(), 1)
    b_acc = 100 * correct[is_b].sum() / max(is_b.sum(), 1)

    # Threshold-curve eval — practical inference policy.
    argmax_p = y_proba[np.arange(n), y_pred]

    taus = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    curve = {}
    for tau in taus:
        # In default mode, "override" means model picks non-zero and crosses tau.
        # In randomized mode, the slot-0 prior is meaningless; just report
        # acc-at-tau (predict-as-model when confident, else "don't know").
        confident = argmax_p >= tau
        # Acc among confident predictions.
        if confident.any():
            tau_acc = 100 * (correct & confident).sum() / confident.sum()
            tau_cov = 100 * confident.sum() / n
        else:
            tau_acc = 0.0
            tau_cov = 0.0
        curve[tau] = {"acc": float(tau_acc), "coverage": float(tau_cov),
                       "confident": int(confident.sum())}

    out = {
        "label": label,
        "n": int(n),
        "overall_acc": float(overall),
        "bucket_A_acc": float(a_acc), "bucket_A_n": int(is_a.sum()),
        "bucket_B_acc": float(b_acc), "bucket_B_n": int(is_b.sum()),
        "threshold_curve": {str(k): v for k, v in curve.items()},
    }

    print(f"\n=== {label} ===")
    print(f"n={n}  overall={overall:.2f}%")
    print(f"  bucket A (orig target=0)   n={int(is_a.sum()):7d}  acc={a_acc:.2f}%")
    print(f"  bucket B (orig target!=0)  n={int(is_b.sum()):7d}  acc={b_acc:.2f}%")
    print(f"threshold-coverage curve:")
    print(f"  {'tau':>5}  {'acc%':>6}  {'cov%':>6}  {'n_conf':>10}")
    for tau, rec in curve.items():
        print(f"  {tau:>5.2f}  {rec['acc']:>5.2f}  {rec['coverage']:>5.2f}  {rec['confident']:>10d}")
    return out


def _train_gbt(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    slot0_weight: float,
    n_estimators: int,
    num_leaves: int,
    learning_rate: float,
    n_jobs: int,
):
    import lightgbm as lgb

    # Per-sample weight: slot0 -> slot0_weight, else 1.0.
    w_train = np.where(y_train == 0, slot0_weight, 1.0).astype(np.float32)

    # Use early stopping on val multi-logloss for sanity.
    print(f"training lightgbm: {X_train.shape[0]:,} train, {X_val.shape[0]:,} val, "
          f"{X_train.shape[1]} features, {N_SLOTS} classes")

    train_ds = lgb.Dataset(X_train, label=y_train, weight=w_train)
    val_ds   = lgb.Dataset(X_val,   label=y_val,   reference=train_ds)
    params = {
        "objective": "multiclass",
        "num_class": N_SLOTS,
        "metric": "multi_logloss",
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbosity": -1,
        "num_threads": n_jobs,
    }
    booster = lgb.train(
        params=params,
        train_set=train_ds,
        valid_sets=[val_ds],
        num_boost_round=n_estimators,
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=True),
            lgb.log_evaluation(period=20),
        ],
    )
    return booster


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output",   type=Path, required=True)
    p.add_argument("--mode", choices=["default", "randomize"], required=True)
    p.add_argument("--no-masks", action="store_true",
                   help="drop enemy flag, type one-hot, and team scalar — model "
                        "must identify targets from raw physical scalars alone")
    p.add_argument("--n-train", type=int, default=500_000,
                   help="cap on total training samples (subsamples shards uniformly)")
    p.add_argument("--n-val", type=int, default=None,
                   help="cap on val samples (default: use all)")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--slot0-weight", type=float, default=0.15,
                   help="per-sample weight on slot-0 training rows (1.0 for randomize mode)")
    p.add_argument("--max-shards-train", type=int, default=None)
    p.add_argument("--max-shards-val", type=int, default=None)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--n-jobs", type=int, default=os.cpu_count() or 8)
    p.add_argument("--token-mask", type=str, default=None,
                   help="Path to a JSON file with a token_mask predicate "
                        "(qnn.bc.token_filter spec). None = no mask.")
    args = p.parse_args()
    token_mask = (json.loads(Path(args.token_mask).read_text())
                  if args.token_mask else None)

    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    randomize = (args.mode == "randomize")
    if randomize:
        # Slot-0 weighting doesn't apply when the slot index is randomized.
        args.slot0_weight = 1.0

    print(f"mode={args.mode}  no_masks={args.no_masks}  seed={args.seed}  "
          f"n_train={args.n_train}  slot0_weight={args.slot0_weight}  "
          f"n_estimators={args.n_estimators}")

    t0 = time.time()
    if token_mask:
        print(f"token_mask: {token_mask}")
    train_shards = _load_split(args.data_dir / "precomputed_train",
                                args.max_shards_train, token_mask=token_mask)
    val_shards   = _load_split(args.data_dir / "precomputed_val",
                                args.max_shards_val,   token_mask=token_mask)
    print(f"loaded {len(train_shards)} train shards, {len(val_shards)} val shards "
          f"in {time.time()-t0:.1f}s")

    # Subsample per-shard so total train ≈ n_train.
    per_shard_train = max(1, args.n_train // max(len(train_shards), 1))
    t0 = time.time()
    fm_train = _build_flat_features(
        train_shards, rng=rng, randomize=randomize,
        max_per_shard=per_shard_train, only_labeled=True,
        drop_masks=args.no_masks,
    )
    print(f"built train matrix {fm_train.X.shape} in {time.time()-t0:.1f}s")

    val_per_shard = (args.n_val // max(len(val_shards), 1)) if args.n_val else None
    t0 = time.time()
    fm_val = _build_flat_features(
        val_shards, rng=rng, randomize=randomize,
        max_per_shard=val_per_shard, only_labeled=True,
        drop_masks=args.no_masks,
    )
    print(f"built val matrix {fm_val.X.shape} in {time.time()-t0:.1f}s")

    # Target distribution sanity.
    for name, fm in [("train", fm_train), ("val", fm_val)]:
        u, c = np.unique(fm.y, return_counts=True)
        top = ", ".join(f"slot{int(uu)}={int(cc)} ({100*cc/fm.y.size:.2f}%)"
                        for uu, cc in zip(u[:5], c[:5]))
        print(f"  {name} target dist (top 5): {top}")

    # Train GBT.
    t0 = time.time()
    booster = _train_gbt(
        fm_train.X, fm_train.y, fm_val.X, fm_val.y,
        slot0_weight=args.slot0_weight,
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_jobs=args.n_jobs,
    )
    print(f"trained in {time.time()-t0:.1f}s")
    booster.save_model(str(args.output / "lgb_model.txt"))

    # Eval.
    t0 = time.time()
    y_proba_val = booster.predict(fm_val.X)
    y_pred_val  = y_proba_val.argmax(axis=1)
    print(f"predicted val in {time.time()-t0:.1f}s")

    results = _eval(
        label=f"{args.mode} (val)",
        y_true=fm_val.y, y_pred=y_pred_val, y_proba=y_proba_val,
        target_was_zero=fm_val.target_was_zero,
    )
    (args.output / "results.json").write_text(json.dumps(results, indent=2))

    # Feature importance (top 30 by gain).
    try:
        imp = booster.feature_importance(importance_type="gain")
        order = np.argsort(imp)[::-1][:30]
        print(f"\nTop 30 features by gain:")
        for r in order:
            print(f"  {fm_train.feature_names[r]:>30s}  gain={imp[r]:.2f}")
    except Exception as e:
        print(f"(feature importance unavailable: {e})")


if __name__ == "__main__":
    main()
