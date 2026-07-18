# labeler — move-axis labeler

Bidirectional inverse dynamics for QWD/MVD.

| module | role |
|---|---|
| `data.py` | torch-free data layer: `FeatureSpec` + `build_features`, press-byte decode, mmap episode loading (`_load_split`), and `materialize_split` → flat `(N,F)` features / `(N,3)` labels / `(N,3)` op_input keep-mask + episode index |
| `model.py` | TCN architecture (`MoveLabeler`); re-exports the feature/spec/decode names from `data.py` |
| `collect.py` | labeler-only native-rate collect (CLI: `python -m qnn.labeler.collect`) |
| `train.py` | TCN training loop (CLI: `python -m qnn.labeler.train`) |
| `gbt.py` | LightGBM per-axis move labeler on the shared data layer (CLI: `python -m qnn.labeler.gbt`) |
| `seg_stats.py` | torch-free segment-parity gate (fb/lr): 20 Hz windowing helpers + onset-rate / duration-bucket-histogram comparison vs truth, with the exact `move_seg` target-derivation semantics |
| `decode.py` | transition-penalized Viterbi decode over per-frame probs (replaces argmax at apply time); `fit_switch_penalty` sweeps λ scored on the segment-parity gate |
| `matched_collect.py` | matched-pair collect: one native-rate replay → slim labeler corpus (forced-MVD features + usercmd truth) + 20 Hz qobs twin with `native_index` mapping |

Both trainers share one featurize / decode / mask implementation (`data.py`):
the TCN `ChunkedDataset` builds features per chunk; the GBT materializes the
whole split flat. The GBT path is CPU-only and torch-free.

### GBT mode

```bash
PYTHONPATH=src python -m qnn.labeler.gbt \
    --data-dir artifacts/collect/qwd_labeler --tag qwd_v1 \
    --num-threads 8 --max-train-frames 3000000
```

Fits one multiclass booster per axis (fb/lr/ud), predicts val in one batched
call per axis (no per-episode loop), and reports per-axis frame accuracy plus a
20 Hz relabel-quality table (windowed-union downsample to 20 Hz, per-axis
agreement + switch-rate vs truth). Velocity-lag features (episode-boundary
clamped) are on by default — they give the frame-local GBT the momentum context
the TCN gets from its convolutions, lifting fb/lr from the ~80%/75% single-frame
ceiling to ~90% (`--no-temporal-lags` to disable). `--lag-look` additionally
lags the 3 look (per-emit view delta) columns — the rotation rhythm air-strafe
lr presses sync with — worth ~+1.3pp lr on the MVD domain
(`move_gbt_mvd_matched320_looklag` vs `move_gbt_mvd_matched320`; data scale
3M→9.2M was flat, `_full`). Outputs boosters + `meta.json`
to `artifacts/labeler/move_gbt_<tag>/`.

### Segment-parity gate (fb/lr)

The a25 `move_seg` head derives its (onset, duration-bucket) training targets
on the fly from per-frame move classes — so a relabeled corpus is judged on
**segment statistics**, not just frame accuracy: an isolated frame flip inside
a long hold creates two false onsets and shatters it into short segments,
inflating onset rate and shifting duration mass into the short Fibonacci
buckets (which the commitment-decode `dur_tilt` is calibrated on). A
~95%-frame-accurate but flickery labeler measures onset ×1.8 / durTV 0.40 on
this gate.

Both trainers report it per epoch/fit on the 20 Hz val streams via
`qnn.labeler.seg_stats` (semantics pinned to `derive_segment_targets` by
`tests/test_seg_stats_parity.py`): `onset_x` = predicted/truth onset rate
(flicker shows up as >1) and `dur_tv` = total-variation distance between
duration-bucket histograms. Full histograms (incl. per-class) land in the GBT
`meta.json` under `seg20` and in the TCN checkpoints. A labeler must hold
onset_x ≈ 1 and low dur_tv **before** its relabels feed `move_seg` training.

Two supporting pieces:

* **Per-episode 20 Hz strides.** Demos record at their native client rate
  (77 Hz and 60 Hz both common), so a global `round(native_hz/20)` stride
  puts mixed corpora in the wrong duration units. On matched corpora the
  gate derives per-episode strides from the qobs twin's `native_index`
  (`data.matched_episode_strides`); plain corpora fall back to `--native-hz`.
* **Decode fit (`decode.py`).** Frame-wise argmax is what inflates onsets;
  the transition-penalized Viterbi decode (uniform switch penalty λ, exact,
  per episode; λ=0 ≡ argmax) is fit per labeler by sweeping λ on the gate.
  The GBT report runs the fit automatically for fb/lr and stores the full
  frontier in `meta.json` under `decode_fit`; the definitive per-model λ is
  the APPLY-space refit (`apply.py --fit`, stored under `decode_fit_apply`).
  Apply-time relabels must use the fitted λ, not argmax.

### Reference model: `move_gbt_mvd_matched1220_final`

Operating relabeler on the 1,140-demo matched corpus (`qwd_matched`,
forced-MVD features vs QWD usercmd truth): `--lag-look`, 8M train frames,
2400 rounds — fb 90.0% / lr 88.2% frame acc. Scaling study (meta.json of
the `mvd_matched*` runs): within-demo frames and demo diversity are FLAT at
fixed capacity; boosting rounds dominate (+1.3pp over 300→2400, log-linear,
halving gains), data adds ~+0.25pp at high capacity. Held-out end-to-end
parity (λ fit on train via `apply --fit`; fb 0.75 / lr 0.25), measured in
the training representation on val:

| axis | 20 Hz agree | onset ratio | duration TV |
|---|---|---|---|
| fb | 89.50% | ×1.011 | 0.013 |
| lr | 87.69% | ×1.034 | 0.014 |

Argmax at the same accuracy measures onset ×1.24/×1.12 — the decode fit is
what closes segment parity. Sidecars for both splits live next to the qobs
shards (`*_act_move_mvdsynth.npy` + `relabel_meta_*.json` provenance).

### Apply (`apply.py`)

The relabel-apply path runs against a matched collect
(`matched_collect.py`, forced-MVD features + usercmd truth + `native_index`
mapping): GBT probs → Viterbi at the fitted λ → exact `native_index` lookup
resamples each 20 Hz qobs frame → rewrites the fb/lr press-byte bits
(ud/jump/attack untouched) into `*_act_move_mvdsynth.npy` sidecars.

```bash
# fit λ in APPLY space on train (point-sampled 20 Hz comparison — the
# training representation; the windowed gate fit is only a proxy):
PYTHONPATH=src python -m qnn.labeler.apply \
    --matched-dir artifacts/collect/qwd_matched \
    --model artifacts/labeler/move_gbt_mvd_matched320_looklag \
    --split precomputed_train --fit
# held-out end-to-end measurement, then --write to emit sidecars:
PYTHONPATH=src python -m qnn.labeler.apply ... --split precomputed_val [--write]
```

`--fit` stores `decode_fit_apply` (chosen λ + full frontier + fit split)
into the model's `meta.json`; measurement-only runs report per-axis 20 Hz
agreement + the segment gate in the actual training representation.

See:
- `project_seq_labeler_axes` — why the labeler isn't a mini-policy
- `project_qwd_rate_distribution` — corpus design (bc_included + trick at >=70 Hz)
- `scripts/move_inference_bakeoff.md` — measured baselines we beat
