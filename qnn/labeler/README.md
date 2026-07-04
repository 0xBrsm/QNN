# labeler — move-axis labeler

Bidirectional inverse dynamics for QWD/MVD.

| module | role |
|---|---|
| `data.py` | torch-free data layer: `FeatureSpec` + `build_features`, press-byte decode, mmap episode loading (`_load_split`), and `materialize_split` → flat `(N,F)` features / `(N,3)` labels / `(N,3)` op_input keep-mask + episode index |
| `model.py` | TCN architecture (`MoveLabeler`); re-exports the feature/spec/decode names from `data.py` |
| `collect.py` | labeler-only native-rate collect (CLI: `python -m qnn.labeler.collect`) |
| `train.py` | TCN training loop (CLI: `python -m qnn.labeler.train`) |
| `gbt.py` | LightGBM per-axis move labeler on the shared data layer (CLI: `python -m qnn.labeler.gbt`) |

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
ceiling to ~90% (`--no-temporal-lags` to disable). Outputs boosters + `meta.json`
to `artifacts/labeler/move_gbt_<tag>/`.

Apply (relabel a forced-MVD collect) is intentionally deferred — it needs the
native-rate forced-MVD collect path and tightened C-rule fire/jump.

See:
- `project_seq_labeler_axes` — why the labeler isn't a mini-policy
- `project_qwd_rate_distribution` — corpus design (bc_included + trick at >=70 Hz)
- `scripts/move_inference_bakeoff.md` — measured baselines we beat
