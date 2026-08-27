# Model Versioning and Promotion Path

Canonical naming and promotion rules for trained models, from training run to
released generation. The name encodes exactly how far an artifact has been
promoted — nothing earns a name segment before passing the gate that segment
represents.

## Name Forms

| Form | Example | Means | Earned by |
|------|---------|-------|-----------|
| Run name | `head_probe_atlas_awposw_seed43` | A trained base checkpoint, candidacy undecided | Training completed |
| `a<T>rc<N>` | `a26rc2` | Release-candidate base model N of tier T | Decode fit passed its gates |
| `a<T>rc<N><letter>` | `a26rc2a` | A deployed decode config of that base model | Actually deployed to the live share (g4) |
| `a<T>rc<N>` (alias) | `a26rc2` | Best deployed letter of the rc family | Letter promoted, drops the letter |
| `a<T>` (bare) | `a25` | The generation's released model | Best rc of the line, promoted at line close |

## Promotion Pipeline

1. **Train a base model.** It is named by its run id (for example
   `head_probe_atlas_awposw_seed43`) and nothing else. Candidate checkpoints
   have no rc name — "candidate" status is not encoded in a name.
2. **Run decode fit** (`qnn.decode_fit`) on the checkpoint. While the fit is
   in flight, refer to the model by its run name.
3. **Fit passes its gates → the checkpoint becomes the tier's next rc
   numeral** (`a26rc1`, `a26rc2`, ...). A failed fit consumes no number: the
   checkpoint stays a run name and the numeral remains available. The rc
   numeral always identifies a base model — a different checkpoint is always
   a different rc number, and refits/guard tweaks on the same checkpoint
   never bump it.
4. **Deploy → letter increment.** Each deployed decode config of an rc gets
   the next letter (`a26rc2a`, `a26rc2b`, ...). Letters count deploys, not
   emits: a staged config that is superseded before it ships is re-emitted
   under the same letter. The version string is baked into the decode config
   (it keys the freeplay waves), so the letter is stamped at emit time — but
   it only advances once the previous letter actually reached the live share (g4).
5. **Best letter drops the letter.** The best deployed config within an rc
   family may be referred to (and deployed) as the bare rc name: `a26rc2`
   means "the settled best of the `a26rc2x` family."
6. **Best rc drops the rc.** When the tier's line closes, the best rc is
   promoted to the bare tier name (`a25`, `a26`) and is that generation's
   released model.

## Rules

- **No rc name before a passed decode fit.** A checkpoint that has not
  passed `qnn.decode_fit` gates is not a release candidate and must not
  carry an rc numeral — in reports, staged filenames, or conversation.
  Decode fit is the candidacy gate ([Decode Config Defaults](decode-config-defaults.md)
  describes what the fit emits).
- **rc number = base model.** New checkpoint → next numeral. Same
  checkpoint, new decode/guard params → same numeral, letter territory.
- **Letters = deploys.** No letter until the config ships. Undeployed
  refits re-emit under the same letter they replace.
- **Failed fits burn nothing.** The numeral is only consumed on gate pass.
- **Aliases are promotions, not synonyms.** Bare `rcN` and bare `aT` are
  earned by being the settled best of their family/line, not shorthand for
  the latest attempt.
- **`--force` overrides a failed gate, loudly.** `qnn.decode_fit --force`
  promotes a config despite a gate FAIL, stamping `forced: true` (and, on
  `--assign-rc`, `rc_forced: true`) into the promoted provenance with the
  operator's justification recorded alongside it. This is a deliberate,
  visible escape hatch for a judgment call (for example `a26rc1b`: the
  reactivity gate failed by a margin judged inside measurement noise) — it
  is not silent, and it does not relax the numeral/letter rules above: a
  forced promotion still earns its rc numeral and deploy letter the same
  way a clean pass does, it just carries the override on its face.

## Process (tooling)

The rules are enforced by `qnn.decode_fit`, not convention:

1. **Fits run under provisional versions.** With no `--version`, the fit
   stages and promotes as `decode.prov-<run_id>-<skill>.json` (for example
   `decode.prov-head_probe_atlas_awposw_seed43-p90.json`). Passing an rc or
   bare-tier name as `--version` is refused at launch; `--version` remains
   available for ablation-arm labels only.
2. **Gate pass promotes the provisional config** (staged → promoted, the
   existing `promote_decode_config` guard — no emit on gate FAIL). The
   checkpoint has now earned its rc numeral.
3. **`--assign-rc` marks the promotion:**

   ```bash
   python -m qnn.decode_fit --run-dir runs/head_probe/<run> --assign-rc a26rc2a
   ```

   This copies the run's latest unassigned promoted config to
   `decode.<rc>.json` with `version` re-stamped and the assignment recorded
   in provenance (`rc_source`, `provisional_version`, `rc_assigned_utc`);
   the provisional source is stamped `rc_assigned` in place. It refuses
   staged configs, gate-FAIL or force-promoted sources, double assignment,
   and bare `rcN`/bare-tier targets (those are promotion aliases, never
   emitted filenames). An existing `decode.<rc>.json` is only overwritten
   under `--replace` — the re-emit-same-letter path for a superseded
   never-deployed artifact; check the live share (g4) first.
4. **Deploy consumes the rc-named config.** The letter names the deploy
   slot and only advances once the previous letter actually shipped.

## Worked Lineage (a25)

- `a25rc1` = `head_probe_move_seg_a1p_pmh60_seed17`; `a25rc2` =
  `head_probe_afmask_p2iso_e12_pk018_seed17`; `a25rc3` =
  `head_probe_p3d_jumpv2_tlfix_nc_seed43`. Three rc numbers, three distinct
  checkpoints.
- `a25rc1a/b/c`, `a25rc2a/b`, `a25rc3a/b/c` were deployed decode/guard
  iterations within their checkpoint's family.
- `a25rc3c` was the last deploy of the line (gate PASS, 2026-07-17); the
  line closed and a25 merged to `main` as release `0.28.0` — `a25` bare is
  that generation's released model.

## Worked Lineage (a26, in flight)

- `a26rc1` = `head_probe_atlas_awposw_seed43` (checkpoint `best_20260717-5t9kvm`,
  the full 72-yaw atlas). `a26rc1a/b/c` are deployed decode iterations within
  that family; `a26rc1b` is the `--force`-promoted config (reactivity gate
  FAIL judged inside measurement noise — see the `--force` rule above);
  `a26rc1c` is a hand-derived, byte-identical-params re-emit of `a26rc1b`
  documenting same-day engine/decode riders (continuous-fire hold-tail, RL
  self-splash atlas-quantization fix, ammo-lockout override, dormant
  convergence-gated crest hold) in its provenance.
- `a26rc2` = `head_probe_atlas24x11_awposw_seed43` (checkpoint
  `best_20260722-98wtxv`, the finalized 24×11 packed atlas) — a distinct
  checkpoint from `a26rc1`, hence the new numeral. `a26rc2a` is its first
  deploy: a clean gate PASS with no `--force`, fit under the reconciled
  split-vector attack law and calibration-family cadence pins.

## a26 Clean Sweep (2026-07-18)

The a26 line briefly predated these rules and carried two mis-assignments:
"a26rc1" named the `pvsfix` checkpoint before any fit ran, and "a26rc2a"
was stamped into the `awposw` staged config whose fit report showed
`gate_passed=False`. Because nothing rc-named ever passed a fit or
deployed, every a26 run record was rewritten to conform on 2026-07-18: rc
strings scrubbed from run narratives (checkpoints are referred to by run
name — `pvsfix`, `awposw`), the staged config restamped to its provisional
version, and the failed fit's freeplay wave cache (keyed to the old config
sha, permanently unreachable) pruned. **No a26 numeral is burned: the
first checkpoint to pass decode fit earns `a26rc1`.** If it terminally fails, rc2 dies with its stamped
  artifacts — the next passing checkpoint takes rc3, since reclaiming a
  numeral that already appears in artifacts would create the same collision
  these rules exist to prevent. Reclaiming applies only to numerals that
  were never stamped anywhere.
