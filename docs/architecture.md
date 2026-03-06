# Architecture

## Pipeline
1. `collect`: Replay demos and write telemetry/packet/summaries + map features.
2. `validate_packets`: Detect packet/tick mismatches for data integrity checks.
3. `train_bc`: Train supervised policy with weighted multi-head action loss.
4. `train_rl`: Fine-tune policy with PPO in a deterministic heading-aware symbolic navigation environment.
5. `eval`: Greedy and/or sampled policy evaluation with sequential or held-out randomized starts.
6. `engine.native_bridge`: Optional native process boundary for future engine-backed rollouts.

## Data Contracts
- `TelemetryTickV1`: per-tick symbolic observation + action label + termination.
- `MapFeaturesV1`: per-region map context and distance-to-goal precompute.
- `PacketEventV1`: decoded packet metadata keyed by episode/tick estimate.
- `EpisodeSummaryV1`: episode-level completion/time/return summary.

## Model
- Feature encoder:
  - Symbolic observation vector (20 dims).
  - Shared 2-layer PyTorch MLP trunk (`tanh`) trained by both BC and PPO.
- Policy heads: `move`, `strafe`, `turn`, `use`.
- Value head for PPO baseline.

## Runtime Split
- Python owns orchestration, corpus processing, offline training loops, and reporting.
- PyTorch owns the policy/value model, optimizer state, checkpoint serialization, and accelerator placement.
- The publishable repo uses a lightweight dev environment for editing/orchestration and a separate AMD ROCm training container under `docker/training/` for local GPU execution.
- Native engine workers are expected to own real-time simulation and emit observations/actions over a JSON-over-stdio boundary.

## Determinism
- Fixed seeds for Python/NumPy.
- Fixed tick replay order for demos.
- Deterministic map graph generation from parsed region IDs.
- Goal completion requires `use` on a terminal `trigger_changelevel` region.
