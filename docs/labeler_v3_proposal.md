# Target Labeler v3 Proposal

This document proposes a third BC target labeler for Quake imitation learning.  The key change is to stop treating weapon awareness as a cone-width problem.  v3 should treat each fire as a weapon-specific shot with delayed outcomes, then decode the demonstrator's engagement target from damage, events, projectile geometry, and temporal follow-through.

The output contract remains unchanged: return one slot index per frame, or `TARGET_IGNORE` when no target should train the supervised pointer head.

## Summary

| Area | Current v2 final | Proposed v3 |
|------|------------------|-------------|
| Unit of attribution | Fire frame | Shot event plus evidence window |
| Primary signal | Sticky cone membership | Outcome-weighted engagement evidence |
| Weapon role | Lead center, range gates, small geometric floors | Hit windows, ray/projectile/splash likelihood, cadence |
| Smoothing | Sticky pid, then engagement extension | Shot-level Viterbi, then evidence-aware frame expansion |
| Expected gain | Small, because opt3 cone remains dominant | Larger, because weapons change the scoring objective |
| Main risk | Low ceiling | More complexity and validation work |

## Current v2 Assessment

The current labeler has strong foundations.

- It fixed the original cone-jitter artifact with a sticky Schmitt trigger.  The opt3 acquire/release constants are explicit in `qnn/bc/target_labeler.py:44` through `qnn/bc/target_labeler.py:49`.
- It releases sticky when the current pid leaves the stream between fires, matching the engine more closely (`qnn/bc/target_labeler.py:398` through `qnn/bc/target_labeler.py:402`).
- It groups same-pid fires by stream continuity (`qnn/bc/target_labeler.py:451` through `qnn/bc/target_labeler.py:469`) and expands engagements into per-frame labels (`qnn/bc/target_labeler.py:477` through `qnn/bc/target_labeler.py:499`).
- It is cheap enough to run during collect.  The collector derives target labels inside each sub-episode at `qnn/bc/collect.py:574` through `qnn/bc/collect.py:577`.

The limitation is that v2 remains a cone labeler.  The weapon-aware path defines per-weapon constants and range gates (`qnn/bc/target_labeler.py:78` through `qnn/bc/target_labeler.py:99`), but the final cone thresholds deliberately use opt3 adaptive geometry as the base (`qnn/bc/target_labeler.py:218` through `qnn/bc/target_labeler.py:238`).  The lead-corrected center is also guarded by `max(cos_lead, cos_current)` (`qnn/bc/target_labeler.py:345` through `qnn/bc/target_labeler.py:354`), which is correct for under-leaded shots but reduces how much lead correction can change.

That explains the measured result: weapon awareness adds only marginal signal because the labeler still asks "which pid is nearest this aim ray at one fire frame?"  A better weapon-aware labeler should ask "which pid did this weapon-specific shot hit, damage, threaten, or track over its natural time window?"

The existing quality test should remain the primary diagnostic.  It classifies within-segment switches as jitter at `<= 8 deg` and legit at `>= 12 deg` (`scripts/analysis/switch_quality.py:8` through `scripts/analysis/switch_quality.py:18`).  v3 should improve that ratio, not just increase label count.

## Why More Cone Tuning Is Unlikely To Win

Cone membership is a weak proxy for target intent:

- LG is an instantaneous 600u beam and should be scored over beam tracking.
- SG/SSG are instantaneous ray bundles with spread.
- NG/SNG are delayed line projectiles.
- RL is a delayed line projectile with splash.
- GL is delayed, ballistic, often bounced, and often used for area denial.

Those differences are temporal and outcome-based.  Turning them into a single angular threshold loses the part of weapon identity that matters most.  The v2 experiments already showed the ceiling: opt3 fixed co-angular jitter, sigma-times-K widened cones and added jitter, and v2 final reverted width to opt3 while keeping only lead center plus range gates.

## Proposed Design

Call the design **Outcome-Weighted Engagement Labeling**.

Pipeline:

1. Build shot records from action fire/weapon streams.
2. Score each candidate pid for each shot using evidence tiers.
3. Decode the shot-pid sequence with sticky transition costs.
4. Expand decoded engagements into frame labels.

The labeler still uses opt3 as a fallback prior, but opt3 is no longer the center of the algorithm.

## Data Dependencies

| Evidence | Status | Use |
|----------|--------|-----|
| `act_attack`, `act_look`, `act_weapon` | Available | Shot records |
| actor rel/vel/half extents/pid | Available | Geometry and slot maps |
| entity events | Available in token spec | Pain/death/event hints |
| projectile tokens | Available in token spec | Optional flight confirmation |
| training damage records | Available in training sidecar | Strong hit attribution |
| QWD `svc_damage` victim extraction | Not currently wired | Future strongest demo signal |

The token spec already exposes event arrays: `entity_event_actions`, `entity_event_sources`, and `entity_event_counts` (`token-spec.md:90` through `token-spec.md:92`).  Each event is an `(action_id, source_id)` pair (`token-spec.md:213` through `token-spec.md:214`).

Training sidecars already include exact damage records.  `TrainingDamageRecordV1` stores attacker entity, target entity, weapon id, flags, and before/after health and armor (`engine/training_protocol.py:36` through `engine/training_protocol.py:49`), and the parser decodes them per frame at `engine/training_protocol.py:175` through `engine/training_protocol.py:178`.

The first v3 prototype should not require new collect data.  It can start with events, geometry, and follow-through.  Direct damage should be an optional high-confidence tier that activates only when a corpus has trusted sidecars.

## Stage 1: Shot Records

Current v2 loops over every `fire == 1` frame (`qnn/bc/target_labeler.py:388` through `qnn/bc/target_labeler.py:393`).  v3 should first group fire frames into weapon-shaped shots.

Shot fields:

| Field | Meaning |
|-------|---------|
| `start_t` | First fire frame |
| `end_t` | Last frame in burst or beam group |
| `weapon` | Dominant weapon id |
| `look_path` | Look vectors around the shot |
| `candidate_pids` | Enemy pids present in the evidence window |
| `cooldown_class` | Instant, rapid, or continuous |

Grouping rule:

```python
if weapon == LG:
    group contiguous fire frames with gaps <= 1
elif weapon in {NG, SNG}:
    group short rapid-fire runs for smoothing
else:
    each fire press is one shot
```

Evidence window:

```python
pre_frames = 8
post_frames = {
    AXE: 3,
    SG: 3,
    SSG: 3,
    NG: 70,
    SNG: 70,
    GL: 105,
    RL: 105,
    LG: beam_length + 3,
}[weapon]
```

GL has a longer physical fuse, but the default label window should stay conservative.  Use the full fuse only for diagnostics or when projectile/event evidence confirms it.

## Stage 2: Evidence Tiers

For every shot and candidate pid, compute:

```python
score =
    100.0 * direct_damage
  +  40.0 * event_hit
  +  12.0 * sim_hit
  +   6.0 * follow_track
  +   3.0 * opt3_prior
  - conflict_penalty
```

The weights are starting points for ablation.  The ordering matters more than exact values: real damage dominates events, events dominate simulated geometry, and simulated geometry dominates the old cone prior.

### Tier 0: Direct Damage

When trusted damage records exist, they should override cone geometry.

```python
records = [
    r for tau in shot_window
    for r in damage_records[tau]
    if r.attacker_entity_num == self_entity_num
    and r.target_entity_num != self_entity_num
    and weapon_compatible(r.weapon_id, shot.weapon)
]

damage(pid) = sum(
    r.damage_health + r.damage_armor
    for r in records
    if entity_to_pid(r.target_entity_num, tau) == pid
)

direct_damage(pid) = log1p(damage(pid))
```

If one pid has clear positive damage, lock the shot:

```python
if best_damage > 0 and best_damage >= 1.5 * second_damage:
    shot_label = best_pid
    confidence = "damage_lock"
```

Weapon awareness enters through hit windows: hitscan damage should be immediate, LG damage can repeat over a beam burst, NG/SNG/RL damage can arrive after flight time, and GL damage may arrive after arc, bounce, or fuse delay.

### Tier 1: Entity Events

If direct damage is unavailable, use token events as weaker outcome evidence.

```python
event_hit(pid) = max(
    event_decay(dt, weapon)
  * action_weight(action)
  * source_compat(source, weapon)
  * ownership_prior(pid, shot, tau)
)

action_weight(PAIN) = 1.0
action_weight(DEATH) = 1.5
action_weight(other) = 0.0
```

Events are noisy because other players can cause pain or death.  Treat them as weighted evidence unless timing, source, and geometry are unambiguous.  A pain event on pid A 100 ms after the demonstrator fires SG at pid A is strong.  A generic death event 1.2s after a GL shot is only a hint.

### Tier 2: Hit Likelihood

When no outcome evidence exists, score physical hit likelihood.  This is not cone membership; it is minimum distance to the weapon's damage volume.

```python
sim_hit(pid) = exp(-0.5 * (miss_distance(pid) / sigma_weapon) ** 2)
             * time_decay(t_closest, weapon)
             * visibility_weight(pid)
```

For hitscan weapons:

```python
d = ray_aabb_distance(ray_origin, look[start_t], actor_box(pid, start_t))
miss = d - spread_radius(weapon, dist)
range_ok = dist <= max_range[weapon]
```

Weapon details:

- Axe requires range <= 64u.
- SG uses the narrow pellet spread.
- SSG uses the wider horizontal spread but should not over-label long-range pellet grazes without event support.
- LG requires range <= 600u and should integrate `ray_hit_likelihood(pid, tau)` over beam frames.

For NG/SNG, solve closest approach between a 1000u/s projectile and candidate linear motion:

```python
projectile(t) = look0 * 1000.0 * t
target(t) = rel0 + vel * t
t_star = clamp(dot(rel0, look0 * 1000.0 - vel) / norm2(look0 * 1000.0 - vel), 0, t_max)
miss = norm(target(t_star) - projectile(t_star)) - target_radius
```

For RL, use closest approach plus splash:

```python
direct_miss = distance_to_capsule - target_radius
splash_miss = distance_to_center - 120.0
sim_hit = max(
    exp(-0.5 * (direct_miss / 24.0) ** 2),
    0.75 * exp(-0.5 * (splash_miss / 48.0) ** 2),
)
```

For GL, combine straight and ballistic approximations, then down-weight unsupported area denial:

```python
straight = projectile_score(speed=600, gravity=0)
arc = projectile_score(speed=600, gravity=800, up_kick=200)
sim_hit = max(straight, arc) * grenade_intent_prior
```

### Tier 3: Follow-Through

Compute whether the demonstrator continues tracking the candidate before and after the shot.

```python
aim_error(t, pid) = angle(look[t], aim_point(pid, t))
pre_track = robust_mean(exp(-aim_error(t, pid)^2 / (2 * pre_sigma^2)))
post_track = robust_mean(exp(-aim_error(t, pid)^2 / (2 * post_sigma^2)))
smooth = exp(-median(abs(delta aim_error)) / smooth_sigma)
follow_track = 0.4 * pre_track + 0.4 * post_track + 0.2 * smooth
```

| Weapon | Pre | Post | Reason |
|--------|-----|------|--------|
| Axe | 3 | 3 | Contact timing |
| SG/SSG | 6 | 8 | Snapshot aim plus recovery |
| NG/SNG | 8 | 16 | Sustained stream |
| RL/GL | 8 | 12 | Lead and expected impact |
| LG | 4 | beam burst | Continuous tracking |

This directly targets random fire and one-frame cone blips.

## Stage 3: Engagement Decoder

Decode shot labels with Viterbi over pids plus `NO_TARGET`.

```python
state_i in {NO_TARGET, pid_1, pid_2, ...}

dp[i, cur] = max_prev(
    dp[i - 1, prev]
  + transition(prev, cur, gap_frames, weapon)
  + emission(shot_i, cur)
)
```

Transition policy:

- Cheap keep when pid stays in stream.
- Expensive switch when evidence is weak.
- Lower switch cost when aim angular velocity is high.
- Strong penalty for A->B->A unless B has outcome evidence.
- Release to `NO_TARGET` when all candidate evidence is weak.

This is more flexible than the current fire loop, which can only keep sticky when the current pid passes the release cone at the next fire (`qnn/bc/target_labeler.py:407` through `qnn/bc/target_labeler.py:420`).  v3 can keep a pid through a weak geometric frame if delayed evidence confirms the shot.

## Frame Label Expansion

After shot decoding, group consecutive same-pid shots into engagements and expand them into frame labels.  Keep the current non-overlap discipline, but make boundaries evidence-aware.

```python
support(t, pid) =
    0.45 * aim_track_frame(t, pid)
  + 0.25 * sticky_decay_from_nearest_shot(t, pid)
  + 0.20 * visibility_or_recency_weight(t, pid)
  + 0.10 * recent_outcome_evidence(t, pid)

while pid_in_stream(t - 1) and support(t - 1, pid) >= frame_floor:
    back_bound -= 1

while pid_in_stream(t + 1) and support(t + 1, pid) >= frame_floor:
    forward_bound += 1
```

For damage-locked shots, allow a short post-impact extension even if aim moves away.  For geometry-only shots, require stronger frame support.

## No-Target Handling

v3 should be more willing to emit `TARGET_IGNORE` for low-evidence shots.

```python
no_target_score =
    spam_prior(weapon, fire_density)
  + weak_geometry_prior(all_candidates)
  + no_outcome_prior(weapon, elapsed_window)
```

GL spam into an empty doorway should be ignored.  RL splash near a visible actor should label that actor.  LG held off target for several frames should release.  This may reduce coverage, but it should improve target-head supervision.

## Why v3 Should Outperform v2

v3 should improve the legit/jitter ratio for concrete reasons:

- **Outcome evidence beats cosine ties.**  If pid A takes damage, pid B being slightly closer to the crosshair at fire time no longer wins.
- **Projectile scoring is temporal.**  RL/NG/SNG labels depend on future closest approach, not only frame `t` aim.
- **Follow-through rejects blips.**  A one-frame A->B->A cone fluctuation has weak post-shot support.
- **Viterbi penalizes oscillation.**  Switches need stronger evidence than sticky keeps.
- **Weapon cadence is used directly.**  LG and nail streams are integrated across bursts instead of treated as independent fire frames.

Expected pattern: fewer jitter switches, flat or higher legit switches, a strong drop in A->B->A oscillations, lower GL spam coverage, and better long-range projectile attribution when simulation or outcome evidence supports a target.

## Validation Plan

Keep the existing switch-quality report and add shot-level diagnostics.

| Metric | Expected direction |
|--------|--------------------|
| `legit / jitter` switch ratio | Up |
| Jitter switches | Down |
| Legit switches | Flat or up |
| A->B->A oscillations | Down |
| Per-weapon labeled fires | Lower for GL spam, stable elsewhere |
| Damage-confirmed attribution | Up when sidecars exist |
| Event-confirmed attribution | Up |
| Geometry-only label share | Down |

Add tables like:

```text
variant    weapon    shots    labeled    damage_lock    event_lock    sim_only    no_target
v2         RL        ...
v3         RL        ...
```

and:

```text
switch_cause      count    jitter    legit    mean_angvel
damage_forced     ...
event_forced      ...
sim_forced        ...
viterbi_switch    ...
fallback_opt3     ...
```

The promotion criterion should be label quality plus BC outcome, not just more weapon-aware differences.

## Concerns And Trade-Offs

- **Damage may be missing.**  The largest gain likely requires trusted hit attribution.  Training sidecars have it; historical QWD shards may not.  v3 must work without damage records and treat damage as optional high-confidence evidence.
- **Events are noisy.**  Pain/death events may be caused by another player or the world.  They should lock only when timing, source, and geometry are tight.
- **Simulation is approximate.**  The obs is view-relative and lacks exact collision geometry.  GL bounces and occlusion should remain likelihood terms, not truth.
- **Offline lookahead can over-teach.**  Bound back-walks by support and report target-head performance separately for damage-locked, event-locked, sim-only, and fallback labels.
- **Complexity is real.**  v3 should be built as opt-in evidence tiers with diagnostics, not as one opaque replacement.

## Migration Plan

### Phase 0: Instrument Only

Build shot records and evidence scores while still writing current v2 labels.  Use existing precomputed shards, action arrays, actor obs arrays, and entity event arrays.  Output per-shot evidence artifacts, per-weapon summaries, and example dumps for high-disagreement shots.  No new collect data is required.

### Phase 1: v3-Lite

Implement event, simulation, follow-through, and Viterbi decoding without direct damage records.  Keep opt3 as fallback prior.

Required code changes:

- new shot builder,
- new evidence scorer,
- new decoder,
- new `switch_quality.py` variant hook,
- no wire change,
- no model change.

### Phase 2: Damage Audit

Audit available corpora for trusted damage.  For training sidecars, verify demonstrator `self_entity_num`, map damage target entity to pid, check weapon ids against `act_weapon`, and compare damage totals against aggregate metrics.  For QWD demos, determine whether `svc_damage` can expose victim pid and amount; if not, decide whether replay can emit a sidecar.

### Phase 3: Damage-Locked v3

Enable direct damage when available.  Lock a shot on clear single-victim damage, choose primary victim by damage for multi-victim explosives, ignore self-damage-only shots unless another victim also took damage, and record all lock decisions for diagnostics.

### Phase 4: BC Ablation

Train and compare:

| Run | Labels |
|-----|--------|
| baseline | current v2 final |
| v3-lite | events + simulation + follow-through |
| v3-damage | direct damage where available, v3-lite fallback |
| v3-strict | only high-confidence labels |

Judge target-head CE, switch quality, per-weapon behavior, eval damage, and live-play target stability.

## Recommended Default Shape

```python
def label_enemy_target_v3(obs, actions, *, evidence=None, config=DEFAULT):
    shots = build_shots(actions)
    tracks = build_candidate_tracks(obs)

    for shot in shots:
        scores = {}
        scores += damage_scores(shot, tracks, evidence.damage)
        scores += event_scores(shot, tracks, obs.events)
        scores += weapon_sim_scores(shot, tracks, obs)
        scores += follow_scores(shot, tracks, actions["look"])
        scores += opt3_prior(shot, tracks)
        shot_scores.append(scores)

    decoded = viterbi_decode(shots, shot_scores, tracks)
    engagements = group_decoded_shots(decoded, tracks)
    return expand_engagements(engagements, tracks)
```

Default evidence weights:

| Component | Weight |
|-----------|--------|
| Direct damage | 100 |
| Death event | 60 |
| Pain event | 40 |
| Weapon simulation | 12 |
| Follow-through | 6 |
| Opt3 prior | 3 |
| Weak switch penalty | 8 |
| A->B->A penalty | 12 |

## Open Questions

1. Can QWD replay expose reliable victim pid and damage amount?
2. How often do actor `PAIN` and `DEATH` events occur near demonstrator fires?
3. Do projectile tokens persist long enough to confirm RL/GL/NG paths?
4. Does v3-lite improve `legit / jitter` before damage is added?
5. How much coverage can be dropped before target-head training degrades?
6. Should explosive multi-victim shots become a future multi-target label?

## Bottom Line

The current v2 is a strong cone labeler, but that is also its ceiling.  Its weapon-aware path still chooses from a per-fire cosine admit test, so weapon information can only make marginal corrections.

v3 should make weapon information change the objective.  Instead of asking "which enemy is inside this weapon-shaped cone?", ask "which enemy does this weapon-specific shot most likely hit, damage, threaten, or track over its natural time window?"  That is where weapon identity becomes load-bearing.
