# Cluster Embeddings

Design document for feature-based cluster embeddings. Replaces the current learned ID-based lookup with a computed feature vector per cluster that generalizes across maps.

## Current State

- Clusters computed by Girvan-Newman edge betweenness on the navmesh graph
- Per-cluster data already available: center, bounds, area_count, exit_count, special_exit_count
- Cluster IDs passed to Python as raw integers
- Python model uses `cluster_embed(256)` learned lookup table — does not generalize across maps
- Route embedding: Python sums `cluster_embed(c)` for intermediate clusters on the route path

## Target Design

### Feature Vector (per cluster, computed at map load)

| Feature | Source | Normalization | Description |
|---------|--------|---------------|-------------|
| exit_count | cluster.exit_count | /max_exits | how many ways out |
| special_exit_count | cluster.special_exit_count | /max_special | teleport/lift/drop/push exits |
| area_count | cluster.area_count | /max_areas | walkable size of the room |
| vertical_pos | cluster.center[2] | map-relative [-1,1] | height on the map |
| has_water | new | 0 or 1 | cluster contains water polygons |
| has_lava | new | 0 or 1 | cluster contains lava polygons |
| has_slime | new | 0 or 1 | cluster contains slime polygons |
| item_density | new | /max_items | count of static items in cluster |

### Projection

C worker computes the feature vector per cluster at map load. Python projects it through a linear layer `cluster_proj(N → d_model)` to get the embedding.

### Usage

- **Self token**: `cluster_proj(features[player_cluster])`
- **Entity token**: `cluster_proj(features[entity_cluster])`
- **Route embedding**: sum of `cluster_proj(features[c])` for intermediate clusters on the path

### Key benefit

Feature vectors generalize across maps. A big central room with 5 exits on DM2 gets a similar embedding to a big central room with 5 exits on DM6. Learned ID lookups cannot do this.

## Prerequisite: Liquid Geometry in Navmesh

Currently, liquid surfaces (water, lava, slime) are excluded from the navmesh entirely.

### Why they're excluded

`QNN_ExtractGeometry` in `qnn_map.c` (line 523) skips faces with `TEX_SPECIAL` flag. In Quake BSP, `TEX_SPECIAL` marks sky, triggers, AND liquid surfaces. All are excluded before geometry reaches Recast.

### Fix

1. In `QNN_ExtractGeometry`, stop skipping `TEX_SPECIAL` faces that are liquid surfaces. Keep skipping sky and triggers. Quake's `CONTENTS_WATER`, `CONTENTS_SLIME`, `CONTENTS_LAVA` can be checked via `SV_PointContents` or by checking the BSP leaf content type at the face position.

2. Tag liquid triangles with distinct Recast area types (`kAreaWater`, `kAreaLava`, `kAreaSlime`) before rasterization, separate from `RC_WALKABLE_AREA`.

3. After navmesh build, per-cluster: scan polygon areas to count liquid types present. Store as cluster features.

4. Optionally: assign higher traversal cost to liquid polygons so pathfinding prefers dry routes but still knows liquid routes exist.

### Impact

- Navmesh will be larger (more polygons for liquid surfaces)
- Clusters will reflect the full room geometry, not just walkable parts
- Route planning can consider liquid shortcuts (swimming through water to reach the other side)
- Cluster features tell the model what hazards are in each room

## Item Density

Cross-reference static items (from BSP baselines) with cluster assignments:
- For each item, find its navmesh area via `QNN_RouteFindArea(item.origin)`
- Look up that area's cluster_id
- Count items per cluster
- Normalize by max item count across all clusters

This requires the store to be initialized before cluster features are computed, or a separate pass over BSP entities.

## Centrality (deferred)

Average shortest-path cost from each cluster to all other clusters. Requires the full cluster-to-cluster distance matrix. Computable from the route graph at init time but potentially expensive for maps with many clusters. Can add later if the simpler features aren't sufficient.
