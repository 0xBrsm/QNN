extern "C" {
#include "qnn.h"
}

#include "qnn_route.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <vector>

namespace {

void SortUniqueInts(std::vector<int> *values)
{
	std::sort(values->begin(), values->end());
	values->erase(std::unique(values->begin(), values->end()), values->end());
}

int ClusterFirstAreaId(const std::vector<int> &area_ids)
{
	if (area_ids.empty())
		return -1;
	return area_ids.front();
}

int ClusterMaxSize(const std::vector<std::vector<int>> &clusters)
{
	int max_size;

	max_size = 0;
	for (const std::vector<int> &cluster : clusters)
	{
		if ((int)cluster.size() > max_size)
			max_size = (int)cluster.size();
	}
	return max_size;
}

/* ── Edge betweenness (Girvan-Newman) ──────────────────────────── */

typedef std::pair<int, int> EdgePair;

EdgePair MakeEdge(int a, int b)
{
	return a < b ? EdgePair(a, b) : EdgePair(b, a);
}

/* Brandes' algorithm for edge betweenness on a subgraph defined by
   area_ids.  O(V*E) — fine for Quake-sized maps (~500 areas). */
void ComputeEdgeBetweenness(
	const std::vector<int> &area_ids,
	const std::vector<std::vector<int>> &walk_neighbors,
	std::map<EdgePair, double> *out)
{
	const size_t n = walk_neighbors.size();
	std::vector<char> in_component(n, 0);

	out->clear();
	for (int id : area_ids)
		in_component[(size_t)id] = 1;

	/* Pre-populate all component edges with 0. */
	for (int a : area_ids)
	{
		for (int b : walk_neighbors[(size_t)a])
		{
			if (in_component[(size_t)b] && b > a)
				(*out)[MakeEdge(a, b)] = 0.0;
		}
	}

	std::vector<std::vector<int>> pred(n);
	std::vector<int> sigma(n);
	std::vector<int> dist(n);
	std::vector<double> delta(n);

	for (int s : area_ids)
	{
		std::stack<int> S;
		std::queue<int> Q;

		for (int id : area_ids)
		{
			pred[(size_t)id].clear();
			sigma[(size_t)id] = 0;
			dist[(size_t)id] = -1;
			delta[(size_t)id] = 0.0;
		}
		sigma[(size_t)s] = 1;
		dist[(size_t)s] = 0;
		Q.push(s);

		while (!Q.empty())
		{
			int v = Q.front();
			Q.pop();
			S.push(v);

			for (int w : walk_neighbors[(size_t)v])
			{
				if (!in_component[(size_t)w])
					continue;
				if (dist[(size_t)w] < 0)
				{
					dist[(size_t)w] = dist[(size_t)v] + 1;
					Q.push(w);
				}
				if (dist[(size_t)w] == dist[(size_t)v] + 1)
				{
					sigma[(size_t)w] += sigma[(size_t)v];
					pred[(size_t)w].push_back(v);
				}
			}
		}

		while (!S.empty())
		{
			int w = S.top();
			S.pop();

			for (int v : pred[(size_t)w])
			{
				double c = ((double)sigma[(size_t)v] / (double)sigma[(size_t)w])
					* (1.0 + delta[(size_t)w]);
				(*out)[MakeEdge(v, w)] += c;
				delta[(size_t)v] += c;
			}
		}
	}

	/* Undirected graph — each shortest path is counted twice. */
	for (auto &kv : *out)
		kv.second /= 2.0;
}

/* Split a walk-connected component at doorways identified by edge
   betweenness.  High-betweenness edges are bottlenecks (doorways);
   low-betweenness edges are room interiors.  We find the natural gap
   in the betweenness distribution and remove everything above it. */
void SplitComponentByBetweenness(
	const std::vector<int> &area_ids,
	const std::vector<std::vector<int>> &walk_neighbors,
	std::vector<std::vector<int>> *out_clusters)
{
	const size_t n = walk_neighbors.size();
	std::map<EdgePair, double> betweenness;

	if (out_clusters == nullptr)
		return;

	ComputeEdgeBetweenness(area_ids, walk_neighbors, &betweenness);
	if (betweenness.empty())
	{
		out_clusters->push_back(area_ids);
		return;
	}

	/* Sort edges by betweenness, descending. */
	std::vector<std::pair<double, EdgePair>> sorted;
	sorted.reserve(betweenness.size());
	for (const auto &kv : betweenness)
		sorted.push_back({kv.second, kv.first});
	std::sort(sorted.begin(), sorted.end(),
		[](const std::pair<double, EdgePair> &a,
		   const std::pair<double, EdgePair> &b)
		{
			if (a.first != b.first) return a.first > b.first;
			return a.second < b.second;
		});

	/* Find the largest relative gap in the sorted betweenness values.
	   Doorway edges have betweenness ~ A*B (product of room sizes on
	   each side).  Interior edges have betweenness ~ O(N).  The gap
	   between these is typically 5-30x for Quake maps. */
	int best_gap_index = -1;
	double best_gap_ratio = 0.0;
	for (size_t i = 0; i + 1 < sorted.size(); ++i)
	{
		double high = sorted[i].first;
		double low = sorted[i + 1].first;

		if (low < 0.001)
			continue;
		double ratio = high / low;
		if (ratio > best_gap_ratio)
		{
			best_gap_ratio = ratio;
			best_gap_index = (int)i;
		}
	}

	/* No significant gap — component is a single room. */
	if (best_gap_ratio < 3.0 || best_gap_index < 0)
	{
		out_clusters->push_back(area_ids);
		return;
	}

	/* Remove all edges above the gap (doorway edges). */
	double threshold = sorted[(size_t)best_gap_index + 1].first;
	std::set<EdgePair> removed;
	for (const auto &ep : sorted)
	{
		if (ep.first <= threshold)
			break;
		removed.insert(ep.second);
	}

	/* Collect connected components from the remaining edges. */
	std::vector<char> in_component(n, 0);
	std::vector<char> visited(n, 0);
	for (int id : area_ids)
		in_component[(size_t)id] = 1;

	for (int start : area_ids)
	{
		if (visited[(size_t)start])
			continue;

		std::vector<int> cluster;
		std::queue<int> q;
		q.push(start);
		visited[(size_t)start] = 1;

		while (!q.empty())
		{
			int v = q.front();
			q.pop();
			cluster.push_back(v);

			for (int w : walk_neighbors[(size_t)v])
			{
				if (!in_component[(size_t)w] || visited[(size_t)w])
					continue;
				if (removed.count(MakeEdge(v, w)))
					continue;
				visited[(size_t)w] = 1;
				q.push(w);
			}
		}

		SortUniqueInts(&cluster);
		out_clusters->push_back(cluster);
	}
}

int ChooseSeedArea(
	const qnn_route_runtime_t *oracle,
	const std::vector<int> &candidate_area_ids,
	const std::vector<int> &selected_area_ids,
	const std::vector<char> &selected_mask,
	const std::vector<int> &special_incidence)
{
	int best_area_id;
	float best_distance;
	int best_special_incidence;

	best_area_id = -1;
	best_distance = -1.0f;
	best_special_incidence = -1;

	/* O(candidate_area_ids * selected_area_ids), which is acceptable for the
	   small area counts in Quake maps. */
	for (int candidate_area_id : candidate_area_ids)
	{
		float min_distance;

		if (candidate_area_id < 0
			|| candidate_area_id >= (int)selected_mask.size()
			|| selected_mask[(size_t)candidate_area_id])
			continue;

		min_distance = 0.0f;
		if (!selected_area_ids.empty())
		{
			min_distance = std::numeric_limits<float>::infinity();
			for (int selected_area_id : selected_area_ids)
			{
				const float distance = RouteDistance(
					oracle->areas[(size_t)candidate_area_id].center,
					oracle->areas[(size_t)selected_area_id].center);

				if (distance < min_distance)
					min_distance = distance;
			}
		}

		if (best_area_id < 0
			|| min_distance > best_distance + kClusterCostEpsilon
			|| (fabsf(min_distance - best_distance) <= kClusterCostEpsilon
				&& special_incidence[(size_t)candidate_area_id] > best_special_incidence)
			|| (fabsf(min_distance - best_distance) <= kClusterCostEpsilon
				&& special_incidence[(size_t)candidate_area_id] == best_special_incidence
				&& candidate_area_id < best_area_id))
		{
			best_area_id = candidate_area_id;
			best_distance = min_distance;
			best_special_incidence = special_incidence[(size_t)candidate_area_id];
		}
	}

	return best_area_id;
}

void PickClusterSeeds(
	const qnn_route_runtime_t *oracle,
	const std::vector<int> &component_area_ids,
	const std::vector<int> &special_incidence,
	int seed_count,
	std::vector<int> *seed_area_ids)
{
	std::vector<int> special_area_ids;
	std::vector<char> selected_mask;

	if (seed_area_ids == nullptr)
		return;

	seed_area_ids->clear();
	if (oracle == nullptr || component_area_ids.empty() || seed_count <= 0)
		return;

	seed_count = std::min(seed_count, (int)component_area_ids.size());
	selected_mask.assign(oracle->areas.size(), 0);

	for (int area_id : component_area_ids)
	{
		if (special_incidence[(size_t)area_id] > 0)
			special_area_ids.push_back(area_id);
	}

	while ((int)seed_area_ids->size() < seed_count && !special_area_ids.empty())
	{
		const int seed_area_id = ChooseSeedArea(
			oracle,
			special_area_ids,
			*seed_area_ids,
			selected_mask,
			special_incidence);

		if (seed_area_id < 0)
			break;
		seed_area_ids->push_back(seed_area_id);
		selected_mask[(size_t)seed_area_id] = 1;
	}

	if (seed_area_ids->empty())
	{
		const int seed_area_id = ChooseSeedArea(
			oracle,
			component_area_ids,
			*seed_area_ids,
			selected_mask,
			special_incidence);

		if (seed_area_id >= 0)
		{
			seed_area_ids->push_back(seed_area_id);
			selected_mask[(size_t)seed_area_id] = 1;
		}
	}

	while ((int)seed_area_ids->size() < seed_count)
	{
		const int seed_area_id = ChooseSeedArea(
			oracle,
			component_area_ids,
			*seed_area_ids,
			selected_mask,
			special_incidence);

		if (seed_area_id < 0)
			break;
		seed_area_ids->push_back(seed_area_id);
		selected_mask[(size_t)seed_area_id] = 1;
	}
}

struct QnnClusterQueueNode
{
	float cost;
	int seed_index;
	int area_id;

	bool operator>(const QnnClusterQueueNode &other) const
	{
		if (cost != other.cost)
			return cost > other.cost;
		if (seed_index != other.seed_index)
			return seed_index > other.seed_index;
		return area_id > other.area_id;
	}
};

void AssignComponentClusters(
	const qnn_route_runtime_t *oracle,
	const std::vector<int> &component_area_ids,
	const std::vector<std::vector<int>> &walk_neighbors,
	const std::vector<int> &seed_area_ids,
	std::vector<int> *cluster_assignment)
{
	std::vector<char> in_component;
	std::vector<float> best_cost;
	std::priority_queue<QnnClusterQueueNode, std::vector<QnnClusterQueueNode>, std::greater<QnnClusterQueueNode>> queue;

	if (oracle == nullptr || cluster_assignment == nullptr)
		return;

	cluster_assignment->assign(oracle->areas.size(), -1);
	in_component.assign(oracle->areas.size(), 0);
	best_cost.assign(oracle->areas.size(), std::numeric_limits<float>::infinity());

	for (int area_id : component_area_ids)
		in_component[(size_t)area_id] = 1;

	for (size_t seed_index = 0; seed_index < seed_area_ids.size(); ++seed_index)
	{
		const int seed_area_id = seed_area_ids[seed_index];

		best_cost[(size_t)seed_area_id] = 0.0f;
		(*cluster_assignment)[(size_t)seed_area_id] = (int)seed_index;
		queue.push({0.0f, (int)seed_index, seed_area_id});
	}

	while (!queue.empty())
	{
		const QnnClusterQueueNode current = queue.top();
		queue.pop();

		if (current.cost > best_cost[(size_t)current.area_id] + kClusterCostEpsilon)
			continue;
		if ((*cluster_assignment)[(size_t)current.area_id] != current.seed_index
			&& fabsf(current.cost - best_cost[(size_t)current.area_id]) <= kClusterCostEpsilon)
			continue;

		for (int neighbor_area_id : walk_neighbors[(size_t)current.area_id])
		{
			float next_cost;

			if (!in_component[(size_t)neighbor_area_id])
				continue;

			next_cost = current.cost + RouteDistance(
				oracle->areas[(size_t)current.area_id].center,
				oracle->areas[(size_t)neighbor_area_id].center);
			if (next_cost + kClusterCostEpsilon < best_cost[(size_t)neighbor_area_id]
				|| (fabsf(next_cost - best_cost[(size_t)neighbor_area_id]) <= kClusterCostEpsilon
					&& current.seed_index < (*cluster_assignment)[(size_t)neighbor_area_id]))
			{
				best_cost[(size_t)neighbor_area_id] = next_cost;
				(*cluster_assignment)[(size_t)neighbor_area_id] = current.seed_index;
				queue.push({next_cost, current.seed_index, neighbor_area_id});
			}
		}
	}
}

void CollectComponentClusters(
	const std::vector<int> &component_area_ids,
	const std::vector<int> &cluster_assignment,
	int cluster_count,
	std::vector<std::vector<int>> *clusters)
{
	if (clusters == nullptr)
		return;

	clusters->assign((size_t)cluster_count, std::vector<int>());
	for (int area_id : component_area_ids)
	{
		const int cluster_index = cluster_assignment[(size_t)area_id];

		if (cluster_index >= 0 && cluster_index < cluster_count)
			(*clusters)[(size_t)cluster_index].push_back(area_id);
	}
	for (std::vector<int> &cluster : *clusters)
		SortUniqueInts(&cluster);
}

int ChooseMergeTarget(
	const std::vector<std::vector<int>> &clusters,
	const std::vector<int> &cluster_assignment,
	const std::vector<std::vector<int>> &walk_neighbors,
	int source_cluster_index)
{
	std::vector<int> shared_edges;
	int best_cluster_index;
	int best_shared_edges;
	int best_overflow;
	int best_target_delta;
	int best_first_area_id;

	shared_edges.assign(clusters.size(), 0);
	for (int area_id : clusters[(size_t)source_cluster_index])
	{
		for (int neighbor_area_id : walk_neighbors[(size_t)area_id])
		{
			const int neighbor_cluster_index = cluster_assignment[(size_t)neighbor_area_id];

			if (neighbor_cluster_index >= 0 && neighbor_cluster_index != source_cluster_index)
				shared_edges[(size_t)neighbor_cluster_index] += 1;
		}
	}

	best_cluster_index = -1;
	best_shared_edges = -1;
	best_overflow = std::numeric_limits<int>::max();
	best_target_delta = std::numeric_limits<int>::max();
	best_first_area_id = std::numeric_limits<int>::max();

	for (size_t cluster_index = 0; cluster_index < clusters.size(); ++cluster_index)
	{
		if ((int)cluster_index == source_cluster_index
			|| clusters[cluster_index].empty()
			|| shared_edges[cluster_index] <= 0)
			continue;

		const int merged_size = (int)clusters[(size_t)source_cluster_index].size() + (int)clusters[cluster_index].size();
		const int overflow = merged_size > kClusterMaxAreaCount ? merged_size - kClusterMaxAreaCount : 0;
		const int target_delta = abs(merged_size - kClusterTargetAreaCount);
		const int first_area_id = ClusterFirstAreaId(clusters[cluster_index]);

		if (best_cluster_index < 0
			|| shared_edges[cluster_index] > best_shared_edges
			|| (shared_edges[cluster_index] == best_shared_edges && overflow < best_overflow)
			|| (shared_edges[cluster_index] == best_shared_edges && overflow == best_overflow && target_delta < best_target_delta)
			|| (shared_edges[cluster_index] == best_shared_edges && overflow == best_overflow && target_delta == best_target_delta
				&& first_area_id < best_first_area_id))
		{
			best_cluster_index = (int)cluster_index;
			best_shared_edges = shared_edges[cluster_index];
			best_overflow = overflow;
			best_target_delta = target_delta;
			best_first_area_id = first_area_id;
		}
	}

	return best_cluster_index;
}

void MergeSmallClusters(
	const std::vector<std::vector<int>> &walk_neighbors,
	std::vector<std::vector<int>> *clusters)
{
	std::vector<int> cluster_assignment;

	if (clusters == nullptr)
		return;

	cluster_assignment.assign(walk_neighbors.size(), -1);
	for (size_t cluster_index = 0; cluster_index < clusters->size(); ++cluster_index)
	{
		for (int area_id : (*clusters)[cluster_index])
			cluster_assignment[(size_t)area_id] = (int)cluster_index;
	}
	for (;;)
	{
		int source_cluster_index;
		int source_size;
		int source_first_area_id;
		int non_empty_cluster_count;

		source_cluster_index = -1;
		source_size = std::numeric_limits<int>::max();
		source_first_area_id = std::numeric_limits<int>::max();
		non_empty_cluster_count = 0;

		for (size_t cluster_index = 0; cluster_index < clusters->size(); ++cluster_index)
		{
			const int cluster_size = (int)(*clusters)[cluster_index].size();
			const int first_area_id = ClusterFirstAreaId((*clusters)[cluster_index]);

			if (cluster_size <= 0)
				continue;
			non_empty_cluster_count += 1;
			if (cluster_size >= kClusterMinAreaCount)
				continue;
			if (source_cluster_index < 0
				|| cluster_size < source_size
				|| (cluster_size == source_size && first_area_id < source_first_area_id))
			{
				source_cluster_index = (int)cluster_index;
				source_size = cluster_size;
				source_first_area_id = first_area_id;
			}
		}

		if (source_cluster_index < 0 || non_empty_cluster_count <= 1)
			break;

		{
			const int target_cluster_index = ChooseMergeTarget(
				*clusters,
				cluster_assignment,
				walk_neighbors,
				source_cluster_index);

			if (target_cluster_index < 0)
				break;

			(*clusters)[(size_t)target_cluster_index].insert(
				(*clusters)[(size_t)target_cluster_index].end(),
				(*clusters)[(size_t)source_cluster_index].begin(),
				(*clusters)[(size_t)source_cluster_index].end());
			SortUniqueInts(&(*clusters)[(size_t)target_cluster_index]);
			(*clusters)[(size_t)source_cluster_index].clear();
			for (int area_id : (*clusters)[(size_t)target_cluster_index])
				cluster_assignment[(size_t)area_id] = target_cluster_index;
		}
	}

	{
		std::vector<std::vector<int>> compacted_clusters;

		compacted_clusters.reserve(clusters->size());
		for (const std::vector<int> &cluster : *clusters)
		{
			if (!cluster.empty())
				compacted_clusters.push_back(cluster);
		}
		*clusters = std::move(compacted_clusters);
	}
}

/* Voronoi fallback — subdivide a large room that has no internal
   doorways into spatially compact clusters. */
void PartitionVoronoi(
	const qnn_route_runtime_t *oracle,
	const std::vector<int> &component_area_ids,
	const std::vector<std::vector<int>> &walk_neighbors,
	const std::vector<int> &special_incidence,
	std::vector<std::vector<int>> *cluster_area_ids)
{
	std::vector<int> seed_area_ids;
	std::vector<int> cluster_assignment;
	std::vector<std::vector<int>> component_clusters;
	int desired_cluster_count;
	const int max_iterations = 100;

	if (cluster_area_ids == nullptr || component_area_ids.empty())
		return;

	desired_cluster_count = std::max(2, ((int)component_area_ids.size() + kClusterTargetAreaCount - 1) / kClusterTargetAreaCount);
	desired_cluster_count = std::min(desired_cluster_count, (int)component_area_ids.size());

	for (int iteration_count = 0; iteration_count < max_iterations; ++iteration_count)
	{
		PickClusterSeeds(
			oracle,
			component_area_ids,
			special_incidence,
			desired_cluster_count,
			&seed_area_ids);
		AssignComponentClusters(
			oracle,
			component_area_ids,
			walk_neighbors,
			seed_area_ids,
			&cluster_assignment);
		CollectComponentClusters(
			component_area_ids,
			cluster_assignment,
			(int)seed_area_ids.size(),
			&component_clusters);

		if (ClusterMaxSize(component_clusters) > kClusterMaxAreaCount
			&& desired_cluster_count < (int)component_area_ids.size())
		{
			desired_cluster_count += 1;
			continue;
		}

		MergeSmallClusters(walk_neighbors, &component_clusters);
		if (ClusterMaxSize(component_clusters) > kClusterMaxAreaCount
			&& desired_cluster_count < (int)component_area_ids.size())
		{
			desired_cluster_count += 1;
			continue;
		}

		for (const std::vector<int> &cluster : component_clusters)
		{
			if (!cluster.empty())
				cluster_area_ids->push_back(cluster);
		}
		return;
	}

	for (const std::vector<int> &cluster : component_clusters)
	{
		if (!cluster.empty())
			cluster_area_ids->push_back(cluster);
	}
}

/* Primary partitioning: split at doorways using edge betweenness,
   then Voronoi fallback for any remaining oversized components. */
void PartitionWalkComponent(
	const qnn_route_runtime_t *oracle,
	const std::vector<int> &component_area_ids,
	const std::vector<std::vector<int>> &walk_neighbors,
	const std::vector<int> &special_incidence,
	std::vector<std::vector<int>> *cluster_area_ids)
{
	std::vector<std::vector<int>> sub_clusters;

	if (cluster_area_ids == nullptr || component_area_ids.empty())
		return;

	/* Too small to split into two viable clusters. */
	if ((int)component_area_ids.size() < 2 * kClusterMinAreaCount)
	{
		cluster_area_ids->push_back(component_area_ids);
		return;
	}

	/* Split at doorways identified by edge betweenness.  No further
	   subdivision — if a room has no internal bottlenecks, it stays
	   as one cluster regardless of polygon count. */
	SplitComponentByBetweenness(component_area_ids, walk_neighbors, &sub_clusters);

	for (const std::vector<int> &sub : sub_clusters)
		cluster_area_ids->push_back(sub);
}

}  // namespace

/* ── Exported cluster/routing functions ─────────────────────────── */

void QNN_ClusterBuild(qnn_route_runtime_t *oracle)
{
	std::vector<std::vector<int>> walk_neighbors;
	std::vector<int> special_incidence;
	std::vector<char> visited;
	std::vector<std::vector<int>> cluster_area_ids;

	if (oracle == nullptr)
		return;

	oracle->clusters.clear();
	if (oracle->areas.empty())
		return;

	walk_neighbors.assign(oracle->areas.size(), std::vector<int>());
	special_incidence.assign(oracle->areas.size(), 0);
	visited.assign(oracle->areas.size(), 0);

	for (QnnNavArea &area : oracle->areas)
		area.cluster_id = -1;

	for (const QnnNavLink &link : oracle->links)
	{
		if (link.src_area_id < 0
			|| link.dst_area_id < 0
			|| link.src_area_id >= (int)oracle->areas.size()
			|| link.dst_area_id >= (int)oracle->areas.size())
			continue;

		if (link.travel_type == QNN_TRAVEL_WALK)
		{
			/* Detour walk adjacency is treated as bidirectional here, so
			   mirror each walk edge for connected-component clustering. */
			walk_neighbors[(size_t)link.src_area_id].push_back(link.dst_area_id);
			walk_neighbors[(size_t)link.dst_area_id].push_back(link.src_area_id);
		}
		else if (IsSpecialTravel(link.travel_type))
		{
			special_incidence[(size_t)link.src_area_id] += 1;
			special_incidence[(size_t)link.dst_area_id] += 1;
		}
	}

	for (std::vector<int> &neighbors : walk_neighbors)
		SortUniqueInts(&neighbors);

	for (size_t area_index = 0; area_index < oracle->areas.size(); ++area_index)
	{
		std::queue<int> queue;
		std::vector<int> component_area_ids;

		if (visited[area_index])
			continue;

		visited[area_index] = 1;
		queue.push((int)area_index);
		while (!queue.empty())
		{
			const int area_id = queue.front();
			queue.pop();
			component_area_ids.push_back(area_id);

			for (int neighbor_area_id : walk_neighbors[(size_t)area_id])
			{
				if (!visited[(size_t)neighbor_area_id])
				{
					visited[(size_t)neighbor_area_id] = 1;
					queue.push(neighbor_area_id);
				}
			}
		}

		SortUniqueInts(&component_area_ids);
		PartitionWalkComponent(
			oracle,
			component_area_ids,
			walk_neighbors,
			special_incidence,
			&cluster_area_ids);
	}

	std::sort(cluster_area_ids.begin(), cluster_area_ids.end(), [](const std::vector<int> &lhs, const std::vector<int> &rhs) {
		return ClusterFirstAreaId(lhs) < ClusterFirstAreaId(rhs);
	});

	oracle->clusters.reserve(cluster_area_ids.size());
	for (size_t cluster_index = 0; cluster_index < cluster_area_ids.size(); ++cluster_index)
	{
		QnnNavCluster cluster;
		float center_sum[3];
		const QnnNavArea &first_area = oracle->areas[(size_t)cluster_area_ids[cluster_index][0]];

		memset(&cluster, 0, sizeof(cluster));
		memset(center_sum, 0, sizeof(center_sum));
		cluster.cluster_id = (int)cluster_index;
		cluster.first_area_id = cluster_area_ids[cluster_index][0];
		cluster.area_count = (int)cluster_area_ids[cluster_index].size();
		memcpy(cluster.bounds_min, first_area.bounds_min, sizeof(cluster.bounds_min));
		memcpy(cluster.bounds_max, first_area.bounds_max, sizeof(cluster.bounds_max));

		for (int area_id : cluster_area_ids[cluster_index])
		{
			QnnNavArea &area = oracle->areas[(size_t)area_id];

			area.cluster_id = cluster.cluster_id;
			center_sum[0] += area.center[0];
			center_sum[1] += area.center[1];
			center_sum[2] += area.center[2];
			for (int axis = 0; axis < 3; ++axis)
			{
				cluster.bounds_min[axis] = std::min(cluster.bounds_min[axis], area.bounds_min[axis]);
				cluster.bounds_max[axis] = std::max(cluster.bounds_max[axis], area.bounds_max[axis]);
			}
		}

		cluster.center[0] = center_sum[0] / (float)cluster.area_count;
		cluster.center[1] = center_sum[1] / (float)cluster.area_count;
		cluster.center[2] = center_sum[2] / (float)cluster.area_count;
		oracle->clusters.push_back(cluster);
	}

	{
		std::vector<std::vector<int>> exit_clusters;
		std::vector<std::vector<int>> special_exit_clusters;

		exit_clusters.assign(oracle->clusters.size(), std::vector<int>());
		special_exit_clusters.assign(oracle->clusters.size(), std::vector<int>());
		for (const QnnNavLink &link : oracle->links)
		{
			const int src_cluster_id = oracle->areas[(size_t)link.src_area_id].cluster_id;
			const int dst_cluster_id = oracle->areas[(size_t)link.dst_area_id].cluster_id;

			if (src_cluster_id < 0 || dst_cluster_id < 0 || src_cluster_id == dst_cluster_id)
				continue;

			exit_clusters[(size_t)src_cluster_id].push_back(dst_cluster_id);
			if (IsSpecialTravel(link.travel_type))
				special_exit_clusters[(size_t)src_cluster_id].push_back(dst_cluster_id);
		}

		for (size_t cluster_index = 0; cluster_index < oracle->clusters.size(); ++cluster_index)
		{
			SortUniqueInts(&exit_clusters[cluster_index]);
			SortUniqueInts(&special_exit_clusters[cluster_index]);
			oracle->clusters[cluster_index].exit_count = (int)exit_clusters[cluster_index].size();
			oracle->clusters[cluster_index].special_exit_count = (int)special_exit_clusters[cluster_index].size();
		}
	}
}

/* Trace the optimal path forward from start_area toward dst using
   Dijkstra best_cost until we leave src_cluster.  Returns the exit
   cluster id and writes the boundary crossing position to out_pos.
   Returns -1 if the path stays in src_cluster the whole way. */
static int TraceExitCluster(
	const qnn_route_runtime_t *oracle,
	int src_cluster,
	int start_area,
	int dst,
	const std::vector<float> &best_cost,
	float *out_pos)
{
	const int area_count = (int)oracle->areas.size();
	int cursor = start_area;

	out_pos[0] = out_pos[1] = out_pos[2] = 0.0f;

	for (int step = 0; step < QNN_ROUTE_MAX_AREAS; ++step)
	{
		if (cursor == dst)
			break;

		/* Find the optimal next hop from cursor toward dst. */
		int best_link_id = -1;
		float best_total = std::numeric_limits<float>::infinity();

		for (int link_id : oracle->outgoing_links[(size_t)cursor])
		{
			const QnnNavLink &link = oracle->links[(size_t)link_id];
			const float total = link.travel_time + best_cost[(size_t)link.dst_area_id];

			if (total + kClusterCostEpsilon < best_total
				|| (fabsf(total - best_total) <= kClusterCostEpsilon
					&& link_id < best_link_id))
			{
				best_total = total;
				best_link_id = link_id;
			}
		}

		if (best_link_id < 0)
			break;

		const QnnNavLink &next_link = oracle->links[(size_t)best_link_id];
		const int next_cluster = oracle->areas[(size_t)next_link.dst_area_id].cluster_id;

		if (next_cluster != src_cluster)
		{
			out_pos[0] = next_link.end_pos[0];
			out_pos[1] = next_link.end_pos[1];
			out_pos[2] = next_link.end_pos[2];
			return next_cluster;
		}

		cursor = next_link.dst_area_id;
		if (cursor < 0 || cursor >= area_count)
			break;
	}

	return -1;
}

void QNN_ClusterBuildRoutingCache(qnn_route_runtime_t *oracle)
{
	const int area_count = (int)oracle->areas.size();
	const size_t table_size = (size_t)area_count * (size_t)area_count;

	struct QnnRouteNode
	{
		float cost;
		int area_id;
		bool operator>(const QnnRouteNode &other) const { return cost > other.cost; }
	};

	oracle->route_entries.assign(table_size, std::vector<QnnRouteEntry>());

	/* Build incoming (reverse) adjacency: for each link src→dst, record
	   the link as an incoming edge of dst. */
	std::vector<std::vector<int>> incoming_links((size_t)area_count);
	for (size_t li = 0; li < oracle->links.size(); ++li)
		incoming_links[(size_t)oracle->links[li].dst_area_id].push_back((int)li);

	std::vector<float> best_cost((size_t)area_count);

	for (int dst = 0; dst < area_count; ++dst)
	{
		std::fill(best_cost.begin(), best_cost.end(), std::numeric_limits<float>::infinity());
		best_cost[(size_t)dst] = 0.0f;

		std::priority_queue<QnnRouteNode, std::vector<QnnRouteNode>, std::greater<QnnRouteNode>> pq;
		pq.push({0.0f, dst});

		while (!pq.empty())
		{
			const QnnRouteNode cur = pq.top();
			pq.pop();
			if (cur.cost > best_cost[(size_t)cur.area_id] + kClusterCostEpsilon)
				continue;

			for (int li : incoming_links[(size_t)cur.area_id])
			{
				const QnnNavLink &link = oracle->links[(size_t)li];
				const float new_cost = cur.cost + link.travel_time;

				if (new_cost + kClusterCostEpsilon < best_cost[(size_t)link.src_area_id])
				{
					best_cost[(size_t)link.src_area_id] = new_cost;
					pq.push({new_cost, link.src_area_id});
				}
			}
		}

		for (int src = 0; src < area_count; ++src)
		{
			if (!std::isfinite(best_cost[(size_t)src]))
				continue;

			const int src_cluster = oracle->areas[(size_t)src].cluster_id;
			const size_t idx = (size_t)src * (size_t)area_count + (size_t)dst;
			std::vector<QnnRouteEntry> &entries = oracle->route_entries[idx];

			/* Keep only the cheapest link per distinct cluster exit.
			   Each entry represents a different doorway the player
			   can leave through to reach dst. */
			for (int link_id : oracle->outgoing_links[(size_t)src])
			{
				const QnnNavLink &link = oracle->links[(size_t)link_id];
				const float remaining_cost = best_cost[(size_t)link.dst_area_id];
				const float total = link.travel_time + remaining_cost;
				int exit_cluster;
				float exit_pos[3];
				bool duplicate;

				if (!std::isfinite(remaining_cost))
					continue;

				/* Find which cluster boundary this link's path
				   crosses first. */
				{
					const int first_dst_cluster = oracle->areas[(size_t)link.dst_area_id].cluster_id;

					if (first_dst_cluster != src_cluster)
					{
						exit_cluster = first_dst_cluster;
						exit_pos[0] = link.end_pos[0];
						exit_pos[1] = link.end_pos[1];
						exit_pos[2] = link.end_pos[2];
					}
					else
					{
						exit_cluster = TraceExitCluster(
							oracle, src_cluster,
							link.dst_area_id, dst,
							best_cost, exit_pos);
					}
				}

				duplicate = false;
				for (size_t ei = 0; ei < entries.size(); ++ei)
				{
					if (entries[ei].exit_cluster == exit_cluster)
					{
						if (total < entries[ei].cost)
						{
							entries[ei].link_id = link.link_id;
							entries[ei].cost = total;
							memcpy(entries[ei].exit_pos, exit_pos, sizeof(exit_pos));
						}
						duplicate = true;
						break;
					}
				}
				if (!duplicate)
				{
					QnnRouteEntry entry;
					entry.link_id = link.link_id;
					entry.cost = total;
					entry.exit_cluster = exit_cluster;
					memcpy(entry.exit_pos, exit_pos, sizeof(exit_pos));
					entries.push_back(entry);
				}
			}
		}
	}
}

const QnnRouteEntry *QNN_ClusterFindBestRouteEntry(const std::vector<QnnRouteEntry> &entries)
{
	const QnnRouteEntry *best_entry;

	best_entry = nullptr;
	for (const QnnRouteEntry &entry : entries)
	{
		if (best_entry == nullptr
			|| entry.cost + kClusterCostEpsilon < best_entry->cost
			|| (fabsf(entry.cost - best_entry->cost) <= kClusterCostEpsilon && entry.link_id < best_entry->link_id))
		{
			best_entry = &entry;
		}
	}
	return best_entry;
}

void QNN_ClusterFillSummaryCounts(qnn_route_runtime_t *oracle)
{
	size_t link_index;

	if (oracle == nullptr)
		return;

	memset(&oracle->summary, 0, sizeof(oracle->summary));
	oracle->summary.area_count = (int)oracle->areas.size();
	oracle->summary.cluster_count = (int)oracle->clusters.size();
	oracle->summary.total_link_count = (int)oracle->links.size();
	if (!oracle->clusters.empty())
	{
		int total_cluster_areas;

		total_cluster_areas = 0;
		oracle->summary.min_cluster_area_count = std::numeric_limits<int>::max();
		for (const QnnNavCluster &cluster : oracle->clusters)
		{
			total_cluster_areas += cluster.area_count;
			oracle->summary.min_cluster_area_count = std::min(oracle->summary.min_cluster_area_count, cluster.area_count);
			oracle->summary.max_cluster_area_count = std::max(oracle->summary.max_cluster_area_count, cluster.area_count);
		}
		oracle->summary.avg_cluster_area_count = (float)total_cluster_areas / (float)oracle->clusters.size();
	}

	for (link_index = 0; link_index < oracle->links.size(); ++link_index)
	{
		switch (oracle->links[link_index].travel_type)
		{
		case QNN_TRAVEL_WALK:
			oracle->summary.walk_link_count += 1;
			break;
		case QNN_TRAVEL_TELEPORT:
			oracle->summary.teleport_link_count += 1;
			break;
		case QNN_TRAVEL_ELEVATOR:
			oracle->summary.lift_link_count += 1;
			break;
		case QNN_TRAVEL_PUSH:
			oracle->summary.push_link_count += 1;
			break;
		case QNN_TRAVEL_DROP:
			oracle->summary.drop_link_count += 1;
			break;
		default:
			break;
		}
	}
}

int QNN_ClusterWalkCachedRoute(
	const qnn_route_runtime_t *oracle,
	int start_area_id,
	int end_area_id,
	int *out_area_ids,
	int *out_link_ids,
	int max_areas)
{
	const int area_count = (int)oracle->areas.size();
	int cursor = start_area_id;
	int area_idx = 0;
	int link_idx = 0;

	while (cursor != end_area_id && area_idx < max_areas)
	{
		const size_t idx = (size_t)cursor * (size_t)area_count + (size_t)end_area_id;
		const QnnRouteEntry *entry = QNN_ClusterFindBestRouteEntry(oracle->route_entries[idx]);
		const int link_id = entry != nullptr ? entry->link_id : -1;

		if (out_area_ids != nullptr)
			out_area_ids[area_idx] = cursor;
		area_idx++;

		if (link_id < 0 || link_id >= (int)oracle->links.size())
			break;
		if (out_link_ids != nullptr && link_idx < max_areas - 1)
			out_link_ids[link_idx++] = link_id;
		cursor = oracle->links[(size_t)link_id].dst_area_id;
		if (cursor < 0 || cursor >= area_count)
			break;
	}

	if (cursor == end_area_id && area_idx < max_areas)
	{
		if (out_area_ids != nullptr)
			out_area_ids[area_idx] = end_area_id;
		area_idx++;
	}

	return area_idx;
}
