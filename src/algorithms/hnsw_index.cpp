
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "hnsw_index.hpp"

#include "../optimizations/simd_operations.hpp"  // direct SIMD kernels for the devirtualized hot path

// HNSWNode implementation
HNSWIndex::HNSWNode::HNSWNode(uint64_t sid, const std::string& k, size_t lvl,
                             size_t max_conn, size_t max_conn_zero, std::pmr::memory_resource* resource)
    : slot_id(sid), key(k), level(lvl), neighbors(resource), neighbor_dists(resource) {
    neighbors.reserve(level + 1);
    neighbor_dists.reserve(level + 1);
    for (size_t i = 0; i <= level; ++i) {
        // Reserve each level's adjacency to its max degree (+1 for the transient
        // over-fill before pruning) so the lists don't reallocate during build.
        size_t cap = (i == 0 ? max_conn_zero : max_conn) + 1;
        std::pmr::vector<size_t> level_neighbors(resource);
        std::pmr::vector<float> level_distances(resource);
        level_neighbors.reserve(cap);
        level_distances.reserve(cap);
        neighbors.push_back(std::move(level_neighbors));
        neighbor_dists.push_back(std::move(level_distances));
    }
}

void HNSWIndex::HNSWNode::addNeighbor(size_t neighbor_id, float distance, size_t at_level) {
    if (at_level < neighbors.size()) {
        neighbors[at_level].push_back(neighbor_id);
        neighbor_dists[at_level].push_back(distance);
    }
}

void HNSWIndex::HNSWNode::removeNeighbor(size_t neighbor_id, size_t at_level) {
    if (at_level < neighbors.size()) {
        auto& level_neighbors = neighbors[at_level];
        auto& level_distances = neighbor_dists[at_level];

        if (auto it = std::ranges::find(level_neighbors, neighbor_id);
            it != level_neighbors.end()) {
            auto idx = static_cast<size_t>(std::distance(level_neighbors.begin(), it));
            level_neighbors.erase(it);
            level_distances.erase(level_distances.begin() + static_cast<std::ptrdiff_t>(idx));
        }
    }
}

void HNSWIndex::HNSWNode::assignNeighbors(size_t at_level, const std::vector<size_t>& ids,
                                          const std::vector<float>& ds) {
    if (at_level >= neighbors.size()) return;
    neighbors[at_level].assign(ids.begin(), ids.end());
    neighbor_dists[at_level].assign(ds.begin(), ds.end());
}

const std::pmr::vector<size_t>& HNSWIndex::HNSWNode::getNeighbors(size_t at_level) const {
    if (at_level < neighbors.size()) {
        return neighbors[at_level];
    }
    static const std::pmr::vector<size_t> empty;
    return empty;
}

const std::pmr::vector<float>& HNSWIndex::HNSWNode::getNeighborDists(size_t at_level) const {
    if (at_level < neighbor_dists.size()) {
        return neighbor_dists[at_level];
    }
    static const std::pmr::vector<float> empty;
    return empty;
}

// HNSWIndex implementation
HNSWIndex::HNSWIndex(size_t dims, size_t M, size_t ef_construction, size_t ef_search,
                     std::shared_ptr<const DistanceMetric> metric, VectorAccessor accessor,
                     AllocationStrategy allocation_strategy, size_t arena_initial_size,
                     uint32_t seed)
    : max_connections(M),
      max_connections_zero(M * 2),
      ef_construction(ef_construction),
      ef_search(ef_search),
      // Guard M<2: log(1)==0 -> ml==inf -> getRandomLevel() casts inf to size_t (UB).
      ml(1.0f / std::log(static_cast<float>(M < 2 ? 2 : M))),
      allocation_strategy(allocation_strategy),
      arena_resource_(&tracking_resource_),  // pool sizes itself; arena_initial_size kept for API compat
      index_resource_(allocation_strategy == AllocationStrategy::Arena
                          ? static_cast<std::pmr::memory_resource*>(&arena_resource_)
                          : static_cast<std::pmr::memory_resource*>(&tracking_resource_)),
      nodes(index_resource_),
      entry_points(index_resource_),
      max_level(0),
      dimensions(dims),
      distance_metric(metric ? std::move(metric) : std::make_shared<EuclideanDistance>()),
      accessor_(std::move(accessor)),
      rng(seed),
      uniform_dist(0.0f, 1.0f) {
    (void)arena_initial_size;  // pool self-sizes; param retained for API compatibility
    // Resolve the concrete metric once (mirrors exactSearch's dynamic_cast) so
    // the hot path dispatches without a virtual call.
    if (dynamic_cast<const EuclideanDistance*>(distance_metric.get())) metric_kind_ = MetricKind::Euclidean;
    else if (dynamic_cast<const ManhattanDistance*>(distance_metric.get())) metric_kind_ = MetricKind::Manhattan;
    else if (dynamic_cast<const CosineSimilarity*>(distance_metric.get())) metric_kind_ = MetricKind::Cosine;
    else metric_kind_ = MetricKind::Virtual;
}

size_t HNSWIndex::getRandomLevel() const {
    float r = uniform_dist(rng);
    if (r <= 0.0f) r = std::numeric_limits<float>::min();
    // Cap the level: a degenerate ml or an unlucky draw could otherwise produce
    // an astronomically large level and OOM on node construction.
    double lvl = -std::log(r) * static_cast<double>(ml);
    if (!(lvl > 0.0)) lvl = 0.0;   // also catches NaN
    if (lvl > 31.0) lvl = 31.0;
    return static_cast<size_t>(lvl);
}

float HNSWIndex::getDistance(const float* a, const float* b) const {
    // Devirtualized dispatch — identical results to the corresponding
    // DistanceMetric::distance_raw / RawDistanceMetric policy, but no vtable
    // lookup in the graph-walk inner loop.
    switch (metric_kind_) {
        case MetricKind::Euclidean:
            return std::sqrt(simd_ops::squared_distance(a, b, dimensions));
        case MetricKind::Manhattan:
            return simd_ops::manhattan_distance(a, b, dimensions);
        case MetricKind::Cosine: {
            const float dot = simd_ops::dot_product(a, b, dimensions);
            const float na = simd_ops::dot_product(a, a, dimensions);
            const float nb = simd_ops::dot_product(b, b, dimensions);
            if (na == 0.0f || nb == 0.0f) return 1.0f;
            return 1.0f - dot / (std::sqrt(na) * std::sqrt(nb));
        }
        case MetricKind::Virtual:
        default:
            if (distance_metric) {
                return distance_metric->distance_raw(std::span<const float>(a, dimensions),
                                                     std::span<const float>(b, dimensions));
            }
            float sum = 0.0f;
            for (size_t i = 0; i < dimensions; ++i) {
                float diff = a[i] - b[i];
                sum += diff * diff;
            }
            return std::sqrt(sum);
    }
}

std::vector<HNSWIndex::SearchCandidate> HNSWIndex::searchLayer(const float* query, size_t ef, size_t level, size_t entry_point) const {
    if (nodes.empty() || entry_point >= nodes.size()) return {};

    std::priority_queue<SearchCandidate, std::vector<SearchCandidate>, std::greater<SearchCandidate>> candidates;
    std::priority_queue<SearchCandidate> result_set;

    // Per-thread versioned visited set: node `id` is visited iff visited[id]==VER.
    // Bumping VER resets the whole set in O(1) with no allocation or hashing per
    // query. thread_local keeps concurrent readers race-free (searches run under
    // a shared ReadGuard, so `nodes` is stable for the call's duration).
    thread_local std::vector<uint32_t> visited;
    thread_local uint32_t visit_ver = 0;
    if (visited.size() < nodes.size()) { visited.assign(nodes.size(), 0); visit_ver = 0; }
    if (++visit_ver == 0) { std::fill(visited.begin(), visited.end(), 0); visit_ver = 1; }  // wrap
    const uint32_t VER = visit_ver;

    float dist = getDistance(query, accessor_(nodes[entry_point].slot_id));
    candidates.push({entry_point, dist});
    result_set.push({entry_point, dist});
    visited[entry_point] = VER;

    while (!candidates.empty()) {
        SearchCandidate current = candidates.top();
        candidates.pop();

        if (current.distance > result_set.top().distance) break;

        const auto& nbrs = nodes[current.node_id].getNeighbors(level);
        for (size_t i = 0; i < nbrs.size(); ++i) {
            size_t neighbor_id = nbrs[i];
            if (neighbor_id >= nodes.size() || visited[neighbor_id] == VER) continue;

            visited[neighbor_id] = VER;

            if (i + 1 < nbrs.size()) {
                size_t next_id = nbrs[i + 1];
                if (next_id < nodes.size()) {
                    __builtin_prefetch(accessor_(nodes[next_id].slot_id), 0, 1);
                }
            }

            float neighbor_dist = getDistance(query, accessor_(nodes[neighbor_id].slot_id));

            if (result_set.size() < ef || neighbor_dist < result_set.top().distance) {
                candidates.push({neighbor_id, neighbor_dist});
                result_set.push({neighbor_id, neighbor_dist});
                if (result_set.size() > ef) result_set.pop();
            }
        }
    }

    std::vector<SearchCandidate> results;
    results.reserve(result_set.size());
    while (!result_set.empty()) {
        results.push_back(result_set.top());
        result_set.pop();
    }
    return results;
}

std::vector<HNSWIndex::SearchCandidate> HNSWIndex::selectNeighbors(
        const std::vector<SearchCandidate>& candidates, size_t M) const {
    if (candidates.size() <= M) return candidates;

    // Malkov-Yashunin Algorithm 4. Process candidates nearest-first; keep e only
    // if no already-selected neighbor r is closer to e than the reference point
    // is (i.e. e is not "dominated"). This preserves diverse / long-range links,
    // which the naive "keep the M closest" rule destroys — the difference between
    // a navigable graph and one where greedy search gets trapped.
    std::vector<SearchCandidate> sorted = candidates;
    std::ranges::sort(sorted, {}, &SearchCandidate::distance);  // ascending dist to reference

    std::vector<SearchCandidate> result;
    result.reserve(M);
    for (const auto& e : sorted) {
        if (result.size() >= M) break;
        const float* e_vec = accessor_(nodes[e.node_id].slot_id);
        bool keep = true;
        for (const auto& r : result) {
            if (getDistance(e_vec, accessor_(nodes[r.node_id].slot_id)) < e.distance) {
                keep = false;  // r dominates e
                break;
            }
        }
        if (keep) result.push_back(e);
    }

    // keepPrunedConnections: if the heuristic under-filled, backfill with the
    // closest remaining candidates so the node isn't under-connected.
    if (result.size() < M) {
        for (const auto& e : sorted) {
            if (result.size() >= M) break;
            bool present = false;
            for (const auto& r : result) {
                if (r.node_id == e.node_id) { present = true; break; }
            }
            if (!present) result.push_back(e);
        }
    }
    return result;
}

void HNSWIndex::pruneNeighbors(size_t node_id, size_t level, size_t M) {
    const auto& nbrs = nodes[node_id].getNeighbors(level);
    if (nbrs.size() <= M) return;
    const auto& dists = nodes[node_id].getNeighborDists(level);

    std::vector<SearchCandidate> cands;
    cands.reserve(nbrs.size());
    for (size_t i = 0; i < nbrs.size(); ++i) cands.push_back({nbrs[i], dists[i]});

    auto selected = selectNeighbors(cands, M);  // reference point is this node
    std::vector<size_t> ids;
    std::vector<float> ds;
    ids.reserve(selected.size());
    ds.reserve(selected.size());
    for (const auto& s : selected) { ids.push_back(s.node_id); ds.push_back(s.distance); }
    nodes[node_id].assignNeighbors(level, ids, ds);
}

void HNSWIndex::addConnections(size_t node_id, const std::vector<SearchCandidate>& candidates, size_t level) {
    if (candidates.empty()) return;

    size_t M = (level == 0) ? max_connections_zero : max_connections;
    auto selected = selectNeighbors(candidates, M);

    for (const auto& sel : selected) {
        if (sel.node_id == node_id) continue;  // never self-connect
        nodes[node_id].addNeighbor(sel.node_id, sel.distance, level);
        nodes[sel.node_id].addNeighbor(node_id, sel.distance, level);
        // "Shrink connections" step: re-select the neighbor's list down to M with
        // the SAME diversifying heuristic (not just the M closest), preserving
        // long-range links and bounding hub-node degree.
        pruneNeighbors(sel.node_id, level, M);
    }
}

void HNSWIndex::insert(uint64_t slot_id, const std::string& key) {
    const float* vec = accessor_(slot_id);

    // If this key already has a live node (e.g. an in-place update that reuses
    // the same slot), tombstone that node so results don't contain duplicates.
    if (auto it = key_to_node_.find(key); it != key_to_node_.end()) {
        deleted_node_ids_.insert(it->second);
    }
    deleted_keys_.erase(key);
    deleted_slots_.erase(slot_id);

    size_t level = getRandomLevel();

    if (nodes.empty()) {
        nodes.emplace_back(slot_id, key, level, max_connections, max_connections_zero, index_resource_);
        key_to_node_[key] = 0;
        max_level = level;
        entry_points.assign(level + 1, 0);
        return;
    }

    // Phase 1: descend from the global top entry point down to level+1 with a
    // greedy (ef=1) walk, chaining the closest node found as the next entry.
    size_t current_ep = entry_points[max_level];
    if (current_ep >= nodes.size()) current_ep = 0;
    for (size_t l = max_level; l > level; --l) {
        auto r = searchLayer(vec, 1, l, current_ep);
        if (!r.empty()) current_ep = r.back().node_id;  // back() == closest
    }

    size_t new_node_id = nodes.size();
    nodes.emplace_back(slot_id, key, level, max_connections, max_connections_zero, index_resource_);
    key_to_node_[key] = new_node_id;

    // Phase 2: connect at levels min(level,max_level)..0, threading the closest
    // candidate at each level as the entry point for the next level down.
    size_t connect_upto = std::min(level, max_level);
    for (size_t l = connect_upto + 1; l-- > 0;) {  // l = connect_upto down to 0
        auto cands = searchLayer(vec, ef_construction, l, current_ep);
        addConnections(new_node_id, cands, l);
        if (!cands.empty()) current_ep = cands.back().node_id;
    }

    // Phase 3: if we drew a taller level, the new node becomes the entry point
    // for every newly-created level (they previously defaulted to node 0/invalid).
    if (level > max_level) {
        entry_points.resize(level + 1);
        for (size_t l = max_level + 1; l <= level; ++l) entry_points[l] = new_node_id;
        max_level = level;
    }
}

void HNSWIndex::remove(const std::string& key) {
    deleted_keys_.insert(key);
    if (auto it = key_to_node_.find(key); it != key_to_node_.end()) {
        deleted_node_ids_.insert(it->second);
        key_to_node_.erase(it);
    }
}

void HNSWIndex::removeSlot(uint64_t slot_id) {
    deleted_slots_.insert(slot_id);
}

std::vector<std::pair<std::string, float>> HNSWIndex::search(const Vector& query, size_t k) const {
    if (nodes.empty()) return {};

    const float* q = query.data_ptr();

    // Descend the hierarchy: greedy ef=1 walk per upper layer, chaining the
    // closest node found as the entry point for the next layer down.
    size_t current_ep = entry_points.empty() ? 0 : entry_points[max_level];
    if (current_ep >= nodes.size()) current_ep = 0;
    for (size_t l = max_level; l > 0; --l) {
        auto r = searchLayer(q, 1, l, current_ep);
        if (!r.empty()) current_ep = r.back().node_id;
    }
    // Base layer: full ef search (at least k so we can return k results).
    auto candidates = searchLayer(q, std::max(ef_search, k), 0, current_ep);

    std::vector<SearchCandidate> filtered;
    filtered.reserve(candidates.size());
    for (const auto& c : candidates) {
        if (c.node_id < nodes.size() &&
            !deleted_node_ids_.contains(c.node_id) &&
            !deleted_keys_.contains(nodes[c.node_id].key) &&
            !deleted_slots_.contains(nodes[c.node_id].slot_id)) {
            filtered.push_back(c);
        }
    }

    size_t result_count = std::min(k, filtered.size());
    std::ranges::partial_sort(filtered,
                              filtered.begin() + static_cast<std::ptrdiff_t>(result_count),
                              {},
                              &SearchCandidate::distance);

    std::vector<std::pair<std::string, float>> results;
    results.reserve(result_count);
    for (size_t i = 0; i < result_count; ++i) {
        results.emplace_back(nodes[filtered[i].node_id].key, filtered[i].distance);
    }
    return results;
}

void HNSWIndex::setEfSearch(size_t ef) {
    ef_search = ef;
}

HNSWIndex::MemoryStatistics HNSWIndex::getMemoryStatistics() const {
    const auto stats = tracking_resource_.getStatistics();
    return MemoryStatistics{
        allocation_strategy,
        stats.allocation_calls,
        stats.deallocation_calls,
        stats.bytes_allocated,
        stats.bytes_deallocated,
        stats.bytes_outstanding,
        stats.peak_bytes_outstanding,
    };
}

HNSWIndex::GraphSnapshot HNSWIndex::exportGraph() const {
    GraphSnapshot snapshot;
    snapshot.dimensions = dimensions;
    snapshot.max_connections = max_connections;
    snapshot.max_connections_zero = max_connections_zero;
    snapshot.ef_construction = ef_construction;
    snapshot.ef_search = ef_search;
    snapshot.max_level = max_level;
    snapshot.entry_points.assign(entry_points.begin(), entry_points.end());
    snapshot.nodes.reserve(nodes.size());

    for (const auto& node : nodes) {
        NodeSnapshot node_snapshot;
        node_snapshot.slot_id = node.slot_id;
        node_snapshot.key = node.key;
        node_snapshot.level = node.level;
        node_snapshot.neighbors.reserve(node.neighbors.size());
        node_snapshot.neighbor_dists.reserve(node.neighbor_dists.size());

        for (const auto& level_neighbors : node.neighbors) {
            node_snapshot.neighbors.emplace_back(level_neighbors.begin(), level_neighbors.end());
        }
        for (const auto& level_distances : node.neighbor_dists) {
            node_snapshot.neighbor_dists.emplace_back(level_distances.begin(), level_distances.end());
        }

        snapshot.nodes.push_back(std::move(node_snapshot));
    }

    snapshot.deleted_keys.reserve(deleted_keys_.size());
    for (const auto& key : deleted_keys_) {
        snapshot.deleted_keys.push_back(key);
    }

    snapshot.deleted_slots.reserve(deleted_slots_.size());
    for (uint64_t slot_id : deleted_slots_) {
        snapshot.deleted_slots.push_back(slot_id);
    }

    return snapshot;
}

void HNSWIndex::importGraph(const GraphSnapshot& snapshot) {
    if (snapshot.dimensions != dimensions) {
        throw std::invalid_argument("HNSW snapshot dimension mismatch");
    }

    max_connections = snapshot.max_connections;
    max_connections_zero = snapshot.max_connections_zero;
    ef_construction = snapshot.ef_construction;
    ef_search = snapshot.ef_search;
    ml = 1.0f / std::log(static_cast<float>(max_connections < 2 ? 2 : max_connections));
    max_level = snapshot.max_level;

    nodes.clear();
    entry_points.clear();
    deleted_keys_.clear();
    deleted_slots_.clear();
    key_to_node_.clear();
    deleted_node_ids_.clear();

    nodes.reserve(snapshot.nodes.size());
    for (const auto& node_snapshot : snapshot.nodes) {
        nodes.emplace_back(node_snapshot.slot_id, node_snapshot.key, node_snapshot.level,
                           max_connections, max_connections_zero, index_resource_);
        auto& node = nodes.back();

        for (size_t level = 0; level < node_snapshot.neighbors.size() && level < node.neighbors.size(); ++level) {
            node.neighbors[level].assign(node_snapshot.neighbors[level].begin(),
                                         node_snapshot.neighbors[level].end());
        }
        for (size_t level = 0; level < node_snapshot.neighbor_dists.size() && level < node.neighbor_dists.size(); ++level) {
            node.neighbor_dists[level].assign(node_snapshot.neighbor_dists[level].begin(),
                                              node_snapshot.neighbor_dists[level].end());
        }
    }

    entry_points.assign(snapshot.entry_points.begin(), snapshot.entry_points.end());
    for (const auto& key : snapshot.deleted_keys) {
        deleted_keys_.insert(key);
    }
    for (uint64_t slot_id : snapshot.deleted_slots) {
        deleted_slots_.insert(slot_id);
    }

    // Rebuild key -> live-node index (last node with a key wins; any earlier
    // same-key nodes are already filtered via deleted_slots_).
    for (size_t i = 0; i < nodes.size(); ++i) {
        key_to_node_[nodes[i].key] = i;
    }
}

void HNSWIndex::printStats() const {
    std::cout << "HNSW Index Statistics:\n";
    std::cout << "  Total nodes: " << nodes.size() << '\n';
    std::cout << "  Max level: " << max_level << '\n';
    std::cout << "  Dimensions: " << dimensions << '\n';
    std::cout << "  Max connections: " << max_connections << '\n';
    std::cout << "  EF construction: " << ef_construction << '\n';
    std::cout << "  EF search: " << ef_search << '\n';

    std::vector<size_t> level_dist = getLevelDistribution();
    std::cout << "  Level distribution:\n";
    for (size_t i = 0; i < level_dist.size(); ++i) {
        std::cout << "    Level " << i << ": " << level_dist[i] << " nodes\n";
    }
}

std::vector<size_t> HNSWIndex::getLevelDistribution() const {
    if (nodes.empty()) return {};
    std::vector<size_t> distribution(max_level + 1, 0);
    for (const auto& node : nodes) {
        if (node.level < distribution.size()) {
            distribution[node.level]++;
        }
    }
    return distribution;
}
