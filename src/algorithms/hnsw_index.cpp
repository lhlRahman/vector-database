
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "hnsw_index.hpp"

// HNSWNode implementation
HNSWIndex::HNSWNode::HNSWNode(uint64_t sid, const std::string& k, size_t lvl)
    : slot_id(sid), key(k), level(lvl) {
    neighbors.resize(level + 1);
    neighbor_dists.resize(level + 1);
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

        auto it = std::find(level_neighbors.begin(), level_neighbors.end(), neighbor_id);
        if (it != level_neighbors.end()) {
            auto idx = static_cast<size_t>(std::distance(level_neighbors.begin(), it));
            level_neighbors.erase(it);
            level_distances.erase(level_distances.begin() + static_cast<std::ptrdiff_t>(idx));
        }
    }
}

const std::vector<size_t>& HNSWIndex::HNSWNode::getNeighbors(size_t at_level) const {
    if (at_level < neighbors.size()) {
        return neighbors[at_level];
    }
    static const std::vector<size_t> empty;
    return empty;
}

const std::vector<float>& HNSWIndex::HNSWNode::getNeighborDists(size_t at_level) const {
    if (at_level < neighbor_dists.size()) {
        return neighbor_dists[at_level];
    }
    static const std::vector<float> empty;
    return empty;
}

// HNSWIndex implementation
HNSWIndex::HNSWIndex(size_t dims, size_t M, size_t ef_construction, size_t ef_search,
                     std::shared_ptr<const DistanceMetric> metric, VectorAccessor accessor)
    : max_connections(M),
      max_connections_zero(M * 2),
      ef_construction(ef_construction),
      ef_search(ef_search),
      ml(1.0f / std::log(static_cast<float>(M))),
      max_level(0),
      dimensions(dims),
      distance_metric(metric ? std::move(metric) : std::make_shared<EuclideanDistance>()),
      accessor_(std::move(accessor)),
      rng(std::random_device{}()),
      uniform_dist(0.0f, 1.0f) {
}

size_t HNSWIndex::getRandomLevel() const {
    float r = uniform_dist(rng);
    if (r <= 0.0f) r = std::numeric_limits<float>::min();
    return static_cast<size_t>(-std::log(r) * ml);
}

float HNSWIndex::getDistance(const float* a, const float* b) const {
    if (distance_metric) {
        return distance_metric->distance_raw(a, b, dimensions);
    }
    float sum = 0.0f;
    for (size_t i = 0; i < dimensions; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

std::vector<HNSWIndex::SearchCandidate> HNSWIndex::searchLayer(const float* query, size_t ef, size_t level) const {
    if (nodes.empty()) return {};

    std::priority_queue<SearchCandidate, std::vector<SearchCandidate>, std::greater<SearchCandidate>> candidates;
    std::priority_queue<SearchCandidate> result_set;
    std::unordered_set<size_t> visited_set;

    size_t entry_level = std::min(level, max_level);
    if (entry_level >= entry_points.size() || entry_points[entry_level] >= nodes.size()) return {};
    size_t current_entry = entry_points[entry_level];

    float dist = getDistance(query, accessor_(nodes[current_entry].slot_id));
    candidates.push({current_entry, dist});
    result_set.push({current_entry, dist});
    visited_set.insert(current_entry);

    while (!candidates.empty()) {
        SearchCandidate current = candidates.top();
        candidates.pop();

        if (current.distance > result_set.top().distance) break;

        const auto& nbrs = nodes[current.node_id].getNeighbors(level);
        for (size_t neighbor_id : nbrs) {
            if (neighbor_id >= nodes.size() || visited_set.count(neighbor_id) > 0) continue;

            visited_set.insert(neighbor_id);

            // Prefetch next neighbor's vector data
            if (&neighbor_id != &nbrs.back()) {
                size_t next_id = *(&neighbor_id + 1);
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

std::vector<HNSWIndex::SearchCandidate> HNSWIndex::searchLayerBase(const float* query, size_t ef) const {
    return searchLayer(query, ef, 0);
}

std::vector<HNSWIndex::SearchCandidate> HNSWIndex::selectNeighbors(
        const std::vector<SearchCandidate>& candidates, size_t M) const {
    if (candidates.size() <= M) return candidates;

    auto sorted = candidates;
    std::partial_sort(sorted.begin(),
                      sorted.begin() + static_cast<std::ptrdiff_t>(M),
                      sorted.end(),
                      [](const SearchCandidate& a, const SearchCandidate& b) {
                          return a.distance < b.distance;
                      });
    sorted.resize(M);
    return sorted;
}

void HNSWIndex::addConnections(size_t node_id, const std::vector<SearchCandidate>& candidates, size_t level) {
    if (candidates.empty()) return;

    size_t M = (level == 0) ? max_connections_zero : max_connections;
    auto selected = selectNeighbors(candidates, M);

    for (const auto& sel : selected) {
        nodes[node_id].addNeighbor(sel.node_id, sel.distance, level);
        nodes[sel.node_id].addNeighbor(node_id, sel.distance, level);
    }
}

void HNSWIndex::insert(uint64_t slot_id, const std::string& key) {
    const float* vec = accessor_(slot_id);
    deleted_keys_.erase(key);

    size_t level = getRandomLevel();

    std::vector<SearchCandidate> candidates;
    if (!nodes.empty()) {
        for (size_t l = max_level; l > level && l > 0; --l) {
            candidates = searchLayer(vec, ef_construction, l);
        }
    }

    size_t new_node_id = nodes.size();
    nodes.emplace_back(slot_id, key, level);

    size_t connect_from = std::min(level, max_level);
    for (size_t l = 0; l <= connect_from; ++l) {
        candidates = searchLayer(vec, ef_construction, l);
        addConnections(new_node_id, candidates, l);
    }

    if (level > max_level) {
        max_level = level;
        entry_points.resize(max_level + 1);
        entry_points[max_level] = new_node_id;
    }

    if (nodes.size() == 1) {
        entry_points.resize(max_level + 1);
        for (size_t l = 0; l <= max_level; ++l) {
            entry_points[l] = 0;
        }
    }
}

void HNSWIndex::remove(const std::string& key) {
    deleted_keys_.insert(key);
}

std::vector<std::pair<std::string, float>> HNSWIndex::search(const Vector& query, size_t k) const {
    if (nodes.empty()) return {};

    const float* q = query.data_ptr();

    std::vector<SearchCandidate> candidates;
    for (size_t l = max_level; l > 0; --l) {
        candidates = searchLayer(q, ef_search, l);
    }
    candidates = searchLayer(q, ef_search, 0);

    std::vector<SearchCandidate> filtered;
    filtered.reserve(candidates.size());
    for (const auto& c : candidates) {
        if (c.node_id < nodes.size() && deleted_keys_.count(nodes[c.node_id].key) == 0) {
            filtered.push_back(c);
        }
    }

    size_t result_count = std::min(k, filtered.size());
    std::partial_sort(filtered.begin(),
                      filtered.begin() + static_cast<std::ptrdiff_t>(result_count),
                      filtered.end(),
                      [](const SearchCandidate& a, const SearchCandidate& b) {
                          return a.distance < b.distance;
                      });

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
