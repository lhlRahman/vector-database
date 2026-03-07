
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
HNSWIndex::HNSWNode::HNSWNode(const Vector& vec, const std::string& k, size_t lvl)
    : vector(vec), key(k), level(lvl) {
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
HNSWIndex::HNSWIndex(size_t dims, size_t M, size_t ef_construction, size_t ef_search, std::shared_ptr<const DistanceMetric> metric)
    : max_connections(M),
      max_connections_zero(M * 2),  // Layer 0 typically has more connections
      ef_construction(ef_construction),
      ef_search(ef_search),
      ml(1.0f / std::log(static_cast<float>(M))),
      max_level(0),
      dimensions(dims),
      distance_metric(metric ? std::move(metric) : std::make_shared<EuclideanDistance>()),
      rng(std::random_device{}()),
      uniform_dist(0.0f, 1.0f) {
}

size_t HNSWIndex::getRandomLevel() const {
    float r = uniform_dist(rng);
    if (r <= 0.0f) r = std::numeric_limits<float>::min();
    return static_cast<size_t>(-std::log(r) * ml);
}

float HNSWIndex::getDistance(const Vector& v1, const Vector& v2) const {
    if (distance_metric) {
        return distance_metric->distance(v1, v2);
    }
    // Fallback to Euclidean distance
    float sum = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

std::vector<float> HNSWIndex::getDistances(const Vector& query, const std::vector<size_t>& node_ids) const {
    std::vector<float> dists;
    dists.reserve(node_ids.size());
    for (size_t node_id : node_ids) {
        dists.push_back(getDistance(query, nodes[node_id].vector));
    }
    return dists;
}

std::vector<HNSWIndex::SearchCandidate> HNSWIndex::searchLayer(const Vector& query, size_t ef, size_t level) const {
    if (nodes.empty()) {
        return {};
    }

    std::priority_queue<SearchCandidate, std::vector<SearchCandidate>, std::greater<SearchCandidate>> candidates;
    std::priority_queue<SearchCandidate> result_set;
    std::unordered_set<size_t> visited_set;

    size_t entry_level = std::min(level, max_level);
    if (entry_level >= entry_points.size() || entry_points[entry_level] >= nodes.size()) {
        return {};
    }
    size_t current_entry = entry_points[entry_level];

    float dist = getDistance(query, nodes[current_entry].vector);
    candidates.push({current_entry, dist});
    result_set.push({current_entry, dist});
    visited_set.insert(current_entry);

    while (!candidates.empty()) {
        SearchCandidate current = candidates.top();
        candidates.pop();

        if (current.distance > result_set.top().distance) {
            break;
        }

        const auto& nbrs = nodes[current.node_id].getNeighbors(level);
        for (size_t neighbor_id : nbrs) {
            if (neighbor_id >= nodes.size() || visited_set.count(neighbor_id) > 0) {
                continue;
            }

            visited_set.insert(neighbor_id);
            float neighbor_dist = getDistance(query, nodes[neighbor_id].vector);

            if (result_set.size() < ef || neighbor_dist < result_set.top().distance) {
                candidates.push({neighbor_id, neighbor_dist});
                result_set.push({neighbor_id, neighbor_dist});

                if (result_set.size() > ef) {
                    result_set.pop();
                }
            }
        }
    }

    // Return candidates WITH distances to avoid recalculation
    std::vector<SearchCandidate> results;
    results.reserve(result_set.size());
    while (!result_set.empty()) {
        results.push_back(result_set.top());
        result_set.pop();
    }

    return results;
}

std::vector<HNSWIndex::SearchCandidate> HNSWIndex::searchLayerBase(const Vector& query, size_t ef) const {
    return searchLayer(query, ef, 0);
}

std::vector<HNSWIndex::SearchCandidate> HNSWIndex::selectNeighbors(
        const std::vector<SearchCandidate>& candidates, size_t M) const {
    if (candidates.size() <= M) {
        return candidates;
    }

    // Distances already computed — just partial_sort
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
    if (candidates.empty()) {
        return;
    }

    size_t M = (level == 0) ? max_connections_zero : max_connections;
    auto selected = selectNeighbors(candidates, M);

    // Add bidirectional connections — distances already available
    for (const auto& sel : selected) {
        nodes[node_id].addNeighbor(sel.node_id, sel.distance, level);
        nodes[sel.node_id].addNeighbor(node_id, sel.distance, level);
    }
}

void HNSWIndex::insert(const Vector& vector, const std::string& key) {
    if (vector.size() != dimensions) {
        throw std::invalid_argument("Vector dimension mismatch");
    }
    deleted_keys_.erase(key);

    // Generate random level for new node
    size_t level = getRandomLevel();

    // Search for nearest neighbors from top levels down
    std::vector<SearchCandidate> candidates;
    if (!nodes.empty()) {
        for (size_t l = max_level; l > level && l > 0; --l) {
            candidates = searchLayer(vector, ef_construction, l);
        }
    }

    // Insert node at its level
    size_t new_node_id = nodes.size();
    nodes.emplace_back(vector, key, level);

    // Add connections at each level from min(level, max_level) down to 0
    size_t connect_from = std::min(level, max_level);
    for (size_t l = 0; l <= connect_from; ++l) {
        candidates = searchLayer(vector, ef_construction, l);
        addConnections(new_node_id, candidates, l);
    }

    // Update entry points if necessary
    if (level > max_level) {
        max_level = level;
        entry_points.resize(max_level + 1);
        entry_points[max_level] = new_node_id;
    }

    // Ensure entry_points has a valid entry for level 0 on first insert
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
    if (nodes.empty()) {
        return {};
    }

    // Search through levels from top to 1
    std::vector<SearchCandidate> candidates;
    for (size_t l = max_level; l > 0; --l) {
        candidates = searchLayer(query, ef_search, l);
    }

    // Search at bottom level — distances already computed
    candidates = searchLayer(query, ef_search, 0);

    // Filter deleted keys (distances already available — no recalculation)
    std::vector<SearchCandidate> filtered;
    filtered.reserve(candidates.size());
    for (const auto& c : candidates) {
        if (c.node_id < nodes.size() && deleted_keys_.count(nodes[c.node_id].key) == 0) {
            filtered.push_back(c);
        }
    }

    // Use partial_sort for top-k instead of full sort
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

    // Level distribution
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
