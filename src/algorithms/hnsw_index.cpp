
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "hnsw_index.hpp"

// HNSWNode implementation
HNSWIndex::HNSWNode::HNSWNode(const Vector& vec, const std::string& k, int lvl)
    : vector(vec), key(k), level(lvl) {
    neighbors.resize(level + 1);
    distances.resize(level + 1);
}

void HNSWIndex::HNSWNode::addNeighbor(size_t neighbor_id, float distance, int level) {
    if (level >= 0 && level < static_cast<int>(neighbors.size())) {
        neighbors[level].push_back(neighbor_id);
        distances[level].push_back(distance);
    }
}

void HNSWIndex::HNSWNode::removeNeighbor(size_t neighbor_id, int level) {
    if (level >= 0 && level < static_cast<int>(neighbors.size())) {
        auto& level_neighbors = neighbors[level];
        auto& level_distances = distances[level];
        
        auto it = std::find(level_neighbors.begin(), level_neighbors.end(), neighbor_id);
        if (it != level_neighbors.end()) {
            size_t index = std::distance(level_neighbors.begin(), it);
            level_neighbors.erase(it);
            level_distances.erase(level_distances.begin() + index);
        }
    }
}

const std::vector<size_t>& HNSWIndex::HNSWNode::getNeighbors(int level) const {
    if (level >= 0 && level < static_cast<int>(neighbors.size())) {
        return neighbors[level];
    }
    static const std::vector<size_t> empty;
    return empty;
}

const std::vector<float>& HNSWIndex::HNSWNode::getDistances(int level) const {
    if (level >= 0 && level < static_cast<int>(distances.size())) {
        return distances[level];
    }
    static const std::vector<float> empty;
    return empty;
}

// HNSWIndex implementation
HNSWIndex::HNSWIndex(size_t dims, size_t M, size_t ef_construction, size_t ef_search, const DistanceMetric* metric)
    : max_connections(M), 
      max_connections_zero(M * 2),  // Layer 0 typically has more connections
      ef_construction(ef_construction),
      ef_search(ef_search),
      ml(1.0f / std::log(static_cast<float>(M))),
      max_level(0),
      dimensions(dims),
      distance_metric(metric ? metric : new EuclideanDistance()),
      rng(std::random_device{}()),
      uniform_dist(0.0f, 1.0f) {
    
    entry_points.push_back(0);  // Initialize with dummy entry point
}

int HNSWIndex::getRandomLevel() const {
    float r = uniform_dist(rng);
    return static_cast<int>(-std::log(r) * ml);
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
    std::vector<float> distances;
    distances.reserve(node_ids.size());
    for (size_t node_id : node_ids) {
        distances.push_back(getDistance(query, nodes[node_id].vector));
    }
    return distances;
}

std::vector<size_t> HNSWIndex::searchLayer(const Vector& query, size_t ef, int level) const {
    if (nodes.empty()) {
        return {};
    }

    std::priority_queue<SearchCandidate, std::vector<SearchCandidate>, std::greater<SearchCandidate>> candidates;
    std::vector<VisitedElement> visited;
    std::unordered_set<size_t> visited_set;

    // Start from entry point at this level
    size_t current_entry = entry_points[std::min(level, static_cast<int>(max_level))];
    if (current_entry >= nodes.size()) {
        return {};
    }

    float dist = getDistance(query, nodes[current_entry].vector);
    candidates.push({current_entry, dist});
    visited.push_back({current_entry, dist});
    visited_set.insert(current_entry);

    while (!candidates.empty()) {
        SearchCandidate current = candidates.top();
        candidates.pop();

        // Check if we can improve
        if (!visited.empty() && current.distance > visited.back().distance) {
            break;
        }

        // Explore neighbors
        const auto& neighbors = nodes[current.node_id].getNeighbors(level);
        for (size_t neighbor_id : neighbors) {
            if (visited_set.find(neighbor_id) != visited_set.end()) {
                continue;
            }

            visited_set.insert(neighbor_id);
            float neighbor_dist = getDistance(query, nodes[neighbor_id].vector);
            
            if (visited.size() < ef || neighbor_dist < visited.back().distance) {
                candidates.push({neighbor_id, neighbor_dist});
                visited.push_back({neighbor_id, neighbor_dist});
                
                // Keep visited list sorted and limited to ef
                std::sort(visited.begin(), visited.end());
                if (visited.size() > ef) {
                    visited.resize(ef);
                }
            }
        }
    }

    // Extract results
    std::vector<size_t> results;
    results.reserve(visited.size());
    for (const auto& elem : visited) {
        results.push_back(elem.node_id);
    }
    
    return results;
}

std::vector<size_t> HNSWIndex::searchLayerBase(const Vector& query, size_t ef) const {
    return searchLayer(query, ef, 0);
}

std::vector<size_t> HNSWIndex::selectNeighbors(const Vector& query, 
                                              const std::vector<size_t>& candidates, 
                                              size_t M, int level) const {
    if (candidates.size() <= M) {
        return candidates;
    }

    // Simple greedy selection (can be improved with more sophisticated algorithms)
    std::vector<std::pair<float, size_t>> candidates_with_dist;
    candidates_with_dist.reserve(candidates.size());
    
    for (size_t candidate_id : candidates) {
        float dist = getDistance(query, nodes[candidate_id].vector);
        candidates_with_dist.emplace_back(dist, candidate_id);
    }
    
    std::sort(candidates_with_dist.begin(), candidates_with_dist.end());
    
    std::vector<size_t> selected;
    selected.reserve(M);
    for (size_t i = 0; i < M; ++i) {
        selected.push_back(candidates_with_dist[i].second);
    }
    
    return selected;
}

std::vector<size_t> HNSWIndex::selectNeighborsSimple(const std::vector<size_t>& candidates, size_t M) const {
    if (candidates.size() <= M) {
        return candidates;
    }
    
    std::vector<size_t> selected(candidates.begin(), candidates.begin() + M);
    return selected;
}

void HNSWIndex::addConnections(size_t node_id, const std::vector<size_t>& candidates, int level) {
    if (candidates.empty()) {
        return;
    }

    size_t M = (level == 0) ? max_connections_zero : max_connections;
    std::vector<size_t> selected = selectNeighbors(nodes[node_id].vector, candidates, M, level);
    
    // Add bidirectional connections
    for (size_t neighbor_id : selected) {
        float dist = getDistance(nodes[node_id].vector, nodes[neighbor_id].vector);
        
        // Add connection from node to neighbor
        nodes[node_id].addNeighbor(neighbor_id, dist, level);
        
        // Add connection from neighbor to node
        nodes[neighbor_id].addNeighbor(node_id, dist, level);
    }
}

void HNSWIndex::insert(const Vector& vector, const std::string& key) {
    if (vector.size() != dimensions) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

    // Generate random level for new node
    int level = getRandomLevel();
    
    // Find entry point (highest level)
    size_t current_entry = 0;
    if (!nodes.empty() && max_level >= 0) {
        current_entry = entry_points[std::min(max_level, static_cast<size_t>(std::numeric_limits<int>::max()))];
    }
    
    // Search for nearest neighbors at each level
    std::vector<size_t> candidates;
    for (int l = max_level; l > level; --l) {
        candidates = searchLayer(vector, ef_construction, l);
        if (!candidates.empty()) {
            current_entry = candidates[0];  // Closest neighbor becomes entry point
        }
    }
    
    // Insert node at its level
    size_t new_node_id = nodes.size();
    nodes.emplace_back(vector, key, level);
    
    // Add connections at each level
    for (int l = std::min(level, static_cast<int>(max_level)); l >= 0; --l) {
        candidates = searchLayer(vector, ef_construction, l);
        addConnections(new_node_id, candidates, l);
    }
    
    // Update entry points if necessary
    if (level > static_cast<int>(max_level)) {
        max_level = level;
        entry_points.resize(max_level + 1);
        entry_points[max_level] = new_node_id;
    }
}

std::vector<std::pair<std::string, float>> HNSWIndex::search(const Vector& query, size_t k) const {
    if (nodes.empty()) {
        return {};
    }

    // Start from highest level entry point
    size_t current_entry = entry_points[max_level];
    
    // Search through levels
    std::vector<size_t> candidates;
    for (int l = max_level; l > 0; --l) {
        candidates = searchLayer(query, ef_search, l);
        if (!candidates.empty()) {
            current_entry = candidates[0];
        }
    }
    
    // Search at bottom level
    candidates = searchLayer(query, ef_search, 0);
    
    // Sort by distance and return top-k results
    std::vector<std::pair<float, size_t>> results_with_dist;
    results_with_dist.reserve(candidates.size());
    
    for (size_t candidate_id : candidates) {
        float dist = getDistance(query, nodes[candidate_id].vector);
        results_with_dist.emplace_back(dist, candidate_id);
    }
    
    std::sort(results_with_dist.begin(), results_with_dist.end());
    
    std::vector<std::pair<std::string, float>> results;
    size_t result_count = std::min(k, results_with_dist.size());
    results.reserve(result_count);
    
    for (size_t i = 0; i < result_count; ++i) {
        size_t node_id = results_with_dist[i].second;
        results.emplace_back(nodes[node_id].key, results_with_dist[i].first);
    }
    
    return results;
}

void HNSWIndex::setEfSearch(size_t ef) {
    ef_search = ef;
}

void HNSWIndex::printStats() const {
    std::cout << "HNSW Index Statistics:" << std::endl;
    std::cout << "  Total nodes: " << nodes.size() << std::endl;
    std::cout << "  Max level: " << max_level << std::endl;
    std::cout << "  Dimensions: " << dimensions << std::endl;
    std::cout << "  Max connections: " << max_connections << std::endl;
    std::cout << "  EF construction: " << ef_construction << std::endl;
    std::cout << "  EF search: " << ef_search << std::endl;
    
    // Level distribution
    std::vector<size_t> level_dist = getLevelDistribution();
    std::cout << "  Level distribution:" << std::endl;
    for (size_t i = 0; i < level_dist.size(); ++i) {
        std::cout << "    Level " << i << ": " << level_dist[i] << " nodes" << std::endl;
    }
}

std::vector<size_t> HNSWIndex::getLevelDistribution() const {
    std::vector<size_t> distribution(max_level + 1, 0);
    for (const auto& node : nodes) {
        if (node.level >= 0 && node.level < static_cast<int>(distribution.size())) {
            distribution[node.level]++;
        }
    }
    return distribution;
}
