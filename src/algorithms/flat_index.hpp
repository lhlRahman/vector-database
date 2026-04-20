#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../core/vector.hpp"
#include "../core/vector_accessor.hpp"
#include "../utils/distance_metric_policies.hpp"

template <RawDistanceMetric MetricPolicy>
class FlatIndex {
public:
    FlatIndex(size_t dimensions, VectorAccessor accessor, MetricPolicy metric = MetricPolicy{})
        : dimensions_(dimensions),
          accessor_(std::move(accessor)),
          metric_(std::move(metric)) {}

    std::vector<std::pair<std::string, float>>
    search(const Vector& query,
           size_t k,
           const std::unordered_map<std::string, uint64_t>& key_to_slot) const {
        const std::span<const float> q(query.data_ptr(), dimensions_);
        std::vector<std::pair<std::string, float>> distances;
        distances.reserve(key_to_slot.size());

        for (const auto& [key, slot_id] : key_to_slot) {
            const std::span<const float> vec(accessor_(slot_id), dimensions_);
            distances.emplace_back(key, metric_.distance(q, vec));
        }

        const size_t actual_k = std::min(k, distances.size());
        std::partial_sort(distances.begin(),
                          distances.begin() + static_cast<std::ptrdiff_t>(actual_k),
                          distances.end(),
                          [](const auto& a, const auto& b) {
                              return a.second < b.second;
                          });
        distances.resize(actual_k);
        return distances;
    }

private:
    size_t dimensions_;
    VectorAccessor accessor_;
    MetricPolicy metric_;
};
