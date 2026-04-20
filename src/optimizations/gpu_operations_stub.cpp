#ifndef __APPLE__

#include "gpu_operations.hpp"

namespace gpu_ops {

bool initialize() { return false; }
bool shutdown() { return true; }
bool is_available() { return false; }

bool set_database_buffer(const float*, size_t, size_t) { return false; }
bool update_database_buffer(const float*, size_t, size_t) { return false; }
void clear_database_buffer() {}
bool has_database_buffer() { return false; }
size_t get_buffer_num_vectors() { return 0; }
size_t get_buffer_dimensions() { return 0; }

std::vector<float> search_euclidean(const Vector&) { return {}; }
std::vector<float> search_dot_product(const Vector&) { return {}; }
std::vector<float> search_cosine(const Vector&) { return {}; }

std::vector<float> batch_dot_products(const Vector&, const std::vector<Vector>&) { return {}; }
std::vector<float> batch_euclidean_distances(const Vector&, const std::vector<Vector>&) { return {}; }

std::vector<std::vector<std::pair<size_t, float>>> batch_knn(
    const std::vector<Vector>&,
    const std::vector<Vector>&,
    size_t) {
    return {};
}

} // namespace gpu_ops

#endif
