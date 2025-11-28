// GPU operations with zero-copy shared memory for Apple Silicon

#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include "../core/vector.hpp"

namespace gpu_ops {

// ==================== Lifecycle ====================
bool initialize();
bool shutdown();
bool is_available();

// ==================== Database Buffer Management ====================
// These functions manage a persistent GPU buffer for the database vectors.
// The buffer uses Apple Silicon's unified memory - zero copy between CPU/GPU!

/**
 * Set the database buffer from contiguous float data.
 * This should be called once when data changes (insert/update/delete).
 * 
 * @param data Pointer to contiguous float array [num_vectors * dimensions]
 * @param num_vectors Number of vectors in the database
 * @param dimensions Dimension of each vector
 * @return true if buffer was set successfully
 */
bool set_database_buffer(const float* data, size_t num_vectors, size_t dimensions);

/**
 * Update the database buffer with new data.
 * Call this after inserts/updates/deletes.
 */
bool update_database_buffer(const float* data, size_t num_vectors, size_t dimensions);

/**
 * Clear the database buffer (free GPU memory).
 */
void clear_database_buffer();

/**
 * Check if database buffer is set.
 */
bool has_database_buffer();

/**
 * Get current buffer info.
 */
size_t get_buffer_num_vectors();
size_t get_buffer_dimensions();

// ==================== Search Operations (use persistent buffer) ====================

/**
 * Search using the persistent database buffer (zero-copy).
 * Much faster than the old API since no data copying is needed.
 * 
 * @param query Query vector
 * @return Vector of distances to each database vector
 */
std::vector<float> search_euclidean(const Vector& query);
std::vector<float> search_dot_product(const Vector& query);
std::vector<float> search_cosine(const Vector& query);

// ==================== Legacy API (copies data each call) ====================
// These are slower but don't require buffer management.

std::vector<float> batch_dot_products(
    const Vector& query,
    const std::vector<Vector>& database
);

std::vector<float> batch_euclidean_distances(
    const Vector& query,
    const std::vector<Vector>& database
);

std::vector<std::vector<std::pair<size_t, float>>> batch_knn(
    const std::vector<Vector>& queries,
    const std::vector<Vector>& database,
    size_t k
);

} // namespace gpu_ops
