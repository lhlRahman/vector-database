#include <metal_stdlib>
using namespace metal;

// ============== Batch Dot Products ==============
// Computes dot(query, database[i]) for ALL i
// Works for ANY vector dimension
kernel void batch_dot_products(
    device const float* query [[buffer(0)]],         // [dimensions]
    device const float* database [[buffer(1)]],      // [num_vectors * dimensions]
    device float* results [[buffer(2)]],             // [num_vectors]
    constant uint& dimensions [[buffer(3)]],         // Vector dimension (128, 256, etc.)
    constant uint& num_vectors [[buffer(4)]],        // Number of database vectors
    uint gid [[thread_position_in_grid]]             // Which database vector this thread handles
) {
    if (gid >= num_vectors) return;
    
    // Pointer to this database vector
    device const float* db_vec = database + (gid * dimensions);
    
    // Compute dot product (loop over ALL dimensions)
    float sum = 0.0f;
    for (uint i = 0; i < dimensions; i++) {
        sum += query[i] * db_vec[i];
    }
    
    results[gid] = sum;
}

// ============== Batch Euclidean Distances ==============
kernel void batch_euclidean_distances(
    device const float* query [[buffer(0)]],
    device const float* database [[buffer(1)]],
    device float* results [[buffer(2)]],
    constant uint& dimensions [[buffer(3)]],
    constant uint& num_vectors [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_vectors) return;
    
    device const float* db_vec = database + (gid * dimensions);
    
    float sum = 0.0f;
    for (uint i = 0; i < dimensions; i++) {
        float diff = query[i] - db_vec[i];
        sum += diff * diff;
    }
    
    results[gid] = sqrt(sum);
}

// ============== Batch Manhattan Distances ==============
kernel void batch_manhattan_distances(
    device const float* query [[buffer(0)]],
    device const float* database [[buffer(1)]],
    device float* results [[buffer(2)]],
    constant uint& dimensions [[buffer(3)]],
    constant uint& num_vectors [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_vectors) return;
    
    device const float* db_vec = database + (gid * dimensions);
    
    float sum = 0.0f;
    for (uint i = 0; i < dimensions; i++) {
        sum += abs(query[i] - db_vec[i]);
    }
    
    results[gid] = sum;
}

// ============== Batch Cosine Distances ==============
kernel void batch_cosine_distances(
    device const float* query [[buffer(0)]],
    device const float* database [[buffer(1)]],
    device float* results [[buffer(2)]],
    constant uint& dimensions [[buffer(3)]],
    constant uint& num_vectors [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_vectors) return;
    
    device const float* db_vec = database + (gid * dimensions);
    
    float dot = 0.0f;
    float query_norm_sq = 0.0f;
    float db_norm_sq = 0.0f;
    
    for (uint i = 0; i < dimensions; i++) {
        dot += query[i] * db_vec[i];
        query_norm_sq += query[i] * query[i];
        db_norm_sq += db_vec[i] * db_vec[i];
    }
    
    float query_norm = sqrt(query_norm_sq);
    float db_norm = sqrt(db_norm_sq);
    
    // Cosine distance = 1 - cosine_similarity
    if (query_norm > 0.0f && db_norm > 0.0f) {
        results[gid] = 1.0f - (dot / (query_norm * db_norm));
    } else {
        results[gid] = 1.0f;  // Max distance if zero vector
    }
}