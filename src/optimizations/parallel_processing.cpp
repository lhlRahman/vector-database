#include "parallel_processing.hpp"

#include <functional>
#include <future>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace parallel_ops {

void batchInsert(VectorDatabase& db, const std::vector<Vector>& vectors, const std::vector<std::string>& keys) {
    if (vectors.size() != keys.size()) {
        throw std::invalid_argument("Number of vectors and keys must match");
    }

    auto task = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            // FIX: Call the 3-argument version of insert to resolve ambiguity.
            db.insert(vectors[i], keys[i], "");
        }
    };

    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1; // Fallback for safety
    size_t batch_size = vectors.size() / num_threads;
    if (batch_size == 0 && vectors.size() > 0) { // Handle cases with more threads than items
        batch_size = 1;
        num_threads = vectors.size();
    }
    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * batch_size;
        size_t end = (i == num_threads - 1) ? vectors.size() : start + batch_size;
        if (start < end) {
            futures.push_back(std::async(std::launch::async, task, start, end));
        }
    }

    for (auto& future : futures) {
        future.get();
    }
}

std::vector<std::vector<std::pair<std::string, float>>> batchSimilaritySearch(
    const VectorDatabase& db, const std::vector<Vector>& queries, size_t k) {
    
    std::vector<std::vector<std::pair<std::string, float>>> results(queries.size());

    auto task = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            results[i] = db.similaritySearch(queries[i], k);
        }
    };

    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;
    size_t batch_size = queries.size() / num_threads;
     if (batch_size == 0 && queries.size() > 0) {
        batch_size = 1;
        num_threads = queries.size();
    }
    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * batch_size;
        size_t end = (i == num_threads - 1) ? queries.size() : start + batch_size;
        if (start < end) {
            futures.push_back(std::async(std::launch::async, task, start, end));
        }
    }

    for (auto& future : futures) {
        future.get();
    }

    return results;
}

void parallel_for_each(std::vector<int>& indices, std::function<void(int)>& func) {
    auto task = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            func(indices[i]);
        }
    };

    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;
    size_t batch_size = indices.size() / num_threads;
    if (batch_size == 0 && indices.size() > 0) {
        batch_size = 1;
        num_threads = indices.size();
    }
    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * batch_size;
        size_t end = (i == num_threads - 1) ? indices.size() : start + batch_size;
        if (start < end) {
            futures.push_back(std::async(std::launch::async, task, start, end));
        }
    }

    for (auto& future : futures) {
        future.get();
    }
}

std::vector<float> parallel_transform(const std::vector<Vector>& queries, const Vector& centroid) {
    std::vector<float> results(queries.size());

    auto task = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            results[i] = std::inner_product(queries[i].begin(), queries[i].end(), centroid.begin(), 0.0f);
        }
    };

    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;
    size_t batch_size = queries.size() / num_threads;
    if (batch_size == 0 && queries.size() > 0) {
        batch_size = 1;
        num_threads = queries.size();
    }
    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * batch_size;
        size_t end = (i == num_threads - 1) ? queries.size() : start + batch_size;
        if (start < end) {
            futures.push_back(std::async(std::launch::async, task, start, end));
        }
    }

    for (auto& future : futures) {
        future.get();
    }

    return results;
}

} // namespace parallel_ops