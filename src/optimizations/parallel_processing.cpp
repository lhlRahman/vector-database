// src/optimizations/parallel_processing.cpp

#include "parallel_processing.hpp"
#include <future>
#include <numeric>

namespace parallel_ops {

void batchInsert(VectorDatabase& db, const std::vector<Vector>& vectors, const std::vector<std::string>& keys) {
    if (vectors.size() != keys.size()) {
        throw std::invalid_argument("Number of vectors and keys must match");
    }

    auto task = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            db.insert(vectors[i], keys[i]);
        }
    };

    size_t num_threads = std::thread::hardware_concurrency();
    size_t batch_size = vectors.size() / num_threads;
    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * batch_size;
        size_t end = (i == num_threads - 1) ? vectors.size() : start + batch_size;
        futures.push_back(std::async(std::launch::async, task, start, end));
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
    size_t batch_size = queries.size() / num_threads;
    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * batch_size;
        size_t end = (i == num_threads - 1) ? queries.size() : start + batch_size;
        futures.push_back(std::async(std::launch::async, task, start, end));
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
    size_t batch_size = indices.size() / num_threads;
    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * batch_size;
        size_t end = (i == num_threads - 1) ? indices.size() : start + batch_size;
        futures.push_back(std::async(std::launch::async, task, start, end));
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
    size_t batch_size = queries.size() / num_threads;
    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * batch_size;
        size_t end = (i == num_threads - 1) ? queries.size() : start + batch_size;
        futures.push_back(std::async(std::launch::async, task, start, end));
    }

    for (auto& future : futures) {
        future.get();
    }

    return results;
}

} // namespace parallel_ops