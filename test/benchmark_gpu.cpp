/**
 * Direct C++ benchmark: GPU vs CPU vs HNSW search performance
 * Bypasses HTTP/JSON overhead for accurate timing
 * 
 * Usage: ./benchmark_gpu [num_vectors] [dimensions] [num_queries] [k]
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>

#include "../src/core/vector_database.hpp"
#include "../src/core/vector.hpp"
#include "../src/algorithms/hnsw_index.hpp"
#include "../src/optimizations/gpu_operations.hpp"

using namespace std::chrono;

// Generate random vectors
std::vector<Vector> generateRandomVectors(size_t count, size_t dims) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<Vector> vectors;
    vectors.reserve(count);
    
    for (size_t i = 0; i < count; i++) {
        std::vector<float> data(dims);
        for (size_t j = 0; j < dims; j++) {
            data[j] = dist(gen);
        }
        vectors.emplace_back(data);
    }
    
    return vectors;
}

// Benchmark function
template<typename Func>
double benchmark(const std::string& name, Func func, int warmup_runs = 1, int timed_runs = 3) {
    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        func();
    }
    
    // Timed runs
    std::vector<double> times;
    for (int i = 0; i < timed_runs; i++) {
        auto start = high_resolution_clock::now();
        func();
        auto end = high_resolution_clock::now();
        double ms = duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }
    
    double avg = 0;
    for (double t : times) avg += t;
    avg /= times.size();
    
    std::cout << "  " << name << ":" << std::endl;
    for (int i = 0; i < timed_runs; i++) {
        std::cout << "    Run " << (i+1) << ": " << std::fixed << std::setprecision(2) << times[i] << "ms" << std::endl;
    }
    std::cout << "    Average: " << std::fixed << std::setprecision(2) << avg << "ms" << std::endl;
    
    return avg;
}

int main(int argc, char* argv[]) {
    // Parse arguments
    size_t num_vectors = (argc > 1) ? std::stoul(argv[1]) : 10000;
    size_t dimensions = (argc > 2) ? std::stoul(argv[2]) : 128;
    size_t num_queries = (argc > 3) ? std::stoul(argv[3]) : 100;
    size_t k = (argc > 4) ? std::stoul(argv[4]) : 10;
    
    std::cout << "========================================" << std::endl;
    std::cout << "GPU vs CPU Direct Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Vectors: " << num_vectors << std::endl;
    std::cout << "Dimensions: " << dimensions << std::endl;
    std::cout << "Queries: " << num_queries << std::endl;
    std::cout << "K: " << k << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Generate database vectors
    std::cout << "Generating " << num_vectors << " database vectors..." << std::endl;
    auto db_vectors = generateRandomVectors(num_vectors, dimensions);
    
    // Generate query vectors
    std::cout << "Generating " << num_queries << " query vectors..." << std::endl;
    auto query_vectors = generateRandomVectors(num_queries, dimensions);
    
    // Create flat vector array for GPU
    std::cout << "Creating flat vector array for GPU..." << std::endl;
    std::vector<float> flat_vectors;
    flat_vectors.reserve(num_vectors * dimensions);
    for (const auto& v : db_vectors) {
        for (size_t i = 0; i < dimensions; i++) {
            flat_vectors.push_back(v[i]);
        }
    }
    
    std::cout << std::endl;
    
    // Initialize GPU
    std::cout << "Initializing GPU..." << std::endl;
    bool gpu_available = gpu_ops::initialize();
    if (gpu_available) {
        std::cout << "GPU initialized successfully!" << std::endl;
        gpu_ops::set_database_buffer(flat_vectors.data(), num_vectors, dimensions);
        std::cout << "GPU buffer set (" << (flat_vectors.size() * sizeof(float) / 1024 / 1024) << " MB)" << std::endl;
    } else {
        std::cout << "GPU not available, skipping GPU tests" << std::endl;
    }
    
    // Build HNSW index
    std::cout << "Building HNSW index..." << std::endl;
    auto hnsw_start = high_resolution_clock::now();
    HNSWIndex hnsw(dimensions, 16, 200);  // M=16, ef_construction=200
    for (size_t i = 0; i < num_vectors; i++) {
        hnsw.insert(db_vectors[i], "vec_" + std::to_string(i));
    }
    auto hnsw_end = high_resolution_clock::now();
    double hnsw_build_ms = duration<double, std::milli>(hnsw_end - hnsw_start).count();
    std::cout << "HNSW index built in " << std::fixed << std::setprecision(0) << hnsw_build_ms << "ms" << std::endl;
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Running Benchmarks" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // ==================== HNSW (Approximate) ====================
    std::cout << "1. HNSW Approximate Search (all " << num_queries << " queries):" << std::endl;
    double hnsw_time = benchmark("HNSW Search", [&]() {
        for (const auto& query : query_vectors) {
            auto results = hnsw.search(query, k);
        }
    });
    
    std::cout << std::endl;
    
    // ==================== CPU Brute Force ====================
    std::cout << "2. CPU Brute Force (all " << num_queries << " queries):" << std::endl;
    double cpu_brute_time = benchmark("CPU Brute Force", [&]() {
        for (const auto& query : query_vectors) {
            std::vector<std::pair<size_t, float>> distances;
            distances.reserve(num_vectors);
            
            for (size_t i = 0; i < num_vectors; i++) {
                float dist = 0;
                for (size_t j = 0; j < dimensions; j++) {
                    float diff = query[j] - db_vectors[i][j];
                    dist += diff * diff;
                }
                distances.emplace_back(i, dist);
            }
            
            // Partial sort to get top-k
            std::partial_sort(distances.begin(), 
                              distances.begin() + std::min(k, distances.size()),
                              distances.end(),
                              [](const auto& a, const auto& b) { return a.second < b.second; });
        }
    });
    
    std::cout << std::endl;
    
    // ==================== GPU Brute Force ====================
    double gpu_time = 0;
    if (gpu_available) {
        std::cout << "3. GPU Brute Force (all " << num_queries << " queries):" << std::endl;
        gpu_time = benchmark("GPU Brute Force", [&]() {
            for (const auto& query : query_vectors) {
                auto distances = gpu_ops::search_euclidean(query);
                
                if (!distances.empty()) {
                    // Partial sort to get top-k
                    std::vector<std::pair<size_t, float>> indexed;
                    indexed.reserve(distances.size());
                    for (size_t i = 0; i < distances.size(); i++) {
                        indexed.emplace_back(i, distances[i]);
                    }
                    std::partial_sort(indexed.begin(),
                                      indexed.begin() + std::min(k, indexed.size()),
                                      indexed.end(),
                                      [](const auto& a, const auto& b) { return a.second < b.second; });
                }
            }
        });
        
        std::cout << std::endl;
    }
    
    // ==================== Results ====================
    std::cout << "========================================" << std::endl;
    std::cout << "Results Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "                     Total Time    Per Query" << std::endl;
    std::cout << "  ─────────────────────────────────────────────" << std::endl;
    std::cout << "  HNSW (approx):     " << std::setw(8) << hnsw_time << "ms   " 
              << std::setw(8) << (hnsw_time / num_queries) << "ms" << std::endl;
    std::cout << "  CPU Brute Force:   " << std::setw(8) << cpu_brute_time << "ms   " 
              << std::setw(8) << (cpu_brute_time / num_queries) << "ms" << std::endl;
    
    if (gpu_available) {
        std::cout << "  GPU Brute Force:   " << std::setw(8) << gpu_time << "ms   " 
                  << std::setw(8) << (gpu_time / num_queries) << "ms" << std::endl;
    }
    std::cout << "  ─────────────────────────────────────────────" << std::endl;
    
    std::cout << std::endl;
    std::cout << "  Comparisons:" << std::endl;
    std::cout << "  • GPU vs CPU Brute: " << std::setprecision(2) << (cpu_brute_time / gpu_time) << "x faster" << std::endl;
    std::cout << "  • GPU vs HNSW:      " << std::setprecision(2) << (hnsw_time / gpu_time) << "x " 
              << (gpu_time < hnsw_time ? "faster" : "slower") << std::endl;
    std::cout << "  • HNSW vs CPU Brute: " << std::setprecision(2) << (cpu_brute_time / hnsw_time) << "x faster" << std::endl;
    
    std::cout << std::endl;
    if (gpu_time < hnsw_time) {
        std::cout << "  GPU wins! Best for exact search at scale." << std::endl;
    } else {
        std::cout << "  HNSW wins! Best for approximate search." << std::endl;
        std::cout << "     (GPU is better when you need 100% exact results)" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Cleanup
    if (gpu_available) {
        gpu_ops::shutdown();
    }
    
    return 0;
}

