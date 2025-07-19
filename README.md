# Vector Database

A high-performance, in-memory vector database with SIMD optimizations, supporting both exact and approximate nearest neighbor search. Built with C++20 and optimized for modern architectures including Apple Silicon (M1/M2) and x86-64.

## 🚀 Features

### Core Functionality
- **High-dimensional vector storage** with efficient indexing
- **Exact nearest neighbor search** using KD-trees
- **Approximate nearest neighbor search** using Locality-Sensitive Hashing (LSH)
- **Batch operations** for efficient bulk insertions and searches
- **Metadata support** for storing additional information with vectors
- **Persistent storage** with save/load functionality

### Performance Optimizations
- **SIMD acceleration** with ARM NEON (Apple Silicon) and AVX2 (x86-64)
- **Query caching** for frequently accessed results
- **Dimensionality reduction** using PCA for large vectors
- **Parallel processing** support for multi-core systems

### API & Integration
- **RESTful HTTP API** for easy integration
- **C++ library** for direct embedding
- **Python client** examples
- **Real-time search** capabilities

## 📊 Performance Metrics

### SIMD Performance (Apple Silicon M1)
- **Dot Product**: 3.7x - 4.9x speedup
- **Vector Addition**: 5.8x - 6.8x speedup
- **Vector Subtraction**: 2.8x - 6.6x speedup

### Search Performance
- **Exact Search**: O(log n) average case with KD-trees
- **Approximate Search**: O(1) average case with LSH
- **Batch Operations**: Optimized for bulk processing

## 🛠️ Use Cases

### Machine Learning & AI
- **Embedding storage** for NLP models
- **Similarity search** for recommendation systems
- **Feature vector databases** for computer vision
- **Neural network embeddings** storage and retrieval

### Data Science
- **High-dimensional data analysis**
- **Clustering and classification**
- **Anomaly detection**
- **Pattern matching**

### Real-time Applications
- **Content recommendation engines**
- **Image similarity search**
- **Document similarity matching**
- **Audio fingerprinting**

## 📦 Installation & Setup

### Prerequisites
- **C++20 compatible compiler** (GCC 10+, Clang 12+, or MSVC 2019+)
- **CMake 3.10+** (for CMake build)
- **Make** (for Makefile build)
- **Python 3.7+** (for Python examples)

### Quick Start

#### Using Makefile (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd vector_database

# Build everything
make all

# Run basic example
./build/basic_usage

# Start the API server
make run-server

# Test with Python client
python examples/api_client_demo.py
```

#### Using CMake
```bash
# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make

# Run examples
./basic_usage
./advanced_features
./simd_benchmark
```

### Build Options

The build system automatically detects your architecture and applies appropriate optimizations:

- **Apple Silicon (M1/M2)**: ARM NEON SIMD instructions
- **x86-64**: AVX2 SIMD instructions
- **Other architectures**: Scalar fallback with compiler optimizations

## 🔧 Usage Examples

### Basic C++ Usage
```cpp
#include "vector_database.hpp"
#include "random_generator.hpp"

int main() {
    // Create a 128-dimensional vector database
    VectorDatabase db(128);
    
    // Insert vectors
    RandomGenerator rng;
    for (int i = 0; i < 1000; ++i) {
        Vector v = rng.generateUniformVector(128);
        db.insert(v, "vector_" + std::to_string(i));
    }
    
    // Search for similar vectors
    Vector query = rng.generateUniformVector(128);
    auto results = db.similaritySearch(query, 5);
    
    for (const auto& [key, distance] : results) {
        std::cout << key << ": " << distance << std::endl;
    }
    
    return 0;
}
```

### Advanced Features
```cpp
// Use approximate search for better performance
VectorDatabase db(128, true);  // Enable LSH

// Add metadata
db.insert(vector, "key", "metadata");

// Batch operations
std::vector<Vector> vectors = {...};
std::vector<std::string> keys = {...};
db.batchInsert(vectors, keys);

// Search with metadata
auto results = db.similaritySearchWithMetadata(query, 5);
```

### REST API Usage
```bash
# Start server
./build/vector_db_server --port 8080 --dimensions 128

# Insert vector
curl -X POST http://localhost:8080/vectors \
  -H "Content-Type: application/json" \
  -d '{"key": "vec1", "vector": [0.1, 0.2, ...]}'

# Search
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, ...], "k": 5}'
```

## 🏗️ Architecture

### Core Components
- **Vector**: High-dimensional vector representation with SIMD operations
- **KDTree**: Exact nearest neighbor search implementation
- **LSHIndex**: Approximate search using Locality-Sensitive Hashing
- **VectorDatabase**: Main database interface with metadata support

### Optimization Layers
- **SIMD Operations**: ARM NEON and AVX2 vectorized operations
- **Query Cache**: LRU cache for frequently accessed results
- **Dimensionality Reduction**: PCA for large vector compression
- **Parallel Processing**: Multi-threaded operations

### API Layer
- **HTTP Server**: RESTful API using cpp-httplib
- **JSON Serialization**: nlohmann/json for data exchange
- **Client Libraries**: C++ and Python examples

## 📈 Benchmarks

Run performance benchmarks to see SIMD optimizations in action:

```bash
# Comprehensive SIMD benchmark
./build/simd_benchmark

# Quick performance test
./build/quick_simd_test

# Search performance benchmark
./build/search_benchmark

# Insertion performance benchmark
./build/insertion_benchmark
```

## 🔍 API Reference

### Core Methods
- `insert(vector, key, metadata?)`: Insert a single vector
- `batchInsert(vectors, keys)`: Insert multiple vectors
- `similaritySearch(query, k)`: Find k nearest neighbors
- `similaritySearchWithMetadata(query, k)`: Search with metadata
- `toggleApproximateSearch(enabled)`: Switch between exact/approximate search
- `saveToFile(filename)`: Persist database to disk
- `loadFromFile(filename)`: Load database from disk

### REST Endpoints
- `POST /vectors`: Insert single vector
- `POST /vectors/batch`: Batch insert
- `GET /vectors`: List all vectors
- `POST /search`: Similarity search
- `GET /health`: Health check
- `PUT /config/approximate`: Toggle search mode

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for complete API reference.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run benchmarks to ensure performance
6. Submit a pull request

## 🙏 Acknowledgments

- **cpp-httplib** for the HTTP server implementation
- **nlohmann/json** for JSON serialization
- **ARM NEON** and **Intel AVX2** for SIMD optimizations

## 📞 Support

- **Issues**: Report bugs and feature requests on GitHub
- **Documentation**: See [docs/](docs/) for detailed guides
- **Examples**: Check [examples/](examples/) for usage patterns
- **Benchmarks**: Review [benchmarks/](benchmarks/) for performance data
