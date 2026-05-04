#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "segment.hpp"

class SegmentedVectorStore {
public:
    struct Config {
        size_t dimensions{0};
        size_t hnsw_m{16};
        size_t hnsw_ef_construction{80};
        size_t hnsw_ef_search{50};
        size_t max_mutable_segment_records{100000};
        size_t max_sealed_segments{16};
        double max_tombstone_ratio{0.25};
        HNSWIndex::AllocationStrategy allocation_strategy{HNSWIndex::AllocationStrategy::Arena};
        size_t arena_initial_size{1024 * 1024};
        std::shared_ptr<const DistanceMetric> metric{std::make_shared<EuclideanDistance>()};
    };

    struct SearchResult {
        std::string key;
        float distance;
        std::string metadata;
    };

    struct Statistics {
        size_t total_vectors;
        size_t mutable_records;
        size_t sealed_segments;
        size_t total_segments;
        size_t total_records;
        size_t total_tombstones;
        size_t wal_bytes;
        size_t vector_bytes;
        size_t hnsw_snapshot_bytes;
        size_t disk_bytes;
        size_t hnsw_allocation_calls;
        size_t hnsw_deallocation_calls;
        size_t hnsw_peak_bytes;
        uint64_t latest_sequence;
    };

    SegmentedVectorStore(std::filesystem::path root, Config config);

    void initialize();
    void shutdown();

    [[nodiscard]] bool insert(const Vector& vector, const std::string& key, const std::string& metadata = "");
    [[nodiscard]] bool update(const Vector& vector, const std::string& key, const std::string& metadata = "");
    [[nodiscard]] bool remove(const std::string& key);
    [[nodiscard]] std::optional<Vector> get(const std::string& key) const;
    [[nodiscard]] std::string getMetadata(const std::string& key) const;
    [[nodiscard]] std::vector<std::pair<std::string, float>> search(const Vector& query, size_t k) const;
    [[nodiscard]] std::vector<SearchResult> searchWithMetadata(const Vector& query, size_t k) const;

    void flush();
    void compact();
    void sealMutableSegment();

    size_t vectorCount() const { return key_locations_.size(); }
    Statistics getStatistics() const;
    std::unordered_map<std::string, Vector> getAllVectors() const;

    void setMetric(std::shared_ptr<const DistanceMetric> metric);
    void configureHNSW(size_t M, size_t ef_construction, size_t ef_search);
    void configureAllocator(HNSWIndex::AllocationStrategy strategy, size_t arena_initial_size);
    void configureSegmentation(size_t max_mutable_segment_records,
                               size_t max_sealed_segments,
                               double max_tombstone_ratio);

private:
    struct Location {
        std::shared_ptr<VectorSegment> segment;
        uint64_t sequence;
    };

    std::filesystem::path root_;
    Config config_;
    std::shared_ptr<VectorSegment> mutable_segment_;
    std::vector<std::shared_ptr<VectorSegment>> sealed_segments_;
    std::unordered_map<std::string, Location> key_locations_;
    uint64_t next_segment_id_{1};
    uint64_t latest_sequence_{0};
    bool initialized_{false};

    VectorSegment::Config segmentConfig() const;
    std::shared_ptr<VectorSegment> createSegment(VectorSegment::State state);
    std::shared_ptr<VectorSegment> loadSegment(const std::string& id, VectorSegment::State state);
    void rebuildKeyLocations();
    void maybeSealMutableSegment();
    void maybeCompact();
    uint64_t nextSequence();

    void writeManifest() const;
    bool readManifest(std::string& mutable_id, std::vector<std::string>& sealed_ids);
    std::string makeSegmentId(uint64_t id) const;
    std::filesystem::path segmentsDir() const;
    std::filesystem::path segmentDir(const std::string& id) const;
    std::filesystem::path manifestPath() const;
    size_t diskBytes() const;
};
