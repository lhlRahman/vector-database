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
        size_t hnsw_ef_construction{200};  // was 80; 200 builds a stronger graph for recall
        size_t hnsw_ef_search{64};         // was 50
        size_t max_mutable_segment_records{100000};
        size_t max_sealed_segments{16};
        double max_tombstone_ratio{0.25};
        HNSWIndex::AllocationStrategy allocation_strategy{HNSWIndex::AllocationStrategy::Arena};
        size_t arena_initial_size{1024 * 1024};
        std::shared_ptr<const DistanceMetric> metric{std::make_shared<EuclideanDistance>()};
        uint32_t hnsw_seed{100};
        uint64_t sequence_reservation_block{1ull << 20};
    };

    struct SearchResult {
        std::string key;
        float distance;
        std::string metadata;
        bool provisional{false};
    };

    struct StagedInsertResult {
        bool applied{false};
        uint64_t lsn{0};
    };

    struct RecordSnapshot {
        std::string key;
        Vector vector;
        std::string metadata;
        uint64_t lsn{0};
        bool provisional{false};
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
        uint64_t visible_lsn;
        uint64_t durable_lsn;
        size_t volatile_records;
    };

    SegmentedVectorStore(std::filesystem::path root, Config config);
    ~SegmentedVectorStore() noexcept;

    SegmentedVectorStore(const SegmentedVectorStore&) = delete;
    SegmentedVectorStore& operator=(const SegmentedVectorStore&) = delete;

    void initialize(bool read_only_recovery = false);
    void shutdown();

    [[nodiscard]] bool insert(const Vector& vector, const std::string& key, const std::string& metadata = "");
    [[nodiscard]] StagedInsertResult stageInsert(
        const Vector& vector, const std::string& key, const std::string& metadata = "");
    // Group-commit batch insert: appends all WAL records, then a single fsync for
    // the whole batch. Returns the number inserted (skips dim-mismatch/duplicate).
    size_t insertBatch(const std::vector<std::string>& keys,
                       const std::vector<Vector>& vectors,
                       const std::vector<std::string>& metadata);
    [[nodiscard]] bool update(const Vector& vector, const std::string& key, const std::string& metadata = "");
    [[nodiscard]] bool remove(const std::string& key);
    [[nodiscard]] std::optional<Vector> get(const std::string& key) const;
    [[nodiscard]] std::string getMetadata(const std::string& key) const;
    [[nodiscard]] std::optional<RecordSnapshot> inspectRecord(
        const std::string& key, bool durable_only) const;
    [[nodiscard]] std::vector<RecordSnapshot> inspectRecords(bool durable_only) const;
    [[nodiscard]] std::vector<std::pair<std::string, float>> search(const Vector& query, size_t k) const;
    [[nodiscard]] std::vector<std::pair<std::string, float>> searchStable(
        const Vector& query, size_t k) const;
    [[nodiscard]] std::vector<SearchResult> searchWithMetadata(const Vector& query, size_t k) const;

    void flush();
    uint64_t commitThrough(uint64_t target_lsn, bool run_maintenance = true);
    void compact();
    void sealMutableSegment();

    size_t vectorCount() const { return key_locations_.size(); }
    uint64_t visibleLsn() const { return visible_lsn_; }
    uint64_t durableLsn() const { return durable_lsn_; }
    uint64_t manifestGeneration() const { return manifest_generation_; }
    size_t volatileCount() const {
        return mutable_segment_ ? mutable_segment_->volatileCount() : 0;
    }
    size_t volatileBytes() const {
        return mutable_segment_ ? mutable_segment_->volatileBytes() : 0;
    }
    bool isVolatile(const std::string& key) const;
    Statistics getStatistics() const;
    std::unordered_map<std::string, Vector> getAllVectors() const;

    void setMetric(std::shared_ptr<const DistanceMetric> metric);
    void configureHNSW(size_t M, size_t ef_construction, size_t ef_search,
                       uint32_t seed = 100);
    void configureAllocator(HNSWIndex::AllocationStrategy strategy, size_t arena_initial_size);
    void configureSegmentation(size_t max_mutable_segment_records,
                               size_t max_sealed_segments,
                               double max_tombstone_ratio);

private:
    class WriterRootLock {
    public:
        WriterRootLock() = default;
        ~WriterRootLock() noexcept;

        WriterRootLock(const WriterRootLock&) = delete;
        WriterRootLock& operator=(const WriterRootLock&) = delete;

        void acquire(const std::filesystem::path& root);
        void release() noexcept;

    private:
        int fd_{-1};
    };

    struct Location {
        std::shared_ptr<VectorSegment> segment;
        uint64_t sequence;
    };

    std::filesystem::path root_;
    Config config_;
    WriterRootLock writer_root_lock_;
    std::shared_ptr<VectorSegment> mutable_segment_;
    std::vector<std::shared_ptr<VectorSegment>> sealed_segments_;
    std::unordered_map<std::string, Location> key_locations_;
    uint64_t next_segment_id_{1};
    uint64_t latest_sequence_{0};
    uint64_t reserved_sequence_hi_{0};
    uint64_t visible_lsn_{0};
    uint64_t durable_lsn_{0};
    uint64_t manifest_generation_{0};
    bool initialized_{false};
    bool read_only_recovery_{false};
    bool maintenance_active_{false};

    VectorSegment::Config segmentConfig() const;
    std::shared_ptr<VectorSegment> createSegment(VectorSegment::State state);
    std::shared_ptr<VectorSegment> loadSegment(
        const std::string& id, VectorSegment::State state, bool read_only_recovery = false);
    void rebuildKeyLocations();
    void maybeSealMutableSegment();
    void maybeCompact();
    uint64_t nextSequence();
    void reserveSequenceBlock();
    uint64_t readSequenceHighwater() const;
    void writeSequenceHighwater(uint64_t highwater) const;

    void writeManifest();
    void writeManifest(
        const std::shared_ptr<VectorSegment>& mutable_segment,
        const std::vector<std::shared_ptr<VectorSegment>>& sealed_segments);
    bool readManifest(std::string& mutable_id, std::vector<std::string>& sealed_ids);
    std::string makeSegmentId(uint64_t id) const;
    std::filesystem::path segmentsDir() const;
    std::filesystem::path segmentDir(const std::string& id) const;
    std::filesystem::path manifestPath() const;
    std::filesystem::path sequenceHighwaterPath() const;
    size_t diskBytes() const;
};
