#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../algorithms/hnsw_index.hpp"
#include "../core/vector.hpp"
#include "../utils/distance_metrics.hpp"

class VectorSegment {
public:
    enum class State {
        Mutable,
        Sealed,
    };

    // Durability is unconditional: every WAL record / tombstone append is
    // followed by fsync before the call returns. There is no "buffered"
    // mode — losing recent writes on power failure was an unacceptable
    // foot-gun for a database that calls itself a database.
    struct Config {
        size_t dimensions{0};
        size_t hnsw_m{16};
        size_t hnsw_ef_construction{80};
        size_t hnsw_ef_search{50};
        HNSWIndex::AllocationStrategy allocation_strategy{HNSWIndex::AllocationStrategy::Arena};
        size_t arena_initial_size{1024 * 1024};
        std::shared_ptr<const DistanceMetric> metric{std::make_shared<EuclideanDistance>()};
    };

    struct SearchResult {
        std::string key;
        float distance;
        std::string metadata;
        uint64_t sequence;
    };

    struct RecordView {
        std::string key;
        std::string metadata;
        const float* data;
        uint64_t sequence;
        bool active{true};
    };

    struct Statistics {
        std::string id;
        State state;
        size_t records;
        size_t live_records;
        size_t tombstones;
        size_t wal_bytes;
        size_t vector_bytes;
        size_t hnsw_snapshot_bytes;
        HNSWIndex::MemoryStatistics hnsw_memory;
    };

    VectorSegment(std::string id, std::filesystem::path directory, Config config, State state);

    const std::string& id() const { return id_; }
    State state() const { return state_; }
    bool isMutable() const { return state_ == State::Mutable; }

    void initializeNew();
    void load();

    [[nodiscard]] bool insert(const Vector& vector, const std::string& key, const std::string& metadata, uint64_t sequence);
    [[nodiscard]] bool update(const Vector& vector, const std::string& key, const std::string& metadata, uint64_t sequence);
    [[nodiscard]] bool insertRecovered(const Vector& vector, const std::string& key, const std::string& metadata, uint64_t sequence);
    [[nodiscard]] bool remove(const std::string& key, uint64_t sequence);
    [[nodiscard]] bool contains(const std::string& key) const;
    [[nodiscard]] std::optional<Vector> get(const std::string& key) const;
    [[nodiscard]] std::string getMetadata(const std::string& key) const;

    [[nodiscard]] std::vector<SearchResult> search(const Vector& query, size_t k) const;
    void forEachLive(const std::function<void(const RecordView&)>& visitor) const;
    void forEachRecord(const std::function<void(const RecordView&)>& visitor) const;

    void seal();
    void flush();

    size_t recordCount() const { return records_.size(); }
    size_t liveCount() const { return live_count_; }
    size_t tombstoneCount() const { return tombstone_count_; }
    double tombstoneRatio() const;
    uint64_t maxSequence() const { return max_sequence_; }
    Statistics getStatistics() const;

private:
    struct Record {
        std::string key;
        std::string metadata;
        std::vector<float> values;
        uint64_t sequence{0};
        bool active{true};
    };

    struct WalEntry {
        enum class Op : uint8_t {
            Insert = 1,
            Delete = 2,
            Update = 3,
        };

        Op op;
        uint64_t sequence;
        std::string key;
        std::string metadata;
        std::vector<float> values;
    };

    std::string id_;
    std::filesystem::path directory_;
    Config config_;
    State state_;
    std::vector<Record> records_;
    std::unordered_map<std::string, size_t> key_to_slot_;
    size_t live_count_{0};
    size_t tombstone_count_{0};
    uint64_t max_sequence_{0};
    std::unique_ptr<HNSWIndex> hnsw_;

    void rebuildIndex();
    bool applyInsert(const Vector& vector, const std::string& key, const std::string& metadata, uint64_t sequence);
    bool applyUpdate(const Vector& vector, const std::string& key, const std::string& metadata, uint64_t sequence);
    bool applyDelete(const std::string& key, uint64_t sequence);
    void appendWalInsert(const Vector& vector, const std::string& key, const std::string& metadata, uint64_t sequence);
    void appendWalUpdate(const Vector& vector, const std::string& key, const std::string& metadata, uint64_t sequence);
    void appendWalDelete(const std::string& key, uint64_t sequence);
    void appendSealedTombstone(const std::string& key, uint64_t sequence);
    std::vector<WalEntry> readWal() const;
    void replayWal();

    void writeVectorsFile() const;
    void readVectorsFile();
    void writeTombstonesFile() const;
    void readTombstonesFile();
    void writeHNSWSnapshot() const;
    bool readHNSWSnapshot();
    void writeMetadata() const;
    void readMetadata();

    const float* vectorPtr(uint64_t slot_id) const;
    std::filesystem::path walPath() const;
    std::filesystem::path vectorsPath() const;
    std::filesystem::path tombstonesPath() const;
    std::filesystem::path hnswPath() const;
    std::filesystem::path metadataPath() const;
};
