#include "segmented_vector_store.hpp"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <sys/file.h>
#include <unordered_map>
#include <unistd.h>

#include "../utils/atomic_write.hpp"

namespace {
constexpr uint32_t kSequenceHighwaterMagic = 0x314e534c;  // "LSN1"
constexpr uint32_t kSequenceHighwaterVersion = 1;
constexpr const char* kWriterLockFilename = ".writer.lock";

struct SequenceHighwaterRecord {
    uint32_t magic;
    uint32_t version;
    uint64_t highwater;
    uint32_t crc32;
};
constexpr size_t kSequenceHighwaterRecordBytes =
    offsetof(SequenceHighwaterRecord, crc32) + sizeof(uint32_t);
static_assert(kSequenceHighwaterRecordBytes == 20);

uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
        }
    }
    return crc;
}

uint32_t sequence_highwater_crc(const SequenceHighwaterRecord& record) {
    return ~crc32_update(
        0xFFFFFFFFu,
        reinterpret_cast<const uint8_t*>(&record),
        offsetof(SequenceHighwaterRecord, crc32));
}

void atomic_text_write(const std::filesystem::path& path, const std::string& contents) {
    vdb::io::atomic_write(path, [&](std::ostream& os) {
        os << contents;
        if (!os.good()) throw std::runtime_error("failed writing manifest: " + path.string());
    });
}

std::vector<std::string> split_csv(const std::string& value) {
    std::vector<std::string> out;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

void committer_failpoint(const char* name) {
    const char* configured = std::getenv("VDB_COMMITTER_FAILPOINT");
    if (configured != nullptr && std::string_view(configured) == name) {
        _exit(86);
    }
}
}

SegmentedVectorStore::WriterRootLock::~WriterRootLock() noexcept {
    release();
}

void SegmentedVectorStore::WriterRootLock::acquire(
    const std::filesystem::path& root) {
    if (fd_ >= 0) {
        throw std::logic_error("segmented storage writer lock is already held");
    }

    const auto lock_path = root / kWriterLockFilename;
    int fd = -1;
    do {
        fd = ::open(lock_path.c_str(),
                    O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW,
                    0644);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
        const int error = errno;
        throw std::runtime_error(
            "cannot open segmented storage writer lock " + lock_path.string() +
            ": " + std::strerror(error));
    }

    for (;;) {
        if (::flock(fd, LOCK_EX | LOCK_NB) == 0) break;
        const int error = errno;
        if (error == EINTR) continue;
        ::close(fd);
        if (error == EWOULDBLOCK || error == EAGAIN) {
            throw std::runtime_error(
                "segmented storage root is already open for writing: " +
                root.string());
        }
        throw std::runtime_error(
            "cannot lock segmented storage root " + root.string() + ": " +
            std::strerror(error));
    }
    fd_ = fd;
}

void SegmentedVectorStore::WriterRootLock::release() noexcept {
    if (fd_ < 0) return;
    (void)::close(fd_);
    fd_ = -1;
}

SegmentedVectorStore::SegmentedVectorStore(std::filesystem::path root, Config config)
    : root_(std::move(root)), config_(std::move(config)) {
}

SegmentedVectorStore::~SegmentedVectorStore() noexcept = default;

void SegmentedVectorStore::initialize(bool read_only_recovery) {
    if (initialized_) return;
    read_only_recovery_ = read_only_recovery;

    if (read_only_recovery_) {
        if (!std::filesystem::is_directory(segmentsDir())) {
            throw std::runtime_error("missing segmented store for read-only recovery");
        }
    } else {
        std::filesystem::create_directories(root_);
        writer_root_lock_.acquire(root_);
    }

    try {
        if (!read_only_recovery_) {
            std::filesystem::create_directories(segmentsDir());
            vdb::io::fsync_dir(root_.parent_path());
            vdb::io::fsync_dir(root_);
        }

        std::string mutable_id;
        std::vector<std::string> sealed_ids;
        const bool have_manifest = readManifest(mutable_id, sealed_ids);
        if (read_only_recovery_ && !have_manifest) {
            throw std::runtime_error("missing manifest for read-only recovery");
        }
        if (have_manifest) {
            sealed_segments_.clear();
            sealed_segments_.reserve(sealed_ids.size());
            for (const auto& id : sealed_ids) {
                sealed_segments_.push_back(loadSegment(
                    id, VectorSegment::State::Sealed, read_only_recovery_));
            }

            if (!mutable_id.empty()) {
                mutable_segment_ = loadSegment(
                    mutable_id, VectorSegment::State::Mutable, read_only_recovery_);
            }
        }

        if (!mutable_segment_ && !read_only_recovery_) {
            mutable_segment_ = createSegment(VectorSegment::State::Mutable);
        }
        if (!mutable_segment_) throw std::runtime_error("manifest has no mutable segment");

        rebuildKeyLocations();
        const uint64_t manifest_durable_lsn = durable_lsn_;
        durable_lsn_ = mutable_segment_ ? mutable_segment_->durableLsn() : 0;
        for (const auto& segment : sealed_segments_) {
            durable_lsn_ = std::max(durable_lsn_, segment->maxSequence());
        }
        durable_lsn_ = std::max(durable_lsn_, manifest_durable_lsn);
        // Writable recovery deliberately discards every unfenced mutable suffix.
        visible_lsn_ = durable_lsn_;
        if (!read_only_recovery_) {
            reserveSequenceBlock();
            writeManifest();
        }
        initialized_ = true;
    } catch (...) {
        writer_root_lock_.release();
        throw;
    }
}

void SegmentedVectorStore::shutdown() {
    if (!initialized_) return;
    if (!read_only_recovery_) flush();
    initialized_ = false;
    writer_root_lock_.release();
}

bool SegmentedVectorStore::insert(const Vector& vector, const std::string& key, const std::string& metadata) {
    StagedInsertResult staged = stageInsert(vector, key, metadata);
    if (!staged.applied) return false;
    commitThrough(staged.lsn);
    return true;
}

SegmentedVectorStore::StagedInsertResult
SegmentedVectorStore::stageInsert(const Vector& vector,
                                  const std::string& key,
                                  const std::string& metadata) {
    if (!initialized_) throw std::runtime_error("segmented store not initialized");
    if (vector.size() != config_.dimensions) throw std::invalid_argument("vector dimension mismatch");
    if (key_locations_.count(key) != 0) return {};

    uint64_t sequence = nextSequence();
    if (!mutable_segment_->stageInsert(vector, key, metadata, sequence)) return {};
    key_locations_[key] = Location{mutable_segment_, sequence};
    visible_lsn_ = sequence;
    return StagedInsertResult{true, sequence};
}

size_t SegmentedVectorStore::insertBatch(const std::vector<std::string>& keys,
                                         const std::vector<Vector>& vectors,
                                         const std::vector<std::string>& metadata) {
    if (!initialized_) throw std::runtime_error("segmented store not initialized");
    size_t committed = 0;
    for (size_t i = 0; i < keys.size() && i < vectors.size(); ++i) {
        const Vector& v = vectors[i];
        const std::string& key = keys[i];
        if (v.size() != config_.dimensions) continue;
        if (key_locations_.count(key) != 0) continue;
        const std::string& meta = (i < metadata.size()) ? metadata[i] : "";
        StagedInsertResult staged = stageInsert(v, key, meta);
        if (staged.applied) {
            ++committed;
        }
    }
    if (visible_lsn_ > durable_lsn_) commitThrough(visible_lsn_);
    return committed;
}

bool SegmentedVectorStore::update(const Vector& vector, const std::string& key, const std::string& metadata) {
    if (!initialized_) throw std::runtime_error("segmented store not initialized");
    if (vector.size() != config_.dimensions) throw std::invalid_argument("vector dimension mismatch");
    if (key_locations_.count(key) == 0) return false;
    if (visible_lsn_ > durable_lsn_) commitThrough(visible_lsn_);

    uint64_t sequence = nextSequence();
    if (!mutable_segment_->update(vector, key, metadata, sequence)) return false;
    key_locations_[key] = Location{mutable_segment_, sequence};
    visible_lsn_ = sequence;
    durable_lsn_ = sequence;

    maybeSealMutableSegment();
    maybeCompact();
    return true;
}

bool SegmentedVectorStore::remove(const std::string& key) {
    if (!initialized_) throw std::runtime_error("segmented store not initialized");
    if (visible_lsn_ > durable_lsn_) commitThrough(visible_lsn_);

    auto it = key_locations_.find(key);
    if (it == key_locations_.end()) return false;

    uint64_t sequence = nextSequence();
    bool removed = it->second.segment->remove(key, sequence);
    if (removed) {
        key_locations_.erase(it);
        visible_lsn_ = sequence;
        durable_lsn_ = sequence;
        maybeCompact();
    }
    return removed;
}

std::optional<Vector> SegmentedVectorStore::get(const std::string& key) const {
    auto it = key_locations_.find(key);
    if (it == key_locations_.end()) return std::nullopt;
    return it->second.segment->get(key);
}

std::string SegmentedVectorStore::getMetadata(const std::string& key) const {
    auto it = key_locations_.find(key);
    if (it == key_locations_.end()) return "";
    return it->second.segment->getMetadata(key);
}

std::optional<SegmentedVectorStore::RecordSnapshot>
SegmentedVectorStore::inspectRecord(const std::string& key, bool durable_only) const {
    auto it = key_locations_.find(key);
    if (it == key_locations_.end()) return std::nullopt;
    if (durable_only && it->second.sequence > durable_lsn_) return std::nullopt;
    auto vector = it->second.segment->get(key);
    if (!vector) return std::nullopt;
    return RecordSnapshot{
        key,
        *vector,
        it->second.segment->getMetadata(key),
        it->second.sequence,
        it->second.sequence > durable_lsn_,
    };
}

std::vector<SegmentedVectorStore::RecordSnapshot>
SegmentedVectorStore::inspectRecords(bool durable_only) const {
    std::vector<std::string> keys;
    keys.reserve(key_locations_.size());
    for (const auto& [key, location] : key_locations_) {
        (void)location;
        keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end());

    std::vector<RecordSnapshot> records;
    records.reserve(keys.size());
    for (const auto& key : keys) {
        auto record = inspectRecord(key, durable_only);
        if (record) records.push_back(std::move(*record));
    }
    return records;
}

std::vector<std::pair<std::string, float>> SegmentedVectorStore::search(const Vector& query, size_t k) const {
    auto with_meta = searchWithMetadata(query, k);
    std::vector<std::pair<std::string, float>> out;
    out.reserve(with_meta.size());
    for (const auto& result : with_meta) {
        out.emplace_back(result.key, result.distance);
    }
    return out;
}

std::vector<std::pair<std::string, float>>
SegmentedVectorStore::searchStable(const Vector& query, size_t k) const {
    if (query.size() != config_.dimensions) throw std::invalid_argument("query dimension mismatch");
    if (k == 0 || key_locations_.empty()) return {};

    const size_t per_segment_k = std::min(std::max(k * 2, k), key_locations_.size());
    std::vector<SearchResult> candidates;
    auto append_results = [&](const std::shared_ptr<VectorSegment>& segment) {
        for (const auto& result : segment->searchStable(query, per_segment_k)) {
            auto current = key_locations_.find(result.key);
            if (current == key_locations_.end() ||
                current->second.segment.get() != segment.get() ||
                current->second.sequence > durable_lsn_) {
                continue;
            }
            candidates.push_back(SearchResult{
                result.key, result.distance, result.metadata, false});
        }
    };

    if (mutable_segment_) append_results(mutable_segment_);
    for (const auto& segment : sealed_segments_) append_results(segment);
    std::ranges::sort(candidates, [](const SearchResult& left, const SearchResult& right) {
        if (left.distance != right.distance) return left.distance < right.distance;
        return left.key < right.key;
    });

    std::vector<std::pair<std::string, float>> results;
    std::set<std::string> seen;
    results.reserve(std::min(k, candidates.size()));
    for (const auto& candidate : candidates) {
        if (!seen.insert(candidate.key).second) continue;
        results.emplace_back(candidate.key, candidate.distance);
        if (results.size() == k) break;
    }
    return results;
}

std::vector<SegmentedVectorStore::SearchResult>
SegmentedVectorStore::searchWithMetadata(const Vector& query, size_t k) const {
    if (query.size() != config_.dimensions) throw std::invalid_argument("query dimension mismatch");
    if (k == 0 || key_locations_.empty()) return {};

    const size_t per_segment_k = std::min(std::max(k * 2, k), key_locations_.size());
    std::vector<SearchResult> candidates;

    auto append_results = [&](const std::shared_ptr<VectorSegment>& segment) {
        for (const auto& result : segment->search(query, per_segment_k)) {
            auto current = key_locations_.find(result.key);
            if (current == key_locations_.end() || current->second.segment.get() != segment.get()) {
                continue;
            }
            candidates.push_back(SearchResult{
                result.key,
                result.distance,
                result.metadata,
                result.sequence > durable_lsn_,
            });
        }
    };

    if (mutable_segment_) append_results(mutable_segment_);
    for (const auto& segment : sealed_segments_) {
        append_results(segment);
    }

    std::ranges::sort(candidates, [](const SearchResult& left, const SearchResult& right) {
        if (left.distance != right.distance) return left.distance < right.distance;
        return left.key < right.key;
    });

    std::vector<SearchResult> results;
    results.reserve(std::min(k, candidates.size()));
    std::set<std::string> seen;
    for (const auto& candidate : candidates) {
        if (seen.insert(candidate.key).second) {
            results.push_back(candidate);
            if (results.size() == k) break;
        }
    }
    return results;
}

bool SegmentedVectorStore::isVolatile(const std::string& key) const {
    auto it = key_locations_.find(key);
    return it != key_locations_.end() && it->second.sequence > durable_lsn_;
}

void SegmentedVectorStore::flush() {
    if (mutable_segment_) commitThrough(visible_lsn_);
    for (const auto& segment : sealed_segments_) {
        segment->flush();
    }
    writeManifest();
}

uint64_t SegmentedVectorStore::commitThrough(uint64_t target_lsn,
                                             bool run_maintenance) {
    if (!initialized_) throw std::runtime_error("segmented store not initialized");
    if (target_lsn > visible_lsn_) {
        throw std::invalid_argument("durability fence exceeds visible frontier");
    }
    if (target_lsn > durable_lsn_) {
        if (!mutable_segment_ || target_lsn != visible_lsn_) {
            throw std::invalid_argument("durability fence must commit the complete weak tail");
        }
        mutable_segment_->commitThrough(target_lsn);
        durable_lsn_ = target_lsn;
    }

    if (run_maintenance && durable_lsn_ == visible_lsn_ && !maintenance_active_) {
        maintenance_active_ = true;
        try {
            maybeSealMutableSegment();
            maybeCompact();
            maintenance_active_ = false;
        } catch (...) {
            maintenance_active_ = false;
            throw;
        }
    }
    return durable_lsn_;
}

void SegmentedVectorStore::compact() {
    if (sealed_segments_.empty()) return;

    bool should_compact = sealed_segments_.size() > 1;
    for (const auto& segment : sealed_segments_) {
        should_compact = should_compact || segment->tombstoneRatio() >= config_.max_tombstone_ratio;
    }
    if (!should_compact) return;

    struct LiveRecord {
        std::string key;
        std::string metadata;
        std::vector<float> values;
        uint64_t sequence;
    };

    std::vector<LiveRecord> live_records;
    for (const auto& segment : sealed_segments_) {
        segment->forEachLive([&](const VectorSegment::RecordView& view) {
            auto location = key_locations_.find(view.key);
            if (location == key_locations_.end() || location->second.segment.get() != segment.get()) {
                return;
            }
            LiveRecord record;
            record.key = view.key;
            record.metadata = view.metadata;
            record.values.assign(view.data, view.data + config_.dimensions);
            record.sequence = view.sequence;
            live_records.push_back(std::move(record));
        });
    }
    std::ranges::sort(live_records, [](const LiveRecord& left, const LiveRecord& right) {
        if (left.sequence != right.sequence) return left.sequence < right.sequence;
        return left.key < right.key;
    });

    auto compacted = createSegment(VectorSegment::State::Mutable);
    for (const auto& record : live_records) {
        if (!compacted->insertRecovered(Vector(record.values), record.key,
                                        record.metadata, record.sequence)) {
            throw std::runtime_error("compaction: failed to insert recovered record for key " + record.key);
        }
    }
    compacted->prepareSeal();

    std::vector<std::filesystem::path> old_dirs;
    old_dirs.reserve(sealed_segments_.size());
    for (const auto& segment : sealed_segments_) {
        old_dirs.push_back(segmentDir(segment->id()));
    }

    std::vector<std::shared_ptr<VectorSegment>> prospective_sealed;
    if (compacted->liveCount() > 0) {
        prospective_sealed.push_back(compacted);
    }

    writeManifest(mutable_segment_, prospective_sealed);
    if (compacted->liveCount() > 0) compacted->activateSeal();
    sealed_segments_.swap(prospective_sealed);
    rebuildKeyLocations();

    if (compacted->liveCount() > 0) {
        try {
            compacted->retireWal();
        } catch (...) {
        }
    } else {
        std::error_code ec;
        std::filesystem::remove_all(segmentDir(compacted->id()), ec);
    }

    for (const auto& dir : old_dirs) {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
    vdb::io::fsync_dir(segmentsDir());
}

void SegmentedVectorStore::sealMutableSegment() {
    if (!mutable_segment_ || mutable_segment_->recordCount() == 0) return;

    auto sealing = mutable_segment_;
    if (visible_lsn_ > durable_lsn_) {
        sealing->commitThrough(visible_lsn_);
        durable_lsn_ = visible_lsn_;
    }
    sealing->prepareSeal();
    committer_failpoint("seal-after-prepare");

    auto replacement = createSegment(VectorSegment::State::Mutable);
    committer_failpoint("seal-after-new-mutable");
    auto prospective_sealed = sealed_segments_;
    prospective_sealed.push_back(sealing);

    // This atomic manifest replacement is the role-transition commit point.
    // Before it, recovery replays `sealing` as mutable from its retained WAL;
    // after it, recovery opens the prepared files and the replacement WAL.
    writeManifest(replacement, prospective_sealed);
    committer_failpoint("seal-after-manifest");

    sealing->activateSeal();
    mutable_segment_ = std::move(replacement);
    sealed_segments_.swap(prospective_sealed);
    committer_failpoint("seal-after-publish");

    // Cleanup is intentionally post-commit. A leftover WAL is harmless because
    // the manifest is authoritative and sealed loads ignore it.
    try {
        sealing->retireWal();
    } catch (...) {
    }
    committer_failpoint("seal-after-retire");
}

SegmentedVectorStore::Statistics SegmentedVectorStore::getStatistics() const {
    Statistics stats{};
    stats.total_vectors = key_locations_.size();
    stats.sealed_segments = sealed_segments_.size();
    stats.total_segments = sealed_segments_.size() + (mutable_segment_ ? 1 : 0);
    stats.latest_sequence = latest_sequence_;
    stats.visible_lsn = visible_lsn_;
    stats.durable_lsn = durable_lsn_;
    stats.volatile_records = volatileCount();

    auto accumulate = [&](const std::shared_ptr<VectorSegment>& segment) {
        auto s = segment->getStatistics();
        stats.total_records += s.records;
        stats.total_tombstones += s.tombstones;
        stats.wal_bytes += s.wal_bytes;
        stats.vector_bytes += s.vector_bytes;
        stats.hnsw_snapshot_bytes += s.hnsw_snapshot_bytes;
        stats.hnsw_allocation_calls += s.hnsw_memory.allocation_calls;
        stats.hnsw_deallocation_calls += s.hnsw_memory.deallocation_calls;
        stats.hnsw_peak_bytes += s.hnsw_memory.peak_bytes_outstanding;
    };

    if (mutable_segment_) {
        stats.mutable_records = mutable_segment_->recordCount();
        accumulate(mutable_segment_);
    }
    for (const auto& segment : sealed_segments_) {
        accumulate(segment);
    }

    stats.disk_bytes = diskBytes();
    return stats;
}

std::unordered_map<std::string, Vector> SegmentedVectorStore::getAllVectors() const {
    std::unordered_map<std::string, Vector> out;
    out.reserve(key_locations_.size());
    for (const auto& [key, location] : key_locations_) {
        auto vector = location.segment->get(key);
        if (vector) out.emplace(key, *vector);
    }
    return out;
}

void SegmentedVectorStore::setMetric(std::shared_ptr<const DistanceMetric> metric) {
    config_.metric = std::move(metric);
    if (!initialized_) return;

    struct LiveRecord {
        std::string key;
        std::string metadata;
        std::vector<float> values;
        uint64_t sequence;
    };

    std::vector<LiveRecord> records;
    records.reserve(key_locations_.size());
    for (const auto& [key, location] : key_locations_) {
        auto vector = location.segment->get(key);
        if (vector) {
            records.push_back(LiveRecord{
                key,
                location.segment->getMetadata(key),
                std::vector<float>(vector->begin(), vector->end()),
                location.sequence,
            });
        }
    }
    std::ranges::sort(records, [](const LiveRecord& left, const LiveRecord& right) {
        if (left.sequence != right.sequence) return left.sequence < right.sequence;
        return left.key < right.key;
    });

    sealed_segments_.clear();
    mutable_segment_ = createSegment(VectorSegment::State::Mutable);
    for (const auto& record : records) {
        // Use insert() (not insertRecovered) so each migrated record is appended
        // to the new segment's WAL and fsync'd. insertRecovered skips the WAL, so
        // the migrated data lived only in memory and was lost on the next reload.
        if (!mutable_segment_->insert(Vector(record.values), record.key,
                                      record.metadata, record.sequence)) {
            throw std::runtime_error("recovery: failed to insert recovered record for key " + record.key);
        }
    }
    rebuildKeyLocations();
    writeManifest();
}

void SegmentedVectorStore::configureHNSW(size_t M, size_t ef_construction, size_t ef_search,
                                         uint32_t seed) {
    config_.hnsw_m = M;
    config_.hnsw_ef_construction = ef_construction;
    config_.hnsw_ef_search = ef_search;
    config_.hnsw_seed = seed;
    if (initialized_ && !read_only_recovery_) writeManifest();
}

void SegmentedVectorStore::configureAllocator(HNSWIndex::AllocationStrategy strategy, size_t arena_initial_size) {
    config_.allocation_strategy = strategy;
    config_.arena_initial_size = arena_initial_size;
}

void SegmentedVectorStore::configureSegmentation(size_t max_mutable_segment_records,
                                                 size_t max_sealed_segments,
                                                 double max_tombstone_ratio) {
    config_.max_mutable_segment_records = max_mutable_segment_records;
    config_.max_sealed_segments = max_sealed_segments;
    config_.max_tombstone_ratio = max_tombstone_ratio;
}

VectorSegment::Config SegmentedVectorStore::segmentConfig() const {
    return VectorSegment::Config{
        config_.dimensions,
        config_.hnsw_m,
        config_.hnsw_ef_construction,
        config_.hnsw_ef_search,
        config_.allocation_strategy,
        config_.arena_initial_size,
        config_.metric,
        config_.hnsw_seed,
    };
}

std::shared_ptr<VectorSegment> SegmentedVectorStore::createSegment(VectorSegment::State state) {
    auto id = makeSegmentId(next_segment_id_++);
    auto segment = std::make_shared<VectorSegment>(id, segmentDir(id), segmentConfig(), state);
    segment->initializeNew();
    return segment;
}

std::shared_ptr<VectorSegment> SegmentedVectorStore::loadSegment(
    const std::string& id, VectorSegment::State state, bool read_only_recovery) {
    auto segment = std::make_shared<VectorSegment>(id, segmentDir(id), segmentConfig(), state);
    segment->load(read_only_recovery);
    latest_sequence_ = std::max(latest_sequence_, segment->maxSequence());

    if (id.rfind("seg_", 0) == 0) {
        uint64_t numeric_id = std::stoull(id.substr(4));
        next_segment_id_ = std::max(next_segment_id_, numeric_id + 1);
    }

    return segment;
}

void SegmentedVectorStore::rebuildKeyLocations() {
    key_locations_.clear();

    struct LatestRecord {
        std::shared_ptr<VectorSegment> segment;
        uint64_t sequence{0};
        bool active{false};
    };

    std::unordered_map<std::string, LatestRecord> latest;

    auto add_records = [&](const std::shared_ptr<VectorSegment>& segment) {
        segment->forEachRecord([&](const VectorSegment::RecordView& view) {
            auto existing = latest.find(view.key);
            if (existing == latest.end()) {
                latest.emplace(view.key, LatestRecord{segment, view.sequence, view.active});
                return;
            }

            if (view.sequence > existing->second.sequence ||
                (view.sequence == existing->second.sequence && view.active)) {
                existing->second = LatestRecord{segment, view.sequence, view.active};
            }
        });
        latest_sequence_ = std::max(latest_sequence_, segment->maxSequence());
    };

    for (const auto& segment : sealed_segments_) add_records(segment);
    if (mutable_segment_) add_records(mutable_segment_);

    for (const auto& [key, record] : latest) {
        if (record.active) {
            key_locations_[key] = Location{record.segment, record.sequence};
        }
    }
}

void SegmentedVectorStore::maybeSealMutableSegment() {
    if (mutable_segment_ &&
        mutable_segment_->recordCount() >= config_.max_mutable_segment_records) {
        sealMutableSegment();
    }
}

void SegmentedVectorStore::maybeCompact() {
    if (sealed_segments_.size() > config_.max_sealed_segments) {
        compact();
        return;
    }

    for (const auto& segment : sealed_segments_) {
        if (segment->tombstoneRatio() >= config_.max_tombstone_ratio) {
            compact();
            return;
        }
    }
}

uint64_t SegmentedVectorStore::nextSequence() {
    if (latest_sequence_ >= reserved_sequence_hi_) {
        reserveSequenceBlock();
    }
    return ++latest_sequence_;
}

void SegmentedVectorStore::reserveSequenceBlock() {
    if (config_.sequence_reservation_block == 0) {
        throw std::invalid_argument("sequence reservation block must be positive");
    }
    const uint64_t persisted = readSequenceHighwater();
    const uint64_t base = std::max({latest_sequence_, reserved_sequence_hi_, persisted});
    if (base > std::numeric_limits<uint64_t>::max() -
                   config_.sequence_reservation_block) {
        throw std::overflow_error("sequence LSN space exhausted");
    }
    const uint64_t next_highwater = base + config_.sequence_reservation_block;
    writeSequenceHighwater(next_highwater);
    latest_sequence_ = base;
    reserved_sequence_hi_ = next_highwater;
}

uint64_t SegmentedVectorStore::readSequenceHighwater() const {
    std::ifstream is(sequenceHighwaterPath(), std::ios::binary);
    if (!is.is_open()) return 0;

    SequenceHighwaterRecord record{};
    is.read(reinterpret_cast<char*>(&record), kSequenceHighwaterRecordBytes);
    if (!is.good() || record.magic != kSequenceHighwaterMagic ||
        record.version != kSequenceHighwaterVersion ||
        record.crc32 != sequence_highwater_crc(record) ||
        is.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("corrupt sequence LSN high-water file");
    }
    return record.highwater;
}

void SegmentedVectorStore::writeSequenceHighwater(uint64_t highwater) const {
    SequenceHighwaterRecord record{
        kSequenceHighwaterMagic,
        kSequenceHighwaterVersion,
        highwater,
        0,
    };
    record.crc32 = sequence_highwater_crc(record);
    vdb::io::atomic_write(sequenceHighwaterPath(), [&](std::ostream& os) {
        os.write(reinterpret_cast<const char*>(&record), kSequenceHighwaterRecordBytes);
        if (!os.good()) throw std::runtime_error("failed writing sequence LSN high-water file");
    });
}

void SegmentedVectorStore::writeManifest() {
    writeManifest(mutable_segment_, sealed_segments_);
}

void SegmentedVectorStore::writeManifest(
    const std::shared_ptr<VectorSegment>& mutable_segment,
    const std::vector<std::shared_ptr<VectorSegment>>& sealed_segments) {
    if (read_only_recovery_) {
        throw std::logic_error("read-only recovery cannot write a manifest");
    }
    if (manifest_generation_ == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("manifest generation exhausted");
    }
    const uint64_t next_generation = manifest_generation_ + 1;
    std::ostringstream os;
    os << "version=1\n";
    os << "manifest_generation=" << next_generation << "\n";
    os << "dimensions=" << config_.dimensions << "\n";
    os << "next_segment_id=" << next_segment_id_ << "\n";
    os << "latest_sequence=" << latest_sequence_ << "\n";
    os << "visible_lsn=" << visible_lsn_ << "\n";
    os << "durable_lsn=" << durable_lsn_ << "\n";
    os << "hnsw_seed=" << config_.hnsw_seed << "\n";
    os << "mutable=" << (mutable_segment ? mutable_segment->id() : "") << "\n";
    os << "sealed=";
    for (size_t i = 0; i < sealed_segments.size(); ++i) {
        if (i > 0) os << ',';
        os << sealed_segments[i]->id();
    }
    os << "\n";
    atomic_text_write(manifestPath(), os.str());
    manifest_generation_ = next_generation;
}

bool SegmentedVectorStore::readManifest(std::string& mutable_id, std::vector<std::string>& sealed_ids) {
    std::ifstream is(manifestPath());
    if (!is.is_open()) return false;

    std::string line;
    while (std::getline(is, line)) {
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        if (key == "dimensions") {
            if (std::stoull(value) != config_.dimensions) {
                throw std::runtime_error("segmented store dimension mismatch");
            }
        } else if (key == "manifest_generation") {
            manifest_generation_ = std::stoull(value);
        } else if (key == "next_segment_id") {
            next_segment_id_ = std::stoull(value);
        } else if (key == "latest_sequence") {
            latest_sequence_ = std::stoull(value);
        } else if (key == "durable_lsn") {
            durable_lsn_ = std::stoull(value);
        } else if (key == "hnsw_seed") {
            const uint64_t seed = std::stoull(value);
            if (seed > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("segmented store HNSW seed is out of range");
            }
            config_.hnsw_seed = static_cast<uint32_t>(seed);
        } else if (key == "mutable") {
            mutable_id = value;
        } else if (key == "sealed") {
            sealed_ids = split_csv(value);
        }
    }
    return true;
}

std::string SegmentedVectorStore::makeSegmentId(uint64_t id) const {
    std::ostringstream os;
    os << "seg_" << std::setw(8) << std::setfill('0') << id;
    return os.str();
}

std::filesystem::path SegmentedVectorStore::segmentsDir() const {
    return root_ / "segments";
}

std::filesystem::path SegmentedVectorStore::segmentDir(const std::string& id) const {
    return segmentsDir() / id;
}

std::filesystem::path SegmentedVectorStore::manifestPath() const {
    return root_ / "manifest.txt";
}

std::filesystem::path SegmentedVectorStore::sequenceHighwaterPath() const {
    return root_ / "lsn.highwater";
}

size_t SegmentedVectorStore::diskBytes() const {
    std::error_code ec;
    if (!std::filesystem::exists(root_, ec)) return 0;

    size_t total = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root_, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        auto size = entry.file_size(ec);
        if (!ec) total += static_cast<size_t>(size);
    }
    return total;
}
