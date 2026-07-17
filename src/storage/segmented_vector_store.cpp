#include "segmented_vector_store.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "../utils/atomic_write.hpp"

namespace {
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
}

SegmentedVectorStore::SegmentedVectorStore(std::filesystem::path root, Config config)
    : root_(std::move(root)), config_(std::move(config)) {
}

void SegmentedVectorStore::initialize() {
    if (initialized_) return;

    std::filesystem::create_directories(segmentsDir());
    vdb::io::fsync_dir(root_.parent_path());
    vdb::io::fsync_dir(root_);

    std::string mutable_id;
    std::vector<std::string> sealed_ids;
    if (readManifest(mutable_id, sealed_ids)) {
        sealed_segments_.clear();
        sealed_segments_.reserve(sealed_ids.size());
        for (const auto& id : sealed_ids) {
            sealed_segments_.push_back(loadSegment(id, VectorSegment::State::Sealed));
        }

        if (!mutable_id.empty()) {
            mutable_segment_ = loadSegment(mutable_id, VectorSegment::State::Mutable);
        }
    }

    if (!mutable_segment_) {
        mutable_segment_ = createSegment(VectorSegment::State::Mutable);
    }

    rebuildKeyLocations();
    writeManifest();
    initialized_ = true;
}

void SegmentedVectorStore::shutdown() {
    if (!initialized_) return;
    flush();
    initialized_ = false;
}

bool SegmentedVectorStore::insert(const Vector& vector, const std::string& key, const std::string& metadata) {
    if (!initialized_) throw std::runtime_error("segmented store not initialized");
    if (vector.size() != config_.dimensions) throw std::invalid_argument("vector dimension mismatch");
    if (key_locations_.count(key) != 0) return false;

    uint64_t sequence = nextSequence();
    if (!mutable_segment_->insert(vector, key, metadata, sequence)) return false;
    key_locations_[key] = Location{mutable_segment_, sequence};

    maybeSealMutableSegment();
    maybeCompact();
    return true;
}

size_t SegmentedVectorStore::insertBatch(const std::vector<std::string>& keys,
                                         const std::vector<Vector>& vectors,
                                         const std::vector<std::string>& metadata) {
    if (!initialized_) throw std::runtime_error("segmented store not initialized");
    size_t committed = 0;
    mutable_segment_->beginDeferredSync();  // defer fsync across the whole batch
    for (size_t i = 0; i < keys.size() && i < vectors.size(); ++i) {
        const Vector& v = vectors[i];
        const std::string& key = keys[i];
        if (v.size() != config_.dimensions) continue;
        if (key_locations_.count(key) != 0) continue;
        const std::string& meta = (i < metadata.size()) ? metadata[i] : "";
        uint64_t sequence = nextSequence();
        if (mutable_segment_->insert(v, key, meta, sequence)) {  // appends WAL (deferred), applies
            key_locations_[key] = Location{mutable_segment_, sequence};
            ++committed;
        }
    }
    mutable_segment_->commitDeferredSync();  // single fsync for the batch (group commit)
    maybeSealMutableSegment();
    maybeCompact();
    return committed;
}

bool SegmentedVectorStore::update(const Vector& vector, const std::string& key, const std::string& metadata) {
    if (!initialized_) throw std::runtime_error("segmented store not initialized");
    if (vector.size() != config_.dimensions) throw std::invalid_argument("vector dimension mismatch");
    if (key_locations_.count(key) == 0) return false;

    uint64_t sequence = nextSequence();
    if (!mutable_segment_->update(vector, key, metadata, sequence)) return false;
    key_locations_[key] = Location{mutable_segment_, sequence};

    maybeSealMutableSegment();
    maybeCompact();
    return true;
}

bool SegmentedVectorStore::remove(const std::string& key) {
    if (!initialized_) throw std::runtime_error("segmented store not initialized");

    auto it = key_locations_.find(key);
    if (it == key_locations_.end()) return false;

    uint64_t sequence = nextSequence();
    bool removed = it->second.segment->remove(key, sequence);
    if (removed) {
        key_locations_.erase(it);
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

std::vector<std::pair<std::string, float>> SegmentedVectorStore::search(const Vector& query, size_t k) const {
    auto with_meta = searchWithMetadata(query, k);
    std::vector<std::pair<std::string, float>> out;
    out.reserve(with_meta.size());
    for (const auto& result : with_meta) {
        out.emplace_back(result.key, result.distance);
    }
    return out;
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
            candidates.push_back(SearchResult{result.key, result.distance, result.metadata});
        }
    };

    if (mutable_segment_) append_results(mutable_segment_);
    for (const auto& segment : sealed_segments_) {
        append_results(segment);
    }

    std::ranges::sort(candidates, {}, &SegmentedVectorStore::SearchResult::distance);

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

void SegmentedVectorStore::flush() {
    if (mutable_segment_) mutable_segment_->flush();
    for (const auto& segment : sealed_segments_) {
        segment->flush();
    }
    writeManifest();
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

    auto compacted = createSegment(VectorSegment::State::Mutable);
    for (const auto& record : live_records) {
        if (!compacted->insertRecovered(Vector(record.values), record.key,
                                        record.metadata, record.sequence)) {
            throw std::runtime_error("compaction: failed to insert recovered record for key " + record.key);
        }
    }
    compacted->seal();

    std::vector<std::filesystem::path> old_dirs;
    old_dirs.reserve(sealed_segments_.size());
    for (const auto& segment : sealed_segments_) {
        old_dirs.push_back(segmentDir(segment->id()));
    }

    sealed_segments_.clear();
    if (compacted->liveCount() > 0) {
        sealed_segments_.push_back(compacted);
    } else {
        std::filesystem::remove_all(segmentDir(compacted->id()));
    }

    rebuildKeyLocations();
    writeManifest();

    for (const auto& dir : old_dirs) {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
}

void SegmentedVectorStore::sealMutableSegment() {
    if (!mutable_segment_ || mutable_segment_->recordCount() == 0) return;

    mutable_segment_->seal();
    sealed_segments_.push_back(mutable_segment_);
    mutable_segment_ = createSegment(VectorSegment::State::Mutable);
    rebuildKeyLocations();
    writeManifest();
}

SegmentedVectorStore::Statistics SegmentedVectorStore::getStatistics() const {
    Statistics stats{};
    stats.total_vectors = key_locations_.size();
    stats.sealed_segments = sealed_segments_.size();
    stats.total_segments = sealed_segments_.size() + (mutable_segment_ ? 1 : 0);
    stats.latest_sequence = latest_sequence_;

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

void SegmentedVectorStore::configureHNSW(size_t M, size_t ef_construction, size_t ef_search) {
    config_.hnsw_m = M;
    config_.hnsw_ef_construction = ef_construction;
    config_.hnsw_ef_search = ef_search;
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
    };
}

std::shared_ptr<VectorSegment> SegmentedVectorStore::createSegment(VectorSegment::State state) {
    auto id = makeSegmentId(next_segment_id_++);
    auto segment = std::make_shared<VectorSegment>(id, segmentDir(id), segmentConfig(), state);
    segment->initializeNew();
    return segment;
}

std::shared_ptr<VectorSegment> SegmentedVectorStore::loadSegment(const std::string& id, VectorSegment::State state) {
    auto segment = std::make_shared<VectorSegment>(id, segmentDir(id), segmentConfig(), state);
    segment->load();
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
    return ++latest_sequence_;
}

void SegmentedVectorStore::writeManifest() const {
    std::ostringstream os;
    os << "version=1\n";
    os << "dimensions=" << config_.dimensions << "\n";
    os << "next_segment_id=" << next_segment_id_ << "\n";
    os << "latest_sequence=" << latest_sequence_ << "\n";
    os << "mutable=" << (mutable_segment_ ? mutable_segment_->id() : "") << "\n";
    os << "sealed=";
    for (size_t i = 0; i < sealed_segments_.size(); ++i) {
        if (i > 0) os << ',';
        os << sealed_segments_[i]->id();
    }
    os << "\n";
    atomic_text_write(manifestPath(), os.str());
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
        } else if (key == "next_segment_id") {
            next_segment_id_ = std::stoull(value);
        } else if (key == "latest_sequence") {
            latest_sequence_ = std::stoull(value);
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
