// Exhaustive byte-cut validation for fenced recall-committer WAL generations.
//
// The fixture contains two one-record generations, one three-record generation,
// and an unfenced two-record tail. Every possible WAL length is opened through
// the production read-only recovery path. A cut may expose only the exact stable
// prefix whose fence is fully present; it may never expose the weak tail or part
// of the three-record generation.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include <unistd.h>

#include "../src/core/vector_database.hpp"
#include "../src/utils/atomic_write.hpp"

namespace {

constexpr size_t kDimensions = 4;
constexpr size_t kIndividuallyStableRecords = 2;
constexpr size_t kGroupedStableRecords = 3;
constexpr size_t kWeakTailRecords = 2;

struct TempTree {
    std::filesystem::path root;

    TempTree() {
        root = std::filesystem::temp_directory_path() /
               ("vdb_committer_cuts_" + std::to_string(::getpid()) + "_" +
                std::to_string(std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count()));
        std::filesystem::create_directories(root);
    }

    ~TempTree() {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }
};

struct ExpectedRecord {
    std::string key;
    std::string metadata;
    std::vector<float> values;
    uint64_t lsn{0};
};

struct FenceBoundary {
    uintmax_t wal_bytes{0};
    size_t record_count{0};
    uint64_t durable_lsn{0};
};

struct FingerprintEntry {
    std::string relative_path;
    std::filesystem::file_type type{std::filesystem::file_type::none};
    uintmax_t size{0};
    std::filesystem::file_time_type write_time{};
    uint64_t content_hash{0};

    bool operator==(const FingerprintEntry&) const = default;
};

using Fingerprint = std::vector<FingerprintEntry>;

struct Counters {
    uintmax_t cuts{0};
    uintmax_t fingerprint_checks{0};
    uintmax_t payload_checks{0};
    uintmax_t absent_stable_checks{0};
    uintmax_t weak_absence_checks{0};
    uintmax_t query_checks{0};
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

std::vector<float> valuesFor(size_t ordinal) {
    std::vector<float> values(kDimensions);
    for (size_t dimension = 0; dimension < kDimensions; ++dimension) {
        values[dimension] = static_cast<float>((ordinal + 1) * 97 + dimension * 13) /
                            1024.0f;
    }
    return values;
}

ExpectedRecord recordFor(size_t ordinal) {
    return ExpectedRecord{
        "r" + std::to_string(ordinal),
        "m" + std::to_string(ordinal),
        valuesFor(ordinal),
        0,
    };
}

bool sameVector(const Vector& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) return false;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::bit_cast<uint32_t>(actual[i]) !=
            std::bit_cast<uint32_t>(expected[i])) {
            return false;
        }
    }
    return true;
}

uint64_t hashFile(const std::filesystem::path& path) {
    constexpr uint64_t kOffsetBasis = 14695981039346656037ull;
    constexpr uint64_t kPrime = 1099511628211ull;
    uint64_t hash = kOffsetBasis;
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) fail("cannot fingerprint " + path.string());
    char buffer[16 * 1024];
    while (input.read(buffer, sizeof(buffer)) || input.gcount() != 0) {
        const auto count = static_cast<size_t>(input.gcount());
        for (size_t i = 0; i < count; ++i) {
            hash ^= static_cast<unsigned char>(buffer[i]);
            hash *= kPrime;
        }
    }
    if (!input.eof()) fail("failed reading fingerprint input " + path.string());
    return hash;
}

Fingerprint fingerprintTree(const std::filesystem::path& root) {
    Fingerprint fingerprint;
    std::error_code error;
    std::filesystem::recursive_directory_iterator iterator(
        root, std::filesystem::directory_options::skip_permission_denied, error);
    const std::filesystem::recursive_directory_iterator end;
    for (; !error && iterator != end; iterator.increment(error)) {
        const auto path = iterator->path();
        const auto status = iterator->symlink_status(error);
        if (error) break;

        FingerprintEntry entry;
        entry.relative_path = path.lexically_relative(root).generic_string();
        entry.type = status.type();
        entry.write_time = std::filesystem::last_write_time(path, error);
        if (error) break;
        if (std::filesystem::is_regular_file(status)) {
            entry.size = std::filesystem::file_size(path, error);
            if (error) break;
            entry.content_hash = hashFile(path);
        }
        fingerprint.push_back(std::move(entry));
    }
    if (error) fail("cannot fingerprint database image: " + error.message());
    std::sort(fingerprint.begin(), fingerprint.end(),
              [](const FingerprintEntry& left, const FingerprintEntry& right) {
                  return left.relative_path < right.relative_path;
              });
    return fingerprint;
}

void copyTree(const std::filesystem::path& source,
              const std::filesystem::path& destination) {
    std::filesystem::create_directories(destination);
    for (const auto& entry : std::filesystem::recursive_directory_iterator(source)) {
        const auto relative = entry.path().lexically_relative(source);
        const auto target = destination / relative;
        if (entry.is_directory()) {
            std::filesystem::create_directories(target);
        } else if (entry.is_regular_file()) {
            std::filesystem::create_directories(target.parent_path());
            std::filesystem::copy_file(
                entry.path(), target, std::filesystem::copy_options::overwrite_existing);
        } else {
            fail("unexpected non-file in database tree: " + entry.path().string());
        }
    }
}

std::filesystem::path findWal(const std::filesystem::path& root) {
    std::vector<std::filesystem::path> matches;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().filename() == "wal.log") {
            matches.push_back(entry.path());
        }
    }
    if (matches.size() != 1) {
        fail("expected exactly one mutable WAL, found " +
             std::to_string(matches.size()));
    }
    return matches.front();
}

std::unique_ptr<VectorDatabase> openDatabase(const std::filesystem::path& path,
                                             vdb::OpenMode mode) {
    auto database = std::make_unique<VectorDatabase>(
        kDimensions, VectorDatabase::SearchMode::Exact, false, false,
        PersistenceConfig{}, false, 0, path.string(),
        VectorDatabase::StorageEngine::Segmented, mode);
    database->configureHNSWAllocator(HNSWIndex::AllocationStrategy::Standard);
    database->initialize();
    return database;
}

void verifyPresent(VectorDatabase& database, const ExpectedRecord& expected) {
    const auto latest = database.inspectRecord(
        expected.key, vdb::ReadVisibility::Latest);
    const auto stable = database.inspectRecord(
        expected.key, vdb::ReadVisibility::Stable);
    require(latest.has_value(), "stable key missing from latest view: " + expected.key);
    require(stable.has_value(), "stable key missing from stable view: " + expected.key);
    for (const auto* actual : {&*latest, &*stable}) {
        require(actual->lsn == expected.lsn,
                "LSN mismatch for " + expected.key);
        require(!actual->provisional,
                "recovered stable key is provisional: " + expected.key);
        require(actual->metadata == expected.metadata,
                "metadata mismatch for " + expected.key);
        require(sameVector(actual->vector, expected.values),
                "vector mismatch for " + expected.key);
    }
}

void verifyAbsent(VectorDatabase& database, const std::string& key) {
    require(!database.inspectRecord(key, vdb::ReadVisibility::Latest),
            "unexpected key in latest view: " + key);
    require(!database.inspectRecord(key, vdb::ReadVisibility::Stable),
            "unexpected key in stable view: " + key);
}

void verifyCut(const std::filesystem::path& image,
               uintmax_t cut,
               const std::vector<ExpectedRecord>& stable_records,
               const std::vector<ExpectedRecord>& weak_records,
               const std::vector<FenceBoundary>& boundaries,
               Counters& counters) {
    size_t expected_count = 0;
    uint64_t expected_lsn = 0;
    for (const auto& boundary : boundaries) {
        if (cut < boundary.wal_bytes) break;
        expected_count = boundary.record_count;
        expected_lsn = boundary.durable_lsn;
    }

    const Fingerprint before = fingerprintTree(image);
    auto database = openDatabase(image, vdb::OpenMode::ReadOnlyRecovery);
    const auto status = database->durabilityStatus();
    require(status.appended_lsn == expected_lsn,
            "recovered appended LSN mismatch");
    require(status.visible_lsn == expected_lsn,
            "recovered visible LSN mismatch");
    require(status.durable_lsn == expected_lsn,
            "recovered durable LSN mismatch");
    require(status.visible_records == expected_count,
            "recovered visible count mismatch");
    require(status.durable_records == expected_count,
            "recovered durable count mismatch");
    require(status.weak_records == 0, "recovery exposed a weak record");
    require(database->vectorCount() == expected_count,
            "vectorCount does not match fenced prefix");

    for (size_t i = 0; i < expected_count; ++i) {
        verifyPresent(*database, stable_records[i]);
        ++counters.payload_checks;
    }
    for (size_t i = expected_count; i < stable_records.size(); ++i) {
        verifyAbsent(*database, stable_records[i].key);
        ++counters.absent_stable_checks;
    }
    for (const auto& record : weak_records) {
        verifyAbsent(*database, record.key);
        ++counters.weak_absence_checks;
    }

    const auto query = database->similaritySearch(
        Vector(std::vector<float>(kDimensions, 0.0f)), stable_records.size() +
                                                        weak_records.size(),
        vdb::ReadVisibility::Stable);
    require(query.results.size() == expected_count,
            "stable query result count does not match fenced prefix");
    for (const auto& result : query.results) {
        const auto match = std::find_if(
            stable_records.begin(), stable_records.begin() + expected_count,
            [&](const ExpectedRecord& record) { return record.key == result.key; });
        require(match != stable_records.begin() + expected_count,
                "stable query returned an unexpected key: " + result.key);
    }
    ++counters.query_checks;

    database.reset();
    const Fingerprint after = fingerprintTree(image);
    require(before == after, "read-only recovery modified the cut image");
    ++counters.fingerprint_checks;
}

void appendWithExpectedAck(VectorDatabase& database,
                           ExpectedRecord& record,
                           vdb::AckMode requested,
                           vdb::AckLevel expected) {
    const auto receipt = database.insertWithAck(
        Vector(record.values), record.key, record.metadata, requested);
    require(receipt.applied, "fixture insert was not applied: " + record.key);
    require(receipt.actual_ack == expected,
            "fixture returned the wrong ACK level: " + record.key);
    require(receipt.lsn != 0, "fixture returned LSN zero: " + record.key);
    require(receipt.provisional == (expected == vdb::AckLevel::Weak),
            "fixture provisional flag disagrees with ACK level: " + record.key);
    record.lsn = receipt.lsn;
}

void run() {
    TempTree temporary;
    const auto source_path = temporary.root / "source";
    const auto image_path = temporary.root / "image";

    auto source = std::make_unique<VectorDatabase>(
        kDimensions, VectorDatabase::SearchMode::Exact, false, false,
        PersistenceConfig{}, false, 0, source_path.string(),
        VectorDatabase::StorageEngine::Segmented, vdb::OpenMode::ReadWrite);
    source->configureHNSWAllocator(HNSWIndex::AllocationStrategy::Standard);
    source->configureSegmentedStorage(1000000, 16, 0.25);

    vdb::RecallCommitConfig config;
    config.enabled = true;
    config.policy = vdb::RecallPolicy::Strict;
    config.epsilon = 0.4;
    config.k_min = 10;
    config.max_tail_records = 4;
    source->configureRecallCommit(config);
    source->initialize();

    const auto source_wal = findWal(source_path);
    std::vector<ExpectedRecord> stable_records;
    std::vector<ExpectedRecord> weak_records;
    std::vector<FenceBoundary> boundaries;

    for (size_t i = 0; i < kIndividuallyStableRecords; ++i) {
        auto record = recordFor(i);
        appendWithExpectedAck(
            *source, record, vdb::AckMode::Stable, vdb::AckLevel::Stable);
        stable_records.push_back(std::move(record));
        boundaries.push_back(FenceBoundary{
            std::filesystem::file_size(source_wal),
            stable_records.size(),
            stable_records.back().lsn,
        });
    }

    for (size_t i = 0; i < kGroupedStableRecords; ++i) {
        auto record = recordFor(kIndividuallyStableRecords + i);
        appendWithExpectedAck(
            *source, record, vdb::AckMode::Weak, vdb::AckLevel::Weak);
        stable_records.push_back(std::move(record));
    }
    const uint64_t group_lsn = source->durabilityFence();
    require(group_lsn == stable_records.back().lsn,
            "explicit fence did not reach the grouped generation");
    boundaries.push_back(FenceBoundary{
        std::filesystem::file_size(source_wal),
        stable_records.size(),
        group_lsn,
    });

    for (size_t i = 0; i < kWeakTailRecords; ++i) {
        auto record = recordFor(stable_records.size() + i);
        appendWithExpectedAck(
            *source, record, vdb::AckMode::Weak, vdb::AckLevel::Weak);
        weak_records.push_back(std::move(record));
    }

    const auto live_status = source->durabilityStatus();
    require(live_status.durable_records == stable_records.size(),
            "fixture stable record count is wrong");
    require(live_status.weak_records == weak_records.size(),
            "fixture weak tail count is wrong");
    require(live_status.durable_lsn == stable_records.back().lsn,
            "fixture durable frontier is wrong");
    require(live_status.visible_lsn == weak_records.back().lsn,
            "fixture visible frontier is wrong");

    const uintmax_t wal_bytes = std::filesystem::file_size(source_wal);
    require(!boundaries.empty() && boundaries.back().wal_bytes < wal_bytes,
            "fixture does not contain an unfenced WAL tail");
    copyTree(source_path, image_path);
    source.reset();  // The source may fence on shutdown; the copied image must not.

    const auto image_wal = findWal(image_path);
    require(std::filesystem::file_size(image_wal) == wal_bytes,
            "copied WAL size changed");

    Counters counters;
    for (uintmax_t cut = wal_bytes;; --cut) {
        std::filesystem::resize_file(image_wal, cut);
        try {
            verifyCut(image_path, cut, stable_records, weak_records,
                      boundaries, counters);
        } catch (const std::exception& error) {
            fail("WAL cut " + std::to_string(cut) + ": " + error.what());
        }
        ++counters.cuts;
        if (cut == 0) break;
    }

    const uintmax_t grouped_generation_bytes =
        boundaries.back().wal_bytes - boundaries[boundaries.size() - 2].wal_bytes;
    const uintmax_t weak_tail_bytes = wal_bytes - boundaries.back().wal_bytes;
    std::cout << "committer_cut_test: PASS"
              << " cuts=" << counters.cuts
              << " wal_bytes=" << wal_bytes
              << " fence_generations=" << boundaries.size()
              << " stable_records=" << stable_records.size()
              << " grouped_generation_records=" << kGroupedStableRecords
              << " grouped_generation_bytes=" << grouped_generation_bytes
              << " weak_tail_records=" << weak_records.size()
              << " weak_tail_bytes=" << weak_tail_bytes
              << " payload_checks=" << counters.payload_checks
              << " absent_stable_checks=" << counters.absent_stable_checks
              << " weak_absence_checks=" << counters.weak_absence_checks
              << " query_checks=" << counters.query_checks
              << " read_only_fingerprints=" << counters.fingerprint_checks
              << '\n';
}

}  // namespace

int main() {
    try {
        vdb::io::set_full_fsync(true);
        run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "committer_cut_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
