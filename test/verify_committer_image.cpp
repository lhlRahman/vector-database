// Read-only verifier for committer crash images and external ACK ledgers.
//
// This executable must never repair an image. It opens with
// OpenMode::ReadOnlyRecovery, which must use the production WAL scanner/search
// path while forbidding directory creation, truncation, manifest replacement,
// background workers, shutdown fences, and cleanup writes. The surrounding
// power-loss harness performs normal filesystem journal recovery, unmounts, and
// remounts read-only before invoking this process.
//
// The ledger format is documented in committer_crash_test.cpp. Verification is
// driven by a named FRONTIER. --allow-through names the following frontier for
// an individual block cut: operations between the two marks are allowed in
// flight but are not required. At an exact named mark, omit --allow-through.
//
// Milestone-3 needs one additional read-only inspection adapter to meet the
// approved "payloads and LSNs, not counts" requirement. The assumed method is:
//
//   vector<RecordSnapshot> VectorDatabase::inspectRecords(
//       ReadVisibility visibility) const;
//
// with RecordSnapshot { string key; Vector vector; string metadata; uint64_t lsn; }.
// Enumeration, rather than key lookup, is required to detect duplicate recovered
// records. If production chooses another name, only inspectRecords() below changes.

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../src/core/vector_database.hpp"

#ifndef VDB_HAS_RECALL_COMMITTER_API
#define VDB_HAS_RECALL_COMMITTER_API 0
#endif

// Enable only when ReadOnlyRecovery is side-effect free, DurabilityStatus also
// exposes appended_lsn and manifest_generation, and inspectRecords() returns the
// production-decoded payload plus its LSN. The current milestone-3 API does not
// yet satisfy those image-verification requirements.
#ifndef VDB_HAS_COMMITTER_IMAGE_VERIFY_API
#define VDB_HAS_COMMITTER_IMAGE_VERIFY_API 0
#endif

namespace {

constexpr int kSkip = 77;

struct Options {
    std::filesystem::path db;
    std::filesystem::path ledger;
    std::string frontier;
    std::optional<std::string> allow_through;
    bool capabilities{false};
    bool validate_only{false};
};

[[noreturn]] void usage(const char* argv0, int status) {
    std::cerr << "usage: " << argv0
              << " --db DIR --ledger FILE --frontier NAME [--allow-through NAME]\n"
              << "       " << argv0 << " --validate-ledger FILE\n"
              << "       " << argv0 << " --capabilities\n";
    std::exit(status);
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string_view {
            if (++i == argc) usage(argv[0], 2);
            return argv[i];
        };
        if (arg == "--db") options.db = value();
        else if (arg == "--ledger") options.ledger = value();
        else if (arg == "--frontier") options.frontier = value();
        else if (arg == "--allow-through") options.allow_through = std::string(value());
        else if (arg == "--validate-ledger") {
            options.ledger = value();
            options.validate_only = true;
        } else if (arg == "--capabilities") options.capabilities = true;
        else if (arg == "--help" || arg == "-h") usage(argv[0], 0);
        else throw std::runtime_error("unknown option: " + arg);
    }
    if (!options.capabilities && options.ledger.empty()) usage(argv[0], 2);
    if (!options.capabilities && !options.validate_only &&
        (options.db.empty() || options.frontier.empty())) usage(argv[0], 2);
    return options;
}

std::vector<std::string_view> split(std::string_view line) {
    std::vector<std::string_view> fields;
    while (!line.empty()) {
        const size_t first = line.find_first_not_of(' ');
        if (first == std::string_view::npos) break;
        line.remove_prefix(first);
        const size_t end = line.find(' ');
        fields.push_back(line.substr(0, end));
        if (end == std::string_view::npos) break;
        line.remove_prefix(end + 1);
    }
    return fields;
}

template <typename UInt>
UInt parseUnsigned(std::string_view text, int base, std::string_view what) {
    UInt value{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value, base);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::runtime_error("invalid " + std::string(what) + ": " + std::string(text));
    }
    return value;
}

size_t parseSize(std::string_view text, std::string_view what) {
    const uint64_t value = parseUnsigned<uint64_t>(text, 10, what);
    if (value > std::numeric_limits<size_t>::max()) throw std::runtime_error("oversized " + std::string(what));
    return static_cast<size_t>(value);
}

bool parseBool(std::string_view text, std::string_view what) {
    if (text == "0") return false;
    if (text == "1") return true;
    throw std::runtime_error("invalid " + std::string(what));
}

bool validName(std::string_view name) {
    return !name.empty() && std::all_of(name.begin(), name.end(), [](unsigned char ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
               (ch >= '0' && ch <= '9') || ch == '_' || ch == '.' || ch == '-';
    });
}

unsigned hexNibble(char ch) {
    if (ch >= '0' && ch <= '9') return static_cast<unsigned>(ch - '0');
    if (ch >= 'a' && ch <= 'f') return static_cast<unsigned>(ch - 'a' + 10);
    if (ch >= 'A' && ch <= 'F') return static_cast<unsigned>(ch - 'A' + 10);
    throw std::runtime_error("invalid hex character");
}

std::string hexDecode(std::string_view encoded) {
    if (encoded == "-") return {};
    if (encoded.size() % 2 != 0) throw std::runtime_error("odd-length hex string");
    std::string value(encoded.size() / 2, '\0');
    for (size_t i = 0; i < value.size(); ++i) {
        value[i] = static_cast<char>((hexNibble(encoded[i * 2]) << 4) |
                                     hexNibble(encoded[i * 2 + 1]));
    }
    return value;
}

double decodeDouble(std::string_view encoded) {
    if (encoded.size() != 16) throw std::runtime_error("double bit pattern must have 16 hex digits");
    return std::bit_cast<double>(parseUnsigned<uint64_t>(encoded, 16, "double bits"));
}

float decodeFloat(std::string_view encoded) {
    if (encoded.size() != 8) throw std::runtime_error("float bit pattern must have 8 hex digits");
    return std::bit_cast<float>(parseUnsigned<uint32_t>(encoded, 16, "float bits"));
}

std::vector<float> decodeVector(std::string_view encoded) {
    if (encoded == "-") return {};
    std::vector<float> values;
    while (!encoded.empty()) {
        const size_t comma = encoded.find(',');
        values.push_back(decodeFloat(encoded.substr(0, comma)));
        if (comma == std::string_view::npos) break;
        encoded.remove_prefix(comma + 1);
        if (encoded.empty()) throw std::runtime_error("empty vector component");
    }
    return values;
}

struct ConfigRecord {
    size_t dimensions{};
    size_t k{};
    uint32_t seed{};
    std::string policy;
    double epsilon{};
    double delta{};
    std::string db_path;
};

struct IntentRecord {
    size_t line{};
    uint64_t ordinal{};
    std::string requested;
    std::string key;
    std::string metadata;
    std::vector<float> values;
};

struct AckRecord {
    size_t line{};
    uint64_t ordinal{};
    bool applied{};
    uint64_t lsn{};
    std::string requested;
    std::string actual;
    uint64_t visible_lsn{};
    uint64_t durable_lsn{};
    size_t durable_count{};
    size_t weak_count{};
    size_t policy_cap{};
    double risk{};
    bool provisional{};
};

struct ResultRecord {
    size_t rank{};
    std::string key;
    float distance{};
};

struct QueryRecord {
    size_t line{};
    uint64_t id{};
    std::string visibility;
    uint64_t snapshot_lsn{};
    uint64_t durable_lsn{};
    uint64_t manifest_generation{};
    size_t k{};
    std::vector<float> values;
    std::vector<ResultRecord> results;
};

struct FrontierRecord {
    size_t line{};
    std::string name;
    uint64_t appended_lsn{};
    uint64_t visible_lsn{};
    uint64_t durable_lsn{};
    size_t durable_count{};
    size_t weak_count{};
    uint64_t manifest_generation{};
};

struct Ledger {
    ConfigRecord config;
    std::vector<IntentRecord> intents;
    std::vector<AckRecord> acks;
    std::vector<QueryRecord> queries;
    std::vector<FrontierRecord> frontiers;
    std::unordered_set<std::string> marks;
    std::optional<int> child_status;
};

template <typename T>
void requireFields(const std::vector<T>& fields, size_t expected, size_t line) {
    if (fields.size() != expected) {
        throw std::runtime_error("ledger line " + std::to_string(line) +
                                 " has " + std::to_string(fields.size()) +
                                 " fields; expected " + std::to_string(expected));
    }
}

Ledger readLedger(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open ledger: " + path.string());
    Ledger ledger;
    bool saw_header = false;
    bool saw_config = false;
    std::unordered_map<uint64_t, size_t> query_index;
    std::unordered_set<uint64_t> intent_ordinals;
    std::unordered_set<uint64_t> ack_ordinals;
    std::unordered_set<std::string> frontier_names;
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) throw std::runtime_error("empty ledger line " + std::to_string(line_number));
        const auto fields = split(line);
        if (fields.empty()) throw std::runtime_error("invalid ledger line");
        const std::string_view kind = fields[0];
        if (!saw_header) {
            requireFields(fields, 2, line_number);
            if (kind != "VDB_COMMITTER_LEDGER" || fields[1] != "1") {
                throw std::runtime_error("unsupported ledger header/version");
            }
            saw_header = true;
            continue;
        }
        if (kind == "CONFIG") {
            requireFields(fields, 8, line_number);
            if (saw_config) throw std::runtime_error("duplicate CONFIG record");
            ledger.config.dimensions = parseSize(fields[1], "dimensions");
            ledger.config.k = parseSize(fields[2], "k");
            ledger.config.seed = parseUnsigned<uint32_t>(fields[3], 10, "seed");
            ledger.config.policy = fields[4];
            ledger.config.epsilon = decodeDouble(fields[5]);
            ledger.config.delta = decodeDouble(fields[6]);
            ledger.config.db_path = hexDecode(fields[7]);
            if (ledger.config.dimensions == 0 || ledger.config.k == 0 ||
                !std::isfinite(ledger.config.epsilon) || ledger.config.epsilon < 0.0 ||
                ledger.config.epsilon > 1.0) {
                throw std::runtime_error("invalid CONFIG values");
            }
            if (ledger.config.policy != "strict" && ledger.config.policy != "exchangeable" &&
                ledger.config.policy != "hypergeometric") {
                throw std::runtime_error("invalid CONFIG policy");
            }
            saw_config = true;
        } else if (kind == "INTENT") {
            requireFields(fields, 6, line_number);
            IntentRecord record;
            record.line = line_number;
            record.ordinal = parseUnsigned<uint64_t>(fields[1], 10, "intent ordinal");
            record.requested = fields[2];
            record.key = hexDecode(fields[3]);
            record.metadata = hexDecode(fields[4]);
            record.values = decodeVector(fields[5]);
            if (!intent_ordinals.insert(record.ordinal).second) throw std::runtime_error("duplicate INTENT ordinal");
            if (record.requested != "stable" && record.requested != "weak") throw std::runtime_error("invalid requested ACK");
            if (record.key.empty()) throw std::runtime_error("empty intent key");
            ledger.intents.push_back(std::move(record));
        } else if (kind == "ACK") {
            requireFields(fields, 13, line_number);
            AckRecord record;
            record.line = line_number;
            record.ordinal = parseUnsigned<uint64_t>(fields[1], 10, "ACK ordinal");
            record.applied = parseBool(fields[2], "ACK applied");
            record.lsn = parseUnsigned<uint64_t>(fields[3], 10, "ACK lsn");
            record.requested = fields[4];
            record.actual = fields[5];
            record.visible_lsn = parseUnsigned<uint64_t>(fields[6], 10, "ACK visible lsn");
            record.durable_lsn = parseUnsigned<uint64_t>(fields[7], 10, "ACK durable lsn");
            record.durable_count = parseSize(fields[8], "ACK durable count");
            record.weak_count = parseSize(fields[9], "ACK weak count");
            record.policy_cap = parseSize(fields[10], "ACK policy cap");
            record.risk = decodeDouble(fields[11]);
            record.provisional = parseBool(fields[12], "ACK provisional");
            if (!ack_ordinals.insert(record.ordinal).second) throw std::runtime_error("duplicate ACK ordinal");
            if (record.actual != "stable" && record.actual != "weak" && record.actual != "none") {
                throw std::runtime_error("invalid actual ACK");
            }
            if (record.requested != "stable" && record.requested != "weak") throw std::runtime_error("invalid requested ACK");
            ledger.acks.push_back(std::move(record));
        } else if (kind == "QUERY") {
            requireFields(fields, 8, line_number);
            QueryRecord record;
            record.line = line_number;
            record.id = parseUnsigned<uint64_t>(fields[1], 10, "query id");
            record.visibility = fields[2];
            record.snapshot_lsn = parseUnsigned<uint64_t>(fields[3], 10, "query snapshot lsn");
            record.durable_lsn = parseUnsigned<uint64_t>(fields[4], 10, "query durable lsn");
            record.manifest_generation = parseUnsigned<uint64_t>(fields[5], 10, "query manifest");
            record.k = parseSize(fields[6], "query k");
            record.values = decodeVector(fields[7]);
            if (record.visibility != "stable" && record.visibility != "latest") throw std::runtime_error("invalid query visibility");
            if (!query_index.emplace(record.id, ledger.queries.size()).second) throw std::runtime_error("duplicate query id");
            ledger.queries.push_back(std::move(record));
        } else if (kind == "RESULT") {
            requireFields(fields, 5, line_number);
            const uint64_t id = parseUnsigned<uint64_t>(fields[1], 10, "result query id");
            const auto query = query_index.find(id);
            if (query == query_index.end()) throw std::runtime_error("RESULT precedes QUERY");
            ResultRecord record;
            record.rank = parseSize(fields[2], "result rank");
            record.key = hexDecode(fields[3]);
            record.distance = decodeFloat(fields[4]);
            auto& results = ledger.queries[query->second].results;
            if (record.rank != results.size()) throw std::runtime_error("non-contiguous result ranks");
            results.push_back(std::move(record));
        } else if (kind == "FRONTIER") {
            requireFields(fields, 8, line_number);
            FrontierRecord record;
            record.line = line_number;
            record.name = fields[1];
            record.appended_lsn = parseUnsigned<uint64_t>(fields[2], 10, "frontier appended lsn");
            record.visible_lsn = parseUnsigned<uint64_t>(fields[3], 10, "frontier visible lsn");
            record.durable_lsn = parseUnsigned<uint64_t>(fields[4], 10, "frontier durable lsn");
            record.durable_count = parseSize(fields[5], "frontier durable count");
            record.weak_count = parseSize(fields[6], "frontier weak count");
            record.manifest_generation = parseUnsigned<uint64_t>(fields[7], 10, "frontier manifest");
            if (!validName(record.name) || !frontier_names.insert(record.name).second) {
                throw std::runtime_error("invalid or duplicate frontier name");
            }
            if (record.appended_lsn < record.visible_lsn || record.visible_lsn < record.durable_lsn) {
                throw std::runtime_error("frontier ordering appended >= visible >= durable violated");
            }
            ledger.frontiers.push_back(std::move(record));
        } else if (kind == "MARK") {
            requireFields(fields, 2, line_number);
            if (!validName(fields[1])) throw std::runtime_error("invalid MARK name");
            if (!ledger.marks.emplace(fields[1]).second) throw std::runtime_error("duplicate MARK");
        } else if (kind == "END") {
            requireFields(fields, 2, line_number);
            if (ledger.child_status) throw std::runtime_error("duplicate END");
            ledger.child_status = static_cast<int>(parseUnsigned<unsigned>(fields[1], 10, "child status"));
        } else if (kind == "ERROR") {
            throw std::runtime_error("workload recorded ERROR at ledger line " + std::to_string(line_number));
        } else {
            throw std::runtime_error("unknown ledger record at line " + std::to_string(line_number));
        }
    }
    if (input.bad()) throw std::runtime_error("I/O error reading ledger");
    if (!saw_header || !saw_config) throw std::runtime_error("incomplete ledger header");
    if (ledger.frontiers.empty()) throw std::runtime_error("ledger has no frontiers");
    if (!ledger.child_status) throw std::runtime_error("ledger has no controller END record");
    if (!ledger.marks.empty()) {
        if (ledger.marks.size() != ledger.frontiers.size()) {
            throw std::runtime_error("physical ledger does not mark every frontier");
        }
        for (const auto& frontier : ledger.frontiers) {
            if (!ledger.marks.contains(frontier.name)) {
                throw std::runtime_error("missing physical mark for frontier " + frontier.name);
            }
        }
    }

    std::unordered_map<uint64_t, const IntentRecord*> intents;
    std::unordered_set<std::string> keys;
    for (const auto& intent : ledger.intents) {
        if (intent.values.size() != ledger.config.dimensions) throw std::runtime_error("intent dimension mismatch");
        if (!keys.insert(intent.key).second) throw std::runtime_error("duplicate intent key");
        intents.emplace(intent.ordinal, &intent);
    }
    uint64_t previous_lsn = 0;
    uint64_t previous_visible = 0;
    uint64_t previous_durable = 0;
    for (const auto& ack : ledger.acks) {
        const auto intent = intents.find(ack.ordinal);
        if (intent == intents.end() || ack.line < intent->second->line) throw std::runtime_error("ACK without preceding INTENT");
        if (ack.requested != intent->second->requested) throw std::runtime_error("ACK request does not match INTENT");
        if (ack.requested == "stable" && ack.actual != "stable") throw std::runtime_error("stable request received weak ACK");
        if (ack.applied) {
            if (ack.lsn == 0 || ack.lsn <= previous_lsn) throw std::runtime_error("non-monotone applied ACK LSN");
            previous_lsn = ack.lsn;
            if (ack.visible_lsn < ack.lsn) throw std::runtime_error("applied ACK is not visible");
            if (ack.actual == "stable" && ack.durable_lsn < ack.lsn) throw std::runtime_error("stable ACK is not durable");
        }
        if (ack.applied && ack.actual == "none") throw std::runtime_error("applied ACK has no level");
        if (!ack.applied && ack.actual != "none") throw std::runtime_error("rejected ACK has a durability level");
        if (ack.actual == "stable" && ack.provisional) throw std::runtime_error("stable ACK marked provisional");
        if (ack.actual == "weak" && !ack.provisional) throw std::runtime_error("weak ACK is not marked provisional");
        if (ack.visible_lsn < previous_visible || ack.durable_lsn < previous_durable) {
            throw std::runtime_error("non-monotone ACK frontier");
        }
        previous_visible = ack.visible_lsn;
        previous_durable = ack.durable_lsn;
        if (ledger.config.policy == "strict" && ack.weak_count > ack.policy_cap) {
            throw std::runtime_error("strict weak-count cap overshoot");
        }
        if (!std::isfinite(ack.risk) || ack.risk < 0.0) throw std::runtime_error("invalid ACK risk");
    }
    uint64_t appended = 0, visible = 0, durable = 0, manifest = 0;
    for (const auto& frontier : ledger.frontiers) {
        if (frontier.appended_lsn < appended || frontier.visible_lsn < visible ||
            frontier.durable_lsn < durable || frontier.manifest_generation < manifest) {
            throw std::runtime_error("non-monotone frontier ledger");
        }
        appended = frontier.appended_lsn;
        visible = frontier.visible_lsn;
        durable = frontier.durable_lsn;
        manifest = frontier.manifest_generation;
    }
    return ledger;
}

const FrontierRecord& findFrontier(const Ledger& ledger, const std::string& name) {
    const auto found = std::find_if(ledger.frontiers.begin(), ledger.frontiers.end(),
                                    [&](const FrontierRecord& record) { return record.name == name; });
    if (found == ledger.frontiers.end()) throw std::runtime_error("frontier not found in ledger: " + name);
    return *found;
}

struct ExpectedSets {
    std::vector<const IntentRecord*> required;
    std::vector<const IntentRecord*> optional_weak;
    std::vector<const IntentRecord*> allowed_inflight;
    std::unordered_map<std::string, uint64_t> expected_lsn;
};

[[maybe_unused]] ExpectedSets classify(const Ledger& ledger, const FrontierRecord& frontier,
                                       const FrontierRecord& upper) {
    if (upper.line < frontier.line) throw std::runtime_error("--allow-through precedes selected frontier");
    std::unordered_map<uint64_t, const IntentRecord*> intents;
    std::unordered_map<uint64_t, const AckRecord*> acks;
    for (const auto& intent : ledger.intents) intents.emplace(intent.ordinal, &intent);
    for (const auto& ack : ledger.acks) acks.emplace(ack.ordinal, &ack);

    ExpectedSets sets;
    for (const auto& intent : ledger.intents) {
        if (intent.line > upper.line) continue;
        const auto ack_it = acks.find(intent.ordinal);
        const AckRecord* ack = ack_it == acks.end() ? nullptr : ack_it->second;
        if (intent.line > frontier.line || !ack || ack->line > frontier.line) {
            sets.allowed_inflight.push_back(&intent);
            continue;
        }
        if (!ack->applied) continue;
        sets.expected_lsn.emplace(intent.key, ack->lsn);
        if (ack->actual == "stable" || ack->lsn <= frontier.durable_lsn) {
            sets.required.push_back(&intent);
        } else {
            sets.optional_weak.push_back(&intent);
        }
    }

    for (const auto& ack : ledger.acks) {
        if (ack.line > frontier.line || !ack.applied) continue;
        if (ack.actual == "stable" && ack.lsn > frontier.durable_lsn) {
            throw std::runtime_error("selected frontier is behind a completed stable ACK");
        }
    }
    return sets;
}

using FileFingerprint = std::vector<std::tuple<std::string, uintmax_t, std::filesystem::file_time_type, unsigned>>;

[[maybe_unused]] FileFingerprint fingerprintTree(const std::filesystem::path& root) {
    FileFingerprint fingerprint;
    std::error_code error;
    if (!std::filesystem::exists(root, error)) return fingerprint;
    std::filesystem::recursive_directory_iterator iterator(
        root, std::filesystem::directory_options::skip_permission_denied, error);
    const std::filesystem::recursive_directory_iterator end;
    for (; !error && iterator != end; iterator.increment(error)) {
        const auto status = iterator->symlink_status(error);
        if (error) break;
        const unsigned type = static_cast<unsigned>(status.type());
        uintmax_t size = 0;
        if (std::filesystem::is_regular_file(status)) size = iterator->file_size(error);
        if (error) break;
        const auto write_time = iterator->last_write_time(error);
        if (error) break;
        fingerprint.emplace_back(iterator->path().lexically_relative(root).generic_string(),
                                 size, write_time, type);
    }
    if (error) throw std::runtime_error("cannot fingerprint image: " + error.message());
    std::sort(fingerprint.begin(), fingerprint.end());
    return fingerprint;
}

#if VDB_HAS_RECALL_COMMITTER_API && VDB_HAS_COMMITTER_IMAGE_VERIFY_API

std::unique_ptr<VectorDatabase> openReadOnly(const Options& options, const ConfigRecord& config) {
    auto db = std::make_unique<VectorDatabase>(
        config.dimensions, VectorDatabase::SearchMode::HNSW, false, false,
        PersistenceConfig{}, false, 0, options.db.string(),
        VectorDatabase::StorageEngine::Segmented, vdb::OpenMode::ReadOnlyRecovery);
    db->configureHNSW(16, 200, 64, config.seed);
    db->initialize();
    return db;
}

struct InspectedRecord {
    std::string key;
    std::vector<float> values;
    std::string metadata;
    uint64_t lsn{};
};

std::vector<InspectedRecord> inspectRecords(VectorDatabase& db) {
    std::vector<InspectedRecord> inspected;
    for (const auto& record : db.inspectRecords(vdb::ReadVisibility::Stable)) {
        std::vector<float> values(record.vector.size());
        for (size_t i = 0; i < values.size(); ++i) values[i] = record.vector[i];
        inspected.push_back(InspectedRecord{
            record.key, std::move(values), record.metadata, record.lsn});
    }
    return inspected;
}

bool sameVector(const std::vector<float>& left, const std::vector<float>& right) {
    if (left.size() != right.size()) return false;
    for (size_t i = 0; i < left.size(); ++i) {
        if (std::bit_cast<uint32_t>(left[i]) != std::bit_cast<uint32_t>(right[i])) return false;
    }
    return true;
}

void verifyPayload(const IntentRecord& expected, const InspectedRecord& actual,
                   std::optional<uint64_t> expected_lsn) {
    if (!sameVector(expected.values, actual.values)) throw std::runtime_error("vector mismatch for " + expected.key);
    if (expected.metadata != actual.metadata) throw std::runtime_error("metadata mismatch for " + expected.key);
    if (expected_lsn && actual.lsn != *expected_lsn) {
        throw std::runtime_error("LSN mismatch for " + expected.key + ": expected " +
                                 std::to_string(*expected_lsn) + ", got " + std::to_string(actual.lsn));
    }
}

void verifyImage(const Options& options, const Ledger& ledger,
                 const FrontierRecord& frontier, const FrontierRecord& upper) {
    const ExpectedSets sets = classify(ledger, frontier, upper);
    const FileFingerprint before = fingerprintTree(options.db);
    auto db = openReadOnly(options, ledger.config);

    const vdb::DurabilityStatus status = db->durabilityStatus();
    if (status.appended_lsn < status.visible_lsn || status.visible_lsn < status.durable_lsn) {
        throw std::runtime_error("recovered frontier ordering is invalid");
    }
    if (status.durable_lsn < frontier.durable_lsn || status.durable_lsn > upper.visible_lsn) {
        throw std::runtime_error("recovered durable LSN is outside the cut's ledger bounds");
    }
    if (status.manifest_generation < frontier.manifest_generation) {
        throw std::runtime_error("recovered manifest generation precedes the ledger frontier");
    }

    const std::vector<InspectedRecord> inspected = inspectRecords(*db);
    std::unordered_map<std::string, const InspectedRecord*> recovered;
    for (const auto& record : inspected) {
        if (!recovered.emplace(record.key, &record).second) {
            throw std::runtime_error("duplicate recovered key: " + record.key);
        }
    }

    std::unordered_set<std::string> allowed_keys;
    for (const auto* expected : sets.required) {
        allowed_keys.insert(expected->key);
        const auto actual = recovered.find(expected->key);
        if (actual == recovered.end()) throw std::runtime_error("required stable key is missing: " + expected->key);
        verifyPayload(*expected, *actual->second, sets.expected_lsn.at(expected->key));
    }

    size_t weak_present = 0;
    for (const auto* expected : sets.optional_weak) {
        allowed_keys.insert(expected->key);
        const auto actual = recovered.find(expected->key);
        if (actual == recovered.end()) continue;
        ++weak_present;
        verifyPayload(*expected, *actual->second, sets.expected_lsn.at(expected->key));
    }
    if (weak_present != 0 && weak_present != sets.optional_weak.size()) {
        throw std::runtime_error("partial weak generation recovered");
    }

    for (const auto* expected : sets.allowed_inflight) {
        allowed_keys.insert(expected->key);
        const auto actual = recovered.find(expected->key);
        if (actual != recovered.end()) verifyPayload(*expected, *actual->second, std::nullopt);
    }

    for (const auto& [key, record] : recovered) {
        (void)record;
        if (!allowed_keys.contains(key)) throw std::runtime_error("unexpected recovered key: " + key);
    }
    if (recovered.size() < sets.required.size()) throw std::runtime_error("recovered key count is below required set");

    // Stable-query replay also exercises the production deterministic merge and
    // seeded HNSW path. Latest queries are recorded for churn analysis but may
    // legitimately lose the optional weak generation.
    for (const auto& query : ledger.queries) {
        if (query.line > frontier.line || query.visibility != "stable" ||
            query.snapshot_lsn > frontier.durable_lsn) continue;
        const VectorDatabase::SearchResponse replay = db->similaritySearch(
            Vector(query.values), query.k, vdb::ReadVisibility::Stable);
        if (replay.results.size() != query.results.size()) throw std::runtime_error("stable query result count changed");
        for (size_t rank = 0; rank < query.results.size(); ++rank) {
            if (replay.results[rank].key != query.results[rank].key ||
                std::bit_cast<uint32_t>(replay.results[rank].distance) !=
                    std::bit_cast<uint32_t>(query.results[rank].distance)) {
                throw std::runtime_error("stable query result changed at rank " + std::to_string(rank));
            }
        }
    }

    db.reset();  // ReadOnlyRecovery destructor must not flush or clean up.
    const FileFingerprint after = fingerprintTree(options.db);
    if (before != after) throw std::runtime_error("read-only recovery modified the database image");
}

#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (options.capabilities) {
#if VDB_HAS_RECALL_COMMITTER_API && VDB_HAS_COMMITTER_IMAGE_VERIFY_API
            std::cout << "read-only committer verification API available\n";
            return 0;
#else
            std::cerr << "SKIP: full read-only committer image verification API is not enabled\n";
            return kSkip;
#endif
        }

        const Ledger ledger = readLedger(options.ledger);
        if (options.validate_only) {
            std::cout << "ledger valid: intents=" << ledger.intents.size()
                      << " acks=" << ledger.acks.size()
                      << " frontiers=" << ledger.frontiers.size() << '\n';
            return 0;
        }
        const FrontierRecord& frontier = findFrontier(ledger, options.frontier);
        const FrontierRecord& upper = options.allow_through
                                          ? findFrontier(ledger, *options.allow_through)
                                          : frontier;
#if !(VDB_HAS_RECALL_COMMITTER_API && VDB_HAS_COMMITTER_IMAGE_VERIFY_API)
        (void)frontier;
        (void)upper;
        std::cerr << "SKIP: side-effect-free recovery and record/LSN inspection are not available\n";
        return kSkip;
#else
        verifyImage(options, ledger, frontier, upper);
        std::cout << "OK frontier=" << frontier.name
                  << " durable_lsn=" << frontier.durable_lsn
                  << " allow_through=" << upper.name << '\n';
        return 0;
#endif
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
