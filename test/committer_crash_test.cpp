// Process-crash and dm-log-writes workload controller for the recall committer.
//
// The external ledger is deliberately plain, versioned, and line framed. Strings
// are hex encoded and floats are stored as IEEE-754 bits, so the checker never
// has to infer payloads from the recovered database:
//
//   VDB_COMMITTER_LEDGER 1
//   CONFIG <dims> <k> <seed> <policy> <epsilon-bits> <delta-bits> <db-hex>
//   INTENT <ordinal> <requested> <key-hex> <metadata-hex> <float-bits-csv>
//   ACK <ordinal> <applied> <lsn> <requested> <actual> <visible> <durable>
//       <durable-count> <weak-count> <policy-cap> <risk-bits> <provisional>
//   QUERY <id> <visibility> <snapshot> <durable> <manifest> <k> <float-bits-csv>
//   RESULT <query-id> <rank> <key-hex> <distance-bits>
//   FRONTIER <name> <appended> <visible> <durable> <durable-count> <weak-count>
//       <manifest-generation>
//   MARK <name>
//   END <child-status>
//
// INTENT is emitted before entering the database. ACK is emitted only after the
// call returns. The parent alone writes and fsyncs the ledger. At every FRONTIER
// the child waits until the parent has persisted the line (and, in physical mode,
// inserted the matching dm-log-writes mark) before it continues. The child owns
// the live VectorDatabase on the heap and terminates with _exit: no destructor or
// shutdown path runs.
//
// Milestone-3 integration contract
// --------------------------------
// Define VDB_HAS_RECALL_COMMITTER_API=1 in vector_database.hpp when these design
// APIs exist in namespace vdb: AckMode, AckLevel, RecallPolicy, ReadVisibility, OpenMode,
// RecallCommitConfig, WriteReceipt, DurabilityStatus, SearchResponse;
// VectorDatabase::configureRecallCommit, insertWithAck, durabilityFence,
// durabilityStatus, and similaritySearch(query, k, visibility). The constructor
// must accept OpenMode as its final compatibility-preserving argument. The two
// small adapters below are the only places that assume result/status field names.

#include <algorithm>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../src/core/vector_database.hpp"
#include "../src/utils/atomic_write.hpp"

#ifndef VDB_HAS_RECALL_COMMITTER_API
#define VDB_HAS_RECALL_COMMITTER_API 0
#endif

namespace {

[[maybe_unused]] constexpr int kSkip = 77;
constexpr size_t kPipeFrameLimit = 64 * 1024;

struct Options {
    std::filesystem::path db;
    std::filesystem::path ledger;
    std::string frontier{"wcap"};
    std::string dm_name;
    std::filesystem::path verifier{"build/verify_committer_image"};
    size_t dimensions{8};
    size_t k{10};
    size_t stable_records{16};
    uint32_t seed{100};
    double epsilon{0.2};
    bool physical{false};
    std::optional<int> expected_child_status;
};

[[noreturn]] void usage(const char* argv0, int status) {
    std::cerr
        << "usage: " << argv0 << " --db DIR --ledger FILE [options]\n"
        << "  --frontier w0|w1|wcap|cap-plus-one|after-fence|FAILPOINT\n"
        << "  --physical --dm-name NAME    add a log-writes mark per frontier\n"
        << "  --verifier PATH              default: build/verify_committer_image\n"
        << "  --expected-child-status N    default: 86 for named failpoints\n"
        << "  --dimensions N --k N --stable-records N --seed N --epsilon X\n";
    std::exit(status);
}

uint64_t parseUnsigned(std::string_view text, const char* option) {
    if (text.empty()) throw std::runtime_error(std::string(option) + " needs a value");
    size_t consumed = 0;
    const auto value = std::stoull(std::string(text), &consumed, 10);
    if (consumed != text.size()) throw std::runtime_error(std::string("invalid ") + option);
    return value;
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
        else if (arg == "--dm-name") options.dm_name = value();
        else if (arg == "--verifier") options.verifier = value();
        else if (arg == "--dimensions") options.dimensions = parseUnsigned(value(), "--dimensions");
        else if (arg == "--k") options.k = parseUnsigned(value(), "--k");
        else if (arg == "--stable-records") options.stable_records = parseUnsigned(value(), "--stable-records");
        else if (arg == "--seed") options.seed = static_cast<uint32_t>(parseUnsigned(value(), "--seed"));
        else if (arg == "--epsilon") options.epsilon = std::stod(std::string(value()));
        else if (arg == "--expected-child-status") {
            const uint64_t status_value = parseUnsigned(value(), "--expected-child-status");
            if (status_value > 255) throw std::runtime_error("child status must be in [0,255]");
            options.expected_child_status = static_cast<int>(status_value);
        }
        else if (arg == "--physical") options.physical = true;
        else if (arg == "--help" || arg == "-h") usage(argv[0], 0);
        else throw std::runtime_error("unknown option: " + arg);
    }
    if (options.db.empty() || options.ledger.empty()) usage(argv[0], 2);
    if (options.dimensions == 0 || options.k == 0) throw std::runtime_error("dimensions and k must be nonzero");
    if (!(options.epsilon >= 0.0 && options.epsilon < 1.0)) throw std::runtime_error("epsilon must be in [0,1)");
    if (options.physical && options.dm_name.empty()) throw std::runtime_error("--physical requires --dm-name");
    return options;
}

bool isNamedFailpoint(const Options& options) {
    return options.frontier != "w0" && options.frontier != "w1" &&
           options.frontier != "wcap" && options.frontier != "cap-plus-one" &&
           options.frontier != "after-fence";
}

void writeAll(int fd, std::string_view bytes) {
    while (!bytes.empty()) {
        const ssize_t written = ::write(fd, bytes.data(), bytes.size());
        if (written < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "write");
        }
        bytes.remove_prefix(static_cast<size_t>(written));
    }
}

void sendLine(int fd, std::string line) {
    if (line.find('\n') != std::string::npos || line.size() + 1 > kPipeFrameLimit) {
        throw std::runtime_error("invalid or oversized ledger frame");
    }
    line.push_back('\n');
    writeAll(fd, line);
}

std::string hexEncode(std::string_view value) {
    static constexpr char digits[] = "0123456789abcdef";
    if (value.empty()) return "-";
    std::string encoded;
    encoded.reserve(value.size() * 2);
    for (unsigned char byte : value) {
        encoded.push_back(digits[byte >> 4]);
        encoded.push_back(digits[byte & 0x0f]);
    }
    return encoded;
}

template <typename UInt>
std::string fixedHex(UInt value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(sizeof(UInt) * 2) << value;
    return out.str();
}

std::string encodeDouble(double value) {
    return fixedHex(std::bit_cast<uint64_t>(value));
}

std::string encodeFloat(float value) {
    return fixedHex(std::bit_cast<uint32_t>(value));
}

std::string encodeVector(const std::vector<float>& values) {
    if (values.empty()) return "-";
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ',';
        out << encodeFloat(values[i]);
    }
    return out.str();
}

bool validName(std::string_view name) {
    return !name.empty() && std::all_of(name.begin(), name.end(), [](unsigned char ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
               (ch >= '0' && ch <= '9') || ch == '_' || ch == '.' || ch == '-';
    });
}

std::vector<float> makeVector(size_t ordinal, size_t dimensions) {
    std::vector<float> values(dimensions);
    // Integer-derived values are exactly representable and independently
    // reproducible by the ledger checker.
    for (size_t d = 0; d < dimensions; ++d) {
        values[d] = static_cast<float>((ordinal + 1) * 257 + d * 17) / 4096.0f;
    }
    return values;
}

class LedgerFile {
public:
    explicit LedgerFile(const std::filesystem::path& path) {
        const auto parent = path.parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "open ledger");
    }

    ~LedgerFile() {
        if (fd_ >= 0) ::close(fd_);
    }

    void append(std::string_view line) {
        writeAll(fd_, line);
        writeAll(fd_, "\n");
        if (::fsync(fd_) != 0) throw std::system_error(errno, std::generic_category(), "fsync ledger");
    }

private:
    int fd_{-1};
};

int runCommand(const std::vector<std::string>& args) {
    if (args.empty()) return 2;
    const pid_t pid = ::fork();
    if (pid < 0) throw std::system_error(errno, std::generic_category(), "fork command");
    if (pid == 0) {
        std::vector<char*> argv;
        argv.reserve(args.size() + 1);
        for (const auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
        argv.push_back(nullptr);
        ::execvp(argv[0], argv.data());
        _exit(127);
    }
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) throw std::system_error(errno, std::generic_category(), "waitpid command");
    }
    if (!WIFEXITED(status)) return 128;
    return WEXITSTATUS(status);
}

void addDeviceMapperMark(const std::string& dm_name, const std::string& mark) {
    if (!validName(mark)) throw std::runtime_error("unsafe dm-log-writes mark: " + mark);
    const int status = runCommand({"dmsetup", "message", dm_name, "0", "mark", mark});
    if (status != 0) throw std::runtime_error("dmsetup mark failed with status " + std::to_string(status));
}

struct ParentResult {
    int child_status{1};
    std::vector<std::string> frontiers;
};

ParentResult collectChild(pid_t child, int event_fd, int control_fd,
                          LedgerFile& ledger, const Options& options) {
    ParentResult result;
    std::string pending;
    char buffer[8192];
    for (;;) {
        const ssize_t count = ::read(event_fd, buffer, sizeof(buffer));
        if (count < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "read child events");
        }
        if (count == 0) break;
        pending.append(buffer, static_cast<size_t>(count));
        for (;;) {
            const size_t newline = pending.find('\n');
            if (newline == std::string::npos) break;
            std::string line = pending.substr(0, newline);
            pending.erase(0, newline + 1);
            ledger.append(line);

            std::istringstream tokens(line);
            std::string kind;
            tokens >> kind;
            if (kind == "FRONTIER") {
                std::string name;
                tokens >> name;
                if (!validName(name)) throw std::runtime_error("malformed FRONTIER line");
                if (options.physical) {
                    addDeviceMapperMark(options.dm_name, name);
                    ledger.append("MARK " + name);
                }
                result.frontiers.push_back(name);
                writeAll(control_fd, "C");
            }
        }
    }
    if (!pending.empty()) throw std::runtime_error("child left a partial ledger frame");
    ::close(event_fd);
    ::close(control_fd);

    int status = 0;
    while (::waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) throw std::system_error(errno, std::generic_category(), "waitpid child");
    }
    if (WIFEXITED(status)) result.child_status = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) result.child_status = 128 + WTERMSIG(status);
    ledger.append("END " + std::to_string(result.child_status));
    return result;
}

void waitForParent(int fd) {
    char command = 0;
    for (;;) {
        const ssize_t count = ::read(fd, &command, 1);
        if (count < 0 && errno == EINTR) continue;
        if (count != 1 || command != 'C') throw std::runtime_error("controller handshake failed");
        return;
    }
}

#if VDB_HAS_RECALL_COMMITTER_API

std::unique_ptr<VectorDatabase> openDatabase(const Options& options, vdb::OpenMode mode) {
    auto db = std::make_unique<VectorDatabase>(
        options.dimensions, VectorDatabase::SearchMode::HNSW, false, false,
        PersistenceConfig{}, false, 0, options.db.string(),
        VectorDatabase::StorageEngine::Segmented, mode);
    db->configureHNSW(16, 200, 64, options.seed);
    db->configureSegmentedStorage(1'000'000, 16, 0.25);
    return db;
}

vdb::RecallCommitConfig recallConfig(const Options& options) {
    vdb::RecallCommitConfig config;
    config.enabled = true;
    config.policy = vdb::RecallPolicy::Strict;
    config.epsilon = options.epsilon;
    config.delta = 0.0;
    config.k_min = options.k;
    config.max_tail_records = std::numeric_limits<size_t>::max();
    config.max_tail_bytes = std::numeric_limits<size_t>::max();
    config.max_tail_age = std::chrono::seconds(60);
    config.group_delay = std::chrono::microseconds(0);
    config.hnsw_seed = options.seed;
    return config;
}

std::string ackName(vdb::AckMode mode) {
    return mode == vdb::AckMode::Stable ? "stable" : "weak";
}

std::string ackName(vdb::AckLevel level) {
    if (level == vdb::AckLevel::Stable) return "stable";
    if (level == vdb::AckLevel::Weak) return "weak";
    return "none";
}

void emitFrontier(int event_fd, int control_fd, std::string name,
                  const vdb::DurabilityStatus& status) {
    if (!validName(name)) throw std::runtime_error("invalid frontier name");
    std::ostringstream line;
    line << "FRONTIER " << name << ' ' << status.appended_lsn << ' '
         << status.visible_lsn << ' ' << status.durable_lsn << ' '
         << status.durable_records << ' ' << status.weak_records << ' '
         << status.manifest_generation;
    sendLine(event_fd, line.str());
    waitForParent(control_fd);
}

vdb::WriteReceipt insertAndRecord(VectorDatabase& db, int event_fd, size_t ordinal,
                                  vdb::AckMode requested, const std::vector<float>& values) {
    const std::string key = "committer-key-" + std::to_string(ordinal);
    const std::string metadata = "payload-" + std::to_string(ordinal);
    sendLine(event_fd, "INTENT " + std::to_string(ordinal) + " " + ackName(requested) +
                           " " + hexEncode(key) + " " + hexEncode(metadata) + " " +
                           encodeVector(values));

    const vdb::WriteReceipt receipt = db.insertWithAck(Vector(values), key, metadata, requested);
    std::ostringstream line;
    line << "ACK " << ordinal << ' ' << (receipt.applied ? 1 : 0) << ' '
         << receipt.lsn << ' ' << ackName(receipt.requested_ack) << ' '
         << ackName(receipt.actual_ack) << ' ' << receipt.visible_lsn << ' '
         << receipt.durable_lsn << ' ' << receipt.durable_count << ' '
         << receipt.weak_count << ' ' << receipt.policy_cap << ' '
         << encodeDouble(receipt.risk_estimate) << ' ' << (receipt.provisional ? 1 : 0);
    sendLine(event_fd, line.str());
    return receipt;
}

void recordQuery(VectorDatabase& db, int event_fd, size_t id,
                 vdb::ReadVisibility visibility, const std::vector<float>& values,
                 size_t k) {
    const VectorDatabase::SearchResponse response = db.similaritySearch(Vector(values), k, visibility);
    const char* visibility_name = visibility == vdb::ReadVisibility::Stable ? "stable" : "latest";
    std::ostringstream query;
    query << "QUERY " << id << ' ' << visibility_name << ' ' << response.snapshot_lsn
          << ' ' << response.durable_lsn << ' ' << response.manifest_generation << ' '
          << k << ' ' << encodeVector(values);
    sendLine(event_fd, query.str());
    for (size_t rank = 0; rank < response.results.size(); ++rank) {
        const auto& result = response.results[rank];
        sendLine(event_fd, "RESULT " + std::to_string(id) + " " +
                               std::to_string(rank) + " " + hexEncode(result.key) +
                               " " + encodeFloat(result.distance));
    }
}

[[noreturn]] void runChild(const Options& options, int event_fd, int control_fd) {
    try {
        sendLine(event_fd, "VDB_COMMITTER_LEDGER 1");
        std::ostringstream config_line;
        config_line << "CONFIG " << options.dimensions << ' ' << options.k << ' '
                    << options.seed << " strict " << encodeDouble(options.epsilon)
                    << ' ' << encodeDouble(0.0) << ' '
                    << hexEncode(options.db.lexically_normal().string());
        sendLine(event_fd, config_line.str());

        // Intentionally outlives this function's normal C++ cleanup path.
        auto* db = openDatabase(options, vdb::OpenMode::ReadWrite).release();
        db->configureRecallCommit(recallConfig(options));
        db->initialize();
        emitFrontier(event_fd, control_fd, "DB_READY", db->durabilityStatus());

        size_t ordinal = 0;
        for (; ordinal < options.stable_records; ++ordinal) {
            insertAndRecord(*db, event_fd, ordinal, vdb::AckMode::Stable,
                            makeVector(ordinal, options.dimensions));
            emitFrontier(event_fd, control_fd, "stable-" + std::to_string(ordinal),
                         db->durabilityStatus());
        }

        const size_t strict_cap = static_cast<size_t>(options.epsilon * options.k);
        size_t weak_target = strict_cap;
        if (options.frontier == "w0") weak_target = 0;
        else if (options.frontier == "w1") weak_target = 1;

        for (size_t weak = 0; weak < weak_target; ++weak, ++ordinal) {
            insertAndRecord(*db, event_fd, ordinal, vdb::AckMode::WeakAllowed,
                            makeVector(ordinal, options.dimensions));
            emitFrontier(event_fd, control_fd, "weak-" + std::to_string(weak + 1),
                         db->durabilityStatus());
        }

        if (options.frontier == "cap-plus-one") {
            insertAndRecord(*db, event_fd, ordinal, vdb::AckMode::WeakAllowed,
                            makeVector(ordinal, options.dimensions));
            emitFrontier(event_fd, control_fd, "cap-plus-one", db->durabilityStatus());
            ++ordinal;
        } else if (options.frontier == "after-fence") {
            const uint64_t fenced_lsn = db->durabilityFence();
            (void)fenced_lsn;
            emitFrontier(event_fd, control_fd, "after-fence", db->durabilityStatus());
        } else if (options.frontier != "w0" && options.frontier != "w1" &&
                   options.frontier != "wcap") {
            // Production failpoints use this environment variable and must _exit
            // at the named point. Returning from durabilityFence is a harness
            // failure, not a successful approximation of the requested cut.
            ::setenv("VDB_COMMITTER_FAILPOINT", options.frontier.c_str(), 1);
            if (options.frontier.starts_with("seal-")) db->sealMutableSegment();
            else (void)db->durabilityFence();
            sendLine(event_fd, "ERROR failpoint-returned " + options.frontier);
            _exit(3);
        }

        const auto query = makeVector(ordinal / 2, options.dimensions);
        recordQuery(*db, event_fd, 0, vdb::ReadVisibility::Stable, query, options.k);
        recordQuery(*db, event_fd, 1, vdb::ReadVisibility::Latest, query, options.k);
        emitFrontier(event_fd, control_fd, "crash", db->durabilityStatus());

        // This is the property under test. db is still live and its committer
        // thread has not been shut down or joined.
        _exit(0);
    } catch (const std::exception& error) {
        try { sendLine(event_fd, "ERROR " + hexEncode(error.what())); } catch (...) {}
        _exit(2);
    } catch (...) {
        _exit(2);
    }
}

int writableRecoveryTwice(const Options& options) {
    for (int attempt = 0; attempt < 2; ++attempt) {
        auto db = openDatabase(options, vdb::OpenMode::ReadWrite);
        db->configureRecallCommit(recallConfig(options));
        db->initialize();
        db->shutdown();
    }
    return 0;
}

#else

[[noreturn]] void runChild(const Options&, int, int) {
    _exit(kSkip);
}

int writableRecoveryTwice(const Options&) {
    return kSkip;
}

#endif

int runVerifier(const Options& options, const std::string& frontier) {
    return runCommand({options.verifier.string(), "--db", options.db.string(),
                       "--ledger", options.ledger.string(), "--frontier", frontier});
}

}  // namespace

int main(int argc, char** argv) {
    try {
        vdb::io::set_full_fsync(true);
        const Options options = parseOptions(argc, argv);
#if !VDB_HAS_RECALL_COMMITTER_API
        std::cerr << "SKIP: recall committer API is not available; define "
                     "VDB_HAS_RECALL_COMMITTER_API when milestone 3 lands\n";
        return kSkip;
#else
        if (std::filesystem::exists(options.ledger)) {
            throw std::runtime_error("refusing to overwrite external ledger: " +
                                     options.ledger.string());
        }
        if (!options.physical) std::filesystem::remove_all(options.db);

        int events[2];
        int controls[2];
        if (::pipe(events) != 0 || ::pipe(controls) != 0) {
            throw std::system_error(errno, std::generic_category(), "pipe");
        }
        const pid_t child = ::fork();
        if (child < 0) throw std::system_error(errno, std::generic_category(), "fork child");
        if (child == 0) {
            ::close(events[0]);
            ::close(controls[1]);
            runChild(options, events[1], controls[0]);
        }

        ::close(events[1]);
        ::close(controls[0]);
        LedgerFile ledger(options.ledger);
        const ParentResult result = collectChild(child, events[0], controls[1], ledger, options);
        const int expected_status = options.expected_child_status.value_or(
            isNamedFailpoint(options) ? 86 : 0);
        if (result.child_status != expected_status) {
            std::cerr << "FAIL: crash child exited with status " << result.child_status
                      << "; expected " << expected_status << "\n";
            return 1;
        }
        if (result.frontiers.empty()) throw std::runtime_error("child recorded no recovery frontier");
        if (options.physical) {
            std::cout << "physical workload complete; ledger=" << options.ledger << "\n";
            return 0;
        }

        int verifier_status = runVerifier(options, result.frontiers.back());
        if (verifier_status != 0) {
            std::cerr << "FAIL: initial read-only ledger verifier returned "
                      << verifier_status << "\n";
            return 1;
        }
        if (writableRecoveryTwice(options) != 0) {
            std::cerr << "FAIL: writable recovery/idempotence check failed\n";
            return 1;
        }
        verifier_status = runVerifier(options, result.frontiers.back());
        if (verifier_status != 0) {
            std::cerr << "FAIL: read-only ledger verifier returned " << verifier_status << "\n";
            return 1;
        }
        std::cout << "PASS: process crash frontier=" << options.frontier
                  << " ledger=" << options.ledger << "\n";
        return 0;
#endif
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
