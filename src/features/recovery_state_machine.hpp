#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "json.hpp" 
class RecoveryStateMachine {
public:
    // High-level phases the system can be in
    enum class State {
        UNINITIALIZED,
        ANALYZING,
        CLEAN,
        RECOVERY_NEEDED,
        RECOVERING,
        RECOVERED,
        CORRUPTED,
        FAILED,
        REPAIR,
        READY,
        ERROR
    };

    // Events that move us between states
    enum class Event {
        START_ANALYSIS,
        ANALYSIS_COMPLETE,
        RECOVERY_START,
        RECOVERY_COMPLETE,
        CORRUPTION_DETECTED,
        FAILURE_DETECTED,
        REPAIR_START,
        MANUAL_INTERVENTION
    };

    struct RecoveryInfo {
        enum class State {
            CLEAN,
            RECOVERY_NEEDED,
            CORRUPTED
        };
        State state = State::CLEAN;

        std::string last_checkpoint_file;
        uint64_t    last_checkpoint_sequence = 0;

        std::vector<std::string> log_files;

        std::string error_message;
    };

    RecoveryStateMachine();

    void processEvent(Event event);

    std::string getStateName() const;
    std::chrono::duration<double> getTimeInCurrentState() const;

    bool isAnalyzing() const { return current_state == State::ANALYZING; }
    bool isReady()     const { return current_state == State::READY; }
    bool isError()     const { return current_state == State::ERROR || current_state == State::FAILED; }
    bool needsRecovery() const { return recovery_info.state == RecoveryInfo::State::RECOVERY_NEEDED; }

    void startRecovery();
    void performAnalysis();
    void attemptRepair();
    void reset();

    void setRecoveryCallbacks(std::function<RecoveryInfo()> analysis_cb,
                              std::function<void(const RecoveryInfo&)> recovery_cb,
                              std::function<void()> repair_cb,
                              std::function<void()> validation_cb);

    void setDirectories(const std::string& data_dir, const std::string& log_dir) {
        data_dir_ = data_dir;
        log_dir_  = log_dir;
    }

    RecoveryInfo getRecoveryInfo() const { return recovery_info; }
    std::string  getErrorMessage() const { return error_message; }

private:
    void transitionTo(State new_state);
    bool canTransition(State from, State to) const;
    std::string getStateNameForState(State state) const;

    void handleUninitialized(Event event);
    void handleAnalyzing(Event event);
    void handleClean(Event event);
    void handleRecoveryNeeded(Event event);
    void handleRecovering(Event event);
    void handleRecovered(Event event);
    void handleCorrupted(Event event);
    void handleFailed(Event event);
    void handleRepair(Event event);
    void handleReady(Event event);
    void handleError(Event event);

    // Default behavior if no callbacks provided
    RecoveryInfo analyzeSystemState();
    void performRecovery(const RecoveryInfo& info);
    void attemptDataRepair();
    void validateRecoveredState();

    // File helpers
    bool validateCheckpointFile(const std::string& filename) const;
    uint64_t readCheckpointSequence(const std::string& filename) const;
    std::vector<std::string> findCommitLogFiles() const;

private:
    State current_state;
    std::chrono::steady_clock::time_point state_entry_time;

    std::map<State, std::function<void(Event)>> state_handlers;

    // External hooks (optional)
    std::function<RecoveryInfo()> analysis_callback;
    std::function<void(const RecoveryInfo&)> recovery_callback;
    std::function<void()> repair_callback;
    std::function<void()> validation_callback;

    // Results of analysis
    RecoveryInfo recovery_info;
    std::string  error_message;

    // Directories to use (configured by caller)
    std::string data_dir_ = "data";
    std::string log_dir_  = "logs";
};

// ---------------- JSON serialization helpers ----------------
inline const char* to_string(RecoveryStateMachine::RecoveryInfo::State s) {
    using S = RecoveryStateMachine::RecoveryInfo::State;
    switch (s) {
        case S::CLEAN:            return "CLEAN";
        case S::RECOVERY_NEEDED:  return "RECOVERY_NEEDED";
        case S::CORRUPTED:        return "CORRUPTED";
        default:                  return "UNKNOWN";
    }
}

inline void to_json(nlohmann::json& j, const RecoveryStateMachine::RecoveryInfo& r) {
    j = nlohmann::json{
        {"state", to_string(r.state)},
        {"last_checkpoint_file", r.last_checkpoint_file},
        {"last_checkpoint_sequence", r.last_checkpoint_sequence},
        {"log_files", r.log_files},
        {"error_message", r.error_message}
    };
}

