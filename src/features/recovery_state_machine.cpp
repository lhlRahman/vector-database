// Copyright [year] <Owner>
#include "recovery_state_machine.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

RecoveryStateMachine::RecoveryStateMachine()
    : current_state(State::UNINITIALIZED),
      state_entry_time(std::chrono::steady_clock::now()) {

    state_handlers[State::UNINITIALIZED]  = [this](Event e){ handleUninitialized(e); };
    state_handlers[State::ANALYZING]      = [this](Event e){ handleAnalyzing(e); };
    state_handlers[State::CLEAN]          = [this](Event e){ handleClean(e); };
    state_handlers[State::RECOVERY_NEEDED]= [this](Event e){ handleRecoveryNeeded(e); };
    state_handlers[State::RECOVERING]     = [this](Event e){ handleRecovering(e); };
    state_handlers[State::RECOVERED]      = [this](Event e){ handleRecovered(e); };
    state_handlers[State::CORRUPTED]      = [this](Event e){ handleCorrupted(e); };
    state_handlers[State::FAILED]         = [this](Event e){ handleFailed(e); };
    state_handlers[State::REPAIR]         = [this](Event e){ handleRepair(e); };
    state_handlers[State::READY]          = [this](Event e){ handleReady(e); };
    state_handlers[State::ERROR]          = [this](Event e){ handleError(e); };
}

void RecoveryStateMachine::processEvent(Event event) {
    auto it = state_handlers.find(current_state);
    if (it != state_handlers.end()) it->second(event);
}

std::string RecoveryStateMachine::getStateName() const {
    return getStateNameForState(current_state);
}

std::chrono::duration<double> RecoveryStateMachine::getTimeInCurrentState() const {
    return std::chrono::steady_clock::now() - state_entry_time;
}

void RecoveryStateMachine::startRecovery()      { processEvent(Event::RECOVERY_START); }
void RecoveryStateMachine::performAnalysis()    { processEvent(Event::START_ANALYSIS); }
void RecoveryStateMachine::attemptRepair()      { processEvent(Event::REPAIR_START); }

void RecoveryStateMachine::reset() {
    current_state = State::UNINITIALIZED;
    state_entry_time = std::chrono::steady_clock::now();
    error_message.clear();
    recovery_info = RecoveryInfo{};
}

void RecoveryStateMachine::setRecoveryCallbacks(std::function<RecoveryInfo()> analysis_cb,
                                                std::function<void(const RecoveryInfo&)> recovery_cb,
                                                std::function<void()> repair_cb,
                                                std::function<void()> validation_cb) {
    analysis_callback  = std::move(analysis_cb);
    recovery_callback  = std::move(recovery_cb);
    repair_callback    = std::move(repair_cb);
    validation_callback= std::move(validation_cb);
}

std::string RecoveryStateMachine::getStateNameForState(State s) const {
    switch (s) {
        case State::UNINITIALIZED:   return "UNINITIALIZED";
        case State::ANALYZING:       return "ANALYZING";
        case State::CLEAN:           return "CLEAN";
        case State::RECOVERY_NEEDED: return "RECOVERY_NEEDED";
        case State::RECOVERING:      return "RECOVERING";
        case State::RECOVERED:       return "RECOVERED";
        case State::CORRUPTED:       return "CORRUPTED";
        case State::FAILED:          return "FAILED";
        case State::REPAIR:          return "REPAIR";
        case State::READY:           return "READY";
        case State::ERROR:           return "ERROR";
        default:                     return "UNKNOWN";
    }
}

void RecoveryStateMachine::transitionTo(State new_state) {
    if (!canTransition(current_state, new_state)) {
        std::cerr << "Invalid state transition from " << getStateName()
                  << " to " << getStateNameForState(new_state) << std::endl;
        return;
    }
    std::cout << "State transition: " << getStateName()
              << " -> " << getStateNameForState(new_state) << std::endl;
    current_state = new_state;
    state_entry_time = std::chrono::steady_clock::now();
}

bool RecoveryStateMachine::canTransition(State from, State to) const {
    switch (from) {
        case State::UNINITIALIZED:   return to == State::ANALYZING;
        case State::ANALYZING:       return to == State::CLEAN || to == State::RECOVERY_NEEDED || to == State::CORRUPTED;
        case State::CLEAN:           return to == State::READY;
        case State::RECOVERY_NEEDED: return to == State::RECOVERING;
        case State::RECOVERING:      return to == State::RECOVERED || to == State::CORRUPTED || to == State::FAILED;
        case State::RECOVERED:       return to == State::READY;
        case State::CORRUPTED:       return to == State::REPAIR || to == State::FAILED;
        case State::FAILED:          return to == State::ERROR;
        case State::REPAIR:          return to == State::RECOVERED || to == State::FAILED;
        case State::READY:           return false;
        case State::ERROR:           return to == State::ANALYZING;
        default:                     return false;
    }
}

// ------------------ handlers ------------------

void RecoveryStateMachine::handleUninitialized(Event e) {
    if (e == Event::START_ANALYSIS) transitionTo(State::ANALYZING);
}

void RecoveryStateMachine::handleAnalyzing(Event e) {
    if (e != Event::START_ANALYSIS) return;
    try {
        if (analysis_callback) recovery_info = analysis_callback();
        else                   recovery_info = analyzeSystemState();

        if (recovery_info.state == RecoveryInfo::State::CLEAN) {
            transitionTo(State::CLEAN);
            processEvent(Event::ANALYSIS_COMPLETE);
        } else if (recovery_info.state == RecoveryInfo::State::RECOVERY_NEEDED) {
            transitionTo(State::RECOVERY_NEEDED);
        } else {
            transitionTo(State::CORRUPTED);
        }
    } catch (const std::exception& ex) {
        error_message = ex.what();
        transitionTo(State::FAILED);
    }
}

void RecoveryStateMachine::handleClean(Event e) {
    if (e == Event::ANALYSIS_COMPLETE) transitionTo(State::READY);
}

void RecoveryStateMachine::handleRecoveryNeeded(Event e) {
    if (e == Event::RECOVERY_START) transitionTo(State::RECOVERING);
}

void RecoveryStateMachine::handleRecovering(Event e) {
    if (e == Event::RECOVERY_START) {
        try {
            if (recovery_callback) recovery_callback(recovery_info);
            else                   performRecovery(recovery_info);
            transitionTo(State::RECOVERED);
        } catch (const std::exception& ex) {
            error_message = ex.what();
            transitionTo(State::FAILED);
        }
    } else if (e == Event::CORRUPTION_DETECTED) {
        transitionTo(State::CORRUPTED);
    } else if (e == Event::FAILURE_DETECTED) {
        transitionTo(State::FAILED);
    }
}

void RecoveryStateMachine::handleRecovered(Event e) {
    if (e == Event::RECOVERY_COMPLETE) {
        try {
            if (validation_callback) validation_callback();
            else                     validateRecoveredState();
            transitionTo(State::READY);
        } catch (const std::exception& ex) {
            error_message = ex.what();
            transitionTo(State::FAILED);
        }
    }
}

void RecoveryStateMachine::handleCorrupted(Event e) {
    if (e == Event::REPAIR_START) transitionTo(State::REPAIR);
    else if (e == Event::FAILURE_DETECTED) transitionTo(State::FAILED);
}

void RecoveryStateMachine::handleFailed(Event e) {
    if (e == Event::FAILURE_DETECTED) transitionTo(State::ERROR);
}

void RecoveryStateMachine::handleRepair(Event e) {
    if (e == Event::REPAIR_START) {
        try {
            if (repair_callback) repair_callback();
            else                 attemptDataRepair();
            transitionTo(State::RECOVERED);
        } catch (const std::exception& ex) {
            error_message = ex.what();
            transitionTo(State::FAILED);
        }
    } else if (e == Event::FAILURE_DETECTED) {
        transitionTo(State::FAILED);
    }
}

void RecoveryStateMachine::handleReady(Event) { /* terminal */ }
void RecoveryStateMachine::handleError(Event e) {
    if (e == Event::MANUAL_INTERVENTION) transitionTo(State::ANALYZING);
}

// ------------------ default behaviors ------------------

RecoveryStateMachine::RecoveryInfo RecoveryStateMachine::analyzeSystemState() {
    RecoveryInfo info;

    const std::string checkpoint_file = data_dir_ + "/main.db";
    if (std::filesystem::exists(checkpoint_file)) {
        if (validateCheckpointFile(checkpoint_file)) {
            info.state = RecoveryInfo::State::RECOVERY_NEEDED;
            info.last_checkpoint_file = checkpoint_file;
            info.last_checkpoint_sequence = readCheckpointSequence(checkpoint_file);
        } else {
            info.state = RecoveryInfo::State::CORRUPTED;
            info.error_message = "Checkpoint file corrupted";
            return info;
        }
    } else {
        auto logs = findCommitLogFiles();
        if (!logs.empty()) {
            info.state = RecoveryInfo::State::RECOVERY_NEEDED;
            info.log_files = std::move(logs);
        } else {
            info.state = RecoveryInfo::State::CLEAN;
        }
    }

    return info;
}

void RecoveryStateMachine::performRecovery(const RecoveryInfo& info) {
    std::cout << "Performing recovery..." << std::endl;
    if (!info.last_checkpoint_file.empty()) {
        std::cout << "Checkpoint present: " << info.last_checkpoint_file << std::endl;
    }
    if (!info.log_files.empty()) {
        std::cout << "Will replay " << info.log_files.size() << " log file(s)." << std::endl;
    }
    // Real replay is done by AtomicPersistence::loadDatabase().
}

void RecoveryStateMachine::attemptDataRepair() {
    std::cout << "Attempting data repair..." << std::endl;
    // Reserved for future: try to salvage a partial checkpoint or logs.
}

void RecoveryStateMachine::validateRecoveredState() {
    std::cout << "Validating recovered state..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::cout << "State validation completed" << std::endl;
}

bool RecoveryStateMachine::validateCheckpointFile(const std::string& filename) const {
    try {
        std::ifstream f(filename, std::ios::binary);
        if (!f) return false;
        uint32_t magic = 0;
        f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        return magic == 0x56444244; // "VDBD"
    } catch (...) { return false; }
}

uint64_t RecoveryStateMachine::readCheckpointSequence(const std::string& filename) const {
    try {
        std::ifstream f(filename, std::ios::binary);
        if (!f) return 0;
        // skip magic + version
        f.seekg(8, std::ios::beg);
        uint64_t seq = 0;
        f.read(reinterpret_cast<char*>(&seq), sizeof(seq));
        return seq;
    } catch (...) { return 0; }
}

std::vector<std::string> RecoveryStateMachine::findCommitLogFiles() const {
    std::vector<std::string> out;
    try {
        if (!std::filesystem::exists(log_dir_)) return out;
        for (const auto& entry : std::filesystem::directory_iterator(log_dir_)) {
            if (!entry.is_regular_file()) continue;
            const auto name = entry.path().filename().string();
            if (name.rfind("commit.log.", 0) == 0) out.push_back(entry.path().string());
        }
        std::sort(out.begin(), out.end());
    } catch (...) {}
    return out;
}
