#ifndef FASTLLM_APISERVER_HOST_OFFLOAD_CONTROL_H
#define FASTLLM_APISERVER_HOST_OFFLOAD_CONTROL_H

#include <algorithm>
#include <cstddef>
#include <string>

namespace fastllm {
namespace apiserver {

    enum class BackendTierState {
        VRAM_READY,
        SUSPENDING,
        RAM_SUSPENDED,
        DISK_SUSPENDED,
        RESUMING,
        DISK_FALLBACK_LOADING,
        ERROR,
    };

    enum class TierControlAction {
        SUSPEND_MEMORY,
        SUSPEND_DISK,
        RESUME,
    };

    enum class TierControlDecision {
        ALLOW,
        METHOD_NOT_ALLOWED,
        FORBIDDEN,
        BUSY,
        ALREADY_SUSPENDED,
        NOT_SUSPENDED,
        TRANSITION_IN_PROGRESS,
        BACKEND_RELOADING,
        BACKEND_SUSPENDED,
        BACKEND_ERROR,
    };

    inline bool ConstantTimeTierTokenEqual(
            const std::string &left,
            const std::string &right) {
        std::size_t difference = left.size() ^ right.size();
        const std::size_t count = std::max(left.size(), right.size());
        for (std::size_t i = 0; i < count; ++i) {
            const unsigned char leftByte = i < left.size() ?
                static_cast<unsigned char>(left[i]) : 0;
            const unsigned char rightByte = i < right.size() ?
                static_cast<unsigned char>(right[i]) : 0;
            difference |= static_cast<std::size_t>(leftByte ^ rightByte);
        }
        return difference == 0;
    }

    inline TierControlDecision EvaluateTierControl(
            TierControlAction action,
            BackendTierState state,
            const std::string &method,
            const std::string &authorization,
            const std::string &expectedToken,
            int activeGenerationRequests,
            int queuedGenerationRequests,
            bool checkpointInProgress) {
        if (method != "POST") {
            return TierControlDecision::METHOD_NOT_ALLOWED;
        }
        static const std::string bearerPrefix = "Bearer ";
        const bool hasBearer =
            authorization.compare(0, bearerPrefix.size(), bearerPrefix) == 0;
        const std::string suppliedToken = hasBearer ?
            authorization.substr(bearerPrefix.size()) : std::string();
        if (expectedToken.empty() || !hasBearer ||
            !ConstantTimeTierTokenEqual(suppliedToken, expectedToken)) {
            return TierControlDecision::FORBIDDEN;
        }
        if (state == BackendTierState::SUSPENDING ||
            state == BackendTierState::RESUMING ||
            state == BackendTierState::DISK_FALLBACK_LOADING) {
            return TierControlDecision::TRANSITION_IN_PROGRESS;
        }
        if (state == BackendTierState::ERROR) {
            return TierControlDecision::BACKEND_ERROR;
        }

        if (action == TierControlAction::RESUME) {
            if (state == BackendTierState::VRAM_READY) {
                return TierControlDecision::NOT_SUSPENDED;
            }
        } else if (state == BackendTierState::RAM_SUSPENDED ||
                   state == BackendTierState::DISK_SUSPENDED) {
            return TierControlDecision::ALREADY_SUSPENDED;
        }

        if (checkpointInProgress || activeGenerationRequests != 0 ||
            queuedGenerationRequests != 0) {
            return TierControlDecision::BUSY;
        }
        return TierControlDecision::ALLOW;
    }

    inline TierControlDecision EvaluateRequestAdmission(
            BackendTierState state) {
        switch (state) {
            case BackendTierState::VRAM_READY:
                return TierControlDecision::ALLOW;
            case BackendTierState::SUSPENDING:
            case BackendTierState::RESUMING:
                return TierControlDecision::BACKEND_RELOADING;
            case BackendTierState::DISK_FALLBACK_LOADING:
                return TierControlDecision::BACKEND_RELOADING;
            case BackendTierState::RAM_SUSPENDED:
            case BackendTierState::DISK_SUSPENDED:
                return TierControlDecision::BACKEND_SUSPENDED;
            case BackendTierState::ERROR:
                return TierControlDecision::BACKEND_ERROR;
        }
        return TierControlDecision::BACKEND_ERROR;
    }

    inline BackendTierState CompleteSuspendTransition(
            bool memoryTierRequested,
            bool ramImageReady) {
        return memoryTierRequested && ramImageReady ?
            BackendTierState::RAM_SUSPENDED :
            BackendTierState::DISK_SUSPENDED;
    }

    inline BackendTierState CompleteResumeTransition(bool success) {
        return success ? BackendTierState::VRAM_READY : BackendTierState::ERROR;
    }

    inline const char *BackendTierStateName(BackendTierState state) {
        switch (state) {
            case BackendTierState::VRAM_READY: return "ready";
            case BackendTierState::SUSPENDING: return "suspending";
            case BackendTierState::RAM_SUSPENDED: return "suspended_host";
            case BackendTierState::DISK_SUSPENDED: return "suspended_disk";
            case BackendTierState::RESUMING: return "resuming";
            case BackendTierState::DISK_FALLBACK_LOADING:
                return "disk_fallback_loading";
            case BackendTierState::ERROR: return "error";
        }
        return "error";
    }

    inline const char *TierControlDecisionMessage(
            TierControlDecision decision) {
        switch (decision) {
            case TierControlDecision::ALLOW:
                return "tier transition allowed";
            case TierControlDecision::METHOD_NOT_ALLOWED:
                return "tier control endpoint requires POST";
            case TierControlDecision::FORBIDDEN:
                return "invalid tier control bearer token";
            case TierControlDecision::BUSY:
                return "tier transition requires an idle server";
            case TierControlDecision::ALREADY_SUSPENDED:
                return "backend is already suspended";
            case TierControlDecision::NOT_SUSPENDED:
                return "backend is not suspended";
            case TierControlDecision::TRANSITION_IN_PROGRESS:
                return "backend tier transition is already in progress";
            case TierControlDecision::BACKEND_RELOADING:
                return "backend is reloading";
            case TierControlDecision::BACKEND_SUSPENDED:
                return "backend is suspended";
            case TierControlDecision::BACKEND_ERROR:
                return "backend tier state is in error";
        }
        return "invalid tier control decision";
    }

} // namespace apiserver
} // namespace fastllm

#endif
