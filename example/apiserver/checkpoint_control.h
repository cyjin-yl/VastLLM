#ifndef FASTLLM_APISERVER_CHECKPOINT_CONTROL_H
#define FASTLLM_APISERVER_CHECKPOINT_CONTROL_H

#include <algorithm>
#include <cstddef>
#include <string>

namespace fastllm {
namespace apiserver {
    enum class CheckpointControlDecision {
        ALLOW,
        METHOD_NOT_ALLOWED,
        FORBIDDEN,
        DISABLED,
        BUSY,
    };

    inline bool ConstantTimeCheckpointTokenEqual(
            const std::string &left,
            const std::string &right) {
        std::size_t difference = left.size() ^ right.size();
        const std::size_t count = std::max(left.size(), right.size());
        for (std::size_t i = 0; i < count; i++) {
            const unsigned char leftByte =
                i < left.size() ? static_cast<unsigned char>(left[i]) : 0;
            const unsigned char rightByte =
                i < right.size() ? static_cast<unsigned char>(right[i]) : 0;
            difference |= static_cast<std::size_t>(leftByte ^ rightByte);
        }
        return difference == 0;
    }

    inline CheckpointControlDecision EvaluateCheckpointControl(
            const std::string &method,
            const std::string &authorization,
            const std::string &expectedToken,
            bool persistenceEnabled,
            int activeGenerationRequests,
            int queuedGenerationRequests) {
        if (method != "POST") {
            return CheckpointControlDecision::METHOD_NOT_ALLOWED;
        }
        static const std::string bearerPrefix = "Bearer ";
        const bool hasBearer =
            authorization.compare(0, bearerPrefix.size(), bearerPrefix) == 0;
        const std::string suppliedToken = hasBearer ?
            authorization.substr(bearerPrefix.size()) : std::string();
        if (expectedToken.empty() || !hasBearer ||
            !ConstantTimeCheckpointTokenEqual(
                suppliedToken, expectedToken)) {
            return CheckpointControlDecision::FORBIDDEN;
        }
        if (!persistenceEnabled) {
            return CheckpointControlDecision::DISABLED;
        }
        if (activeGenerationRequests != 0 ||
            queuedGenerationRequests != 0) {
            return CheckpointControlDecision::BUSY;
        }
        return CheckpointControlDecision::ALLOW;
    }

    inline const char *CheckpointControlDecisionMessage(
            CheckpointControlDecision decision) {
        switch (decision) {
            case CheckpointControlDecision::ALLOW:
                return "prefix-cache checkpoint allowed";
            case CheckpointControlDecision::METHOD_NOT_ALLOWED:
                return "checkpoint endpoint requires POST";
            case CheckpointControlDecision::FORBIDDEN:
                return "invalid checkpoint bearer token";
            case CheckpointControlDecision::DISABLED:
                return "persistent prefix cache is disabled";
            case CheckpointControlDecision::BUSY:
                return "prefix-cache checkpoint requires an idle server";
        }
        return "invalid checkpoint-control decision";
    }
}
}

#endif
