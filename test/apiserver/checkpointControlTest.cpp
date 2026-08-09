#include "example/apiserver/checkpoint_control.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
    using fastllm::apiserver::CheckpointControlDecision;

    void Expect(bool condition, const std::string &message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    void ExpectDecision(
            const std::string &method,
            const std::string &authorization,
            const std::string &expectedToken,
            bool persistenceEnabled,
            int activeGenerationRequests,
            int queuedGenerationRequests,
            CheckpointControlDecision expected) {
        const CheckpointControlDecision actual =
            fastllm::apiserver::EvaluateCheckpointControl(
                method,
                authorization,
                expectedToken,
                persistenceEnabled,
                activeGenerationRequests,
                queuedGenerationRequests);
        Expect(actual == expected, "unexpected checkpoint-control decision");
        const std::string message =
            fastllm::apiserver::CheckpointControlDecisionMessage(actual);
        Expect(expectedToken.empty() ||
                   message.find(expectedToken) == std::string::npos,
               "checkpoint-control diagnostic leaked the bearer token");
    }
}

int main() {
    try {
        const std::string secret = "secret";
        ExpectDecision("GET", "Bearer secret", secret, true, 0, 0,
                       CheckpointControlDecision::METHOD_NOT_ALLOWED);
        ExpectDecision("POST", "", secret, true, 0, 0,
                       CheckpointControlDecision::FORBIDDEN);
        ExpectDecision("POST", "Bearer secreu", secret, true, 0, 0,
                       CheckpointControlDecision::FORBIDDEN);
        ExpectDecision("POST", "Bearer secretx", secret, true, 0, 0,
                       CheckpointControlDecision::FORBIDDEN);
        ExpectDecision("POST", "Bearer secret", "", true, 0, 0,
                       CheckpointControlDecision::FORBIDDEN);
        ExpectDecision("POST", "Bearer secret", secret, false, 0, 0,
                       CheckpointControlDecision::DISABLED);
        ExpectDecision("POST", "Bearer secret", secret, true, 1, 0,
                       CheckpointControlDecision::BUSY);
        ExpectDecision("POST", "Bearer secret", secret, true, 0, 1,
                       CheckpointControlDecision::BUSY);
        ExpectDecision("POST", "Bearer secret", secret, true, 0, 0,
                       CheckpointControlDecision::ALLOW);
        std::cout << "checkpoint control regression: PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "checkpoint control regression: FAIL: "
                  << error.what() << "\n";
        return 1;
    }
}
