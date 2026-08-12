#include "example/apiserver/host_offload_control.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using fastllm::apiserver::BackendTierState;
using fastllm::apiserver::TierControlAction;
using fastllm::apiserver::TierControlDecision;

void Expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void ExpectDecision(
        TierControlAction action,
        BackendTierState state,
        const std::string &method,
        const std::string &authorization,
        const std::string &expectedToken,
        int active,
        int queued,
        bool checkpointInProgress,
        TierControlDecision expected) {
    const auto actual = fastllm::apiserver::EvaluateTierControl(
        action, state, method, authorization, expectedToken,
        active, queued, checkpointInProgress);
    Expect(actual == expected, "unexpected tier-control decision");
    const std::string message =
        fastllm::apiserver::TierControlDecisionMessage(actual);
    Expect(expectedToken.empty() ||
               message.find(expectedToken) == std::string::npos,
           "tier-control diagnostic leaked the bearer token");
}

void TestEndpointAuthorizationAndExclusion() {
    const std::string secret = "secret";
    ExpectDecision(TierControlAction::SUSPEND_MEMORY,
                   BackendTierState::VRAM_READY,
                   "GET", "Bearer secret", secret, 0, 0, false,
                   TierControlDecision::METHOD_NOT_ALLOWED);
    ExpectDecision(TierControlAction::SUSPEND_MEMORY,
                   BackendTierState::VRAM_READY,
                   "POST", "", secret, 0, 0, false,
                   TierControlDecision::FORBIDDEN);
    ExpectDecision(TierControlAction::SUSPEND_MEMORY,
                   BackendTierState::VRAM_READY,
                   "POST", "Bearer secret", secret, 1, 0, false,
                   TierControlDecision::BUSY);
    ExpectDecision(TierControlAction::SUSPEND_MEMORY,
                   BackendTierState::VRAM_READY,
                   "POST", "Bearer secret", secret, 0, 1, false,
                   TierControlDecision::BUSY);
    ExpectDecision(TierControlAction::SUSPEND_MEMORY,
                   BackendTierState::VRAM_READY,
                   "POST", "Bearer secret", secret, 0, 0, true,
                   TierControlDecision::BUSY);
    ExpectDecision(TierControlAction::SUSPEND_MEMORY,
                   BackendTierState::VRAM_READY,
                   "POST", "Bearer secret", secret, 0, 0, false,
                   TierControlDecision::ALLOW);
}

void TestThreeTierStateRules() {
    const std::string secret = "secret";
    ExpectDecision(TierControlAction::SUSPEND_MEMORY,
                   BackendTierState::RAM_SUSPENDED,
                   "POST", "Bearer secret", secret, 0, 0, false,
                   TierControlDecision::ALREADY_SUSPENDED);
    ExpectDecision(TierControlAction::SUSPEND_DISK,
                   BackendTierState::DISK_SUSPENDED,
                   "POST", "Bearer secret", secret, 0, 0, false,
                   TierControlDecision::ALREADY_SUSPENDED);
    ExpectDecision(TierControlAction::RESUME,
                   BackendTierState::VRAM_READY,
                   "POST", "Bearer secret", secret, 0, 0, false,
                   TierControlDecision::NOT_SUSPENDED);
    ExpectDecision(TierControlAction::RESUME,
                   BackendTierState::RAM_SUSPENDED,
                   "POST", "Bearer secret", secret, 0, 0, false,
                   TierControlDecision::ALLOW);
    ExpectDecision(TierControlAction::RESUME,
                   BackendTierState::DISK_SUSPENDED,
                   "POST", "Bearer secret", secret, 0, 0, false,
                   TierControlDecision::ALLOW);
    ExpectDecision(TierControlAction::RESUME,
                   BackendTierState::SUSPENDING,
                   "POST", "Bearer secret", secret, 0, 0, false,
                   TierControlDecision::TRANSITION_IN_PROGRESS);
    ExpectDecision(TierControlAction::RESUME,
                   BackendTierState::DISK_FALLBACK_LOADING,
                   "POST", "Bearer secret", secret, 0, 0, false,
                   TierControlDecision::TRANSITION_IN_PROGRESS);
}

void TestRequestAdmissionAndTransitionResults() {
    Expect(fastllm::apiserver::EvaluateRequestAdmission(
               BackendTierState::VRAM_READY) ==
               TierControlDecision::ALLOW,
           "VRAM READY must admit requests");
    Expect(fastllm::apiserver::EvaluateRequestAdmission(
               BackendTierState::SUSPENDING) ==
               TierControlDecision::BACKEND_RELOADING,
           "suspend transition must reject requests as reloading");
    Expect(fastllm::apiserver::EvaluateRequestAdmission(
               BackendTierState::RESUMING) ==
               TierControlDecision::BACKEND_RELOADING,
           "resume transition must reject requests as reloading");
    Expect(fastllm::apiserver::EvaluateRequestAdmission(
               BackendTierState::DISK_FALLBACK_LOADING) ==
               TierControlDecision::BACKEND_RELOADING,
           "disk fallback load must reject requests as reloading");
    Expect(fastllm::apiserver::EvaluateRequestAdmission(
               BackendTierState::RAM_SUSPENDED) ==
               TierControlDecision::BACKEND_SUSPENDED,
           "RAM tier must reject requests as suspended");
    Expect(fastllm::apiserver::EvaluateRequestAdmission(
               BackendTierState::DISK_SUSPENDED) ==
               TierControlDecision::BACKEND_SUSPENDED,
           "disk tier must reject requests as suspended");

    Expect(fastllm::apiserver::CompleteSuspendTransition(
               true, true) == BackendTierState::RAM_SUSPENDED,
           "successful memory suspend must enter RAM tier");
    Expect(fastllm::apiserver::CompleteSuspendTransition(
               true, false) == BackendTierState::DISK_SUSPENDED,
           "failed RAM image must descend to disk tier");
    Expect(fastllm::apiserver::CompleteSuspendTransition(
               false, false) == BackendTierState::DISK_SUSPENDED,
           "explicit disk suspend must enter disk tier");
    Expect(fastllm::apiserver::CompleteResumeTransition(true) ==
               BackendTierState::VRAM_READY,
           "successful resume must return to VRAM READY");
    Expect(fastllm::apiserver::CompleteResumeTransition(false) ==
               BackendTierState::ERROR,
           "failed disk fallback must enter ERROR");
    Expect(std::string(fastllm::apiserver::BackendTierStateName(
               BackendTierState::RAM_SUSPENDED)) == "suspended_host",
           "RAM state name must be observable");
    Expect(std::string(fastllm::apiserver::BackendTierStateName(
               BackendTierState::DISK_FALLBACK_LOADING)) ==
               "disk_fallback_loading",
           "disk fallback state name must be observable");
}

} // namespace

int main() {
    try {
        TestEndpointAuthorizationAndExclusion();
        TestThreeTierStateRules();
        TestRequestAdmissionAndTransitionResults();
        std::cout << "host offload control: PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "host offload control: FAIL: " << error.what() << "\n";
        return 1;
    }
}
