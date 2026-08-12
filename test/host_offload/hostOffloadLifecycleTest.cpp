#include "host_offload.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

constexpr uint64_t KiB = UINT64_C(1024);

void Expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Fixture {
    std::unordered_map<std::string, fastllm::Data> weights;
    fastllm::WeightMaterializationPlan plan;
    std::filesystem::path sourcePath;
    std::unordered_map<std::string, std::vector<uint8_t>> expected;

    Fixture() {
        sourcePath = std::filesystem::temp_directory_path() /
            ("fastllm-host-offload-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
             ".gguf");
        {
            std::ofstream output(sourcePath, std::ios::binary);
            const std::string bytes(64, 's');
            output.write(bytes.data(), bytes.size());
        }

        fastllm::WeightSourceIdentity identity;
        std::string error;
        Expect(fastllm::CaptureWeightSourceIdentity(
                   sourcePath.string(), 1234, identity, &error),
               "capture source identity: " + error);

        fastllm::WeightMaterializationRecipe inputA;
        inputA.id = 1;
        inputA.outputName = "source.a";
        inputA.source = identity;
        inputA.sourceBytes = 16;
        Expect(plan.AddRecipe(inputA, &error), "add source.a recipe: " + error);

        fastllm::WeightMaterializationRecipe inputB = inputA;
        inputB.id = 2;
        inputB.outputName = "source.b";
        inputB.sourceOffset = 16;
        Expect(plan.AddRecipe(inputB, &error), "add source.b recipe: " + error);

        fastllm::WeightMaterializationRecipe derived;
        derived.id = 3;
        derived.kind = fastllm::WeightRecipeKind::MERGE;
        derived.outputName = "derived.weight";
        derived.inputIds = {1, 2};
        derived.mergeType = "concat";
        Expect(plan.AddRecipe(derived, &error), "add derived recipe: " + error);

        fastllm::WeightMaterializationRecipe ordinary = inputA;
        ordinary.id = 4;
        ordinary.outputName = "ordinary.weight";
        ordinary.sourceOffset = 32;
        Expect(plan.AddRecipe(ordinary, &error), "add ordinary recipe: " + error);

        AddWeight("must.weight", 11);
        AddWeight("derived.weight", 37);
        AddWeight("ordinary.weight", 73);
    }

    ~Fixture() {
        std::error_code ignored;
        std::filesystem::remove(sourcePath, ignored);
    }

    void AddWeight(const std::string &name, uint8_t seed) {
        std::vector<float> values(KiB);
        std::vector<uint8_t> bytes(values.size() * sizeof(float));
        for (size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<uint8_t>(seed + i * 17);
        }
        std::memcpy(values.data(), bytes.data(), bytes.size());
        auto inserted = weights.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(name),
            std::forward_as_tuple(
                fastllm::DataType::FLOAT32,
                std::vector<int>{1, static_cast<int>(values.size())},
                values));
        fastllm::Data &data = inserted.first->second;
        data.name = name;
        data.isModelWeight = true;
        data.ToDevice(fastllm::DataDevice::CUDA);
        expected.emplace(name, std::move(bytes));
    }

    bool Reload(const std::vector<std::string> &names, std::string *error) {
        for (const std::string &name : names) {
            auto weight = weights.find(name);
            auto bytes = expected.find(name);
            if (weight == weights.end() || bytes == expected.end()) {
                if (error != nullptr) {
                    *error = "unknown weight " + name;
                }
                return false;
            }
            fastllm::Data &data = weight->second;
            data.cpuData = new uint8_t[data.expansionBytes];
            std::memcpy(data.cpuData, bytes->second.data(), data.expansionBytes);
            data.dataDevice = fastllm::DataDevice::CPU;
            data.dataDeviceIds.clear();
            data.ToDevice(fastllm::DataDevice::CUDA);
        }
        return true;
    }
};

void ExpectAllCuda(const Fixture &fixture) {
    for (const auto &entry : fixture.weights) {
        Expect(entry.second.cudaData != nullptr && entry.second.cpuData == nullptr,
               entry.first + " must have one CUDA copy");
    }
}

void TestFullFitRoundTrip() {
    Fixture fixture;
    fastllm::HostCacheBudget budget(12 * KiB, 0, [] { return 64 * KiB; });
    fastllm::HostOffloadManager manager(
        fixture.weights, fixture.plan, budget,
        [&](const std::vector<std::string> &names, std::string *error) {
            return fixture.Reload(names, error);
        });

    const auto suspended = manager.Suspend(7);
    Expect(suspended.outcome == fastllm::HostOffloadOutcome::SUSPENDED_HOST,
           "full-fit suspend must use host tier: " + suspended.reason);
    Expect(suspended.cachedBytes == 12 * KiB && suspended.sourceEvictedBytes == 0,
           "full-fit suspend must cache every final tensor");
    Expect(budget.ResidentBytes() == 12 * KiB,
           "full-fit reservations must equal the configured budget");
    for (const auto &entry : fixture.weights) {
        Expect(entry.second.cudaData == nullptr && entry.second.cpuData != nullptr,
               entry.first + " must be host-only while suspended");
    }

    const auto resumed = manager.Resume(7);
    Expect(resumed.outcome == fastllm::HostOffloadOutcome::READY,
           "full-fit resume must return READY: " + resumed.reason);
    ExpectAllCuda(fixture);
    Expect(budget.ResidentBytes() == 0,
           "READY must release all manager-owned host reservations");
}

void TestPartialFitUsesPriorityOrder() {
    Fixture fixture;
    fastllm::HostCacheBudget budget(8 * KiB, 0, [] { return 64 * KiB; });
    fastllm::HostOffloadManager manager(
        fixture.weights, fixture.plan, budget,
        [&](const std::vector<std::string> &names, std::string *error) {
            return fixture.Reload(names, error);
        });

    const auto suspended = manager.Suspend(9);
    Expect(suspended.outcome == fastllm::HostOffloadOutcome::SUSPENDED_HOST,
           "partial-fit suspend must remain in host tier: " + suspended.reason);
    Expect(suspended.cachedBytes == 8 * KiB &&
               suspended.sourceEvictedBytes == 4 * KiB,
           "partial-fit bytes must split 8 KiB cached / 4 KiB rebuilt");
    Expect(manager.SourceEvictedWeights() ==
               std::vector<std::string>{"ordinary.weight"},
           "must-cache and derived weights must win over ordinary weights");
    Expect(fixture.weights.at("must.weight").cpuData != nullptr,
           "must-cache tensor must remain in RAM");
    Expect(fixture.weights.at("derived.weight").cpuData != nullptr,
           "derived tensor must remain in RAM before ordinary tensor");
    const auto &ordinary = fixture.weights.at("ordinary.weight");
    Expect(ordinary.cpuData == nullptr && ordinary.cudaData == nullptr,
           "ordinary source-rebuildable tensor must release its payload");

    const auto resumed = manager.Resume(9);
    Expect(resumed.outcome == fastllm::HostOffloadOutcome::READY,
           "partial-fit resume must return READY: " + resumed.reason);
    Expect(resumed.rebuiltBytes == 4 * KiB,
           "resume must report source-rebuilt bytes");
    ExpectAllCuda(fixture);
    Expect(budget.ResidentBytes() == 0,
           "partial READY must release all host reservations");
}

void TestBelowMustCacheFallsBackWithoutMutation() {
    for (uint64_t bytes : {2 * KiB, UINT64_C(0)}) {
        Fixture fixture;
        fastllm::HostCacheBudget budget(bytes, 0, [] { return 64 * KiB; });
        fastllm::HostOffloadManager manager(
            fixture.weights, fixture.plan, budget,
            [&](const std::vector<std::string> &names, std::string *error) {
                return fixture.Reload(names, error);
            });

        const auto result = manager.Suspend(11);
        Expect(result.outcome ==
                   fastllm::HostOffloadOutcome::DISK_FALLBACK_REQUIRED,
               "budget below must-cache must request disk fallback");
        ExpectAllCuda(fixture);
        Expect(budget.ResidentBytes() == 0,
               "failed preflight must release every reservation");
    }
}

void TestGenerationAndChecksumFailuresRequireDiskFallback() {
    {
        Fixture fixture;
        fastllm::HostCacheBudget budget(12 * KiB, 0, [] { return 64 * KiB; });
        fastllm::HostOffloadManager manager(
            fixture.weights, fixture.plan, budget,
            [&](const std::vector<std::string> &names, std::string *error) {
                return fixture.Reload(names, error);
            });
        Expect(manager.Suspend(13).outcome ==
                   fastllm::HostOffloadOutcome::SUSPENDED_HOST,
               "generation fixture must suspend");
        const auto mismatch = manager.Resume(14);
        Expect(mismatch.outcome ==
                   fastllm::HostOffloadOutcome::DISK_FALLBACK_REQUIRED,
               "generation mismatch must require full disk recovery");
    }

    {
        Fixture fixture;
        fastllm::HostCacheBudget budget(12 * KiB, 0, [] { return 64 * KiB; });
        fastllm::HostOffloadManager manager(
            fixture.weights, fixture.plan, budget,
            [&](const std::vector<std::string> &names, std::string *error) {
                return fixture.Reload(names, error);
            });
        Expect(manager.Suspend(15).outcome ==
                   fastllm::HostOffloadOutcome::SUSPENDED_HOST,
               "checksum fixture must suspend");
        fixture.weights.at("must.weight").cpuData[0] ^= 0xff;
        const auto corrupted = manager.Resume(15);
        Expect(corrupted.outcome ==
                   fastllm::HostOffloadOutcome::DISK_FALLBACK_REQUIRED,
               "host checksum mismatch must require full disk recovery");
        Expect(corrupted.reason.find("checksum") != std::string::npos,
               "checksum fallback reason must identify corruption");
    }
}

void TestGgufAuxiliaryCachesRetireBeforeSuspend() {
    Fixture fixture;
    fastllm::Data &weight = fixture.weights.at("ordinary.weight");
    weight.isGGUFData = true;
    void *auxiliary = FastllmCudaMalloc(256);
    Expect(auxiliary != nullptr, "allocate GGUF auxiliary CUDA cache");
    weight.extraCudaData.push_back(auxiliary);
    weight.extraCudaHalfData.push_back(auxiliary);


    fastllm::HostCacheBudget budget(12 * KiB, 0, [] { return 64 * KiB; });
    fastllm::HostOffloadManager manager(
        fixture.weights, fixture.plan, budget,
        [&](const std::vector<std::string> &names, std::string *reloadError) {
            return fixture.Reload(names, reloadError);
        });
    Expect(manager.Suspend(17).outcome ==
               fastllm::HostOffloadOutcome::SUSPENDED_HOST,
           "GGUF weight must suspend after auxiliary cache teardown");
    Expect(weight.extraCudaData.empty() &&
               weight.extraCudaHalfData.empty(),
           "host suspend must retire GGUF auxiliary CUDA caches");
    Expect(manager.Resume(17).outcome ==
               fastllm::HostOffloadOutcome::READY,
           "GGUF weight must resume after auxiliary cache teardown");
}

} // namespace

int main() {
#ifdef USE_CUDA
    try {
        TestFullFitRoundTrip();
        TestPartialFitUsesPriorityOrder();
        TestBelowMustCacheFallsBackWithoutMutation();
        TestGenerationAndChecksumFailuresRequireDiskFallback();
        TestGgufAuxiliaryCachesRetireBeforeSuspend();
        std::cout << "host offload lifecycle: PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "host offload lifecycle: FAIL: " << error.what() << "\n";
        return 1;
    }
#else
    std::cout << "host offload lifecycle: SKIP (CUDA disabled)\n";
    return 0;
#endif
}
