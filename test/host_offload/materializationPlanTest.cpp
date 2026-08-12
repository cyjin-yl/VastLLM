#include "host_offload.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::filesystem::path CreateSourceFile() {
    const auto nonce = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
        ("fastllm-host-offload-" + std::to_string(nonce) + ".gguf");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write("GGUF-test-payload", 17);
    output.close();
    return path;
}

fastllm::WeightMaterializationRecipe DirectRecipe(
        uint64_t id, const std::string &name,
        const fastllm::WeightSourceIdentity &source,
        uint64_t offset) {
    fastllm::WeightMaterializationRecipe recipe;
    recipe.id = id;
    recipe.kind = fastllm::WeightRecipeKind::GGUF_DIRECT;
    recipe.outputName = name;
    recipe.outputType = fastllm::DataType::FLOAT32;
    recipe.outputDims = {2, 2};
    recipe.source = source;
    recipe.sourceOffset = offset;
    recipe.sourceBytes = 4 * sizeof(float);
    recipe.ggmlType = 0;
    return recipe;
}

void TestSourceIdentityRejectsChangedFile() {
    const auto path = CreateSourceFile();
    std::string error;
    fastllm::WeightSourceIdentity identity;
    Expect(fastllm::CaptureWeightSourceIdentity(
               path.string(), UINT64_C(0x1020304050607080), identity, &error),
           "source identity capture failed: " + error);
    Expect(fastllm::ValidateWeightSourceIdentity(
               identity, UINT64_C(0x1020304050607080), &error),
           "unchanged source identity must validate: " + error);
    Expect(std::filesystem::path(identity.canonicalPath).is_absolute(),
           "source identity path must be canonical and absolute");

    {
        std::ofstream output(path, std::ios::binary | std::ios::app);
        output.put('x');
    }
    error.clear();
    Expect(!fastllm::ValidateWeightSourceIdentity(
               identity, UINT64_C(0x1020304050607080), &error),
           "changed file size must invalidate the source identity");
    Expect(!error.empty(), "source mismatch must explain the reason");

    std::filesystem::remove(path);
}

void TestTensorTableFingerprintMismatchIsRejected() {
    const auto path = CreateSourceFile();
    std::string error;
    fastllm::WeightSourceIdentity identity;
    Expect(fastllm::CaptureWeightSourceIdentity(
               path.string(), UINT64_C(0x1111), identity, &error),
           "source identity capture failed: " + error);
    Expect(!fastllm::ValidateWeightSourceIdentity(
               identity, UINT64_C(0x2222), &error),
           "changed tensor-table fingerprint must invalidate the source");
    std::filesystem::remove(path);
}

void TestReloadClosureIsDependencyOrdered() {
    const auto path = CreateSourceFile();
    std::string error;
    fastllm::WeightSourceIdentity identity;
    Expect(fastllm::CaptureWeightSourceIdentity(
               path.string(), UINT64_C(0xabc), identity, &error),
           "source identity capture failed: " + error);

    fastllm::WeightMaterializationPlan plan;
    Expect(plan.AddRecipe(DirectRecipe(1, "q.weight", identity, 64), &error),
           "first direct recipe failed: " + error);
    Expect(plan.AddRecipe(DirectRecipe(2, "k.weight", identity, 128), &error),
           "second direct recipe failed: " + error);

    fastllm::WeightMaterializationRecipe merge;
    merge.id = 3;
    merge.kind = fastllm::WeightRecipeKind::MERGE;
    merge.outputName = "qk.weight";
    merge.outputType = fastllm::DataType::FLOAT32;
    merge.outputDims = {4, 2};
    merge.inputIds = {1, 2};
    merge.mergeType = "linear";
    Expect(plan.AddRecipe(merge, &error), "merge recipe failed: " + error);

    const auto closure = plan.BuildReloadClosure({"qk.weight"}, &error);
    Expect(error.empty(), "valid closure must not report an error: " + error);
    Expect(closure.size() == 3,
           "merge closure must include both sources and output");
    Expect(closure[0]->id == 1 && closure[1]->id == 2 && closure[2]->id == 3,
           "reload closure must be deterministic dependency order");
    Expect(plan.FindByOutput("qk.weight") != nullptr,
           "final output lookup must be stable");

    std::filesystem::remove(path);
}

void TestInvalidRecipeGraphIsRejected() {
    fastllm::WeightMaterializationPlan plan;
    std::string error;

    fastllm::WeightMaterializationRecipe missing;
    missing.id = 7;
    missing.kind = fastllm::WeightRecipeKind::MERGE;
    missing.outputName = "missing-input.weight";
    missing.outputType = fastllm::DataType::FLOAT32;
    missing.outputDims = {1};
    missing.inputIds = {999};
    missing.mergeType = "linear";
    Expect(!plan.AddRecipe(missing, &error),
           "merge recipe with an unknown dependency must be rejected");

    const auto path = CreateSourceFile();
    fastllm::WeightSourceIdentity identity;
    error.clear();
    Expect(fastllm::CaptureWeightSourceIdentity(
               path.string(), UINT64_C(0xabc), identity, &error),
           "source identity capture failed: " + error);
    Expect(plan.AddRecipe(DirectRecipe(1, "dup.weight", identity, 64), &error),
           "first direct recipe failed: " + error);
    error.clear();
    Expect(!plan.AddRecipe(DirectRecipe(2, "dup.weight", identity, 128), &error),
           "duplicate output names must be rejected");
    std::filesystem::remove(path);
}

} // namespace

int main() {
    try {
        TestSourceIdentityRejectsChangedFile();
        TestTensorTableFingerprintMismatchIsRejected();
        TestReloadClosureIsDependencyOrdered();
        TestInvalidRecipeGraphIsRejected();
        std::cout << "materialization plan: PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "materialization plan: FAIL: " << error.what() << "\n";
        return 1;
    }
}
