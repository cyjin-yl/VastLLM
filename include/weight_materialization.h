#ifndef FASTLLM_WEIGHT_MATERIALIZATION_H
#define FASTLLM_WEIGHT_MATERIALIZATION_H

#include "fastllm.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace fastllm {

    struct WeightSourceIdentity {
        std::string canonicalPath;
        uint64_t device = 0;
        uint64_t inode = 0;
        uint64_t fileSize = 0;
        int64_t modifiedSeconds = 0;
        int64_t modifiedNanoseconds = 0;
        uint64_t tensorTableFingerprint = 0;
    };

    enum class WeightRecipeKind {
        GGUF_DIRECT,
        MERGE,
    };

    struct WeightMaterializationRecipe {
        uint64_t id = 0;
        WeightRecipeKind kind = WeightRecipeKind::GGUF_DIRECT;
        std::string outputName;
        DataType outputType = DataType::FLOAT32;
        std::vector<int> outputDims;
        WeightSourceIdentity source;
        uint64_t sourceOffset = 0;
        uint64_t sourceBytes = 0;
        int ggmlType = 0;
        std::string sourceTensorName;
        std::string sourceArch;
        int replaceType = 0;
        int untileNumKHeads = 0;
        int untileNumVHeads = 0;
        int untileVRowStart = 0;
        bool untileComposeNegLog = false;
        std::vector<uint64_t> inputIds;
        std::string mergeType;
        uint64_t finalBytes = 0;
        uint64_t finalChecksum = 0;
        bool checksumAvailable = false;
    };

    bool CaptureWeightSourceIdentity(
        const std::string &path,
        uint64_t tensorTableFingerprint,
        WeightSourceIdentity &identity,
        std::string *error = nullptr);

    bool ValidateWeightSourceIdentity(
        const WeightSourceIdentity &identity,
        uint64_t tensorTableFingerprint,
        std::string *error = nullptr);

    class WeightMaterializationPlan {
    public:
        bool AddRecipe(const WeightMaterializationRecipe &recipe,
                       std::string *error = nullptr);
        const WeightMaterializationRecipe *FindById(uint64_t id) const;
        const WeightMaterializationRecipe *FindByOutput(
            const std::string &outputName) const;
        std::vector<const WeightMaterializationRecipe*> BuildReloadClosure(
            const std::vector<std::string> &outputs,
            std::string *error = nullptr) const;
        void Clear();
        size_t Size() const;

    private:
        std::vector<WeightMaterializationRecipe> recipes;
        std::map<uint64_t, size_t> byId;
        std::map<std::string, size_t> byOutput;
    };

} // namespace fastllm

#endif
