#include "weight_materialization.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <functional>
#include <set>
#include <sstream>
#include <sys/stat.h>

namespace fastllm {
namespace {

    void SetRecipeError(std::string *error, const std::string &message) {
        if (error != nullptr) {
            *error = message;
        }
    }

    bool ReadSourceIdentity(const std::string &path,
                            uint64_t tensorTableFingerprint,
                            WeightSourceIdentity &identity,
                            std::string *error) {
        std::error_code pathError;
        const std::filesystem::path canonical =
            std::filesystem::canonical(path, pathError);
        if (pathError) {
            SetRecipeError(error, "cannot canonicalize weight source '" + path +
                                  "': " + pathError.message());
            return false;
        }

        struct stat status {};
        if (::stat(canonical.c_str(), &status) != 0) {
            SetRecipeError(error, "cannot stat weight source '" +
                                  canonical.string() + "': " +
                                  std::strerror(errno));
            return false;
        }
        if (!S_ISREG(status.st_mode)) {
            SetRecipeError(error, "weight source is not a regular file: " +
                                  canonical.string());
            return false;
        }

        identity.canonicalPath = canonical.string();
        identity.device = static_cast<uint64_t>(status.st_dev);
        identity.inode = static_cast<uint64_t>(status.st_ino);
        identity.fileSize = static_cast<uint64_t>(status.st_size);
        identity.modifiedSeconds = static_cast<int64_t>(status.st_mtim.tv_sec);
        identity.modifiedNanoseconds = static_cast<int64_t>(status.st_mtim.tv_nsec);
        identity.tensorTableFingerprint = tensorTableFingerprint;
        return true;
    }

} // namespace

bool CaptureWeightSourceIdentity(const std::string &path,
                                 uint64_t tensorTableFingerprint,
                                 WeightSourceIdentity &identity,
                                 std::string *error) {
    if (error != nullptr) {
        error->clear();
    }
    WeightSourceIdentity observed;
    if (!ReadSourceIdentity(path, tensorTableFingerprint, observed, error)) {
        return false;
    }
    identity = std::move(observed);
    return true;
}

bool ValidateWeightSourceIdentity(const WeightSourceIdentity &identity,
                                  uint64_t tensorTableFingerprint,
                                  std::string *error) {
    if (error != nullptr) {
        error->clear();
    }
    if (identity.canonicalPath.empty()) {
        SetRecipeError(error, "weight source identity has no path");
        return false;
    }
    if (tensorTableFingerprint != identity.tensorTableFingerprint) {
        SetRecipeError(error, "weight source tensor table fingerprint changed");
        return false;
    }

    WeightSourceIdentity observed;
    if (!ReadSourceIdentity(identity.canonicalPath,
                            tensorTableFingerprint, observed, error)) {
        return false;
    }
    if (observed.canonicalPath != identity.canonicalPath ||
        observed.device != identity.device ||
        observed.inode != identity.inode ||
        observed.fileSize != identity.fileSize ||
        observed.modifiedSeconds != identity.modifiedSeconds ||
        observed.modifiedNanoseconds != identity.modifiedNanoseconds) {
        SetRecipeError(error, "weight source identity changed since model load");
        return false;
    }
    return true;
}

bool WeightMaterializationPlan::AddRecipe(
        const WeightMaterializationRecipe &recipe,
        std::string *error) {
    if (error != nullptr) {
        error->clear();
    }
    if (recipe.id == 0) {
        SetRecipeError(error, "materialization recipe id must be nonzero");
        return false;
    }
    if (recipe.outputName.empty()) {
        SetRecipeError(error, "materialization recipe output name is empty");
        return false;
    }
    if (byId.find(recipe.id) != byId.end()) {
        SetRecipeError(error, "duplicate materialization recipe id");
        return false;
    }
    if (byOutput.find(recipe.outputName) != byOutput.end()) {
        SetRecipeError(error, "duplicate materialization recipe output: " +
                              recipe.outputName);
        return false;
    }
    if (recipe.kind == WeightRecipeKind::MERGE) {
        if (recipe.inputIds.empty()) {
            SetRecipeError(error, "merge materialization recipe has no inputs");
            return false;
        }
        for (uint64_t inputId : recipe.inputIds) {
            if (byId.find(inputId) == byId.end()) {
                SetRecipeError(error, "merge materialization recipe references "
                                      "an unknown input id");
                return false;
            }
        }
    } else if (!recipe.inputIds.empty()) {
        SetRecipeError(error, "direct materialization recipe cannot have inputs");
        return false;
    }

    const size_t index = recipes.size();
    recipes.push_back(recipe);
    byId.emplace(recipe.id, index);
    byOutput.emplace(recipe.outputName, index);
    return true;
}

const WeightMaterializationRecipe *WeightMaterializationPlan::FindById(
        uint64_t id) const {
    const auto it = byId.find(id);
    return it == byId.end() ? nullptr : &recipes[it->second];
}

const WeightMaterializationRecipe *WeightMaterializationPlan::FindByOutput(
        const std::string &outputName) const {
    const auto it = byOutput.find(outputName);
    return it == byOutput.end() ? nullptr : &recipes[it->second];
}

std::vector<const WeightMaterializationRecipe*>
WeightMaterializationPlan::BuildReloadClosure(
        const std::vector<std::string> &outputs,
        std::string *error) const {
    if (error != nullptr) {
        error->clear();
    }
    std::vector<const WeightMaterializationRecipe*> ordered;
    std::set<uint64_t> visiting;
    std::set<uint64_t> visited;

    std::function<bool(const WeightMaterializationRecipe*)> visit =
        [&](const WeightMaterializationRecipe *recipe) {
            if (visited.find(recipe->id) != visited.end()) {
                return true;
            }
            if (!visiting.insert(recipe->id).second) {
                SetRecipeError(error, "materialization recipe graph has a cycle");
                return false;
            }
            for (uint64_t inputId : recipe->inputIds) {
                const WeightMaterializationRecipe *input = FindById(inputId);
                if (input == nullptr) {
                    SetRecipeError(error, "materialization recipe dependency is missing");
                    return false;
                }
                if (!visit(input)) {
                    return false;
                }
            }
            visiting.erase(recipe->id);
            visited.insert(recipe->id);
            ordered.push_back(recipe);
            return true;
        };

    for (const std::string &output : outputs) {
        const WeightMaterializationRecipe *recipe = FindByOutput(output);
        if (recipe == nullptr) {
            SetRecipeError(error, "no materialization recipe for output: " + output);
            return {};
        }
        if (!visit(recipe)) {
            return {};
        }
    }
    return ordered;
}

void WeightMaterializationPlan::Clear() {
    recipes.clear();
    byId.clear();
    byOutput.clear();
}

size_t WeightMaterializationPlan::Size() const {
    return recipes.size();
}

} // namespace fastllm
