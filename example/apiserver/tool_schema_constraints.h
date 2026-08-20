#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "fastllm.h"
#include "json11.hpp"

inline void AppendUniqueToolSchemaName(
    std::vector<std::string> &names, const std::string &name) {
    if (!name.empty() &&
        std::find(names.begin(), names.end(), name) == names.end()) {
        names.push_back(name);
    }
}

inline void CollectOpenAIToolSchemaFields(
    const json11::Json &schema,
    std::vector<std::string> &allowed,
    std::vector<std::string> &required,
    std::map<std::string, std::string> *constants = nullptr) {
    const auto &properties = schema["properties"];
    if (properties.is_object()) {
        for (const auto &item : properties.object_items()) {
            AppendUniqueToolSchemaName(allowed, item.first);
            if (constants == nullptr || !item.second.is_object()) {
                continue;
            }
            const auto &constant = item.second["const"];
            if (constant.is_string()) {
                (*constants)[item.first] = constant.string_value();
                continue;
            }
            const auto &values = item.second["enum"];
            if (values.is_array() && values.array_items().size() == 1 &&
                values[0].is_string()) {
                (*constants)[item.first] = values[0].string_value();
            }
        }
    }
    const auto &requiredJson = schema["required"];
    if (requiredJson.is_array()) {
        for (const auto &item : requiredJson.array_items()) {
            if (item.is_string()) {
                AppendUniqueToolSchemaName(
                    required, item.string_value());
            }
        }
    }
}

inline bool CompileOpenAIToolSchemaConstraints(
    const std::string &toolName,
    const json11::Json &parameters,
    fastllm::GenerationConfig &config) {
    if (toolName.empty() || !parameters.is_object()) {
        return false;
    }
    std::vector<std::string> baseAllowed;
    std::vector<std::string> baseRequired;
    CollectOpenAIToolSchemaFields(
        parameters, baseAllowed, baseRequired);

    const json11::Json *unionJson = nullptr;
    if (parameters["oneOf"].is_array()) {
        unionJson = &parameters["oneOf"];
    } else if (parameters["anyOf"].is_array()) {
        unionJson = &parameters["anyOf"];
    }

    std::vector<std::string> allAllowed = baseAllowed;
    std::vector<fastllm::ToolCallParameterSchemaBranch> branches;
    if (unionJson != nullptr) {
        for (const auto &branchJson : unionJson->array_items()) {
            if (!branchJson.is_object()) {
                continue;
            }
            fastllm::ToolCallParameterSchemaBranch branch;
            branch.allowedNames = baseAllowed;
            branch.requiredNames = baseRequired;
            CollectOpenAIToolSchemaFields(
                branchJson, branch.allowedNames,
                branch.requiredNames, &branch.constValues);
            for (const auto &name : branch.allowedNames) {
                AppendUniqueToolSchemaName(allAllowed, name);
            }
            if (!branch.allowedNames.empty()) {
                branches.push_back(std::move(branch));
            }
        }
    }

    if (allAllowed.empty()) {
        return false;
    }
    config.tool_call_allowed_parameter_names[toolName] =
        std::move(allAllowed);
    if (!baseRequired.empty()) {
        config.tool_call_required_parameter_names[toolName] =
            std::move(baseRequired);
    }
    if (!branches.empty()) {
        config.tool_call_parameter_schema_branches[toolName] =
            std::move(branches);
    }
    return true;
}
