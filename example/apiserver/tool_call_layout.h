#pragma once

#include <cctype>
#include <string>

#include "fastllm.h"

namespace fastllm {

inline bool CompileToolCallGrammarLayout(
        const std::string &rendered,
        ToolCallGrammarLayout &layout,
        std::string &error) {
    static const std::string toolSentinel = "__FL_TOOL_SENTINEL__";
    static const std::string argASentinel = "__FL_ARG_A__";
    static const std::string argBSentinel = "__FL_ARG_B__";
    static const std::string valueASentinel = "__FL_VALUE_A__";
    static const std::string valueBSentinel = "__FL_VALUE_B__";
    static const std::string toolOpen = "<tool_call>";
    static const std::string parameterCloseMarker = "</parameter>";
    static const std::string functionCloseMarker = "</function>";
    static const std::string toolCloseMarker = "</tool_call>";

    layout = ToolCallGrammarLayout();
    const size_t toolName = rendered.rfind(toolSentinel);
    const size_t toolCallOpen = toolName == std::string::npos ?
        std::string::npos : rendered.rfind(toolOpen, toolName);
    const size_t argA = toolName == std::string::npos ?
        std::string::npos : rendered.find(
            argASentinel, toolName + toolSentinel.size());
    const size_t valueA = argA == std::string::npos ?
        std::string::npos : rendered.find(
            valueASentinel, argA + argASentinel.size());
    const size_t parameterCloseA = valueA == std::string::npos ?
        std::string::npos : rendered.find(
            parameterCloseMarker, valueA + valueASentinel.size());
    const size_t argB = parameterCloseA == std::string::npos ?
        std::string::npos : rendered.find(
            argBSentinel,
            parameterCloseA + parameterCloseMarker.size());
    const size_t valueB = argB == std::string::npos ?
        std::string::npos : rendered.find(
            valueBSentinel, argB + argBSentinel.size());
    const size_t parameterCloseB = valueB == std::string::npos ?
        std::string::npos : rendered.find(
            parameterCloseMarker, valueB + valueBSentinel.size());
    const size_t functionClose = parameterCloseB == std::string::npos ?
        std::string::npos : rendered.find(
            functionCloseMarker,
            parameterCloseB + parameterCloseMarker.size());
    const size_t toolCallClose = functionClose == std::string::npos ?
        std::string::npos : rendered.find(
            toolCloseMarker,
            functionClose + functionCloseMarker.size());
    if (toolCallOpen == std::string::npos ||
        argA == std::string::npos || valueA == std::string::npos ||
        parameterCloseA == std::string::npos ||
        argB == std::string::npos || valueB == std::string::npos ||
        parameterCloseB == std::string::npos ||
        functionClose == std::string::npos ||
        toolCallClose == std::string::npos) {
        error = "rendered template does not contain the sentinel tool layout";
        return false;
    }

    layout.functionPrefix = rendered.substr(
        toolCallOpen + toolOpen.size(),
        toolName - (toolCallOpen + toolOpen.size()));
    const std::string betweenToolAndArg = rendered.substr(
        toolName + toolSentinel.size(),
        argA - (toolName + toolSentinel.size()));
    const std::string betweenArgAndValue = rendered.substr(
        argA + argASentinel.size(),
        valueA - (argA + argASentinel.size()));
    size_t layoutStart = 0;
    while (layoutStart < betweenArgAndValue.size() &&
           !std::isspace(static_cast<unsigned char>(
               betweenArgAndValue[layoutStart]))) {
        layoutStart++;
    }
    if (layoutStart == 0) {
        error = "cannot derive the tool-name terminator";
        return false;
    }
    layout.nameTerminator =
        betweenArgAndValue.substr(0, layoutStart);
    layout.parameterValuePrefix =
        betweenArgAndValue.substr(layoutStart);
    if (betweenToolAndArg.compare(
            0, layout.nameTerminator.size(),
            layout.nameTerminator) != 0) {
        error = "function and parameter name terminators differ";
        return false;
    }
    layout.parameterPrefix = betweenToolAndArg.substr(
        layout.nameTerminator.size());
    const std::string betweenParameters = rendered.substr(
        parameterCloseA + parameterCloseMarker.size(),
        argB - (parameterCloseA + parameterCloseMarker.size()));
    if (betweenParameters != layout.parameterPrefix) {
        error = "first and subsequent parameter layouts differ";
        return false;
    }
    layout.parameterClose = rendered.substr(
        valueA + valueASentinel.size(),
        parameterCloseA + parameterCloseMarker.size() -
            (valueA + valueASentinel.size()));
    layout.functionClose = rendered.substr(
        parameterCloseB + parameterCloseMarker.size(),
        functionClose + functionCloseMarker.size() -
            (parameterCloseB + parameterCloseMarker.size()));
    layout.toolCallClose = rendered.substr(
        functionClose + functionCloseMarker.size(),
        toolCallClose + toolCloseMarker.size() -
            (functionClose + functionCloseMarker.size()));
    if (layout.functionPrefix.empty() ||
        layout.nameTerminator.empty() ||
        layout.parameterPrefix.empty() ||
        layout.parameterClose.empty() ||
        layout.functionClose.empty() ||
        layout.toolCallClose.empty()) {
        error = "compiled tool layout contains an empty structural segment";
        return false;
    }
    layout.valid = true;
    error.clear();
    return true;
}

} // namespace fastllm
