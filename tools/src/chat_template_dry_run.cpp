#include "template.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

bool ReadTextFile(const std::string &path, std::string &content,
                  std::string &error) {
    std::ifstream input(path, std::ios::binary);
    if (!input.good()) {
        error = "cannot open " + path;
        return false;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) {
        error = "cannot read " + path;
        return false;
    }
    content = buffer.str();
    return true;
}

int InvalidProbe(const std::string &message) {
    std::cerr << "chat_template_dry_run: " << message << "\n";
    return 2;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        return InvalidProbe(
            "usage: chat_template_dry_run <template.jinja> <payloads.json>");
    }

    std::string templateText;
    std::string corpusText;
    std::string error;
    if (!ReadTextFile(argv[1], templateText, error) ||
        !ReadTextFile(argv[2], corpusText, error)) {
        return InvalidProbe(error);
    }

    const json11::Json corpus = json11::Json::parse(corpusText, error);
    if (!error.empty()) {
        return InvalidProbe("invalid payload corpus JSON: " + error);
    }
    if (!corpus.is_array()) {
        return InvalidProbe("payload corpus root must be an array");
    }

    bool failed = false;
    for (const auto &item : corpus.array_items()) {
        if (!item.is_object() || !item["name"].is_string() ||
            item["name"].string_value().empty() ||
            !item["context"].is_object()) {
            return InvalidProbe(
                "each case must be {\"name\": non-empty string, "
                "\"context\": object}");
        }
        const std::string name = item["name"].string_value();
        const fastllm::ChatTemplateDryRunResult result =
            fastllm::DryRunChatTemplate(
                templateText, fastllm::JinjaVarFromJson(item["context"]));
        json11::Json::object output {
            {"name", name},
            {"ok", result.ok}
        };
        if (result.ok) {
            output["rendered"] = result.rendered;
        } else {
            output["error"] = result.error;
            failed = true;
        }
        std::string serialized = json11::Json(output).dump();
        serialized.erase(
            std::remove_if(
                serialized.begin(), serialized.end(),
                [](char c) { return c == '\n' || c == '\r' || c == '\t'; }),
            serialized.end());
        std::cout << serialized << "\n";
    }
    return failed ? 1 : 0;
}
