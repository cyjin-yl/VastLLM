#include "template.h"
#include "models/basellm.h"

#include <cstdio>
#include <string>

namespace {

int failures = 0;
int checks = 0;

void Check(bool ok, const char *message) {
    checks++;
    if (ok) {
        std::printf("  ok   %s\n", message);
    } else {
        std::printf("  FAIL %s\n", message);
        failures++;
    }
}
struct TemplateProbeModel : fastllm::basellm {
    int makeInputCalls = 0;

    std::string MakeInput(const std::string &, int,
                          const std::string &) override {
        makeInputCalls++;
        return "legacy-fallback";
    }

    std::string MakeHistory(const std::string &, int,
                            const std::string &,
                            const std::string &) override {
        return "";
    }
};


json11::Json ParseJson(const std::string &text) {
    std::string error;
    json11::Json value = json11::Json::parse(text, error);
    if (!error.empty()) {
        std::printf("fixture parse error: %s\n", error.c_str());
        failures++;
    }
    return value;
}

void TestNestedJsonConversionAndRender() {
    std::printf("[1] nested JSON context uses the production Jinja renderer\n");
    const json11::Json json = ParseJson(R"JSON({
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"function": {"name": "read"}}],
        "tool_name": "read",
        "enable_thinking": true,
        "nullable": null
    })JSON");
    const fastllm::JinjaVar context = fastllm::JinjaVarFromJson(json);

    Check(context.type == fastllm::JinjaVar::JinjaDict,
          "root object converts to JinjaDict");
    Check(context.dictValue.at("messages").type ==
              fastllm::JinjaVar::JinjaArray,
          "nested messages convert to JinjaArray");
    Check(context.dictValue.at("tools").arrayValue.at(0)
              .dictValue.at("function").dictValue.at("name").stringValue ==
              "read",
          "nested tool name survives JSON conversion");
    Check(context.dictValue.at("enable_thinking").BoolValue(),
          "JSON true remains truthy");
    Check(context.dictValue.at("nullable").type ==
              fastllm::JinjaVar::JinjaNone,
          "JSON null remains JinjaNone");

    const std::string templ =
        "{% for m in messages %}{{ m.role }}={{ m.content }};{% endfor %}"
        "{% if enable_thinking %}think{% endif %}"
        "|{{ tool_name }}";
    const fastllm::ChatTemplateDryRunResult result =
        fastllm::DryRunChatTemplate(templ, context);
    Check(result.ok, "valid template renders successfully");
    if (result.rendered != "user=hello;think|read") {
        std::printf("       actual: %s\n", result.rendered.c_str());
    }
    Check(result.rendered == "user=hello;think|read",
          "rendered output matches exactly");
    Check(result.error.empty(), "successful render has no error");
}

void TestInvalidTemplateIsExplicit() {
    std::printf("[2] invalid template returns an explicit error\n");
    const fastllm::JinjaVar context = fastllm::JinjaVarFromJson(
        ParseJson(R"JSON({"messages": []})JSON"));
    const fastllm::ChatTemplateDryRunResult result =
        fastllm::DryRunChatTemplate("{{ messages[0].content ", context);
    Check(!result.ok, "invalid template is rejected");
    Check(result.rendered.empty(), "failed render has no output");
    Check(!result.error.empty(), "failed render includes a reason");
}

void TestContextsStayIsolated() {
    std::printf("[3] repeated dry-runs do not leak context\n");
    const std::string templ = "{{ value }}";
    const fastllm::ChatTemplateDryRunResult first =
        fastllm::DryRunChatTemplate(
            templ, fastllm::JinjaVarFromJson(ParseJson("{\"value\":\"A\"}")));
    const fastllm::ChatTemplateDryRunResult second =
        fastllm::DryRunChatTemplate(
            templ, fastllm::JinjaVarFromJson(ParseJson("{\"value\":\"B\"}")));
    Check(first.ok && second.ok, "both independent contexts render");
    Check(first.rendered == "A" && second.rendered == "B",
          "each render sees only its own context");
}

void TestNestedInlineConditionalExpression() {
    std::printf("[4] official-template inline conditional works inside parentheses\n");
    const std::string templ =
        "{{ '<s>' + (reasoning_instructions + '\\n\\n' "
        "if reasoning_instructions else '') + content }}";
    const fastllm::ChatTemplateDryRunResult truthy =
        fastllm::DryRunChatTemplate(
            templ,
            fastllm::JinjaVarFromJson(ParseJson(
                "{\"reasoning_instructions\":\"R\",\"content\":\"body\"}")));
    const fastllm::ChatTemplateDryRunResult falsy =
        fastllm::DryRunChatTemplate(
            templ,
            fastllm::JinjaVarFromJson(ParseJson(
                "{\"reasoning_instructions\":\"\",\"content\":\"body\"}")));
    Check(truthy.ok && falsy.ok,
          "both inline-conditional branches parse");
    Check(truthy.rendered == "<s>R\n\nbody",
          "truthy branch preserves prefix and suffix concatenation");
    Check(falsy.rendered == "<s>body",
          "falsy branch preserves prefix and suffix concatenation");
}


void TestRuntimeTemplateErrorNeverFallsBack() {
    std::printf("[5] runtime Jinja failure is explicit and never calls MakeInput\n");
    TemplateProbeModel model;
    model.weight.tokenizer.chatTemplate = "{{ messages[0].content ";
    const fastllm::ChatMessages messages = {{"user", "hello"}};
    const long long before = fastllm::GetChatTemplateRenderErrorCount();
    bool threw = false;
    std::string error;
    try {
        (void)model.ApplyChatTemplate(messages);
    } catch (const std::exception &caught) {
        threw = true;
        error = caught.what();
    }
    const long long after = fastllm::GetChatTemplateRenderErrorCount();
    Check(threw, "invalid configured template throws to the request boundary");
    Check(error.find("chat_template") != std::string::npos,
          "exception names chat_template explicitly");
    Check(model.makeInputCalls == 0,
          "invalid configured template never calls MakeInput");
    Check(after - before == 1,
          "runtime template error counter increases exactly once");
}

void TestAbsentTemplateKeepsLegacyModelPath() {
    std::printf("[6] models without a configured template keep MakeInput behavior\n");
    TemplateProbeModel model;
    model.weight.tokenizer.chatTemplate.clear();
    const fastllm::ChatMessages messages = {{"user", "hello"}};
    const std::string rendered =
        model.ApplyChatTemplate(messages);
    Check(rendered == "legacy-fallback",
          "absent template uses the model-specific prompt builder");
    Check(model.makeInputCalls == 1,
          "absent template calls MakeInput exactly once");
}
}  // namespace

int main() {
    std::printf("== chat template weight-free validation ==\n");
    TestNestedJsonConversionAndRender();
    TestInvalidTemplateIsExplicit();
    TestContextsStayIsolated();
    TestNestedInlineConditionalExpression();
    TestRuntimeTemplateErrorNeverFallsBack();
    TestAbsentTemplateKeepsLegacyModelPath();
    std::printf("== %d/%d passed ==\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
