// tool_call 语法状态机硬不变量测试(C++ 真实代码路径, 对应
// v100-perfs/tests/toolcall_grammar_invariants.py 的 python 复刻)。
// 直接驱动 basellm::EvaluateToolCallConstraintText + 合成 vocab。
#include "models/basellm.h"
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace fastllm;

struct StubModel : basellm {
    std::string MakeInput(const std::string &, int,
                          const std::string &) override { return ""; }
    std::string MakeHistory(const std::string &, int, const std::string &,
                            const std::string &) override { return ""; }
};

// 合成 vocab(形态对齐 Qwen BPE: 标签整词 + 碎片)
static const std::vector<std::string> VOCAB = {
    "<tool_call>", "</tool_call>", "<function=", "</function>",
    "<parameter=", "</parameter>", "<", "</", "<p", "<pa", "<par",
    "<parameter", "</pa", "</param", "</parame", "</f", "</fu",
    "</functio", "</function", "</tool", "</tool_", "list_dir",
    "read_file", "list", "_dir", "path", "max_depth", "max", "_depth",
    "pa", "at", "h", "pattern", "pat", "tern", "paath", "ma", "ax",
    "ax_", "aax", "m", "a", "x", "t", "e", "r", "n", "d", "p", "i",
    "s", "l", "_", "th", "ath", ">", "/etc", "/", "etc", "2", "\n",
    " ", "  ", "hello", "1", "10", "txt", ".", "readme", "><",
};

static GenerationConfig MakeConfig() {
    GenerationConfig cfg;
    cfg.tool_call_name_constraint_enabled = true;
    cfg.tool_call_allowed_names = {"list_dir", "read_file"};
    cfg.tool_call_invoke_name_prefixes = {"<function="};
    cfg.tool_call_name_terminator = ">";
    cfg.tool_call_parameter_name_constraint_enabled = true;
    cfg.tool_call_allowed_parameter_names = {
        {"list_dir", {"path", "max_depth"}},
        {"read_file", {"path"}},
    };
    cfg.tool_call_parameter_name_prefixes = {"<parameter="};
    cfg.tool_call_required_parameter_constraint_enabled = true;
    cfg.tool_call_required_parameter_names = {
        {"list_dir", {"path"}},
        {"read_file", {"path"}},
    };
    return cfg;
}

struct EvalResult {
    std::vector<int> allowed;   // empty => 不 mask
    std::vector<int> blocked;
};

static EvalResult Eval(StubModel &m, const std::string &text,
                       const GenerationConfig &cfg) {
    EvalResult r;
    m.EvaluateToolCallConstraintText(text, cfg, r.allowed, &r.blocked);
    return r;
}

// can_spell: 从 startText 出发逐 token DFS, target 是否可拼出
static bool CanSpell(StubModel &m, const std::string &startText,
                     const std::string &target, const GenerationConfig &cfg,
                     int depth = 0) {
    if (target.empty()) return true;
    if (depth > 40) return false;
    EvalResult r = Eval(m, startText, cfg);
    if (r.allowed.empty() && r.blocked.empty()) {
        if (getenv("CS_TRACE"))
            printf("[CS] FREE at text-tail '%.40s' (target rem '%.20s')\n",
                   startText.c_str() + (startText.size() > 40 ? startText.size() - 40 : 0),
                   target.c_str());
        return true;  // 自由
    }
    std::set<int> allowSet(r.allowed.begin(), r.allowed.end());
    std::set<int> blockSet(r.blocked.begin(), r.blocked.end());
    for (size_t tid = 0; tid < VOCAB.size(); tid++) {
        const std::string &tok = VOCAB[tid];
        if (tok.empty()) continue;
        if (target.compare(0, tok.size(), tok) != 0) continue;
        if (!r.allowed.empty() && !allowSet.count((int)tid)) continue;
        if (blockSet.count((int)tid)) continue;
        if (CanSpell(m, startText + tok, target.substr(tok.size()), cfg,
                     depth + 1)) {
            return true;
        }
    }
    return false;
}

static int fails = 0;
static void Check(const char *name, bool cond, const char *detail = "") {
    printf("[%s] %s %s\n", cond ? "PASS" : "FAIL", name, detail);
    if (!cond) fails++;
}

int main() {
    StubModel m;
    // 注入合成 vocab
    for (size_t i = 0; i < VOCAB.size(); i++) {
        m.weight.tokenizer.tokenToStringDict[(int)i] = VOCAB[i];
    }
    GenerationConfig cfg = MakeConfig();

    // ---- I1: 多参数块间隙(已写完 path 块) ----
    std::string P1 =
        "<tool_call>\n<function=list_dir>\n<parameter=path>/etc</parameter>";
    Check("I1.2 </function> 可拼出", CanSpell(m, P1, "</function>", cfg));
    Check("I1.3 <parameter=max_depth> 链路可拼出",
          CanSpell(m, P1, "<parameter=max_depth>", cfg));
    Check("I1.4 非 schema 名裸 pattern 不可拼",
          !CanSpell(m, P1, "pattern", cfg));
    Check("I1.5 裸 max_depth(无前缀)不可拼",
          !CanSpell(m, P1, "max_depth", cfg));

    // ---- I2: S2 -> S3 回转 ----
    std::string P2 = P1 + "\n<parameter=";
    Check("I2.2 schema 名 path 可拼", CanSpell(m, P2, "path>", cfg));
    Check("I2.3 schema 名 max_depth 可拼",
          CanSpell(m, P2, "max_depth>", cfg));
    Check("I2.4 pattern 不可拼(根因 A)",
          !CanSpell(m, P2, "pattern>", cfg));
    Check("I2.5 paath 不可拼", !CanSpell(m, P2, "paath>", cfg));
    Check("I2.6 maax_depth 不可拼", !CanSpell(m, P2, "maax_depth>", cfg));

    // ---- I3: 缺必填时 S2 不放行 </function>(修复 B) ----
    std::string P4 =
        "<tool_call>\n<function=list_dir>\n<parameter=max_depth>2</parameter>";
    Check("I3.2 缺 path 时 </function> 不可拼",
          !CanSpell(m, P4, "</function>", cfg));
    Check("I3.3 缺 path 时 <parameter=path> 可拼",
          CanSpell(m, P4, "<parameter=path>", cfg));
    std::string P5 = P1 + "\n<parameter=max_depth>2</parameter>";
    Check("I3.4 必填齐后 </function> 可拼",
          CanSpell(m, P5, "</function>", cfg));

    // ---- I4: 空值黑名单 ----
    std::string P6 = "<tool_call>\n<function=list_dir>\n<parameter=path>";
    Check("I4.2 空值 </parameter> 不可拼",
          !CanSpell(m, P6, "</parameter>", cfg));
    Check("I4.3 全空白值仍不可闭合",
          !CanSpell(m, P6, "  </parameter>", cfg));
    Check("I4.4 空值可写非空白值", CanSpell(m, P6, "/etc", cfg));
    Check("I4.5 非空值后 </parameter> 可拼",
          CanSpell(m, P6 + "/etc", "</parameter>", cfg));

    // ---- I5: 全流程逐步可生成 ----
    std::vector<std::string> seq = {
        "<tool_call>", "<function=", "list", "_dir", ">",
        "<parameter=", "path", ">", "/etc", "</parameter>",
        "<parameter=", "max", "_depth", ">", "2", "</parameter>",
        "</function>", "</tool_call>",
    };
    std::string acc;
    bool allOk = true;
    for (size_t i = 0; i + 1 < seq.size(); i++) {
        EvalResult r = Eval(m, acc, cfg);
        int tid = -1;
        for (size_t j = 0; j < VOCAB.size(); j++)
            if (VOCAB[j] == seq[i]) tid = (int)j;
        if (!r.allowed.empty()) {
            bool in = false;
            for (int id : r.allowed) if (id == tid) in = true;
            if (!in) {
                printf("[FAIL] I5 token 被拒 '%s' (acc 尾 %.30s)\n",
                       seq[i].c_str(), acc.c_str());
                allOk = false;
                break;
            }
        }
        for (int id : r.blocked) {
            if (id == tid) {
                printf("[FAIL] I5 token 被黑名单 '%s'\n", seq[i].c_str());
                allOk = false;
                break;
            }
        }
        if (!allOk) break;
        acc += seq[i];
    }
    Check("I5 全流程可生成", allOk);

    // ---- I7: 回归场景(循环形态) ----
    std::string loop = "<tool_call>\n<function=list_dir>\n"
        "<parameter=path>/etc</parameter>\n"
        "<parameter=path>/etc</parameter>\n"
        "<parameter=path>/etc</parameter>";
    Check("I7.1 重复闭合后 </function> 可达(可跳出)",
          CanSpell(m, loop, "</function>", cfg));
    std::string bad = P1 + "\n<parameter=";
    Check("I7.2 循环起点 pattern 不可拼",
          !CanSpell(m, bad, "pattern>", cfg));

    printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
