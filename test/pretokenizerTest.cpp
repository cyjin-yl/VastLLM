// 预分词(pre-tokenizer) + 分词的正确性回归测试。
//
// 【上游BUMP勿回退】这个测试是为"从上游 bump 之后能立刻知道预分词有没有被 bump 掉"
// 而写的, 不要删。它覆盖的线上故障是**静默**的: 分词结果解码回字符串一模一样,
// 只是 token 序列与训练时不同, 模型因此在分布外推理 —— "能编译 + 不崩 + 输出是
// 合法中文"完全判断不出对错, 必须把切分结果固化成断言。
//
// 覆盖的真实故障:
//   1. 从来没读过 GGUF 的 tokenizer.ggml.pre  -> 整段文本一起 BPE, 合并跨词/跨数字/跨标点
//      现场: /home/ezra/Documents/Proto-UI -> /home/eze/Documents/PotouI
//   2. \p{N} 是**单个数字**成块(数字逐位切), 漏掉这条所有长数字都会切错
//   3. qwen35 与 qwen2 的唯一差别: 字母串是否吃 \p{M}(组合符)
//
// 期望值的来源(两个互相独立的 oracle, 不是我自己算的):
//   - 切分期望: 用 Python `regex` 模块直接跑 qwen35 的原始正则得到
//       (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//       |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}
//       | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//   - token id 期望: llama.cpp 的 llama-tokenize 对同一份文本、同一个 GGUF 跑出来的
//       llama-tokenize -m <gguf> -f <text> --ids --no-bos --no-escape
//
// 构建: cmake -DUNIT_TEST=ON .. && cmake --build . --target testPreTokenizer
// 运行: ./testPreTokenizer [gguf路径] [--corpus <语料清单>]
//       不带参数时只跑"切分"部分(不需要模型), 内建 10 条用例全部硬编码;
//       带 GGUF 路径(或设置环境变量 FASTLLM_TEST_GGUF)时额外跑端到端 token id 对拍;
//       再带 --corpus 时对一份更大的语料做批量对拍, 并给出"关掉预分词 / 启用预分词"
//       两组的整句命中率(用来量化这次修复到底修好了多少)。
// 退出码非 0 = 有用例失败。
//
// 语料与基准文件不放在本仓(它们和具体 GGUF 的词表绑定), 放在:
//     /home/ezra/projects/EzraVastLLM/pretokenizer-corpus/
//       corpus/*.txt        45 篇语料(中英混排/路径/长数字/空白换行/缩写/代码/日志/长 prompt)
//       corpus/*.ids        llama-tokenize 跑出来的基准 token id
//       corpus_list.txt     喂给 --corpus 的清单(每行一个不带后缀的路径)
//       regen_golden.sh     换模型或换 llama.cpp 版本后重新生成基准
// 实测(2026-08-20, Qwen3.8-27B-Uncensored-Cyber-Q5_K_M-plus-mtp.gguf):
//     关掉预分词 33/45 篇整句一致  ->  启用预分词 45/45 篇整句一致

#include "fastllm.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool ok, const std::string &what) {
    g_checks++;
    if (ok) {
        printf("  ok   %s\n", what.c_str());
    } else {
        printf("  FAIL %s\n", what.c_str());
        g_failures++;
    }
}

// 把不可见字符转义出来, 失败信息才看得懂
std::string Visible(const std::string &s) {
    std::string r;
    for (size_t i = 0; i < s.size(); i++) {
        const unsigned char c = (unsigned char)s[i];
        if (c == '\n') r += "\\n";
        else if (c == '\r') r += "\\r";
        else if (c == '\t') r += "\\t";
        else r += (char)c;
    }
    return r;
}

std::string JoinPieces(const std::vector<std::string> &v) {
    std::string r = "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i) r += ", ";
        r += "\"" + Visible(v[i]) + "\"";
    }
    return r + "]";
}

std::string JoinIds(const std::vector<int> &v) {
    std::string r = "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i) r += ", ";
        r += std::to_string(v[i]);
    }
    return r + "]";
}

struct Case {
    const char *name;
    std::string text;
    std::vector<std::string> pieces;   // qwen35 预分词的期望切块
    std::vector<int> ids;              // llama-tokenize 的期望 token id
};

std::vector<Case> BuildCases() {
    std::vector<Case> c;
    // 01_basic
    c.push_back(Case{"01_basic", "Hello, world!",
        {"Hello", ",", " world", "!"},
        {9419, 11, 1814, 0}});
    // 02_path
    c.push_back(Case{"02_path", "/home/ezra/Documents/Proto-UI",
        {"/home", "/ezra", "/Documents", "/Proto", "-UI"},
        {17674, 14, 9825, 926, 52889, 14, 30510, 12, 2202}});
    // 03_cjk_mix
    c.push_back(Case{"03_cjk_mix", "\xe4\xb8\xad\xe8\x8b\xb1\xe6\xb7\xb7\xe6\x8e\x92 mixed \xe6\x96\x87\xe6\x9c\xac test 123 \xe7\xbb\x93\xe6\x9d\x9f",
        {"\xe4\xb8\xad\xe8\x8b\xb1\xe6\xb7\xb7\xe6\x8e\x92", " mixed", " \xe6\x96\x87\xe6\x9c\xac", " test", " ", "1", "2", "3", " \xe7\xbb\x93\xe6\x9d\x9f"},
        {124891, 97518, 96428, 9238, 220, 109120, 1228, 220, 16, 17, 18, 220, 98162}});
    // 04_digits
    c.push_back(Case{"04_digits", "1234567890 and 42 and 007",
        {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", " and", " ", "4", "2", " and", " ", "0", "0", "7"},
        {16, 17, 18, 19, 20, 21, 22, 23, 24, 15, 321, 220, 19, 17, 321, 220, 15, 15, 22}});
    // 05_whitespace
    c.push_back(Case{"05_whitespace", "a  b\n\nc\td   ",
        {"a", " ", " b", "\n\n", "c", "\td", "   "},
        {64, 220, 292, 271, 66, 2611, 262}});
    // 06_contract
    c.push_back(Case{"06_contract", "It's Bob's, they're, we've, I'm, we'll, I'd DON'T",
        {"It", "'s", " Bob", "'s", ",", " they", "'re", ",", " we", "'ve", ",", " I", "'m", ",", " we", "'ll", ",", " I", "'d", " DON", "'T"},
        {2064, 579, 13853, 579, 11, 781, 2224, 11, 567, 2908, 11, 353, 2688, 11, 567, 3172, 11, 353, 4035, 42803, 16813}});
    // 07_code
    c.push_back(Case{"07_code", "def foo_bar(x=1): return x*2  # comment",
        {"def", " foo", "_bar", "(x", "=", "1", "):", " return", " x", "*", "2", " ", " #", " comment"},
        {727, 14785, 13975, 2007, 28, 16, 1590, 460, 830, 9, 17, 220, 653, 3847}});
    // 08_words
    c.push_back(Case{"08_words", "Proto-UI prototypes brutalist",
        {"Proto", "-UI", " prototypes", " brutalist"},
        {30510, 12, 2202, 45070, 26860, 375}});
    // 09_accents
    c.push_back(Case{"09_accents", "cafe\xcc\x81 nai\xcc\x88ve \xc3\xa9l\xc3\xa8ve \xe0\xa4\x95\xe0\xa5\x8d\xe0\xa4\xb7 \xe0\xb8\x81\xe0\xb8\xb3",
        {"cafe\xcc\x81", " nai\xcc\x88ve", " \xc3\xa9l\xc3\xa8ve", " \xe0\xa4\x95\xe0\xa5\x8d\xe0\xa4\xb7", " \xe0\xb8\x81\xe0\xb8\xb3"},
        {895, 1795, 52033, 238883, 136, 230, 571, 30865, 75569, 165216, 156644}});
    // 10_mixnewline
    c.push_back(Case{"10_mixnewline", "line1\r\nline2\n\n   line3   \n",
        {"line", "1", "\r\n", "line", "2", "\n\n", "  ", " line", "3", "   \n"},
        {1021, 16, 317, 1021, 17, 271, 256, 1500, 18, 5690}});
    return c;
}


// ---------------------------------------------------------------- 用例 1: 切分

void TestPreTokenizeSplit(const std::vector<Case> &cases) {
    printf("[1] qwen35 预分词切分 (期望值来自 Python regex 跑原始正则)\n");
    fastllm::Tokenizer tok;
    tok.SetPreTokenizer("qwen35");
    for (const Case &cs : cases) {
        const std::vector<std::string> got = tok.PreTokenizeSplit(cs.text);
        const bool ok = (got == cs.pieces);
        Check(ok, std::string(cs.name) + " 切分 " + (ok ? JoinPieces(got)
              : (JoinPieces(got) + "  != 期望 " + JoinPieces(cs.pieces))));
        // 切块必须无损: 拼回去要和原文逐字节相同
        std::string joined;
        for (const std::string &p : got) joined += p;
        Check(joined == cs.text, std::string(cs.name) + " 切块拼回原文无损");
    }
}

// ---------------------------------------------------------------- 用例 2: 回退

void TestFallback() {
    printf("[2] 未知/缺失 pre 的安全回退: 保持不切分, 不静默改行为\n");
    const std::string text = "Hello, /home/ezra/Documents/Proto-UI 12345";

    fastllm::Tokenizer none;
    Check(none.preTokenizerType == fastllm::Tokenizer::PRE_TOKENIZER_NONE,
          "默认(从没调用过 SetPreTokenizer)是 PRE_TOKENIZER_NONE");
    Check(none.PreTokenizeSplit(text) == std::vector<std::string>(1, text),
          "默认不切分, 原样返回整串");

    fastllm::Tokenizer empty;
    empty.SetPreTokenizer("");
    Check(empty.preTokenizerType == fastllm::Tokenizer::PRE_TOKENIZER_NONE,
          "pre 缺失 -> PRE_TOKENIZER_NONE");
    Check(empty.PreTokenizeSplit(text) == std::vector<std::string>(1, text),
          "pre 缺失 -> 不切分");

    fastllm::Tokenizer unknown;
    unknown.SetPreTokenizer("some-pre-we-never-heard-of");
    Check(unknown.preTokenizerType == fastllm::Tokenizer::PRE_TOKENIZER_NONE,
          "未知 pre -> PRE_TOKENIZER_NONE");
    Check(unknown.PreTokenizeSplit(text) == std::vector<std::string>(1, text),
          "未知 pre -> 不切分");
    Check(unknown.preTokenizerName == "some-pre-we-never-heard-of",
          "未知 pre 的原值被记下来了(便于日志排查)");
}

// -------------------------------------------------- 用例 3: qwen2 与 qwen35 的差别

void TestQwen2VsQwen35() {
    printf("[3] qwen2 vs qwen35: 字母串是否吃 \\p{M} 组合符\n");
    // "e" + U+0301 COMBINING ACUTE ACCENT
    const std::string text = "cafe\xcc\x81 x";

    fastllm::Tokenizer q35;
    q35.SetPreTokenizer("qwen35");
    const std::vector<std::string> got35 = q35.PreTokenizeSplit(text);
    Check(got35 == std::vector<std::string>({"cafe\xcc\x81", " x"}),
          "qwen35: [\\p{L}\\p{M}]+ 把组合符并进单词 -> " + JoinPieces(got35));

    fastllm::Tokenizer q2;
    q2.SetPreTokenizer("qwen2");
    const std::vector<std::string> got2 = q2.PreTokenizeSplit(text);
    Check(got2 == std::vector<std::string>({"cafe", "\xcc\x81", " x"}),
          "qwen2: \\p{L}+ 不吃组合符, 组合符自成一块 -> " + JoinPieces(got2));
}

// -------------------------------------------- 用例 4: <FLM_FIX_TOKEN_n> 不能被切开

void TestFixTokenAtomic() {
    printf("[4] fastllm 内部的 <FLM_FIX_TOKEN_n> 必须整块保留\n");
    fastllm::Tokenizer tok;
    tok.SetPreTokenizer("qwen35");
    const std::vector<std::string> got =
        tok.PreTokenizeSplit("abc<FLM_FIX_TOKEN_12345>def");
    bool found = false;
    for (const std::string &p : got) {
        if (p == "<FLM_FIX_TOKEN_12345>") found = true;
    }
    Check(found, "<FLM_FIX_TOKEN_12345> 作为独立块存在 -> " + JoinPieces(got));
    std::string joined;
    for (const std::string &p : got) joined += p;
    Check(joined == "abc<FLM_FIX_TOKEN_12345>def", "拼回原文无损");
}

// ------------------------------------------ 用例 5: 端到端 token id 与 llama.cpp 对拍

std::vector<int> EncodeToIds(fastllm::Tokenizer &tok, const std::string &s) {
    fastllm::Data d = tok.Encode(s);
    std::vector<int> ids;
    for (int i = 0; i < (int)d.Count(0); i++) {
        ids.push_back((int)((float *)d.cpuData)[i]);
    }
    return ids;
}

// 完全照 src/model.cpp 里 GGUF 分支的做法搭一个 tokenizer(顺序也一样):
// merges 排名 -> score, tokens -> trie, token_type 3/4 -> special, byteAsChar,
// 最后设置 pre。usePre = false 用来复现"修复前"的行为做对照。
bool BuildTokenizerFromGGUF(const std::string &path, fastllm::Tokenizer &tok,
                            bool usePre, std::string &preName) {
    json11::Json config;
    fastllm::ReadGGUFMetaData(path, config);
    json11::Json params = config["params"];
    const auto &tokenItems = params["tokenizer.ggml.tokens"].array_items();
    if (tokenItems.empty()) {
        printf("  !! GGUF 里没有 tokenizer.ggml.tokens, 跳过\n");
        return false;
    }

    std::unordered_map<std::string, int> mergeRank;
    const auto &mergeItems = params["tokenizer.ggml.merges"].array_items();
    mergeRank.reserve(mergeItems.size() * 2);
    for (int mi = 0; mi < (int)mergeItems.size(); mi++) {
        const std::string &rule = mergeItems[mi].string_value();
        const size_t sep = rule.find(' ');
        if (sep == std::string::npos || sep == 0 || sep + 1 >= rule.size()) {
            continue;
        }
        const std::string merged = rule.substr(0, sep) + rule.substr(sep + 1);
        if (mergeRank.find(merged) == mergeRank.end()) {
            mergeRank[merged] = mi;
        }
    }

    int idx = 0;
    for (auto &it : tokenItems) {
        float score = 1.0f;
        if (!mergeRank.empty()) {
            auto r = mergeRank.find(it.string_value());
            score = r != mergeRank.end() ? -(float)r->second : -1e9f;
        }
        tok.Insert(it.string_value(), idx, score);
        idx++;
    }

    const auto &typeItems = params["tokenizer.ggml.token_type"].array_items();
    if (typeItems.size() == tokenItems.size()) {
        std::map<std::string, int> specials;
        for (int i = 0; i < (int)tokenItems.size(); i++) {
            const int t = typeItems[i].int_value();
            if (t == 3 || t == 4) {
                specials[tokenItems[i].string_value()] = i;
            }
        }
        if (!specials.empty()) {
            tok.SetSpecialTokens(specials);
        }
    }
    tok.byteAsChar = true;

    preName = params["tokenizer.ggml.pre"].is_string()
                  ? params["tokenizer.ggml.pre"].string_value()
                  : std::string();
    tok.SetPreTokenizer(usePre ? preName : std::string("__disabled__"));
    printf("  词表 %d 条, merges %d 条, pre = \"%s\"%s\n",
           (int)tokenItems.size(), (int)mergeItems.size(), preName.c_str(),
           usePre ? "" : "  (本次故意关掉预分词, 用于复现修复前的行为)");
    return true;
}

void TestAgainstLlamaCpp(const std::string &ggufPath,
                         const std::vector<Case> &cases) {
    printf("[5] 端到端 token id 对拍 llama.cpp (%s)\n", ggufPath.c_str());

    // ---- 对照组: 关掉预分词(= 本次修复之前的行为) ----
    int beforeMatch = 0;
    {
        fastllm::Tokenizer old;
        std::string preName;
        if (!BuildTokenizerFromGGUF(ggufPath, old, false, preName)) {
            return;
        }
        for (const Case &cs : cases) {
            if (EncodeToIds(old, cs.text) == cs.ids) beforeMatch++;
        }
    }

    // ---- 实验组: 按 GGUF 里的 pre 启用预分词 ----
    fastllm::Tokenizer tok;
    std::string preName;
    if (!BuildTokenizerFromGGUF(ggufPath, tok, true, preName)) {
        return;
    }
    int afterMatch = 0;
    for (const Case &cs : cases) {
        const std::vector<int> got = EncodeToIds(tok, cs.text);
        const bool ok = (got == cs.ids);
        if (ok) afterMatch++;
        Check(ok, std::string(cs.name) + " token id " +
              (ok ? JoinIds(got) : (JoinIds(got) + "  != llama.cpp " + JoinIds(cs.ids))));
    }
    printf("  整句 exact match: 关掉预分词 %d/%d  ->  启用预分词 %d/%d\n",
           beforeMatch, (int)cases.size(), afterMatch, (int)cases.size());
}


// ---------------------------- 批量对拍模式(可选): --corpus <目录>
//
// 目录里每个 X.txt 配一个 X.ids, 后者由 llama.cpp 生成:
//   llama-tokenize -m <gguf> -f X.txt --ids --no-bos --no-escape --log-disable | tail -1
// 用来在**更大的语料**上给出"修复前 / 修复后"的真实整句命中率,
// 10 条内建用例太短, 体现不出差距。
std::vector<int> ParseIds(const std::string &line) {
    std::vector<int> ids;
    size_t i = 0;
    while (i < line.size()) {
        if ((line[i] >= '0' && line[i] <= '9') || line[i] == '-') {
            size_t j = i;
            if (line[j] == '-') j++;
            while (j < line.size() && line[j] >= '0' && line[j] <= '9') j++;
            ids.push_back(atoi(line.substr(i, j - i).c_str()));
            i = j;
        } else {
            i++;
        }
    }
    return ids;
}

bool ReadWholeFile(const std::string &path, std::string &out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    char buf[65536];
    size_t n;
    out.clear();
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    fclose(f);
    return true;
}

void RunCorpus(const std::string &ggufPath, const std::string &listFile) {
    printf("[6] 批量对拍(语料清单 %s)\n", listFile.c_str());
    std::string list;
    if (!ReadWholeFile(listFile, list)) {
        printf("  !! 读不到清单文件\n");
        g_failures++;
        return;
    }
    std::vector<std::string> stems;
    size_t p0 = 0;
    while (p0 <= list.size()) {
        size_t p1 = list.find('\n', p0);
        if (p1 == std::string::npos) p1 = list.size();
        std::string line = list.substr(p0, p1 - p0);
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (!line.empty()) stems.push_back(line);
        if (p1 == list.size()) break;
        p0 = p1 + 1;
    }

    fastllm::Tokenizer off, on;
    std::string preName;
    if (!BuildTokenizerFromGGUF(ggufPath, off, false, preName)) return;
    if (!BuildTokenizerFromGGUF(ggufPath, on, true, preName)) return;

    int total = 0, offMatch = 0, onMatch = 0;
    long long offTokDiff = 0, onTokDiff = 0;
    for (const std::string &stem : stems) {
        std::string text, goldLine;
        if (!ReadWholeFile(stem + ".txt", text) ||
            !ReadWholeFile(stem + ".ids", goldLine)) {
            printf("  !! 缺文件: %s\n", stem.c_str());
            g_failures++;
            continue;
        }
        const std::vector<int> gold = ParseIds(goldLine);
        const std::vector<int> a = EncodeToIds(off, text);
        const std::vector<int> b = EncodeToIds(on, text);
        total++;
        if (a == gold) {
            offMatch++;
        } else {
            // 关掉预分词就对不上的篇目 —— 正是本次修复真正解决的那些
            printf("       (关掉预分词时不一致: %s, fastllm %d tok / llama.cpp %d tok)\n",
                   stem.c_str(), (int)a.size(), (int)gold.size());
        }
        if (b == gold) onMatch++;
        offTokDiff += (long long)a.size() - (long long)gold.size();
        onTokDiff += (long long)b.size() - (long long)gold.size();
        if (b != gold) {
            printf("  FAIL %s\n       fastllm  %s\n       llama.cpp %s\n",
                   stem.c_str(), JoinIds(b).c_str(), JoinIds(gold).c_str());
        }
    }
    printf("  语料 %d 篇: 关掉预分词 %d 篇整句一致, 启用预分词 %d 篇整句一致\n",
           total, offMatch, onMatch);
    printf("  token 数与 llama.cpp 的累计差: 关掉 %+lld, 启用 %+lld\n",
           offTokDiff, onTokDiff);
    Check(total > 0 && onMatch == total, "批量对拍: 每一篇都与 llama.cpp 逐 token 一致");
}

}  // namespace

int main(int argc, char **argv) {
    printf("==== 预分词 / 分词 正确性回归 ====\n");
    const std::vector<Case> cases = BuildCases();

    TestPreTokenizeSplit(cases);
    TestFallback();
    TestQwen2VsQwen35();
    TestFixTokenAtomic();

    std::string ggufPath;
    if (argc > 1) {
        ggufPath = argv[1];
    } else if (const char *env = getenv("FASTLLM_TEST_GGUF")) {
        ggufPath = env;
    }
    if (ggufPath.empty()) {
        printf("[5] 跳过端到端 token id 对拍: 没给 GGUF 路径\n"
               "    用法: ./testPreTokenizer <gguf路径>  或设置 FASTLLM_TEST_GGUF\n");
    } else {
        TestAgainstLlamaCpp(ggufPath, cases);
        for (int i = 1; i + 1 < argc; i++) {
            if (strcmp(argv[i], "--corpus") == 0) {
                RunCorpus(ggufPath, argv[i + 1]);
            }
        }
    }

    printf("---- %d/%d 通过 ----\n", g_checks - g_failures, g_checks);
    if (g_failures > 0) {
        printf("有 %d 项失败: 预分词被改坏了, 或者从上游 bump 时被回退了。\n"
               "回退的后果: BPE 在整段文本上合并, 切分与 HF/llama.cpp 不一致,\n"
               "            模型逐字复述走样(路径/标识符拼错), 而且**完全静默**。\n",
               g_failures);
    }
    return g_failures == 0 ? 0 : 1;
}
