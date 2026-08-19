//
// Created by tylunasli on 8/8/25.
//

#include "utils.h"

// 【上游BUMP勿回退】预分词需要 Unicode 码点类别表, 见该头文件顶部说明。
#include "unicode_categories.h"

#include "fastllm.h"

#include <cstring>
#include <cmath>
#include <cfloat>
#include <thread>
#include <algorithm>

#ifdef PY_API
#include <pybind11/embed.h>
namespace py = pybind11;
#endif
namespace fastllm {

    Tokenizer::TrieNode::TrieNode() {
        this->tokenId = -999999;
        this->score = 0.0f;
    }

    Tokenizer::Tokenizer() {
        root = new TrieNode();
        int n = 0;
        wchar_t special_token = L'\x0';
        for (; special_token < L'!'; special_token++, n++) {
            byteCharDict[L'\x100' + n] = special_token;
            charByteDict[special_token] = L'\x100' + n;
        }
        for (special_token = L'\x7F'; special_token < L'\xA1'; special_token++, n++) {
            byteCharDict[L'\x100' + n] = special_token;
            charByteDict[special_token] = L'\x100' + n;
        }
        byteCharDict[L'\x100' + n++] = L'\xAD';
        charByteDict[L'\xAD'] = L'\x100' + (n - 1);
    }

    Tokenizer::~Tokenizer() {
        Clear();
        delete root;
    }

    void Tokenizer::Clear() {
        std::vector <TrieNode*> q;
        q.push_back(root);
        for (int i = 0; i < q.size(); i++) {
            TrieNode *now = q[i];
            for (auto it : now->next) {
                q.push_back(it.second);
            }
        }
        if (specialRoot != nullptr) {
            q.push_back(specialRoot);
            for (int i = q.size() - 1; i < q.size(); i++) {
                TrieNode *now = q[i];
                for (auto it : now->next) {
                    q.push_back(it.second);
                }
            }
        }
        for (TrieNode * node : q)
            delete node;
        q.clear();
        root = new TrieNode();
        specialRoot = nullptr;
        tokenToStringDict.clear();
        tokenToScoreDict.clear();
        stringToTokenDict.clear();
    }

    void Tokenizer::Insert(const std::string &s, int tokenId, float score) {
        TrieNode *now = this->root;
        for (int i = 0; i < s.size(); i++) {
            if (now->next.find(s[i]) == now->next.end()) {
                now->next[s[i]] = new TrieNode();
            }
            now = now->next[s[i]];
        }
        now->tokenId = tokenId;
        now->score = score;
        tokenToStringDict[tokenId] = s;
        tokenToScoreDict[tokenId] = score;
        stringToTokenDict[s] = tokenId;
    }

    void Tokenizer::SetSpecialTokens(const std::map<std::string, int>& specialTokenMap) {
        if (specialRoot == nullptr)
            specialRoot = new TrieNode();
        for (auto &it : specialTokenMap) {
            TrieNode *now = this->specialRoot;
            std::string normalized = Normalize(it.first, false);
            for (int i = 0; i < normalized.size(); i++) {
                if (now->next.find(normalized[i]) == now->next.end()) {
                    now->next[normalized[i]] = new TrieNode();
                }
                now = now->next[normalized[i]];
            }
            now->tokenId = it.second;
            now->score = 0.0f;
            tokenToStringDict[it.second] = it.first;
            stringToTokenDict[it.first] = it.second;
            specialTokens.push_back(it.first);
        }
    }

    void Tokenizer::SetTokenizerConfig(const json11::Json &config) {
        this->tokenizerConfig = config;
        if (config["chat_template"].is_string()) {
            this->chatTemplate = config["chat_template"].string_value();
        }
    }

    void Tokenizer::TryMergePairs(std::vector<Symbol> &symbols, int l, int r, std::priority_queue <SymbolPairs> &q) {
        if (l == -1 || r == -1 || symbols[l].len == 0 || symbols[r].len == 0) {
            return;
        }
        auto now = symbols[l].node;
        char *s = symbols[r].s;
        int pos = symbols[r].pos, len = symbols[r].len;
        for (int i = pos; i < pos + len; i++) {
            if (now->next.find(s[i]) != now->next.end()) {
                now = now->next[s[i]];
            } else {
                return;
            }
        }
        if (now->tokenId == -999999) {
            return;
        }
        q.push(SymbolPairs(now->score, l, r, symbols[l].len + symbols[r].len));
    }

    int Tokenizer::GetRank(std::vector <Symbol> &symbols, PartitionLinkNode *cur, int skip) {
        auto nxt = cur->Skip(skip + 2);
        if (nxt == nullptr) {
            return std::numeric_limits<int>::max();
        }
        auto s = symbols[0].s + symbols[0].pos;
        std::string key(s + cur->cur->first, s + nxt->cur->first);
        if (stringToTokenDict.find(key) != stringToTokenDict.end()) {
            return stringToTokenDict[key];
        }
        return std::numeric_limits<int>::max();
    }

    int Tokenizer::GetRank(std::vector<Symbol> &symbols,  std::vector<std::pair<int, int>> &partitions, int idx, int skip) {
        if (idx + skip + 2 >= partitions.size()) {
            return std::numeric_limits<int>::max();
        }
        auto s = symbols[0].s + symbols[0].pos;
        std::string key(s + partitions[idx].first, s + partitions[idx + skip + 2].first);
        if (stringToTokenDict.find(key) != stringToTokenDict.end()) {
            return stringToTokenDict[key];
        }
        return std::numeric_limits<int>::max();
    }

    std::string Tokenizer::Normalize(const std::string &ori, const bool addDummyPrefix) {
        if (this->byteAsChar) {
            std::wstring ws(ori.size(), L' ');
            for (int i=0; i < ori.length(); i++) {
                wchar_t wi = static_cast<wchar_t>(static_cast<unsigned char>(ori[i]));
                if (charByteDict.find(wi) != charByteDict.end()) {
                    wi = charByteDict[wi];
                }
                ws[i] = wi;
            }
            return converter.to_bytes(ws);
        }
        if (blankRepeatCount > 1) {
            std::string blankReplaced(ori);
            for (int c = blankRepeatCount; c > 1; c--) {
                std::string blank("<|blank_" + std::to_string(c) + "|>");
                size_t pos = 0;
                while ((pos = blankReplaced.find(std::string(c, ' '), pos)) != std::string::npos) {
                    blankReplaced.replace(pos, c, blank);
                    pos += blank.length();
                }
            }
        }
        std::string blank = "";
        blank += 226, blank += 150, blank += 129;
        std::string s = (addDummyPrefix && this->addDummyPrefix) ? blank : "";
        if (15 < ori.size() && ori.substr(0, 15) == "<FLM_FIX_TOKEN_") {
            s = "";
        }
        for (int i = 0; i < ori.size(); i++) {
            if (ori[i] == ' ') {
                if (!(this->removeExtraWhitespaces && i > 0 && ori[i - 1] == ' ')) {
                    s += blank;
                }
            } else if (!this->tokenizerConfig["replacements"].is_null()) {
                json11::Json replacement = tokenizerConfig["replacements"][ori.substr(i, 1)];
                if (replacement.is_string())
                    s += replacement.string_value();
            } else {
                s += ori[i];
            }
        }
        return s;
    }


    // ============ 【上游BUMP勿回退】预分词(pre-tokenizer)正则切分 ============
    //
    // 原来是什么:
    //   Tokenizer::Encode 只按 special token 把输入切开, 剩下的**整段**文本
    //   直接交给 Normalize + BytePairEncode。在 src/ include/ 里 grep
    //   "tokenizer.ggml.pre" 是**零命中** —— 这个 GGUF 键从来没有被读过。
    //
    // 为什么错:
    //   BPE 的语义是"在一个块(word)内部反复合并相邻符号"。HF tokenizers /
    //   llama.cpp / vLLM 都会先用 tokenizer.json 里的 pre-tokenizer 正则把文本
    //   切成块(一个单词、一个数字、一串标点、一段空白 ...), BPE 只在块内进行,
    //   合并**永远跨不出块边界**。本仓少了这一步, BPE 就能跨词边界、跨数字、
    //   跨标点任意合并, 产生训练语料里根本不存在的 token 序列。
    //   这种切分"合法但非规范": 解码回字符串一模一样, 所以任何字符串层面的
    //   自检都发现不了; 但模型拿到的是分布外的 token 序列 -> 逐字复述走样。
    //
    // 现场特征:
    //   修好 merges 排名(见 src/model.cpp 的同名标记)之后逐词正确率恢复,
    //   但整句 exact match 仍然只有 2/5。出错点集中在"字母紧挨数字/标点/
    //   连字符"以及长数字上, 生产实测:
    //     /home/ezra/Documents/Proto-UI  ->  /home/eze/Documents/PotouI
    //
    // 正确做法:
    //   读 GGUF 的 tokenizer.ggml.pre, 按对应正则先切块, 再逐块 BPE。
    //   本模型 pre = "qwen35", 正则取自 llama.cpp src/llama-vocab.cpp 的
    //   LLAMA_VOCAB_PRE_TYPE_QWEN35 分支:
    //     (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
    //     |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    //     |\p{N}
    //     | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
    //     |\s*[\r\n]+
    //     |\s+(?!\S)
    //     |\s+
    //   注意 \p{N} 这一支是**单个**数字成块 => 数字逐位切(12345 -> 1|2|3|4|5)。
    //   漏掉这条, 所有长数字的切分都会和 HF 不一样, 而且很难从字符串上看出来。
    //
    // 实现方式:
    //   std::regex **不支持** \p{L} / \p{M} / \p{N} 这类 Unicode 属性类
    //   (libstdc++ 直接不认), 所以不能拿 std::regex 硬套。这里照搬 llama.cpp
    //   的做法: 先把 UTF-8 解成码点, 查码点类别表
    //   (include/utils/unicode_categories.h, 数据从 llama.cpp
    //   src/unicode-data.cpp 抽取), 再用手写状态机做与正则**等价**的切分。
    //   状态机逐行移植自 llama.cpp src/unicode.cpp 的
    //   unicode_regex_split_custom_qwen35 / unicode_regex_split_custom_qwen2。
    //   **分支顺序不能调整**: 正则的 | 是有序选择(leftmost-alternation),
    //   换顺序就是换语义。
    //
    // 安全回退:
    //   pre 缺失或不认识 -> preTokenizerType 保持 PRE_TOKENIZER_NONE,
    //   PreTokenizeSplit 原样返回 {s}, 行为与上游逐字节一致, 并且**打印一行提示**。
    //   绝对不要改成静默忽略: 这个 bug 之所以拖了这么久, 就是因为它完全静默。
    //
    // 回归测试: test/pretokenizerTest.cpp (cmake --target testPreTokenizer)。

    namespace {

        // 码点 -> 类别位 的查表。0x110000 项 * 2 字节 = 2.2MB, 首次使用时构建。
        // 构建过程与 llama.cpp 的 unicode_cpt_flags_array() 一一对应:
        // 先按 [start_i, start_{i+1}) 铺开区间标志, 再叠加 WHITESPACE 位。
        const std::vector<uint16_t> &UnicodeCptFlagsTable() {
            static const std::vector<uint16_t> table = []() {
                std::vector<uint16_t> t((size_t)kUnicodeMaxCodepoints,
                                        (uint16_t)FASTLLM_CPT_UNDEFINED);
                for (size_t i = 1; i < kUnicodeRangesFlagsCount; i++) {
                    const uint32_t ini = kUnicodeRangesFlags[i - 1].start;
                    const uint32_t end = kUnicodeRangesFlags[i].start;
                    const uint16_t flags = kUnicodeRangesFlags[i - 1].flags;
                    for (uint32_t cpt = ini;
                         cpt < end && cpt < kUnicodeMaxCodepoints; cpt++) {
                        t[cpt] = flags;
                    }
                }
                for (size_t i = 0; i < kUnicodeWhitespaceCount; i++) {
                    t[kUnicodeWhitespace[i]] |= (uint16_t)FASTLLM_CPT_WHITESPACE;
                }
                return t;
            }();
            return table;
        }

        inline uint16_t UnicodeCptFlags(uint32_t cpt) {
            const std::vector<uint16_t> &t = UnicodeCptFlagsTable();
            return cpt < t.size() ? t[cpt] : (uint16_t)FASTLLM_CPT_UNDEFINED;
        }

        // qwen 系正则里的缩写分支写作 '[sS] / '[tT] / '[rR][eE] ... , 只涉及
        // ASCII 大小写。llama.cpp 那边调的是通用 unicode_tolower, 但核对过
        // 它的 unicode_map_lowercase(1433 条): 没有任何非 ASCII 码点会小写成
        // s/t/m/d/r/v/l/e, 所以 ASCII 版与上游**完全等价**。
        inline uint32_t AsciiToLower(uint32_t cpt) {
            return (cpt >= 'A' && cpt <= 'Z') ? cpt + 32 : cpt;
        }

        // UTF-8 -> 码点序列, 同时记下每个码点在原串里的**字节**起点
        // (byteOffsets 比 cpts 多一项, 末项 = s.size(), 方便取右开区间)。
        // 非法字节的处理与 llama.cpp unicode_cpts_from_utf8 一致:
        // 产出 U+FFFD 并只前进 1 字节 —— 这样"码点区间 <-> 字节区间"永远
        // 一一对应, 切块时不会丢字节、也不会把一个多字节字符劈成两半。
        void UnicodeCptsFromUtf8(const std::string &s,
                                 std::vector<uint32_t> &cpts,
                                 std::vector<size_t> &byteOffsets) {
            cpts.clear();
            byteOffsets.clear();
            cpts.reserve(s.size());
            byteOffsets.reserve(s.size() + 1);
            size_t offset = 0;
            while (offset < s.size()) {
                const uint8_t c0 = (uint8_t)s[offset];
                size_t len = 0;
                uint32_t cpt = 0;
                if ((c0 & 0x80) == 0x00) {
                    cpt = c0;
                    len = 1;
                } else if ((c0 & 0xE0) == 0xC0) {
                    if (offset + 1 < s.size() &&
                        ((uint8_t)s[offset + 1] & 0xC0) == 0x80) {
                        cpt = ((uint32_t)(c0 & 0x1F) << 6) |
                              ((uint32_t)((uint8_t)s[offset + 1] & 0x3F));
                        len = 2;
                    }
                } else if ((c0 & 0xF0) == 0xE0) {
                    if (offset + 2 < s.size() &&
                        ((uint8_t)s[offset + 1] & 0xC0) == 0x80 &&
                        ((uint8_t)s[offset + 2] & 0xC0) == 0x80) {
                        cpt = ((uint32_t)(c0 & 0x0F) << 12) |
                              ((uint32_t)((uint8_t)s[offset + 1] & 0x3F) << 6) |
                              ((uint32_t)((uint8_t)s[offset + 2] & 0x3F));
                        len = 3;
                    }
                } else if ((c0 & 0xF8) == 0xF0) {
                    if (offset + 3 < s.size() &&
                        ((uint8_t)s[offset + 1] & 0xC0) == 0x80 &&
                        ((uint8_t)s[offset + 2] & 0xC0) == 0x80 &&
                        ((uint8_t)s[offset + 3] & 0xC0) == 0x80) {
                        cpt = ((uint32_t)(c0 & 0x07) << 18) |
                              ((uint32_t)((uint8_t)s[offset + 1] & 0x3F) << 12) |
                              ((uint32_t)((uint8_t)s[offset + 2] & 0x3F) << 6) |
                              ((uint32_t)((uint8_t)s[offset + 3] & 0x3F));
                        len = 4;
                    }
                }
                if (len == 0) {   // 非法 UTF-8: 吐一个替换字符, 只吃 1 字节
                    cpt = 0xFFFD;
                    len = 1;
                }
                byteOffsets.push_back(offset);
                cpts.push_back(cpt);
                offset += len;
            }
            byteOffsets.push_back(s.size());
        }

        // 与 qwen2 / qwen35 预分词正则等价的手写状态机。
        // withAccentMark = true  -> qwen35: 字母串吃组合符, [\p{L}\p{M}]+
        // withAccentMark = false -> qwen2 : 字母串只吃 \p{L}+
        // 返回每个块的**码点长度**, 顺序即原文顺序, 总和 == cpts.size()。
        //
        // 移植自 llama.cpp src/unicode.cpp unicode_regex_split_custom_qwen35。
        // 上游是"对一串已有 offsets 逐段再切", 我们只有一条正则, 所以段就是全串。
        std::vector<size_t> PreTokenSplitQwen(const std::vector<uint32_t> &cpts,
                                              bool withAccentMark) {
            std::vector<size_t> offsets;
            offsets.reserve(cpts.size());

            const size_t offsetEnd = cpts.size();
            static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;

            auto getCpt = [&](size_t pos) -> uint32_t {
                return pos < offsetEnd ? cpts[pos] : OUT_OF_RANGE;
            };
            // 越界位置返回 0(所有位都不置) —— 上游用 unicode_cpt_flags{} 表示,
            // 下面 "flags != 0" 的判断正是靠这个区分"越界"和"未定义码点"。
            auto getFlags = [&](size_t pos) -> uint16_t {
                return pos < offsetEnd ? UnicodeCptFlags(cpts[pos]) : (uint16_t)0;
            };
            auto isLetterLike = [&](size_t pos) -> bool {
                const uint16_t f = getFlags(pos);
                return (f & FASTLLM_CPT_LETTER) != 0 ||
                       (withAccentMark && (f & FASTLLM_CPT_ACCENT_MARK) != 0);
            };
            // [^\s\p{L}\p{M}\p{N}] 的取反判定
            const uint16_t notPunctMask = (uint16_t)(
                FASTLLM_CPT_WHITESPACE | FASTLLM_CPT_LETTER | FASTLLM_CPT_NUMBER |
                (withAccentMark ? FASTLLM_CPT_ACCENT_MARK : 0));
            auto isPunctLike = [&](uint16_t f) -> bool {
                return (f & notPunctMask) == 0;
            };

            size_t prevEnd = 0;
            auto addToken = [&](size_t end) -> size_t {
                const size_t len = end - prevEnd;
                if (len > 0) {
                    offsets.push_back(len);
                }
                prevEnd = end;
                return len;
            };

            for (size_t pos = 0; pos < offsetEnd; /* pos 在各分支里推进 */) {
                const uint32_t cpt = getCpt(pos);
                const uint16_t flags = getFlags(pos);

                // regex: (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
                if (cpt == '\'' && pos + 1 < offsetEnd) {
                    const uint32_t n1 = AsciiToLower(getCpt(pos + 1));
                    if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                        pos += addToken(pos + 2);
                        continue;
                    }
                    if (pos + 2 < offsetEnd) {
                        const uint32_t n2 = AsciiToLower(getCpt(pos + 2));
                        if ((n1 == 'r' && n2 == 'e') ||
                            (n1 == 'v' && n2 == 'e') ||
                            (n1 == 'l' && n2 == 'l')) {
                            pos += addToken(pos + 3);
                            continue;
                        }
                    }
                }

                // regex: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
                // 允许前面挂一个"非换行非字母非数字"的引导字符, 这就是为什么
                // " hello" 会整块留在一起(前导空格属于单词)。
                if (!(cpt == '\r' || cpt == '\n' ||
                      (flags & FASTLLM_CPT_NUMBER) != 0)) {
                    if (isLetterLike(pos) || isLetterLike(pos + 1)) {
                        pos++;
                        while (isLetterLike(pos)) {
                            pos++;
                        }
                        addToken(pos);
                        continue;
                    }
                }

                // regex: \p{N}  —— **一个**数字一块, 不是 \p{N}+
                if ((flags & FASTLLM_CPT_NUMBER) != 0) {
                    pos++;
                    addToken(pos);
                    continue;
                }

                // regex: <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
                uint16_t flags2 = (cpt == ' ' ? getFlags(pos + 1) : flags);
                if (isPunctLike(flags2) && flags != 0) {
                    pos += (cpt == ' ') ? 1 : 0;
                    while (isPunctLike(flags2) && flags2 != 0) {
                        flags2 = getFlags(++pos);
                    }
                    uint32_t cpt2 = getCpt(pos);
                    while (cpt2 == '\r' || cpt2 == '\n') {
                        cpt2 = getCpt(++pos);
                    }
                    addToken(pos);
                    continue;
                }

                size_t numWhitespaces = 0;
                size_t lastEndRorN = 0;
                while ((getFlags(pos + numWhitespaces) & FASTLLM_CPT_WHITESPACE) != 0) {
                    const uint32_t cpt2 = getCpt(pos + numWhitespaces);
                    if (cpt2 == '\r' || cpt2 == '\n') {
                        lastEndRorN = pos + numWhitespaces + 1;
                    }
                    numWhitespaces++;
                }

                // regex: \s*[\r\n]+
                if (lastEndRorN > 0) {
                    pos = lastEndRorN;
                    addToken(pos);
                    continue;
                }

                // regex: \s+(?!\S)  —— 一串空白后面还跟着非空白时, 留最后一个
                // 空白给下一个块当"前导空格"
                if (numWhitespaces > 1 && getCpt(pos + numWhitespaces) != OUT_OF_RANGE) {
                    pos += numWhitespaces - 1;
                    addToken(pos);
                    continue;
                }

                // regex: \s+
                if (numWhitespaces > 0) {
                    pos += numWhitespaces;
                    addToken(pos);
                    continue;
                }

                // 所有分支都没命中: 单个码点自成一块
                addToken(++pos);
            }

            return offsets;
        }

        // 把一段"确定不含 <FLM_FIX_TOKEN_n>"的文本切块并追加到 out。
        void PreTokenizeAppendPlain(const std::string &seg,
                                    int preType,
                                    std::vector<std::string> &out) {
            std::vector<uint32_t> cpts;
            std::vector<size_t> byteOffsets;
            UnicodeCptsFromUtf8(seg, cpts, byteOffsets);
            const std::vector<size_t> lens = PreTokenSplitQwen(
                cpts, preType == Tokenizer::PreTokenizerType::PRE_TOKENIZER_QWEN35);
            size_t cur = 0;
            for (size_t len : lens) {
                const size_t b0 = byteOffsets[cur];
                const size_t b1 = byteOffsets[cur + len];
                out.push_back(seg.substr(b0, b1 - b0));
                cur += len;
            }
        }

    }  // namespace

    void Tokenizer::SetPreTokenizer(const std::string &pre) {
        // 【上游BUMP勿回退】pre 值到正则的映射抄自 llama.cpp src/llama-vocab.cpp
        // 里 tokenizer_pre -> LLAMA_VOCAB_PRE_TYPE_* 的那串 if-else。
        // 新增模型时照着上游补分支, 不要"先当成 qwen35 用着" —— 用错正则比
        // 不切分更难查。
        this->preTokenizerName = pre;
        if (pre == "qwen35") {
            this->preTokenizerType = PreTokenizerType::PRE_TOKENIZER_QWEN35;
            printf("Load tokenizer pre = %s (预分词正则已启用: qwen35)\n", pre.c_str());
        } else if (pre == "qwen2" || pre == "deepseek-r1-qwen" ||
                   pre == "kormo" || pre == "f2llmv2") {
            this->preTokenizerType = PreTokenizerType::PRE_TOKENIZER_QWEN2;
            printf("Load tokenizer pre = %s (预分词正则已启用: qwen2)\n", pre.c_str());
        } else if (pre.empty()) {
            this->preTokenizerType = PreTokenizerType::PRE_TOKENIZER_NONE;
            printf("Warning: GGUF 里没有 tokenizer.ggml.pre, 预分词切分未启用。\n"
                   "         BPE 将在整段文本上合并, 切分结果可能与 HF/llama.cpp 不一致,\n"
                   "         表现为逐字复述走样(路径/标识符拼错)。\n");
        } else {
            this->preTokenizerType = PreTokenizerType::PRE_TOKENIZER_NONE;
            printf("Warning: 不认识的 tokenizer.ggml.pre = \"%s\", 预分词切分未启用。\n"
                   "         BPE 将在整段文本上合并, 切分结果可能与 HF/llama.cpp 不一致。\n"
                   "         要支持它, 请照 llama.cpp src/unicode.cpp 的\n"
                   "         unicode_regex_split_custom_* 补一个分支。\n", pre.c_str());
        }
        fflush(stdout);
    }

    std::vector<std::string> Tokenizer::PreTokenizeSplit(const std::string &s) const {
        // 【上游BUMP勿回退】未启用预分词时必须原样返回整串 ——
        // 这样 Encode 里的调用点对老模型是逐字节等价的空操作。
        if (this->preTokenizerType == PreTokenizerType::PRE_TOKENIZER_NONE ||
            s.empty()) {
            return std::vector<std::string>(1, s);
        }

        // fastllm 内部用 <FLM_FIX_TOKEN_数字> 直接指定 token id(BytePairEncode
        // 里解析)。这串必须整块传下去, 被预分词切开就失效了, 所以先摘出来。
        std::vector<std::string> ret;
        const std::string fixMark = "<FLM_FIX_TOKEN_";
        size_t i = 0;
        while (i < s.size()) {
            size_t fixStart = s.find(fixMark, i);
            size_t fixEnd = std::string::npos;
            while (fixStart != std::string::npos) {
                size_t j = fixStart + fixMark.size();
                while (j < s.size() && s[j] >= '0' && s[j] <= '9') {
                    j++;
                }
                if (j > fixStart + fixMark.size() && j < s.size() && s[j] == '>') {
                    fixEnd = j + 1;
                    break;
                }
                fixStart = s.find(fixMark, fixStart + 1);
            }
            const size_t plainEnd = (fixStart == std::string::npos) ? s.size() : fixStart;
            if (plainEnd > i) {
                PreTokenizeAppendPlain(s.substr(i, plainEnd - i),
                                       (int)this->preTokenizerType, ret);
            }
            if (fixStart == std::string::npos) {
                break;
            }
            ret.push_back(s.substr(fixStart, fixEnd - fixStart));
            i = fixEnd;
        }
        return ret;
    }
    // ========== 【上游BUMP勿回退】预分词实现到此为止 ==========


    bool isDigitOrChar(char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    }

    std::vector<float> Tokenizer::UnigramEncode(const std::string &s) {
        // SymbolPairs.l 表示上一个位置
        // SymbolPairs.r 表示选择上一个位置的第几个TrieNode
        std::vector<TrieNode *> specialIds;
        std::vector<std::vector<TrieNode *>> lattice(s.size() + 1, std::vector<TrieNode *>());
        std::vector<std::vector<SymbolPairs>> latticeScores(s.size() + 1, std::vector<SymbolPairs>());
        for (int i = 0; i < s.size(); i++) {
            if (i + 3 < s.size() && s[i] == '<' && s[i + 1] == 'F' && s[i + 2] == 'L' && s[i + 3] == 'M') {
                if (i + 15 < s.size() && s.substr(i, 15) == "<FLM_FIX_TOKEN_") {
                    int start = i;
                    i += 15;
                    TrieNode *fixNode = new TrieNode();
                    fixNode->tokenId = 0;
                    fixNode->score = -0.1;
                    while (s[i] >= '0' && s[i] <= '9') {
                        fixNode->tokenId = fixNode->tokenId * 10 + s[i] - '0';
                        i++;
                    }
                    specialIds.push_back(fixNode);
                    lattice[start].push_back(fixNode);
                    latticeScores[start].push_back(SymbolPairs(0.F, i + 1, 0, i - start));
                    continue;
                }
            }
            if (this->specialRoot != nullptr) {
                TrieNode *now = this->specialRoot;
                int next = i;
                for (; next < s.size(); next++) {
                    if (now->next.find(s[next]) == now->next.end())
                        break;
                    now = now->next[s[next]];
                }
                if (now->tokenId != -999999 && next > i) {
                    lattice[i].push_back(now);
                    latticeScores[i].push_back(SymbolPairs(now->score, next, -1, next - i));
                    i = next - 1;
                    continue;
                }
            }

            TrieNode *now = this->root;
            for (int j = i; j < s.size(); j++) {
                if (now->next.find(s[j]) != now->next.end()) {
                    now = now->next[s[j]];
                    if (now->tokenId != -999999) {
                        lattice[i].push_back(now);
                        latticeScores[i].push_back(SymbolPairs(now->score, j + 1, -1, j - i + 1));
                    }
                } else {
                    break;
                }
            }
            if (latticeScores[i].empty()) {
                // 未识别的字符
                uint8_t c = (uint8_t) (s[i]);
                std::string now = "<0x00>";
                now[3] = (c / 16 > 9 ? ('A' + c / 16 - 10) : ('0' + c / 16));
                now[4] = (c % 16 > 9 ? ('A' + c % 16 - 10) : ('0' + c % 16));
                if (stringToTokenDict.find(now) != stringToTokenDict.end()) {
                    TrieNode *byte = new TrieNode();
                    byte->tokenId = stringToTokenDict[now];
                    byte->score = FLT_MAX - 10.0f;
                    specialIds.push_back(byte);
                    lattice[i].push_back(byte);
                    latticeScores[i].push_back(SymbolPairs(0.F, i + 1, -1, 1));
                }
            }
        }
        TrieNode *empty = new TrieNode();
        specialIds.push_back(empty);
        lattice[s.size()].push_back(empty);
        latticeScores[s.size()].push_back(SymbolPairs(0.F, s.size(), -1, 0));
        // viterbi 求解
        for (int i = 0; i < s.size(); i++) {
            for (int j = 0; j < latticeScores[i].size(); j++) {
                int jNext = i + latticeScores[i][j].size;
                for (int k = 0; k < latticeScores[jNext].size(); k++) {
                    float newScore = latticeScores[i][j].score + lattice[jNext][k]->score;
                    if (latticeScores[jNext][k].r == -1 || latticeScores[jNext][k].score < newScore) {
                        latticeScores[jNext][k].l = i;
                        latticeScores[jNext][k].r = j;
                        latticeScores[jNext][k].score = newScore;
                    }
                }
            }
        }
        std::vector<float> v;
        int pos = s.size();
        int row = latticeScores[s.size()][0].l, column = latticeScores[s.size()][0].r;
        while (column != -1) {
            SymbolPairs& node = latticeScores[row][column];
            v.push_back(lattice[row][column]->tokenId);
            row = node.l;
            column = node.r;
        }
        std::reverse(v.begin(), v.end());
        for (TrieNode * node : specialIds)
            delete node;
        return v;
    }

    std::vector<float> Tokenizer::BytePairEncode(const std::string &s) {
            std::vector<Symbol> symbols;
            for (int i = 0; i < s.size(); i++) {
                if (i + 3 < s.size() && s[i] == '<' && s[i + 1] == 'F' && s[i + 2] == 'L' && s[i + 3] == 'M') {
                    if (i + 15 < s.size() && s.substr(i, 15) == "<FLM_FIX_TOKEN_") {
                        i += 15;
                        int now = 0;
                        while (s[i] >= '0' && s[i] <= '9') {
                            now = now * 10 + s[i] - '0';
                            i++;
                        }
                        symbols.push_back(Symbol(nullptr, (char *) s.data(), i, 0, (int) symbols.size() - 1,
                                                 (int) symbols.size() + 1, now));
                        continue;
                    }
                }

                if (this->specialRoot != nullptr) {
                    TrieNode *now = this->specialRoot;
                    int next = i;
                    for (; next < s.size(); next++) {
                        if (now->next.find(s[next]) == now->next.end())
                            break;
                        now = now->next[s[next]];
                    }
                    if (now->tokenId != -999999 && next > i) {
                        symbols.push_back(Symbol(nullptr, (char *)s.data(), i, 0, (int) symbols.size() - 1,
                                          (int) symbols.size() + 1, now->tokenId));
                        i = next - 1;
                        continue;
                    }
                }

                int tokenId = -999999, pos = i - 1;
                TrieNode *now = this->root;
                for (int j = i; j < s.size(); j++) {
                    if (now->next.find(s[j]) != now->next.end()) {
                        now = now->next[s[j]];
                        if (now->tokenId != -999999) {
                            tokenId = now->tokenId;
                            pos = j;
                            break;
                        }
                    } else {
                        break;
                    }
                }
                if (pos >= i) {
                    symbols.push_back(Symbol(now, (char *) s.data(), i, pos - i + 1, (int) symbols.size() - 1,
                                             (int) symbols.size() + 1, -999999));
                    i = pos;
                } else {
                    symbols.push_back(Symbol(nullptr, (char *) s.data(), i, 0, (int) symbols.size() - 1,
                                             (int) symbols.size() + 1, -999999));
                }
            }
            symbols.back().next = -1;

            std::priority_queue<SymbolPairs> workQueue;
            for (int i = 1; i < symbols.size(); i++) {
                TryMergePairs(symbols, i - 1, i, workQueue);
            }

            while (!workQueue.empty()) {
                auto top = workQueue.top();
                workQueue.pop();
                if (symbols[top.l].len == 0 || symbols[top.r].len == 0 ||
                    symbols[top.l].len + symbols[top.r].len != top.size) {
                    continue;
                }

                for (int i = symbols[top.r].pos; i < symbols[top.r].pos + symbols[top.r].len; i++) {
                    symbols[top.l].node = symbols[top.l].node->next[symbols[top.r].s[i]];
                }
                symbols[top.l].len += symbols[top.r].len;
                symbols[top.r].len = 0;
                symbols[top.l].next = symbols[top.r].next;
                if (symbols[top.r].next >= 0) {
                    symbols[symbols[top.r].next].prev = top.l;
                }

                TryMergePairs(symbols, symbols[top.l].prev, top.l, workQueue);
                TryMergePairs(symbols, top.l, symbols[top.l].next, workQueue);
            }

            std::vector<float> v;
            for (int i = 0; i < symbols.size(); i++) {
                if (symbols[i].len > 0) {
                    v.push_back(symbols[i].node->tokenId);
                } else if (symbols[i].node == nullptr) {
                    if (symbols[i].fixId != -999999) {
                        v.push_back(symbols[i].fixId);
                    } else {
                        // 未识别的字符
                        uint8_t c = (uint8_t) (symbols[i].s[symbols[i].pos]);
                        std::string now = "<0x00>";
                        now[3] = (c / 16 > 9 ? ('A' + c / 16 - 10) : ('0' + c / 16));
                        now[4] = (c % 16 > 9 ? ('A' + c % 16 - 10) : ('0' + c % 16));
                        if (stringToTokenDict.find(now) != stringToTokenDict.end()) {
                            v.push_back(stringToTokenDict[now]);
                        }
                    }
                }
            }
            return v;
    }

    Data Tokenizer::Encode(const std::string &ori) {
        if (this->type == TokenizerType::GLM) {
            const std::map<std::string, int> glmSpecialTokens = {{"[MASK]", 50003}, {"<|startofpiece|>", 50006}, {"<|endofpiece|>", 50007}, {"[sMASK]", 50008}, {"[gMASK]", 50009}};
            if (this->specialTokens.empty())
                SetSpecialTokens(glmSpecialTokens);
        }
        if (this->type == TokenizerType::BPE || this->type == TokenizerType::GLM) {
            std::string &s = const_cast<std::string &>(ori);
            std::vector<float> v;
            if (!this->specialTokens.empty()) {
                int findPos = 0;
                while (findPos < (int)s.length()) {
                    int nextSpecialToken = -1;
                    int nextSpecialTokenPos = -1;
                    int nextSpecialTokenLen = -1;
                    for (auto &token : this->specialTokens) {
                        int ind = s.find(token, findPos);
                        if (ind >= 0 && (nextSpecialTokenPos < 0 || ind < nextSpecialTokenPos)) {
                            nextSpecialTokenPos = ind;
                            nextSpecialToken = stringToTokenDict[token];
                            nextSpecialTokenLen = token.length();
                        }
                    }
                    std::string subStr;
                    if (nextSpecialTokenPos < 0) {
                        subStr = s.substr(findPos);
                        findPos = s.length();
                    } else {
                        subStr = s.substr(findPos, nextSpecialTokenPos - findPos);
                        findPos = nextSpecialTokenPos + nextSpecialTokenLen;
                    }
                    if (subStr.length() > 0) {
#ifdef USE_SENTENCEPIECE
                        if (spProcessor != nullptr) {
                            std::vector<int> ids;
                            spProcessor->Encode(subStr, &ids);
                            for (int id : ids) {
                                v.push_back(id);
                            }
                        } else
#endif
                        {
                            // 【上游BUMP勿回退】这里必须先做**预分词切分**再 BPE。
                            // 上游是直接 BytePairEncode(Normalize(subStr)) ——
                            // 整段文本一起做 BPE, 合并能跨词/跨数字/跨标点,
                            // 切出来的 token 序列与 HF/llama.cpp 不同(合法但非规范),
                            // 模型因此在训练分布之外推理 -> 逐字复述走样
                            // (实测 /home/ezra/Documents/Proto-UI ->
                            //        /home/eze/Documents/PotouI)。
                            // PreTokenizeSplit 在未配置 tokenizer.ggml.pre 时
                            // 原样返回 {subStr}, 对老模型是等价空操作。
                            const std::vector<std::string> pieces =
                                PreTokenizeSplit(subStr);
                            for (const std::string &piece : pieces) {
                                if (piece.empty()) {
                                    continue;
                                }
                                std::string ns = Normalize(piece, false);
                                std::vector<float> &&subTokenIds = BytePairEncode(ns);
                                v.insert(v.end(), subTokenIds.begin(), subTokenIds.end());
                            }
                        }
                    }
                    if (nextSpecialTokenPos >= 0) {
                        v.push_back(nextSpecialToken);
                    }
                }
            } else {
                // 【上游BUMP勿回退】同上: 没有 special token 的路径也要先预分词。
                // 注意 addDummyPrefix 只对**第一块**生效 —— 它模拟的是
                // SentencePiece 在句首补一个空格, 每块都补会多出一堆空格。
                if (this->preTokenizerType == PreTokenizerType::PRE_TOKENIZER_NONE) {
                    std::string ns = Normalize(ori);
                    v = BytePairEncode(ns);
                } else {
                    bool firstPiece = true;
                    const std::vector<std::string> pieces = PreTokenizeSplit(ori);
                    for (const std::string &piece : pieces) {
                        if (piece.empty()) {
                            continue;
                        }
                        std::string ns = Normalize(piece, firstPiece);
                        firstPiece = false;
                        std::vector<float> &&subTokenIds = BytePairEncode(ns);
                        v.insert(v.end(), subTokenIds.begin(), subTokenIds.end());
                    }
                }
            }
            return Data (DataType::FLOAT32, {1, (int)v.size()}, v);
        } else if (this->type == TokenizerType::QWEN) {
            std::map<std::string, int> specialTokens = {{"<|im_start|>", 151644}, {"<|im_end|>", 151645}, {"<|endoftext|>", 151643}};
            for (const std::string &token : this->specialTokens) {
                auto tokenId = stringToTokenDict.find(token);
                if (tokenId != stringToTokenDict.end()) {
                    specialTokens[token] = tokenId->second;
                }
            }
            for (int i = 0; i < ori.size(); i++) {
                if (i + 3 < ori.size() && ori[i] == '<' && ori[i + 1] == 'F' && ori[i + 2] == 'L' && ori[i + 3] == 'M') {
                    if (i + 15 < ori.size() && ori.substr(i, 15) == "<FLM_FIX_TOKEN_") {
                        i += 15;
                        int now = 0;
                        while (ori[i] >= '0' && ori[i] <= '9') {
                            now = now * 10 + ori[i] - '0';
                            i++;
                        }
                        specialTokens["<FLM_FIX_TOKEN_" + std::to_string(now) + ">"] = now;
                        continue;
                    }
                }
            }
            
            // comment these special tokens for now
            // for (int i = 0; i < 205; i++) {
            //     specialTokens.insert("<|extra_" + std::to_string(i) + "|>");
            // }

            std::vector<std::pair<int, int>> sep;
            for (auto &token : specialTokens) {
                int pos = 0;
                while ((pos = ori.find(token.first, pos)) != std::string::npos) {
                    sep.push_back({pos, token.first.size()});
                    pos += token.first.size();
                }
            }
            sep.push_back({ori.size(), 1}); // use this to tokenize the last few words
            std::sort(sep.begin(), sep.end(), std::greater<std::pair<int, int>>());

            std::vector<Symbol> symbols;
            std::vector<float> v;

            for (int i = 0; i <= ori.size(); i++) {
                if (i == sep.back().first) {
                    if (!symbols.empty()) {
                        symbols.back().next = -1;
                        std::string cur = ori.substr(i - symbols.size(), symbols.size());
                        std::vector<std::pair<int, int>> partitions(symbols.size() + 1);
                        std::vector <PartitionLinkNode> nodes(symbols.size() + 1);
                        for (int j = 0; j <= (int) symbols.size(); j++) {
                            partitions[j] = std::make_pair(j, std::numeric_limits<int>::max());
                        }
                        for (int j = 0; j <= (int) symbols.size(); j++) {
                            nodes[j].cur = &partitions[j];
                            if (j > 0) {
                                nodes[j].prev = &nodes[j - 1];
                            }
                            if (j + 1 < nodes.size()) {
                                nodes[j].next = &nodes[j + 1];
                            }
                            nodes[j].id = j;
                        }
                        for (int j = 0; j < partitions.size() - 2; j++) {
                            partitions[j].second = GetRank(symbols, partitions, j, 0);
                        }
                        std::set <std::pair <int, int> > pq;
                        for (int j = 0; j < nodes.size(); j++) {
                            pq.insert(std::make_pair(nodes[j].cur->second, j));
                        }
                        int del = 0;
                        while (partitions.size() - del > 1) {
                            int min_rank = pq.begin()->first;
                            auto sel = &nodes[pq.begin()->second];

                            if (min_rank != std::numeric_limits<int>::max()) {
                                pq.erase(std::make_pair(sel->cur->second, sel->id));
                                sel->cur->second = GetRank(symbols, sel, 1);
                                pq.insert(std::make_pair(sel->cur->second, sel->id));
                                if (sel->prev != nullptr) {
                                    pq.erase(std::make_pair(sel->prev->cur->second, sel->prev->id));
                                    sel->prev->cur->second = GetRank(symbols, sel->prev, 1);
                                    pq.insert(std::make_pair(sel->prev->cur->second, sel->prev->id));
                                }
                                pq.erase(std::make_pair(sel->next->cur->second, sel->next->id));
                                sel->next = sel->next->next;
                                sel->next->prev = sel;
                                del++;
                            } else {
                                break;
                            }
                        }
                        auto it = &nodes[0];
                        while (it != nullptr && it->next != nullptr) {
                            std::string key = cur.substr(it->cur->first, it->next->cur->first - it->cur->first);
                            v.push_back((float) stringToTokenDict[key]);
                            it = it->next;
                        }
                        symbols.clear();
                    }

                    std::string special = ori.substr(sep.back().first, sep.back().second);
                    if (specialTokens.find(special) != specialTokens.end()) {
                        v.push_back(specialTokens[special]);
                    }

                    i += sep.back().second - 1;
                    sep.pop_back();

                    continue;
                }

                int tokenId = -999999, pos = i - 1;
                TrieNode *now = this->root;
                for (int j = i; j < ori.size(); j++) {
                    if (now->next.find(ori[j]) != now->next.end()) {
                        now = now->next[ori[j]];
                        if (now->tokenId != -999999) {
                            tokenId = now->tokenId;
                            pos = j;
                            break;
                        }
                    } else {
                        break;
                    }
                }
                if (pos >= i) {
                    symbols.push_back(Symbol(now, (char *) ori.data(), i, pos - i + 1, (int) symbols.size() - 1,
                                             (int) symbols.size() + 1, -999999));
                    i = pos;
                } else {
                    symbols.push_back(Symbol(nullptr, (char *) ori.data(), i, 0, (int) symbols.size() - 1,
                                             (int) symbols.size() + 1, -999999));
                }
            }

            return Data (DataType::FLOAT32, {1, (int)v.size()}, v);
        } else if (this->type == TokenizerType::BERT) {
            std::vector <float> v;
            for (int i = 0; i < ori.size(); i++) {
                int tokenId = -999999, pos = i - 1;
                TrieNode *now = this->root;

                if (i > 0 && isDigitOrChar(ori[i - 1]) && isDigitOrChar(ori[i])) {
                    now = now->next['#']->next['#'];
                }
                for (int j = i; j < ori.size(); j++) {
                    if (now->next.find(ori[j]) != now->next.end()) {
                        now = now->next[ori[j]];
                        if (now->tokenId != -999999) {
                            tokenId = now->tokenId;
                            pos = j;
                        }
                    } else {
                        break;
                    }
                }
                if (pos >= i) {
                    i = pos;
                    v.push_back(tokenId);
                }
            }

            return Data (DataType::FLOAT32, {1, (int)v.size()}, v);
        } else {
            std::vector <float> v;
            for (int i = 0; i < ori.size(); i++) {
                int tokenId = -999999, pos = i - 1;
                TrieNode *now = this->root;
                for (int j = i; j < ori.size(); j++) {
                    if (now->next.find(ori[j]) != now->next.end()) {
                        now = now->next[ori[j]];
                        if (now->tokenId != -999999) {
                            tokenId = now->tokenId;
                            pos = j;
                        }
                    } else {
                        break;
                    }
                }
                if (pos >= i) {
                    i = pos;
                    v.push_back(tokenId);
                }
            }

            return Data (DataType::FLOAT32, {1, (int)v.size()}, v);
        }
    }

    std::string Tokenizer::DecodeTokens(const std::vector<int> &tokens) {
        std::string ret = "";
        for (int i = 0; i < tokens.size(); i++) {
            std::string s = tokenToStringDict[tokens[i]];
            if (s.size() == 6 && s.substr(0, 3) == "<0x" && s.back() == '>') {
                int c = 0;
                for (int i = 3; i < 5; i++) {
                    c *= 16;
                    if (s[i] >= '0' && s[i] <= '9') {
                        c += (s[i] - '0');
                    } else {
                        c += (s[i] - 'A' + 10);
                    }
                }

                s = " ";
                s[0] = c;
            }
            if (s == "<n>") {
                ret += "\n";
            } else if (s == "<|tab|>") {
                ret += "\t";
            } else {
                ret += s;
            }
        }

        std::string blank = "";
        blank += 226, blank += 150, blank += 129;
        while (true) {
            std::string::size_type pos(0);
            if ((pos = ret.find(blank)) != std::string::npos)
                ret.replace(pos, blank.length(), " ");
            else break;
        }
        if (this->byteAsChar) {
            std::wstring wret = converter.from_bytes(ret);
            std::string decoded(wret.size(), ' ');
            for (int i=0; i < wret.length(); i++) {
                if (byteCharDict.find(wret[i]) != byteCharDict.end()) {
                    wret[i] = byteCharDict[wret[i]];
                }
                decoded[i] = static_cast<char>(wret[i]);
            }
            ret = decoded;
        }
        int pos = ret.find("<|blank_");
        if (pos != -1) {
            int space_num = atoi(ret.substr(8, ret.size() - 10).c_str());
            return std::string(space_num, ' ');
        }

        return ret;
    }

    std::string Tokenizer::Decode(const Data &data) {
        std::vector <int> tokens;
        for (int i = 0; i < data.Count(0); i++) {
            tokens.push_back((int) ((float *) data.cpuData)[i]);
        }
#ifdef USE_SENTENCEPIECE
        if (spProcessor != nullptr) {
            std::string result;
            spProcessor->Decode(tokens, &result);
            return result;
        }
#endif
        return DecodeTokens(tokens);
    }

    int Tokenizer::GetTokenId(const std::string &s) {
        AssertInFastLLM(stringToTokenDict.find(s) != stringToTokenDict.end(), 
                        "Tokenizer.GetTokenId error: can't find token \"" + s + "\"");
        return stringToTokenDict[s];
    }

    std::string Tokenizer::GetToken(int id) {
        AssertInFastLLM(tokenToStringDict.find(id) != tokenToStringDict.end(), 
                        "Tokenizer.GetToken error: can't find tokenid \"" + std::to_string(id) + "\"");
        return this->DecodeTokens(std::vector <int> {id}).c_str();
    }

}
