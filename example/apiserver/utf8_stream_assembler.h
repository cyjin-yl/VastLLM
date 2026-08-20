#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

// Tokenizer::DecodeTokens may return raw byte fragments for byte-level BPE.
// A single token is therefore not guaranteed to be valid UTF-8 even when the
// concatenated token sequence is. Buffer only an incomplete trailing sequence;
// emit complete code points immediately so streaming latency is unchanged.
class Utf8StreamAssembler {
public:
    std::string Push(const std::string &bytes) {
        pending_ += bytes;
        return Drain(false);
    }

    std::string Flush() {
        return Drain(true);
    }

    size_t ReplacementCount() const {
        return replacementCount_;
    }

private:
    static int SequenceLength(uint8_t lead) {
        if (lead < 0x80) {
            return 1;
        }
        if (lead >= 0xC2 && lead <= 0xDF) {
            return 2;
        }
        if (lead >= 0xE0 && lead <= 0xEF) {
            return 3;
        }
        if (lead >= 0xF0 && lead <= 0xF4) {
            return 4;
        }
        return 0;
    }

    static bool IsContinuation(uint8_t byte) {
        return byte >= 0x80 && byte <= 0xBF;
    }

    static bool IsValidSequenceByte(
            uint8_t lead, uint8_t byte, int offset) {
        if (!IsContinuation(byte)) {
            return false;
        }
        if (offset != 1) {
            return true;
        }
        if (lead == 0xE0) {
            return byte >= 0xA0;
        }
        if (lead == 0xED) {
            return byte <= 0x9F;
        }
        if (lead == 0xF0) {
            return byte >= 0x90;
        }
        if (lead == 0xF4) {
            return byte <= 0x8F;
        }
        return true;
    }

    void AppendReplacement(std::string &output) {
        output.append("\xEF\xBF\xBD", 3);
        replacementCount_++;
    }

    std::string Drain(bool final) {
        std::string output;
        output.reserve(pending_.size());
        size_t offset = 0;
        while (offset < pending_.size()) {
            const uint8_t lead =
                static_cast<uint8_t>(pending_[offset]);
            const int length = SequenceLength(lead);
            if (length == 1) {
                output.push_back(pending_[offset++]);
                continue;
            }
            if (length == 0) {
                AppendReplacement(output);
                offset++;
                continue;
            }

            const size_t available = pending_.size() - offset;
            const size_t inspect = std::min<size_t>(available, length);
            bool validPrefix = true;
            for (size_t index = 1; index < inspect; index++) {
                if (!IsValidSequenceByte(
                        lead,
                        static_cast<uint8_t>(pending_[offset + index]),
                        static_cast<int>(index))) {
                    validPrefix = false;
                    break;
                }
            }
            if (!validPrefix) {
                AppendReplacement(output);
                offset++;
                continue;
            }
            if (available < static_cast<size_t>(length)) {
                if (!final) {
                    break;
                }
                AppendReplacement(output);
                offset = pending_.size();
                break;
            }

            output.append(pending_, offset, length);
            offset += length;
        }
        pending_.erase(0, offset);
        return output;
    }

    std::string pending_;
    size_t replacementCount_ = 0;
};
