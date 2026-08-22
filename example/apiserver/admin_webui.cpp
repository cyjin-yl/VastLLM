// AdminWebUi 的实现。见 admin_webui.h 顶部的设计说明。
#include "admin_webui.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

namespace fastllm {
    namespace apiserver {

        namespace {

            // 常量时间比较防时序侧信道。
            bool ConstantTimeEqual(const std::string &left,
                                   const std::string &right) {
                std::size_t diff = left.size() ^ right.size();
                const std::size_t maxLen =
                    left.size() > right.size() ? left.size() : right.size();
                for (std::size_t i = 0; i < maxLen; i++) {
                    const unsigned char a =
                        i < left.size() ? (unsigned char)left[i] : 0u;
                    const unsigned char b =
                        i < right.size() ? (unsigned char)right[i] : 0u;
                    diff |= (unsigned char)(a ^ b);
                }
                return diff == 0;
            }

            // 大小写不敏感的 header 名匹配。
            bool HeaderNameEquals(const std::string &name,
                                  const char *expected) {
                if (name.size() != std::strlen(expected)) {
                    return false;
                }
                for (std::size_t i = 0; i < name.size(); i++) {
                    const char a = name[i] >= 'A' && name[i] <= 'Z'
                        ? (char)(name[i] - 'A' + 'a') : name[i];
                    const char b = expected[i] >= 'A' && expected[i] <= 'Z'
                        ? (char)(expected[i] - 'A' + 'a') : expected[i];
                    if (a != b) {
                        return false;
                    }
                }
                return true;
            }

            std::string TrimHeaderValue(const std::string &value) {
                std::size_t begin = 0;
                std::size_t end = value.size();
                while (begin < end && (value[begin] == ' ' ||
                                       value[begin] == '\t')) {
                    begin++;
                }
                while (end > begin && (value[end - 1] == ' ' ||
                                       value[end - 1] == '\t')) {
                    end--;
                }
                return value.substr(begin, end - begin);
            }

            // ---- profile 编辑辅助 ----

            // 判定值是否为布尔形态: 0/1/true/false/yes/no/on/off(大小写不限)。
            bool LooksBoolean(const std::string &value) {
                if (value == "0" || value == "1") {
                    return true;
                }
                std::string lowered;
                lowered.reserve(value.size());
                for (char ch : value) {
                    lowered += (char)std::tolower((unsigned char)ch);
                }
                return lowered == "true" || lowered == "false" ||
                       lowered == "yes" || lowered == "no" ||
                       lowered == "on" || lowered == "off";
            }

            // 行内注释起点: 前面带空格的 '#', 或行首 '#'。
            size_t FindInlineComment(const std::string &line) {
                for (size_t i = 1; i < line.size(); i++) {
                    if (line[i] == '#' && (line[i - 1] == ' ' ||
                                           line[i - 1] == '\t')) {
                        return i;
                    }
                }
                if (!line.empty() && line[0] == '#') {
                    return 0;
                }
                return std::string::npos;
            }

            // 剥掉成对引号。
            std::string StripQuotes(const std::string &value) {
                if (value.size() >= 2) {
                    const char first = value.front();
                    if ((first == '\'' || first == '"') &&
                        value.back() == first) {
                        return value.substr(1, value.size() - 2);
                    }
                }
                return value;
            }

            // 读文件全部行(去掉 \r)。
            bool ReadAllLines(const std::string &path,
                              std::vector<std::string> &lines) {
                lines.clear();
                std::ifstream file(path, std::ios::binary);
                if (!file) {
                    return false;
                }
                std::string line;
                while (std::getline(file, line)) {
                    if (!line.empty() && line.back() == '\r') {
                        line.pop_back();
                    }
                    lines.push_back(line);
                }
                return true;
            }

        }

        bool AdminWebUiCheckAuth(
                const std::unordered_map<std::string, std::string> &headers) {
            const char *configured = std::getenv("AUTH_TOKEN");
            const std::string expectedToken =
                configured == nullptr ? std::string() : std::string(configured);
            if (expectedToken.empty()) {
                return false;   // fail closed
            }
            const std::string bearerPrefix = "Bearer ";
            for (const auto &header : headers) {
                if (!HeaderNameEquals(header.first, "Authorization")) {
                    continue;
                }
                const std::string authorization = TrimHeaderValue(header.second);
                if (authorization.size() <= bearerPrefix.size() ||
                    authorization.compare(0, bearerPrefix.size(),
                                          bearerPrefix) != 0) {
                    return false;
                }
                return ConstantTimeEqual(authorization.substr(
                    bearerPrefix.size()), expectedToken);
            }
            return false;
        }

        bool AdminWebUiReadLogTail(const std::string &path,
                                   size_t maxLines, size_t maxBytes,
                                   std::string &out) {
            out.clear();
            if (maxLines == 0 || maxBytes == 0) {
                return false;
            }
            std::ifstream file(path, std::ios::binary);
            if (!file) {
                return false;
            }
            file.seekg(0, std::ios::end);
            const std::streamoff fileSize = file.tellg();
            if (fileSize < 0) {
                return false;
            }
            const std::streamoff readBytes =
                (std::streamoff)std::min<size_t>((size_t)fileSize, maxBytes);
            file.seekg(fileSize - readBytes, std::ios::beg);
            std::string data((size_t)readBytes, '\0');
            file.read(&data[0], readBytes);
            data.resize((size_t)file.gcount());
            size_t newlines = 0;
            std::size_t cutBegin = 0;
            for (std::size_t i = data.size(); i-- > 0;) {
                if (data[i] == '\n') {
                    newlines++;
                    if (newlines > maxLines) {
                        cutBegin = i + 1;
                        break;
                    }
                }
            }
            out = data.substr(cutBegin);
            if (!out.empty() && out.back() != '\n') {
                out += '\n';
            }
            return true;
        }

        std::vector<AdminProfileInfo> AdminWebUiListProfiles(
                const std::string &activeProxyLogFile) {
            std::vector<AdminProfileInfo> profiles;
            const char *projectDir = std::getenv("PROJECT_DIR");
            const char *overrideDir =
                std::getenv("FASTLLM_ADMIN_PROFILES_DIR");
            std::string dir;
            if (overrideDir != nullptr && overrideDir[0] != 0) {
                dir = overrideDir;
            } else if (projectDir != nullptr && projectDir[0] != 0) {
                dir = std::string(projectDir) +
                      "/runtime/fastllm-native-profiles";
            } else {
                return profiles;
            }
            DIR *handle = opendir(dir.c_str());
            if (handle == nullptr) {
                return profiles;
            }
            struct dirent *entry = nullptr;
            while ((entry = readdir(handle)) != nullptr) {
                const std::string name = entry->d_name;
                if (name.size() <= 4 ||
                    name.compare(name.size() - 4, 4, ".env") != 0) {
                    continue;
                }
                if (name[0] == '.' || name.rfind(".bak-", 0) == 0) {
                    continue;
                }
                AdminProfileInfo info;
                info.name = name;
                info.path = dir + "/" + name;
                struct stat st{};
                if (stat(info.path.c_str(), &st) == 0) {
                    info.mtime = (long long)st.st_mtime;
                }
                if (!activeProxyLogFile.empty()) {
                    std::string probe = name.substr(0, name.size() - 4);
                    const std::size_t dash = probe.find('-');
                    if (dash != std::string::npos &&
                        probe.compare(0, 4, "q38-") == 0) {
                        probe = probe.substr(dash + 1);
                    }
                    if (activeProxyLogFile.find(probe) !=
                            std::string::npos) {
                        info.active = true;
                    }
                }
                profiles.push_back(std::move(info));
            }
            closedir(handle);
            std::sort(profiles.begin(), profiles.end(),
                      [](const AdminProfileInfo &l, const AdminProfileInfo &r) {
                          return l.mtime > r.mtime;
                      });
            return profiles;
        }

        bool AdminWebUiParseProfile(const std::string &path,
                                    std::vector<AdminProfileKey> &keys,
                                    std::string &error) {
            keys.clear();
            error.clear();
            std::vector<std::string> lines;
            if (!ReadAllLines(path, lines)) {
                error = "cannot open profile: " + path;
                return false;
            }
            for (size_t i = 0; i < lines.size(); i++) {
                const std::string &raw = lines[i];
                const size_t firstNonWs = raw.find_first_not_of(" \t");
                if (firstNonWs == std::string::npos ||
                    raw[firstNonWs] == '#') {
                    continue;
                }
                const size_t eq = raw.find('=');
                if (eq == std::string::npos || eq == 0) {
                    continue;
                }
                AdminProfileKey entry;
                entry.key = raw.substr(0, eq);
                while (!entry.key.empty() &&
                       (entry.key.back() == ' ' || entry.key.back() == '\t')) {
                    entry.key.pop_back();
                }
                if (entry.key.empty()) {
                    continue;
                }
                const size_t valueBegin = eq + 1;
                const size_t commentPos = FindInlineComment(raw);
                const size_t valueEnd = commentPos == std::string::npos
                    ? raw.size() : commentPos;
                std::string rawValue = raw.substr(valueBegin,
                                                  valueEnd - valueBegin);
                while (!rawValue.empty() &&
                       (rawValue.back() == ' ' || rawValue.back() == '\t')) {
                    rawValue.pop_back();
                }
                entry.value = StripQuotes(rawValue);
                entry.boolean = LooksBoolean(entry.value);
                if (commentPos != std::string::npos) {
                    entry.comment = raw.substr(commentPos);
                }
                entry.lineIndex = i;
                keys.push_back(std::move(entry));
            }
            return true;
        }

        bool AdminWebUiWriteProfileKeys(
                const std::string &path,
                const std::vector<std::pair<std::string, std::string>> &updates,
                std::string &error) {
            error.clear();
            if (updates.empty()) {
                error = "no updates provided";
                return false;
            }
            std::vector<std::string> lines;
            if (!ReadAllLines(path, lines)) {
                error = "cannot open profile: " + path;
                return false;
            }
            std::vector<std::pair<std::string, std::string>> pending = updates;
            for (size_t i = 0; i < lines.size() && !pending.empty(); i++) {
                const std::string &raw = lines[i];
                const size_t firstNonWs = raw.find_first_not_of(" \t");
                if (firstNonWs == std::string::npos ||
                    raw[firstNonWs] == '#') {
                    continue;
                }
                const size_t eq = raw.find('=');
                if (eq == std::string::npos || eq == 0) {
                    continue;
                }
                std::string key = raw.substr(0, eq);
                while (!key.empty() &&
                       (key.back() == ' ' || key.back() == '\t')) {
                    key.pop_back();
                }
                for (size_t u = 0; u < pending.size(); u++) {
                    if (pending[u].first != key) {
                        continue;
                    }
                    const std::string &newValue = pending[u].second;
                    const size_t commentPos = FindInlineComment(raw);
                    const std::string comment = commentPos == std::string::npos
                        ? std::string() : raw.substr(commentPos);
                    std::string replacement = raw.substr(0, eq + 1);
                    const bool needsQuote =
                        newValue.empty() ||
                        newValue.find_first_of(" \t'\"#") != std::string::npos;
                    if (needsQuote) {
                        std::string escaped;
                        for (char ch : newValue) {
                            if (ch == '\'') {
                                escaped += "'\"'\"'";
                            } else {
                                escaped += ch;
                            }
                        }
                        replacement += "'" + escaped + "'";
                    } else {
                        replacement += newValue;
                    }
                    if (!comment.empty()) {
                        replacement += comment;
                    }
                    lines[i] = replacement;
                    pending.erase(pending.begin() + (long)u);
                    break;
                }
            }
            if (!pending.empty()) {
                lines.push_back("");
                for (const auto &entry : pending) {
                    lines.push_back(entry.first + "=" + entry.second);
                }
            }
            const std::string tempPath = path + ".tmp-" +
                std::to_string((long long)::getpid());
            {
                std::ofstream out(tempPath, std::ios::binary |
                                           std::ios::trunc);
                if (!out) {
                    error = "cannot create temp file: " + tempPath;
                    return false;
                }
                for (const std::string &l : lines) {
                    out << l << '\n';
                }
                out.flush();
                if (!out) {
                    error = "write failed: " + tempPath;
                    ::unlink(tempPath.c_str());
                    return false;
                }
            }
            if (::rename(tempPath.c_str(), path.c_str()) != 0) {
                error = std::string("rename failed: ") + strerror(errno);
                ::unlink(tempPath.c_str());
                return false;
            }
            return true;
        }

        bool AdminWebUiSpawnDetached(const std::string &command) {
            const pid_t pid = fork();
            if (pid < 0) {
                return false;
            }
            if (pid == 0) {
                setsid();
                const int devnull = open("/dev/null", O_RDWR);
                if (devnull >= 0) {
                    dup2(devnull, STDIN_FILENO);
                    dup2(devnull, STDOUT_FILENO);
                    dup2(devnull, STDERR_FILENO);
                    if (devnull > STDERR_FILENO) {
                        close(devnull);
                    }
                }
                execl("/bin/sh", "sh", "-c", command.c_str(), (char*)nullptr);
                _exit(127);   // execl 失败才到这里
            }
            return true;
        }
    }
}
