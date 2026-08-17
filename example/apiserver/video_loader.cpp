#include "video_loader.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
    constexpr size_t kMaxVideoBytes = 512u * 1024u * 1024u;  // data URL 载荷上限

    double GetEnvDouble(const char *name, double fallback) {
        const char *raw = getenv(name);
        if (raw == nullptr || *raw == '\0') {
            return fallback;
        }
        char *end = nullptr;
        double value = strtod(raw, &end);
        return (end != raw && value > 0.0) ? value : fallback;
    }

    int GetEnvInt(const char *name, int fallback) {
        const char *raw = getenv(name);
        if (raw == nullptr || *raw == '\0') {
            return fallback;
        }
        char *end = nullptr;
        long value = strtol(raw, &end, 10);
        return (end != raw && value > 0) ? (int) value : fallback;
    }

    bool DecodeBase64Loose(const std::string &encoded,
                           std::vector<uint8_t> &decoded,
                           std::string &error) {
        static const int8_t table[256] = {
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
            52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-2,-1,-1,
            -1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,
            15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
            -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
            41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
        };
        decoded.clear();
        if (encoded.empty() || encoded.size() % 4 != 0 ||
            encoded.size() > (kMaxVideoBytes * 4 / 3 + 8)) {
            error = "video data URL has invalid or oversized base64 payload";
            return false;
        }
        decoded.reserve(encoded.size() / 4 * 3);
        for (size_t offset = 0; offset < encoded.size(); offset += 4) {
            int values[4];
            for (int i = 0; i < 4; i++) {
                values[i] = table[(uint8_t) encoded[offset + i]];
            }
            const bool finalBlock = offset + 4 == encoded.size();
            if (values[0] < 0 || values[1] < 0 || values[2] == -1 ||
                values[3] == -1 || values[0] == -2 || values[1] == -2 ||
                (!finalBlock && (values[2] == -2 || values[3] == -2)) ||
                (values[2] == -2 && values[3] != -2)) {
                error = "video data URL contains invalid base64";
                decoded.clear();
                return false;
            }
            const uint32_t packed =
                ((uint32_t) values[0] << 18) |
                ((uint32_t) values[1] << 12) |
                ((uint32_t) std::max(0, values[2]) << 6) |
                (uint32_t) std::max(0, values[3]);
            decoded.push_back((uint8_t) (packed >> 16));
            if (values[2] != -2) {
                decoded.push_back((uint8_t) (packed >> 8));
            }
            if (values[3] != -2) {
                decoded.push_back((uint8_t) packed);
            }
        }
        return true;
    }

    // data: URL 载荷落临时文件,返回路径;其他 URL 原样返回(ffmpeg 直接读)。
    bool MaterializeVideoInput(const std::string &url,
                               std::string &path,
                               bool &isTempFile,
                               std::string &error) {
        isTempFile = false;
        if (url.rfind("data:", 0) != 0) {
            path = url;
            return true;
        }
        const size_t comma = url.find(',');
        if (comma == std::string::npos ||
            url.substr(0, comma).find(";base64") == std::string::npos) {
            error = "video data URL must use base64 encoding";
            return false;
        }
        std::vector<uint8_t> bytes;
        if (!DecodeBase64Loose(url.substr(comma + 1), bytes, error)) {
            return false;
        }
        char tmpl[] = "/tmp/fastllm_video_XXXXXX";
        int fd = mkstemp(tmpl);
        if (fd < 0) {
            error = "cannot create temp file for video payload";
            return false;
        }
        size_t written = 0;
        while (written < bytes.size()) {
            ssize_t n = write(fd, bytes.data() + written, bytes.size() - written);
            if (n <= 0) {
                close(fd);
                unlink(tmpl);
                error = "cannot write temp file for video payload";
                return false;
            }
            written += (size_t) n;
        }
        close(fd);
        path = tmpl;
        isTempFile = true;
        return true;
    }

    // fork+execvp 运行命令,捕获 stdout;避免 shell 注入。
    bool RunCaptureStdout(const std::vector<std::string> &argv,
                          std::vector<uint8_t> &out,
                          std::string &error) {
        int pipeFd[2];
        if (pipe(pipeFd) != 0) {
            error = "pipe() failed";
            return false;
        }
        pid_t pid = fork();
        if (pid < 0) {
            close(pipeFd[0]);
            close(pipeFd[1]);
            error = "fork() failed";
            return false;
        }
        if (pid == 0) {
            // 子进程跑系统 ffmpeg/ffprobe;apiserver 继承的
            // LD_LIBRARY_PATH(如 conda env lib)会让它们加载到
            // 不兼容的系统库(libpangoft2/libfontconfig 符号缺失)。
            // 仅清子进程环境,不影响 apiserver 自身。
            unsetenv("LD_LIBRARY_PATH");
            dup2(pipeFd[1], STDOUT_FILENO);
            // stderr 并入同一管道:ffprobe/ffmpeg 失败原因必须可见,
            // 之前丢进 /dev/null 导致线上只能看到 "exited abnormally"。
            dup2(pipeFd[1], STDERR_FILENO);
            close(pipeFd[0]);
            close(pipeFd[1]);
            std::vector<char*> args;
            for (const auto &s : argv) {
                args.push_back(const_cast<char*>(s.c_str()));
            }
            args.push_back(nullptr);
            execvp(args[0], args.data());
            _exit(127);
        }
        close(pipeFd[1]);
        out.clear();
        uint8_t buf[65536];
        while (true) {
            ssize_t n = read(pipeFd[0], buf, sizeof(buf));
            if (n == 0) {
                break;
            }
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                break;
            }
            out.insert(out.end(), buf, buf + n);
            if (out.size() > kMaxVideoBytes * 4) {
                break;  // 防爆:raw 帧最多 ~GB 级,超过视为异常
            }
        }
        close(pipeFd[0]);
        int status = 0;
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            error = argv[0] + " exited abnormally";
            if (!out.empty()) {
                // 附带子进程 stdout/stderr 尾部,便于定位失败原因
                size_t tailStart = out.size() > 384 ? out.size() - 384 : 0;
                error += ": ";
                error.append(reinterpret_cast<const char*>(out.data() + tailStart),
                             out.size() - tailStart);
            }
            return false;
        }
        return true;
    }

    bool ProbeVideoSize(const std::string &input,
                        int &width,
                        int &height,
                        std::string &error) {
        std::vector<uint8_t> out;
        if (!RunCaptureStdout(
                {"ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-of", "csv=p=0", input},
                out, error)) {
            error = "ffprobe failed: " + error;
            return false;
        }
        std::string text(out.begin(), out.end());
        int w = 0, h = 0;
        if (sscanf(text.c_str(), "%d,%d", &w, &h) != 2 || w <= 0 || h <= 0) {
            error = "ffprobe returned no usable video stream";
            return false;
        }
        width = w;
        height = h;
        return true;
    }
}

bool LoadOpenAIVideoUrl(const std::string &url,
                        OpenAIDecodedVideo &video,
                        std::string &error) {
    error.clear();
    const double fps = GetEnvDouble("FASTLLM_VIDEO_FPS", 2.0);
    const int maxFrames = GetEnvInt("FASTLLM_VIDEO_MAX_FRAMES", 32);
    const int maxEdge = GetEnvInt("FASTLLM_VIDEO_MAX_EDGE", 768);

    std::string input;
    bool isTempFile = false;
    if (!MaterializeVideoInput(url, input, isTempFile, error)) {
        return false;
    }
    struct TempGuard {
        const std::string &path;
        bool active;
        ~TempGuard() { if (active) unlink(path.c_str()); }
    } guard{input, isTempFile};

    int srcW = 0, srcH = 0;
    if (!ProbeVideoSize(input, srcW, srcH, error)) {
        return false;
    }

    // 帧长边压到 maxEdge 以内(等比),引擎侧再做 32 对齐的精确缩放。
    int dstW = srcW, dstH = srcH;
    if (std::max(srcW, srcH) > maxEdge) {
        const double scale = (double) maxEdge / (double) std::max(srcW, srcH);
        dstW = std::max(2, (int) (srcW * scale) & ~1);
        dstH = std::max(2, (int) (srcH * scale) & ~1);
    }

    char fpsArg[32];
    snprintf(fpsArg, sizeof(fpsArg), "fps=%.3f", fps);
    char framesArg[16];
    snprintf(framesArg, sizeof(framesArg), "%d", maxFrames);
    std::string vf = std::string(fpsArg);
    if (dstW != srcW || dstH != srcH) {
        char scaleArg[64];
        snprintf(scaleArg, sizeof(scaleArg), ",scale=%d:%d", dstW, dstH);
        vf += scaleArg;
    }

    std::vector<uint8_t> raw;
    if (!RunCaptureStdout(
            {"ffmpeg", "-nostdin", "-v", "error", "-i", input,
             "-vf", vf, "-frames:v", framesArg,
             "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"},
            raw, error)) {
        error = "ffmpeg frame extraction failed: " + error;
        return false;
    }

    const size_t frameBytes = (size_t) dstW * dstH * 3;
    if (frameBytes == 0 || raw.size() < frameBytes ||
        raw.size() % frameBytes != 0) {
        error = "ffmpeg produced no complete frames";
        return false;
    }
    const int frameCount = (int) (raw.size() / frameBytes);

    video.width = dstW;
    video.height = dstH;
    video.frameCount = frameCount;
    video.rgb.resize(raw.size());
    for (size_t i = 0; i < raw.size(); i++) {
        video.rgb[i] = (float) raw[i];
    }
    printf("[video] decoded %d frames %dx%d (src %dx%d, fps=%.2f, cap=%d)\n",
           frameCount, dstW, dstH, srcW, srcH, fps, maxFrames);
    fflush(stdout);
    return true;
}
