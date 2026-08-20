// Provide by Jacques CHEN (http://whchen.net/index.php/About.html)
// HTML file reference from ChatGLM-MNN （https://github.com/wangzhaode/ChatGLM-MNN)

#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>
#include <stdlib.h>
#include <string>
#include <mutex>
#include <chrono>
#include <ctime>
#include <cstdarg>

// 带时间戳的日志行：[YYYY-MM-DD HH:MM:SS] ...
static void LogTs(const char *fmt, ...) {
    char ts[32];
    std::time_t t = std::time(nullptr);
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    printf("[%s] ", ts);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    fflush(stdout);
}

/*
 * Headers
 */

#ifdef _WIN32
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif //_CRT_SECURE_NO_WARNINGS

#ifndef _CRT_NONSTDC_NO_DEPRECATE
#define _CRT_NONSTDC_NO_DEPRECATE
#endif //_CRT_NONSTDC_NO_DEPRECATE

#if defined(_MSC_VER)
#if _MSC_VER < 1900
#error Sorry, Visual Studio versions prior to 2015 are not supported
#endif

#pragma comment(lib, "ws2_32.lib")

#ifdef _WIN64
using ssize_t = __int64;
#else
using ssize_t = long;
#endif
#endif // _MSC_VER

#ifndef S_ISREG
#define S_ISREG(m) (((m)&S_IFREG) == S_IFREG)
#endif // S_ISREG

#ifndef S_ISDIR
#define S_ISDIR(m) (((m)&S_IFDIR) == S_IFDIR)
#endif // S_ISDIR

#ifndef NOMINMAX
#define NOMINMAX
#endif // NOMINMAX

#include <io.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#ifndef WSA_FLAG_NO_HANDLE_INHERIT
#define WSA_FLAG_NO_HANDLE_INHERIT 0x80
#endif

#ifndef strcasecmp
#define strcasecmp _stricmp
#endif // strcasecmp

using socket_t = SOCKET;
#ifdef CPPHTTPLIB_USE_POLL
#define poll(fds, nfds, timeout) WSAPoll(fds, nfds, timeout)
#endif

#else // not _WIN32

#include <arpa/inet.h>
#ifndef _AIX
#include <ifaddrs.h>
#endif
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#ifdef __linux__
#include <resolv.h>
#endif
#include <netinet/tcp.h>
#ifdef CPPHTTPLIB_USE_POLL
#include <poll.h>
#endif
#include <csignal>
#include <pthread.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using socket_t = int;
#ifndef INVALID_SOCKET
#define INVALID_SOCKET (-1)
#endif
#endif //_WIN32

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>
#include <chrono>
#include "model.h"
#include "http_request_reader.h"
#include "http_response.h"
#include "socket_writer.h"
#include "openai_output_parser.h"
#include "output_token_limit.h"
#include "stop_parser.h"
#include "image_loader.h"
#include "video_loader.h"
#include "openai_multimodal_request.h"
#include "checkpoint_control.h"
#include "utils/stop_string_matcher.h"
#include "host_offload.h"
#include "fastllm-kernel-route.h"

class MultimodalInputGuard {
public:
    std::map<std::string, std::vector<fastllm::Data*> > inputs;

    ~MultimodalInputGuard() {
        if (!released) {
            for (auto &entry : inputs) {
                for (auto *tensor : entry.second) {
                    delete tensor;
                }
            }
        }
    }

    void Release() {
        released = true;
    }

private:
    bool released = false;
};

long long _GetCurrentTime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::seconds>(duration).count();
}

std::string GenerateRandomID() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    std::stringstream ss;
    for (int i = 0; i < 36; ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            ss << '-';
        }
        ss << std::hex << dis(gen);
    }
    return ss.str();
}

std::map <std::string, fastllm::DataType> dataTypeDict = {
    {"float32", fastllm::DataType::FLOAT32},
    {"half", fastllm::DataType::FLOAT16},
    {"float16", fastllm::DataType::FLOAT16},
    {"int8", fastllm::DataType::INT8},
    {"int4", fastllm::DataType::INT4_NOZERO},
    {"int4z", fastllm::DataType::INT4},
    {"int4g", fastllm::DataType::INT4_GROUP}
};

struct APIConfig {
    std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
    std::string modelName = "fastllm";
    std::string multimodalProjectorPath;

    int threads = 4; // 使用的线程数
    bool lowMemMode = false; // 是否使用低内存模式
    bool cudaEmbedding = false; // 是否使用cudaEmbedding
    int port = 8080; // 端口号
    int tokens = -1; // token容量限制
    int chunkedPrefillSize = -1; // 分块 prefill 切片大小；-1 使用模型默认值
    int batch = 256; // batch数限制
    int defaultMaxTokens = kDefaultOutputTokenLimit; // 请求省略max_tokens时的输出上限
    fastllm::DataType dtype = fastllm::DataType::FLOAT16;
    fastllm::DataType atype = fastllm::DataType::FLOAT32;
    fastllm::DataType kvCacheDtype = fastllm::DataType::DATA_AUTO_NONE;
    int groupCnt = -1;

    std::map <std::string, int> devices;
};
APIConfig config;

// vLLM 风格的滚动窗口吞吐统计:prefill 在首 token 时入账,decode 每 token
// 实时入账(长请求不遮罩),metrics 线程每 15s 读清。
// decode tok/s 口径 = 窗口 tokens / 窗口墙钟秒(同 vLLM Avg throughput)。
std::atomic<uint64_t> gWinPrefillTokens{0};
std::atomic<uint64_t> gWinDecodeTokens{0};
std::atomic<uint64_t> gWinPrefillUs{0};
std::atomic<uint64_t> gWinFinishedReqs{0};

static inline void NotePrefillDone(int promptTokens, double prefillMs) {
    gWinPrefillTokens.fetch_add((uint64_t) std::max(0, promptTokens),
                                std::memory_order_relaxed);
    gWinPrefillUs.fetch_add((uint64_t) std::max(0.0, prefillMs) * 1000,
                            std::memory_order_relaxed);
}


bool PrepareServerPersistentPrefixCache(
        fastllm::basellm *model) {
    fastllm::PersistentPrefixCacheStatus status;
    std::string error;
    const bool prepared =
        fastllm::PreparePersistentPrefixCacheFromEnv(
            model, status, &error);
    if (!prepared) {
        std::fprintf(
            stderr,
            "[Prefix-persist] restore skipped; cold start: %s\n",
            error.empty() ? "unknown error" : error.c_str());
        return false;
    }
    if (status.enabled) {
        std::fprintf(
            stderr,
            "[Prefix-persist] enabled, loaded generation %llu\n",
            (unsigned long long)status.loadedGeneration);
    }
    return true;
}

#ifndef _WIN32
volatile sig_atomic_t shutdownSignal = 0;
volatile sig_atomic_t serverSocketForSignal = -1;

void HandleShutdownSignal(int signalNumber) {
    shutdownSignal = signalNumber;
    const int socketFd = (int)serverSocketForSignal;
    serverSocketForSignal = -1;
    if (socketFd >= 0) {
        close(socketFd);
    }
}
#endif

void ToNext(char * &cur, const std::string &target, std::string &v) {
    v = "";
    while (*cur != 0) {
        bool stop = true;
        for (int i = 0; i < target.size(); i++) {
            if (cur[i] != target[i]) {
                stop = false;
                break;
            }
        }
        if (stop && target.size() > 0) {
            cur += target.size();
            break;
        } else {
            v += *(cur++);
        }
    }
}

struct HttpRequest {
    std::string method;
    std::string route;
    std::string type;
    std::unordered_map <std::string, std::string> headers;
    std::string body;

    void Init (char *buffer) {
        char *old = buffer;
        headers.clear();
        ToNext(buffer, " ", method);
        ToNext(buffer, " ", route);
        ToNext(buffer, "\r\n", type);
        while (true) {
            if (buffer[0] == 0 || ((long long)(buffer - old)) > 8 * 1024 * 1024) {
                break;
            }
            if (buffer[0] == '\r' && buffer[1] == '\n') {
                buffer += 2;
                ToNext(buffer, "", body);
                break;
            } else {
                std::string key;
                ToNext(buffer, ":", key);
                ToNext(buffer, "\r\n", headers[key]);
            }
        }
    }

    bool IsValid (char *buffer, int size) {
        char *old = buffer;
        headers.clear();
        ToNext(buffer, " ", method);
        ToNext(buffer, " ", route);
        ToNext(buffer, "\r\n", type);
        while (true) {
            if (buffer[0] == 0 || ((long long)(buffer - old)) > 8 * 1024 * 1024) {
                break;
            }
            if (buffer[0] == '\r' && buffer[1] == '\n') {
                if (headers.find("Content-Length") != headers.end()) {
                    if (size - ((long long)(buffer - old)) - 2 >= atoi(headers["Content-Length"].c_str())) {
                        return true;
                    } else {
                        return false;
                    }
                }
            } else {
                std::string key;
                ToNext(buffer, ":", key);
                ToNext(buffer, "\r\n", headers[key]);
            }
        }
        return false;
    }

    void Print() {
        for (auto &it : headers) {
            printf("%s: %s\n", it.first.c_str(), it.second.c_str());
        }
        printf("body: %s\n", body.c_str());
    }
} httpChecker;

std::string TrimHttpHeaderValue(const std::string &value) {
    size_t begin = 0;
    while (begin < value.size() &&
           std::isspace((unsigned char)value[begin])) {
        begin++;
    }
    size_t end = value.size();
    while (end > begin &&
           std::isspace((unsigned char)value[end - 1])) {
        end--;
    }
    return value.substr(begin, end - begin);
}

bool HttpHeaderNameEquals(const std::string &left,
                          const char *right) {
    const size_t rightSize = std::strlen(right);
    if (left.size() != rightSize) {
        return false;
    }
    for (size_t i = 0; i < rightSize; i++) {
        if (std::tolower((unsigned char)left[i]) !=
            std::tolower((unsigned char)right[i])) {
            return false;
        }
    }
    return true;
}



struct WorkNode {
    int client;
    HttpRequest request;
    json11::Json config;
    std::string error;

    void Init(char *buffer, int client) {
        this->client = client;
        request.Init(buffer);
        config = json11::Json::parse(request.body, this->error);
    }
};

// close 后置 -1：请求线程异常退出路径需要知道 socket 是否已关闭,
// 避免对已复用的 fd 二次 close。
static void CloseNodeClient(WorkNode *node) {
    if (node != nullptr && node->client >= 0) {
        // 这里原本写成了 CloseNodeClient(node) —— 自递归且 node->client 不变,
        // 守卫永远成立 => 无限递归爆栈 => SIGSEGV。ad019956 把各处的
        // close(node->client) 重构进本函数时写错了。生产上三次段错误的
        // backtrace 全是本函数的无限自调用。
        close(node->client);
        node->client = -1;
    }
}

// 健康检查/元数据类路由: 不碰模型、不做推理, 因此不该占用推理并发额度。
//
// 为什么需要区分(2026-08-20 生产事故的放大器):
//   maxActivateQueryNumber = min(256, --batch), 生产是 --batch 1 => 1。
//   派发闸门 activateQueryNumber < maxActivateQueryNumber 对**所有**路由生效,
//   于是只要有一个请求在生成, /health 和 /version 就一直排队, 客户端超时。
//   后果: 上游代理无法区分"后端在忙"与"后端已僵死" —— 本次 MTPLoop 自死锁
//   期间, 代理始终显示 backend=READY, 故障因此拖了一个小时才被发现。
// 注意不能把 /admin/* 放进来: 它们会 suspend/resume 模型, 必须串行。
static bool IsLightweightRoute(const std::string &rawRoute) {
    std::string route = rawRoute;
    if (route.size() > 1 && route.back() == '/') {
        route.pop_back();
    }
    return route == "/health" || route == "/version" ||
           route == "/props" || route == "/config.json";
}

struct WorkQueue {
    std::unique_ptr<fastllm::basellm> model;
    int maxActivateQueryNumber = 256;
    int activateQueryNumber = 0;
    // 轻量路由走独立队列, 否则会被队头的推理请求挡住(head-of-line blocking)
    std::queue <WorkNode*> lightQ;
    int activateLightNumber = 0;
    int totalQueryNumber = 0;
    std::mutex locker;
    std::condition_variable cv;
    std::queue <WorkNode*> q;
    std::thread *loop = nullptr;
    bool checkpointInProgress = false;
    bool stopping = false;
    std::atomic<uint64_t> cacheMutationEpoch {0};
    std::atomic<bool> pagedManagersPreallocated {false};
    std::atomic<uint64_t> checkpointedMutationEpoch {0};
    // 模型级 suspend/resume：suspend 释放全部 GPU 内存（进程常驻），
    // resume 从 GGUF + 磁盘前缀缓存重建模型。执行期间禁止派发新请求。
    std::atomic<bool> suspended {false};
    std::atomic<bool> suspendInProgress {false};
    std::atomic<bool> resumeInProgress {false};
    std::unique_ptr<fastllm::HostOffloadManager> hostOffloadManager;
    uint64_t hostOffloadGeneration = 0;
    std::string suspendedTier = "disk";

    void ConfigureHostOffloadManager() {
        hostOffloadManager.reset();
        if (model == nullptr ||
            !fastllm::HostCacheBudget::SharedBudgetEnabled() ||
            model->GetWeightMaterializationPlan().Size() == 0) {
            return;
        }
        hostOffloadManager.reset(new fastllm::HostOffloadManager(
            model->weight.weight,
            model->GetWeightMaterializationPlan(),
            fastllm::HostCacheBudget::Global(),
            [this](const std::vector<std::string> &names,
                   std::string *error) {
                return model != nullptr &&
                    model->ReloadGGUFWeightSubset(names, error);
            }));
    }

    void Push(char *buffer, int client) {
        WorkNode *node = new WorkNode();
        node->Init(buffer, client);
        const bool light = IsLightweightRoute(node->request.route);
        locker.lock();
        if (light) {
            lightQ.push(node);
        } else {
            q.push(node);
        }
        locker.unlock();

        cv.notify_all();
    }

    bool BeginExclusiveCheckpoint(std::string &reason) {
        std::lock_guard<std::mutex> lock(locker);
        if (checkpointInProgress) {
            reason = "a prefix-cache checkpoint is already running";
            return false;
        }
        if (activateQueryNumber != 1 || !q.empty()) {
            reason =
                "prefix-cache checkpoint requires no active or queued requests";
            return false;
        }
        checkpointInProgress = true;
        return true;
    }

    void EndExclusiveCheckpoint() {
        {
            std::lock_guard<std::mutex> lock(locker);
            checkpointInProgress = false;
        }
        cv.notify_all();
    }

    void Start() {
        {
            std::lock_guard<std::mutex> lock(locker);
            stopping = false;
        }
        // vLLM 风格周期指标:每 15s 打印一次窗口吞吐 + 当前队列状态,
        // 完全静默期(无活动无队列)不刷屏。
        std::thread([this]() {
            while (true) {
                std::this_thread::sleep_for(std::chrono::seconds(15));
                if (this->stopping) {
                    return;
                }
                const uint64_t pt = gWinPrefillTokens.exchange(0);
                const uint64_t dt = gWinDecodeTokens.exchange(0);
                const uint64_t pus = gWinPrefillUs.exchange(0);
                const uint64_t fin = gWinFinishedReqs.exchange(0);
                int active, queued, total;
                {
                    std::lock_guard<std::mutex> lock(this->locker);
                    active = this->activateQueryNumber;
                    queued = (int) this->q.size();
                    total = this->totalQueryNumber;
                }
                if (fin == 0 && active == 0 && queued == 0) {
                    continue;
                }
                // 缓存占用:L1=内存 trie 前缀页,L2=磁盘持久缓存;kv_pool=页池水位
                std::string cacheDesc;
                if (!suspended.load(std::memory_order_acquire) &&
                    !suspendInProgress.load(std::memory_order_acquire) &&
                    model != nullptr) {
                    uint64_t totalPages = 0, usedPages = 0, triePages = 0;
                    uint64_t logicalMaxPages = 0;
                    int pageLen = 0;
                    model->GetPagedCachePoolStats(
                        totalPages, usedPages, triePages, pageLen,
                        &logicalMaxPages);
                    char buf[256];
                    // 【上游BUMP勿回退】kv_pool 的分母是**已分配的物理页**
                    // (dims[0] 之和), budget 才是 --tokens 换算出的逻辑上限
                    // (maxPages 之和)。两者必须分开显示:
                    // 页池是懒分配的(初始 min(128, maxPages), 之后 Grow),
                    // 早期 total 远小于 budget 是**正常**的, 不是"池子快满了"。
                    // 混用这两个口径曾让空载被读成 94% 占用(见 basellm.cpp
                    // GetPagedCachePoolStats 的长注释)。
                    std::snprintf(buf, sizeof(buf),
                        " | kv_pool=%llu/%llu pg (%.0f%%, budget=%llu pg) "
                        "L1trie=%llu pg (~%llu tok)",
                        (unsigned long long) usedPages,
                        (unsigned long long) totalPages,
                        totalPages > 0 ?
                            100.0 * (double) usedPages / (double) totalPages :
                            0.0,
                        (unsigned long long) logicalMaxPages,
                        (unsigned long long) triePages,
                        (unsigned long long) (triePages *
                            (uint64_t) std::max(0, pageLen)));
                    cacheDesc = buf;
                    // 显存对账: nvidia-smi 的"已用"要能被下面几项解释,
                    // 剩下的 other 就是还没找到主的部分(线上曾有 7.6GB 无主)。
                    // alloc_free 是 fastllm 自己缓存分配器攥着的空闲块 ——
                    // 对 CUDA 而言它仍算"已用", 所以会出现"手里闲着几 GB
                    // 却报显存不足"。
                    fastllm::VramBreakdown vram;
                    if (fastllm::GetVramBreakdown(vram)) {
                        const long long otherB =
                            (long long)vram.usedBytes -
                            (long long)vram.pagedPoolBytes -
                            (long long)vram.allocBusyBytes -
                            (long long)vram.allocFreeBytes;
                        std::snprintf(buf, sizeof(buf),
                            " | vram=%llu/%lluMB(pool=%llu alloc_busy=%llu "
                            "alloc_free=%llu graph_pin=%llu other=%lld)",
                            (unsigned long long)(vram.usedBytes / 1048576),
                            (unsigned long long)(vram.totalBytes / 1048576),
                            (unsigned long long)(vram.pagedPoolBytes / 1048576),
                            (unsigned long long)(vram.allocBusyBytes / 1048576),
                            (unsigned long long)(vram.allocFreeBytes / 1048576),
                            (unsigned long long)(vram.graphPinnedBytes / 1048576),
                            (long long)(otherB / 1048576));
                        cacheDesc += buf;
                    }
                    const auto disk =
                        fastllm::GetPersistentPrefixCacheStatus();
                    if (disk.enabled) {
                        std::snprintf(buf, sizeof(buf),
                            " L2disk=%.1fMB gen=%llu ckpt=%llu hits=%llu",
                            (double) disk.payloadBytes / 1048576.0,
                            (unsigned long long) disk.loadedGeneration,
                            (unsigned long long) disk.checkpointCount,
                            (unsigned long long) disk.restoreHitCount);
                        cacheDesc += buf;
                    }
                }
                printf("[metrics] running=%d pending=%d (total=%d) | "
                       "prefill %llu tok (%.1f tok/s) | "
                       "decode %llu tok (%.1f tok/s) | done %llu req%s\n",
                       active, queued, total,
                       (unsigned long long) pt,
                       pus > 0 ? (double) pt * 1e6 / (double) pus : 0.0,
                       (unsigned long long) dt,
                       (double) dt / 15.0,  // 窗口墙钟口径(同 vLLM)
                       (unsigned long long) fin, cacheDesc.c_str());
                // 算子路由普查。每类算子有多份实现, 靠一串 if 静默选路;
                // 不打出来就只能靠读代码猜"到底走了哪条", 已经猜错过两次
                // (见 include/fastllm-kernel-route.h 的说明)。
                // 累计口径(不是 15 秒窗口口径), 只在显式打开明细统计时刷日志,
                // 免得占版面; 常态下用 curl /props | jq .kernel_routes 看。
                if (fastllm::KernelRouteShapeStatsEnabled()) {
                    printf("%s\n", fastllm::FormatKernelRouteCensus().c_str());
                }
                fflush(stdout);
            }
        }).detach();
        loop = new std::thread ([] (WorkQueue *ts) {
            while (true) {
                std::unique_lock<std::mutex> lock(ts->locker);
                ts->cv.wait(lock, [ts]() {
                    const bool canDispatch =
                        !ts->checkpointInProgress &&
                        !ts->suspendInProgress &&
                        !ts->resumeInProgress &&
                        ts->activateQueryNumber <
                            ts->maxActivateQueryNumber &&
                        !ts->q.empty();
                    // 轻量路由不看推理额度, 只要没在 checkpoint/suspend 就能发
                    const bool canDispatchLight =
                        !ts->checkpointInProgress &&
                        !ts->suspendInProgress &&
                        !ts->resumeInProgress &&
                        !ts->lightQ.empty();
                    const bool drained =
                        ts->stopping &&
                        !ts->checkpointInProgress &&
                        ts->activateQueryNumber == 0 &&
                        ts->activateLightNumber == 0 &&
                        ts->q.empty() && ts->lightQ.empty();
                    return canDispatch || canDispatchLight || drained;
                });
                if (ts->stopping &&
                    ts->activateQueryNumber == 0 &&
                    ts->activateLightNumber == 0 &&
                    ts->q.empty() && ts->lightQ.empty()) {
                    return;
                }
                // 请求线程体。轻量路由(健康检查/元数据)与推理请求共用同一段
                // 逻辑, 只是记在不同的并发计数上 —— 见 IsLightweightRoute 的说明。
                auto runNode = [ts](WorkNode *now, bool light) {
                        // 请求线程兜底:Deal 内任何异常(如 KV 页池 Grow OOM)
                        // 不得逃逸——detached 线程异常 = std::terminate 全进程自杀。
                        // 记 500 + 关闭连接,调度簿记照常,进程继续服务。
                        try {
                            std::string route = now->request.route;
                            if (route.size() > 1 && route.back() == '/') {
                                route.pop_back();
                            }
                            const bool mayMutatePrefixCache =
                                now->request.method == "POST" &&
                                (route == "/generate" ||
                                 route == "/v1/chat/completions");
                            ts->Deal(now);
                            if (mayMutatePrefixCache) {
                                ts->cacheMutationEpoch.fetch_add(
                                    1, std::memory_order_relaxed);
                            }
                        } catch (const std::exception &exc) {
                            fprintf(stderr,
                                    "[WorkQueue] request thread exception: %s\n",
                                    exc.what());
                            if (now->client >= 0) {
                                WriteFixedJsonResponse(
                                    now->client, 500,
                                    OpenAIHttpError(
                                        std::string("internal error: ") +
                                            exc.what(),
                                        "server_error", "internal_error"));
                            }
                        } catch (...) {
                            fprintf(stderr,
                                    "[WorkQueue] request thread unknown exception\n");
                            if (now->client >= 0) {
                                WriteFixedJsonResponse(
                                    now->client, 500,
                                    OpenAIHttpError(
                                        "internal error", "server_error",
                                        "internal_error"));
                            }
                        }
                        printf("Response client %d finish\n",
                               now->client);
                        delete now;
                        {
                        std::lock_guard<std::mutex> lock(ts->locker);
                        if (light) {
                            ts->activateLightNumber--;
                        } else {
                            ts->activateQueryNumber--;
                        }
                    }
                    ts->cv.notify_all();
                };

                // 先发轻量路由: 不占推理额度, 也不能被队头的长请求挡住。
                // 这样 /health 在后端满负荷时依然秒回, 上游代理才能区分
                // "在忙"和"已僵死"。
                while (!ts->checkpointInProgress &&
                       !ts->suspendInProgress &&
                       !ts->resumeInProgress &&
                       !ts->lightQ.empty()) {
                    WorkNode *now = ts->lightQ.front();
                    ts->lightQ.pop();
                    if (SocketPeerDisconnected(now->client)) {
                        CloseNodeClient(now);
                        delete now;
                        continue;
                    }
                    ts->activateLightNumber++;
                    std::thread(runNode, now, true).detach();
                }

                while (!ts->checkpointInProgress &&
                       !ts->suspendInProgress &&
                       !ts->resumeInProgress &&
                       ts->activateQueryNumber <
                           ts->maxActivateQueryNumber &&
                       !ts->q.empty()) {
                    WorkNode *now = ts->q.front();
                    ts->q.pop();
                    // 排队期间客户端很可能已经走了(agent 超时、用户 Ctrl-C、
                    // 上游 proxy 断流)。此时再去算它是纯亏:
                    //   - 一个 262K 的 prefill 要独占 GPU 几分钟;
                    //   - 它占的 KV 页在整个生成期间不释放, 把页池顶到水位线,
                    //     进而诱发 PagedCacheManager::Grow 失败(实测会连带
                    //     中断所有在飞请求);
                    //   - 真实请求被它挤在队列后面干等。
                    // 生产上见过后端 running=1 pending=5 全是这种僵尸请求。
                    // 探活只是一次 select(0 超时)+MSG_PEEK, 不消耗数据也不阻塞。
                    if (SocketPeerDisconnected(now->client)) {
                        printf("[queue] 丢弃已断开的排队请求 client=%d\n",
                               now->client);
                        fflush(stdout);
                        CloseNodeClient(now);
                        delete now;
                        continue;
                    }
                    ts->activateQueryNumber++;
                    ts->totalQueryNumber++;
                    printf("totalQueryNumber = %d\n",
                           ts->totalQueryNumber);
                    std::thread(runNode, now, false).detach();
                }
            }
        }, this);
    }

    void StopAndDrain() {
        {
            std::lock_guard<std::mutex> lock(locker);
            stopping = true;
        }
        cv.notify_all();
        if (loop != nullptr && loop->joinable()) {
            loop->join();
        }
        delete loop;
        loop = nullptr;
    }

    void Deal(WorkNode *node) {
        auto *req = &node->request;
        std::string route = req->route;
        if (route.size() > 1 && route.back() == '/') {
            route.pop_back();
        }
        // json11 的 dump() 在本仓库被定制为 pretty-print(\n\t 缩进,供 .tfdl
        // 模型文件用)。SSE 客户端(OpenAI 兼容)要求每个 chunk 是单行紧凑
        // JSON;字符串值中的 \n/\t 均已转义,剥掉裸换行/制表符是安全的。
        auto compactJsonDump = [](const json11::Json &value) {
            std::string pretty = value.dump();
            std::string compact;
            compact.reserve(pretty.size());
            for (char ch : pretty) {
                if (ch != '\n' && ch != '\t') {
                    compact += ch;
                }
            }
            return compact;
        };
        auto writeJsonAndClose = [&](int status, const json11::Json &body,
                                     const std::vector<std::pair<std::string, std::string>> &headers = {}) {
            WriteFixedJsonResponse(node->client, status, body, headers);
            CloseNodeClient(node);
        };
        auto writeMethodNotAllowed = [&](const std::string &allowed) {
            writeJsonAndClose(
                405,
                OpenAIHttpError("Method " + req->method + " is not allowed for " + route + ".",
                                "invalid_request_error", "method_not_allowed"),
                {{"Allow", allowed}});
        };
        // 从请求头读取 Bearer 控制令牌（与 checkpoint 端点一致）。
        auto readControlAuthorization = [&]() -> std::string {
            std::string authorization;
            for (const auto &header : req->headers) {
                if (HttpHeaderNameEquals(
                        header.first, "Authorization")) {
                    authorization =
                        TrimHttpHeaderValue(header.second);
                    break;
                }
            }
            const char *configured = std::getenv(
                "FASTLLM_PREFIX_CACHE_CONTROL_TOKEN");
            const std::string expectedToken =
                configured == nullptr ? std::string() :
                    std::string(configured);
            const std::string bearerPrefix = "Bearer ";
            if (expectedToken.empty() ||
                authorization.size() <= bearerPrefix.size() ||
                authorization.compare(0, bearerPrefix.size(),
                                      bearerPrefix) != 0) {
                return std::string();
            }
            const std::string presented =
                authorization.substr(bearerPrefix.size());
            if (!fastllm::apiserver::ConstantTimeCheckpointTokenEqual(
                    presented, expectedToken)) {
                return std::string();
            }
            return presented;
        };

        // suspend 后模型对象被释放，除 resume/health/suspend 外的路由一律 503，
        // 防止对空模型解引用。
        if (suspended.load(std::memory_order_acquire) &&
            route != "/admin/resume" &&
            route != "/admin/suspend" &&
            route != "/health") {
            writeJsonAndClose(
                503,
                OpenAIHttpError(
                    "backend is suspended; call POST /admin/resume",
                    "server_error", "backend_suspended"));
            return;
        }

        if (route == "/admin/suspend") {
            if (req->method != "POST") {
                writeMethodNotAllowed("POST");
                return;
            }
            if (readControlAuthorization().empty()) {
                writeJsonAndClose(
                    403,
                    OpenAIHttpError(
                        "missing or invalid control token",
                        "authentication_error",
                        "invalid_control_token"));
                return;
            }
            std::string tier = "memory";
            if (node->config["tier"].is_string()) {
                tier = node->config["tier"].string_value();
            }
            if (tier != "memory" && tier != "disk") {
                writeJsonAndClose(
                    400,
                    OpenAIHttpError(
                        "tier must be \"memory\" or \"disk\"",
                        "invalid_request_error", "invalid_tier"));
                return;
            }
            if (suspended.load(std::memory_order_acquire) ||
                resumeInProgress.load(std::memory_order_acquire)) {
                writeJsonAndClose(
                    409,
                    OpenAIHttpError(
                        "backend is already suspended",
                        "conflict_error", "backend_suspended"));
                return;
            }
            suspendInProgress.store(true, std::memory_order_release);
            {
                std::lock_guard<std::mutex> lock(locker);
                if (checkpointInProgress ||
                    activateQueryNumber != 1 || !q.empty()) {
                    suspendInProgress.store(
                        false, std::memory_order_release);
                    writeJsonAndClose(
                        409,
                        OpenAIHttpError(
                            "suspend requires no active or queued requests",
                            "conflict_error",
                            "suspend_requires_idle_server"));
                    return;
                }
            }
            const auto suspendStarted =
                std::chrono::steady_clock::now();
            // 先把最新前缀缓存落盘（仅当 checkpoint 之后又有 KV 变更），
            // 这样 resume 能从磁盘恢复 KV 状态。
            if (fastllm::GetPersistentPrefixCacheStatus().enabled &&
                cacheMutationEpoch.load(std::memory_order_relaxed) !=
                    checkpointedMutationEpoch.load(
                        std::memory_order_relaxed)) {
                fastllm::PersistentPrefixCheckpointStats stats;
                std::string checkpointError;
                if (!fastllm::CheckpointPersistentPrefixCache(
                        model.get(), stats, &checkpointError)) {
                    std::fprintf(
                        stderr,
                        "[suspend] prefix-cache checkpoint failed: %s\n",
                        checkpointError.empty() ?
                            "unknown error" : checkpointError.c_str());
                } else {
                    checkpointedMutationEpoch.store(
                        cacheMutationEpoch.load(
                            std::memory_order_relaxed),
                        std::memory_order_relaxed);
                }
            }
            bool memoryTierReady = false;
            fastllm::HostOffloadTransitionResult hostResult;
            if (tier == "memory" && hostOffloadManager != nullptr) {
                model->PrepareHostWeightSuspend();
                hostResult = hostOffloadManager->Suspend(
                    ++hostOffloadGeneration);
                memoryTierReady =
                    hostResult.outcome ==
                    fastllm::HostOffloadOutcome::SUSPENDED_HOST;
                if (memoryTierReady) {
                    fastllm::ReleaseCudaIdlePoolMemory();
                } else {
                    std::fprintf(
                        stderr,
                        "[suspend] RAM tier unavailable, falling back to "
                        "disk: %s\n",
                        hostResult.reason.c_str());
                }
            }
            if (!memoryTierReady) {
                hostOffloadManager.reset();
                model.reset();
                fastllm::ClearAllPagedCacheManagers();
                fastllm::ReleaseCudaIdlePoolMemory();
                suspendedTier = "disk";
            } else {
                suspendedTier = "memory";
            }
            suspended.store(true, std::memory_order_release);
            suspendInProgress.store(false, std::memory_order_release);
            cv.notify_all();
            const double durationMs =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() -
                        suspendStarted).count();
            writeJsonAndClose(200, json11::Json::object {
                {"status", "ok"},
                {"tier", suspendedTier},
                {"requested_tier", tier},
                {"state", "suspended"},
                {"duration_ms", durationMs},
                {"cached_bytes", (double)hostResult.cachedBytes},
                {"source_evicted_bytes",
                 (double)hostResult.sourceEvictedBytes},
                {"cache_hit_ratio", hostResult.cacheHitRatio},
                {"fallback_reason",
                 memoryTierReady ? std::string() : hostResult.reason}
            });
            return;
        }

        if (route == "/admin/resume") {
            if (req->method != "POST") {
                writeMethodNotAllowed("POST");
                return;
            }
            if (readControlAuthorization().empty()) {
                writeJsonAndClose(
                    403,
                    OpenAIHttpError(
                        "missing or invalid control token",
                        "authentication_error",
                        "invalid_control_token"));
                return;
            }
            if (!suspended.load(std::memory_order_acquire) ||
                suspendInProgress.load(std::memory_order_acquire)) {
                writeJsonAndClose(
                    409,
                    OpenAIHttpError(
                        "backend is not suspended",
                        "conflict_error", "backend_not_suspended"));
                return;
            }
            resumeInProgress.store(true, std::memory_order_release);
            {
                std::lock_guard<std::mutex> lock(locker);
                if (checkpointInProgress ||
                    activateQueryNumber != 1 || !q.empty()) {
                    resumeInProgress.store(
                        false, std::memory_order_release);
                    writeJsonAndClose(
                        409,
                        OpenAIHttpError(
                            "resume requires no active or queued requests",
                            "conflict_error",
                            "resume_requires_idle_server"));
                    return;
                }
            }
            const auto resumeStarted =
                std::chrono::steady_clock::now();
            std::string resumeError;
            double cacheHitRatio = 0.0;
            uint64_t rebuiltBytes = 0;
            try {
                if (suspendedTier == "memory" &&
                    model != nullptr &&
                    hostOffloadManager != nullptr) {
                    const auto hostResult =
                        hostOffloadManager->Resume(
                            hostOffloadGeneration);
                    cacheHitRatio = hostResult.cacheHitRatio;
                    rebuiltBytes = hostResult.rebuiltBytes;
                    if (hostResult.outcome ==
                        fastllm::HostOffloadOutcome::READY) {
                        model->RestoreAfterHostWeightResume();
                        PrepareServerPersistentPrefixCache(model.get());
                    } else {
                        std::fprintf(
                            stderr,
                            "[resume] RAM tier failed, rebuilding from "
                            "disk: %s\n",
                            hostResult.reason.c_str());
                        hostOffloadManager.reset();
                        model.reset();
                        fastllm::ClearAllPagedCacheManagers();
                        fastllm::ReleaseCudaIdlePoolMemory();
                        suspendedTier = "disk";
                        model = fastllm::CreateLLMModelFromFile(
                            ::config.path,
                            ::config.multimodalProjectorPath);
                        model->SetTokenLimit(::config.tokens);
                        if (::config.chunkedPrefillSize > 0) {
                            model->SetChunkedPrefillSize(
                                ::config.chunkedPrefillSize);
                        }
                        model->SetDataType(::config.atype);
                        if (::config.kvCacheDtype !=
                            fastllm::DataType::DATA_AUTO_NONE) {
                            model->SetKVCacheDataType(
                                ::config.kvCacheDtype);
                        }
                        model->maxBatch =
                            maxActivateQueryNumber;
                        PrepareServerPersistentPrefixCache(
                            model.get());
                        ConfigureHostOffloadManager();
                    }
                } else {
                    model = fastllm::CreateLLMModelFromFile(
                        ::config.path,
                        ::config.multimodalProjectorPath);
                    model->SetTokenLimit(::config.tokens);
                    if (::config.chunkedPrefillSize > 0) {
                        model->SetChunkedPrefillSize(
                            ::config.chunkedPrefillSize);
                    }
                    model->SetDataType(::config.atype);
                    if (::config.kvCacheDtype !=
                        fastllm::DataType::DATA_AUTO_NONE) {
                        model->SetKVCacheDataType(
                            ::config.kvCacheDtype);
                    }
                    model->maxBatch = maxActivateQueryNumber;
                    PrepareServerPersistentPrefixCache(model.get());
                    ConfigureHostOffloadManager();
                }
                pagedManagersPreallocated.store(
                    false, std::memory_order_relaxed);
                suspended.store(false, std::memory_order_release);
            } catch (const std::exception &error) {
                resumeError = error.what();
            } catch (...) {
                resumeError = "unknown error";
            }
            resumeInProgress.store(false, std::memory_order_release);
            cv.notify_all();
            if (!resumeError.empty()) {
                std::fprintf(
                    stderr, "[resume] model restore failed: %s\n",
                    resumeError.c_str());
                writeJsonAndClose(
                    500,
                    OpenAIHttpError(
                        "resume failed: " + resumeError,
                        "server_error", "resume_failed"));
                return;
            }
            const double durationMs =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() -
                        resumeStarted).count();
            writeJsonAndClose(200, json11::Json::object {
                {"status", "ok"},
                {"state", "ready"},
                {"duration_ms", durationMs},
                {"tier", suspendedTier},
                {"cache_hit_ratio", cacheHitRatio},
                {"rebuilt_bytes", (double)rebuiltBytes}
            });
            return;
        }

        if (route == "/admin/prefix-cache/checkpoint") {
            std::string authorization;
            for (const auto &header : req->headers) {
                if (HttpHeaderNameEquals(
                        header.first, "Authorization")) {
                    authorization =
                        TrimHttpHeaderValue(header.second);
                    break;
                }
            }
            const char *configured = std::getenv(
                "FASTLLM_PREFIX_CACHE_CONTROL_TOKEN");
            const std::string expectedToken =
                configured == nullptr ? std::string() :
                    std::string(configured);
            const fastllm::PersistentPrefixCacheStatus before =
                fastllm::GetPersistentPrefixCacheStatus();
            int activeGenerationRequests = 0;
            int queuedGenerationRequests = 0;
            {
                std::lock_guard<std::mutex> lock(locker);
                activeGenerationRequests =
                    std::max(0, activateQueryNumber - 1);
                queuedGenerationRequests =
                    static_cast<int>(q.size());
            }
            const fastllm::apiserver::CheckpointControlDecision
                decision =
                    fastllm::apiserver::EvaluateCheckpointControl(
                        req->method,
                        authorization,
                        expectedToken,
                        before.enabled,
                        activeGenerationRequests,
                        queuedGenerationRequests);
            if (decision ==
                fastllm::apiserver::CheckpointControlDecision::
                    METHOD_NOT_ALLOWED) {
                writeMethodNotAllowed("POST");
                return;
            }
            if (decision ==
                fastllm::apiserver::CheckpointControlDecision::
                    FORBIDDEN) {
                writeJsonAndClose(
                    403,
                    OpenAIHttpError(
                        fastllm::apiserver::
                            CheckpointControlDecisionMessage(decision),
                        "authentication_error",
                        "invalid_checkpoint_token"));
                return;
            }
            if (decision ==
                fastllm::apiserver::CheckpointControlDecision::
                    DISABLED) {
                writeJsonAndClose(
                    503,
                    OpenAIHttpError(
                        fastllm::apiserver::
                            CheckpointControlDecisionMessage(decision),
                        "server_error",
                        "prefix_cache_disabled"));
                return;
            }
            if (decision ==
                fastllm::apiserver::CheckpointControlDecision::
                    BUSY) {
                writeJsonAndClose(
                    409,
                    OpenAIHttpError(
                        fastllm::apiserver::
                            CheckpointControlDecisionMessage(decision),
                        "conflict_error",
                        "checkpoint_requires_idle_server"));
                return;
            }
            std::string busyReason;
            if (!BeginExclusiveCheckpoint(busyReason)) {
                writeJsonAndClose(
                    409,
                    OpenAIHttpError(
                        busyReason,
                        "conflict_error",
                        "checkpoint_requires_idle_server"));
                return;
            }
            struct CheckpointGuard {
                WorkQueue *queue;
                ~CheckpointGuard() {
                    queue->EndExclusiveCheckpoint();
                }
            } checkpointGuard {this};
            fastllm::PersistentPrefixCheckpointStats stats;
            std::string checkpointError;
            const bool checkpointed =
                fastllm::CheckpointPersistentPrefixCache(
                    model.get(), stats, &checkpointError);
            (void)checkpointGuard;
            if (!checkpointed) {
                writeJsonAndClose(
                    500,
                    OpenAIHttpError(
                        checkpointError.empty() ?
                            "persistent prefix checkpoint failed" :
                            checkpointError,
                        "server_error",
                        "prefix_cache_checkpoint_failed"));
                return;
            }
            checkpointedMutationEpoch.store(
                cacheMutationEpoch.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
            const fastllm::PersistentPrefixCacheStatus after =
                fastllm::GetPersistentPrefixCacheStatus();
            writeJsonAndClose(200, json11::Json::object {
                {"status", "ok"},
                {"generation", (double)stats.generation},
                {"pages", (double)stats.pages},
                {"bytes", (double)stats.bytes},
                {"duration_ms", stats.durationMs},
                {"checkpoint_count",
                    (double)after.checkpointCount},
                {"restore_hit_count",
                    (double)after.restoreHitCount}
            });
            return;
        }

        if (route == "/health" || route == "/version" || route == "/props") {
            if (req->method != "GET") {
                writeMethodNotAllowed("GET");
                return;
            }
            if (route == "/health") {
                int activeRequests = 0;
                int queuedRequests = 0;
                {
                    std::lock_guard<std::mutex> lock(locker);
                    activeRequests = std::max(0, activateQueryNumber - 1);
                    queuedRequests = static_cast<int>(q.size());
                }
                const bool suspendedNow =
                    suspended.load(std::memory_order_acquire);
                writeJsonAndClose(200, json11::Json::object {
                    {"status", "ok"},
                    {"ready", !suspendedNow},
                    {"suspended", suspendedNow},
                    {"tier_state",
                     suspendedNow
                         ? (suspendedTier == "memory"
                                ? "suspended_host"
                                : "suspended_disk")
                         : "ready"},
                    {"accepting", activeRequests < maxActivateQueryNumber},
                    {"active_requests", activeRequests},
                    {"queued_requests", queuedRequests},
                    {"model", ::config.modelName}
                });
                return;
            }
            if (route == "/version") {
                writeJsonAndClose(200, json11::Json::object {
                    {"name", "fastllm"},
                    {"version", "unknown"},
                    {"build", "unknown"}
                });
                return;
            }
            const std::string kvCacheDtype =
                ::config.kvCacheDtype == fastllm::DataType::DATA_AUTO_NONE
                    ? "auto"
                    : fastllm::GetDataTypeName(::config.kvCacheDtype);
            const auto zstdMetrics =
                fastllm::GetPagedCacheCpuSnapshotZstdMetrics();
            const auto persistentStatus =
                fastllm::GetPersistentPrefixCacheStatus();
            const auto prefixStats =
                fastllm::GetPrefixCacheStatsSnapshot();
            // 算子路由普查: 每类算子实际命中了哪条 kernel。
            // 判断标准(本模型 Qwen3.8-27B / V100 / turbo3 KV 下的预期):
            //   gguf.mmvq                 decode 主力(n<=8), 应占绝大多数调用
            //   gguf.dequant_fp16_gemm    prefill 主力(n>8), 每 chunk 重新展开权重
            //   gguf.sm70_iq4xs_mmq       只在 n in [8,64] 出现, 常态为 0 是正常的
            //   attn.native_fallback      V100 无 FlashInfer, 注意力应该全落这里
            //   attn.sm70_paged_xqa       要求分页 KV 为 float16; turbo3 下恒为 0
            //   attn.sm70_flash_prefill   要求分页 KV 为 fp8_e4m3; turbo3 下恒为 0
            // 某一格出乎意料地为 0 或非 0, 就说明选路判定和预期不一致。
            json11::Json::object kernelRoutes;
            {
                const auto routeTotals = fastllm::GetKernelRouteTotals();
                for (int i = 0; i < (int)routeTotals.size(); i++) {
                    if (routeTotals[i].calls == 0) {
                        continue;
                    }
                    kernelRoutes[fastllm::GetKernelRouteName(
                            (fastllm::KernelRoute)i)] =
                        json11::Json::object {
                            {"calls", (double)routeTotals[i].calls},
                            {"tokens", (double)routeTotals[i].tokens},
                            {"min_n", (double)routeTotals[i].minN},
                            {"max_n", (double)routeTotals[i].maxN}
                        };
                }
            }
            json11::Json::array kernelRouteShapes;
            for (const auto &shape : fastllm::GetKernelRouteShapes()) {
                kernelRouteShapes.push_back(json11::Json::object {
                    {"route", fastllm::GetKernelRouteName(shape.route)},
                    {"ggml_type", (double)shape.ggmlType},
                    {"n", (double)shape.n},
                    {"m", (double)shape.m},
                    {"k", (double)shape.k},
                    {"calls", (double)shape.calls}
                });
            }
            writeJsonAndClose(200, json11::Json::object {
                {"model", ::config.modelName},
                {"max_batch", ::config.batch},
                {"token_pool", ::config.tokens},
                {"kernel_routes", kernelRoutes},
                {"kernel_route_shapes", kernelRouteShapes},
                {"kernel_route_shape_stats_enabled",
                    fastllm::KernelRouteShapeStatsEnabled()},
                {"kv_cache_dtype", kvCacheDtype},
                {"activation_dtype", fastllm::GetDataTypeName(::config.atype)},
                {"default_max_tokens", ::config.defaultMaxTokens},
                {"multimodal_projector_loaded",
                    !::config.multimodalProjectorPath.empty()},
                {"cpu_request_swap_enabled",
                    model->IsCpuRequestSwapEnabled()},
                {"cpu_request_swap_zstd_enabled",
                    model->IsCpuRequestSwapZstdEnabled()},
                {"cpu_request_swap_disk_enabled",
                    model->IsCpuRequestSwapDiskEnabled()},
                {"cpu_request_swap_suspended",
                    (int)model->GetSuspendedResponseContextCount()},
                {"cpu_request_swap_disk_spills",
                    (double)model->GetCpuRequestSwapDiskSpillCount()},
                {"cpu_request_swap_disk_restores",
                    (double)model->GetCpuRequestSwapDiskRestoreCount()},
                {"cpu_request_swap_disk_write_bytes",
                    (double)model->GetCpuRequestSwapDiskWriteBytes()},
                {"cpu_request_swap_disk_read_bytes",
                    (double)model->GetCpuRequestSwapDiskReadBytes()},
                {"cpu_request_swap_disk_read_mib_per_second",
                    model->
                        GetCpuRequestSwapDiskReadMegabytesPerSecond()},
                {"paged_cache_snapshot_zstd_compress_calls",
                    (double)zstdMetrics.compressCalls},
                {"paged_cache_snapshot_zstd_compress_input_bytes",
                    (double)zstdMetrics.compressInputBytes},
                {"paged_cache_snapshot_zstd_compress_output_bytes",
                    (double)zstdMetrics.compressOutputBytes},
                {"paged_cache_snapshot_zstd_compress_seconds",
                    (double)zstdMetrics.compressNanoseconds / 1.0e9},
                {"paged_cache_snapshot_zstd_decompress_calls",
                    (double)zstdMetrics.decompressCalls},
                {"paged_cache_snapshot_zstd_decompress_input_bytes",
                    (double)zstdMetrics.decompressInputBytes},
                {"paged_cache_snapshot_zstd_decompress_output_bytes",
                    (double)zstdMetrics.decompressOutputBytes},
                {"paged_cache_snapshot_zstd_decompress_seconds",
                    (double)zstdMetrics.decompressNanoseconds / 1.0e9},
                {"prefix_cache_cpu_tier_enabled",
                    fastllm::PagedPrefixCacheCpuTierEnabled()},
                {"prefix_cache_disk_tier_enabled",
                    fastllm::PagedPrefixCacheDiskTierEnabled()},
                {"prefix_cache_gpu_hit_pages",
                    (double)fastllm::
                        GetPagedPrefixCacheGpuHitPages()},
                {"prefix_cache_cpu_hit_pages",
                    (double)fastllm::
                        GetPagedPrefixCacheCpuHitPages()},
                {"prefix_cache_cpu_tier_bytes",
                    (double)fastllm::
                        GetPagedPrefixCacheCpuTierBytes()},
                {"prefix_cache_disk_write_bytes",
                    (double)fastllm::
                        GetPagedPrefixCacheDiskWriteBytes()},
                {"prefix_cache_disk_live_bytes",
                    (double)fastllm::
                        GetPagedPrefixCacheDiskLiveBytes()},
                {"prefix_cache_disk_read_bytes",
                    (double)fastllm::
                        GetPagedPrefixCacheDiskReadBytes()},
                {"prefix_cache_disk_hits",
                    (double)fastllm::
                        GetPagedPrefixCacheDiskHitCount()},
                {"prefix_cache_disk_read_mib_per_second",
                    fastllm::
                        GetPagedPrefixCacheDiskReadMegabytesPerSecond()},
                {"prefix_cache_recompute_tokens_per_second",
                    fastllm::
                        GetPagedPrefixCacheRecomputeTokensPerSecond()},
                {"prefix_cache_zstd_decompress_mib_per_second",
                    fastllm::
                        GetPagedPrefixCacheZstdDecompressMegabytesPerSecond()},
                {"prefix_cache_zstd_compress_calls",
                    (double)fastllm::
                        GetPagedPrefixCacheZstdCompressCalls()},
                {"prefix_cache_zstd_compress_input_bytes",
                    (double)fastllm::
                        GetPagedPrefixCacheZstdCompressInputBytes()},
                {"prefix_cache_zstd_compress_output_bytes",
                    (double)fastllm::
                        GetPagedPrefixCacheZstdCompressOutputBytes()},
                {"prefix_cache_zstd_compress_seconds",
                    fastllm::
                        GetPagedPrefixCacheZstdCompressSeconds()},
                {"prefix_cache_zstd_decompress_calls",
                    (double)fastllm::
                        GetPagedPrefixCacheZstdDecompressCalls()},
                {"prefix_cache_zstd_decompress_input_bytes",
                    (double)fastllm::
                        GetPagedPrefixCacheZstdDecompressInputBytes()},
                {"prefix_cache_zstd_decompress_output_bytes",
                    (double)fastllm::
                        GetPagedPrefixCacheZstdDecompressOutputBytes()},
                {"prefix_cache_zstd_decompress_seconds",
                    fastllm::
                        GetPagedPrefixCacheZstdDecompressSeconds()},
                {"prefix_cache_persistence_enabled",
                    persistentStatus.enabled},
                {"prefix_cache_persistence_loaded_generation",
                    (double)persistentStatus.loadedGeneration},
                {"prefix_cache_persistence_checkpoint_count",
                    (double)persistentStatus.checkpointCount},
                {"prefix_cache_persistence_restore_hits",
                    (double)persistentStatus.restoreHitCount},
                {"prefix_cache_persistence_payload_bytes",
                    (double)persistentStatus.payloadBytes},
                {"prefix_cache_persistence_last_duration_ms",
                    persistentStatus.lastDurationMs},
                {"prefix_cache_persistence_last_error",
                    persistentStatus.lastError},
                // ---- 可观测性统计(FASTLLM_PREFIX_CACHE_STATS 门控累计) ----
                {"prefix_cache_stats_enabled",
                    fastllm::PrefixCacheStatsEnabled()},
                {"prefix_cache_stats_requests",
                    (double)prefixStats.requests},
                {"prefix_cache_stats_hit_requests",
                    (double)prefixStats.hitRequests},
                {"prefix_cache_stats_query_tokens",
                    (double)prefixStats.queryTokens},
                {"prefix_cache_stats_hit_tokens",
                    (double)prefixStats.hitTokens},
                {"prefix_cache_stats_hit_tokens_mem_trie",
                    (double)prefixStats.hitTokensMemTrie},
                {"prefix_cache_stats_hit_tokens_cpu_tier",
                    (double)prefixStats.hitTokensCpuTier},
                {"prefix_cache_stats_hit_tokens_disk",
                    (double)prefixStats.hitTokensDisk},
                {"prefix_cache_stats_miss_no_record",
                    (double)prefixStats.missNoRecord},
                // no_record 以前是个大杂烩; 下面四项把它拆开。尤其
                // miss_extra_missing != 0 表示 paged trie 其实命中了,
                // 只是 GDN/linear 快照对不上而整条被丢掉 —— 和"没记录过"
                // 是完全相反的两种问题。
                {"prefix_cache_stats_miss_probe_empty",
                    (double)prefixStats.missProbeEmpty},
                {"prefix_cache_stats_miss_layer_min",
                    (double)prefixStats.missLayerMin},
                {"prefix_cache_stats_miss_single_page",
                    (double)prefixStats.missSinglePage},
                {"prefix_cache_stats_miss_extra_missing",
                    (double)prefixStats.missExtraMissing},
                {"prefix_cache_stats_miss_multimodal_disabled",
                    (double)prefixStats.missMultimodalDisabled},
                {"prefix_cache_stats_miss_remainder_has_image",
                    (double)prefixStats.missRemainderHasImage},
                {"prefix_cache_stats_miss_mm_delta_unavailable",
                    (double)prefixStats.missMmDeltaUnavailable},
                {"prefix_cache_stats_miss_evicted",
                    (double)prefixStats.missEvicted},
                {"prefix_cache_stats_miss_below_threshold",
                    (double)prefixStats.missBelowThreshold},
                {"prefix_cache_stats_miss_generation_mismatch",
                    (double)prefixStats.missGenerationMismatch},
                {"prefix_cache_stats_miss_restore_failed",
                    (double)prefixStats.missRestoreFailed},
                {"prefix_cache_stats_miss_other",
                    (double)prefixStats.missOther},
                {"prefix_cache_stats_record_accepted",
                    (double)prefixStats.recordAccepted},
                {"prefix_cache_stats_record_rejected_min_hits_tokens",
                    (double)prefixStats.recordRejectedMinHitsTokens},
                {"prefix_cache_stats_record_rejected_capacity",
                    (double)prefixStats.recordRejectedCapacity},
                {"prefix_cache_stats_record_rejected_no_space",
                    (double)prefixStats.recordRejectedNoSpace},
                {"prefix_cache_stats_record_rejected_other",
                    (double)prefixStats.recordRejectedOther},
                {"prefix_cache_stats_evict_trie_nodes",
                    (double)prefixStats.evictTrieNodes},
                {"prefix_cache_stats_evict_cpu_tier_calls",
                    (double)prefixStats.evictCpuTierCalls},
                {"prefix_cache_stats_evict_cpu_tier_bytes",
                    (double)prefixStats.evictCpuTierBytes},
                {"prefix_cache_stats_resident_mem_trie_bytes",
                    (double)prefixStats.memTrieResidentBytes},
                {"prefix_cache_stats_resident_cpu_tier_bytes",
                    (double)prefixStats.cpuTierResidentBytes},
                {"prefix_cache_stats_resident_disk_bytes",
                    (double)prefixStats.diskResidentBytes},
                // 记录侧挂的 manager != 查询侧查的 manager 的次数。
                // 非 0 = 前缀链被记进了永远不会被查询的池子, 命中率必然
                // 恒为 0, 而且不会有任何报错(vision 前向曾经就是这样)。
                {"prefix_cache_stats_record_manager_lookup_mismatch",
                    (double)prefixStats.recordManagerLookupMismatch},
                {"prefix_cache_stats_record_skip_manager_null",
                    (double)prefixStats.recordSkipManagerNull},
                {"prefix_cache_stats_record_skip_manager_no_page_index",
                    (double)prefixStats.recordSkipManagerNoPageIndex},
                {"prefix_cache_stats_record_skip_manager_wrong_type",
                    (double)prefixStats.recordSkipManagerWrongType},
                {"prefix_cache_stats_query_calls",
                    (double)prefixStats.queryCalls},
                {"prefix_cache_stats_query_matched_pages",
                    (double)prefixStats.queryMatchedPages},
                {"prefix_cache_stats_query_break_no_child",
                    (double)prefixStats.queryBreakNoChild},
                {"prefix_cache_stats_query_break_edge_mismatch",
                    (double)prefixStats.queryBreakEdgeMismatch},
                {"prefix_cache_stats_query_break_materialize",
                    (double)prefixStats.queryBreakMaterialize},
                {"prefix_cache_stats_query_break_generation",
                    (double)prefixStats.queryBreakGeneration},
                {"prefix_cache_stats_query_full_match",
                    (double)prefixStats.queryFullMatch},
                // ---- 工具调用语法约束统计(ROOT CAUSE #3) ----
                {"toolcall_grammar_enabled",
                    fastllm::ToolCallGrammarEnabled()},
                {"toolcall_blocks_total",
                    (double)fastllm::GetToolCallGrammarStatsSnapshot()
                            .blocksTotal},
                {"toolcall_malformed_total",
                    (double)fastllm::GetToolCallGrammarStatsSnapshot()
                            .malformedTotal},
                {"toolcall_repaired_total",
                    (double)fastllm::GetToolCallGrammarStatsSnapshot()
                            .repairedTotal},
                {"toolcall_constraint_steps",
                    (double)fastllm::GetToolCallGrammarStatsSnapshot()
                            .constraintSteps},
                {"toolcall_constraint_masked_tokens",
                    (double)fastllm::GetToolCallGrammarStatsSnapshot()
                            .maskedTokens},
                {"backend", "fastllm"}
            });
            return;
        }

        const bool generateRoute = route == "/generate";
        const bool chatRoute = route == "/v1/chat/completions";
        if ((generateRoute || chatRoute) && req->method != "POST") {
            writeMethodNotAllowed("POST");
            return;
        }
        if (!generateRoute && !chatRoute) {
            writeJsonAndClose(
                404,
                OpenAIHttpError("Route " + route + " was not found.",
                                "invalid_request_error", "not_found"));
            return;
        }
        if (generateRoute) {
            std::string message;

            if (node->error == "") {
                if (node->config["prompt"].is_null()) {
                    node->error = "prompt is empty!";
                }
            }
            if (node->error != "") {
                printf("error body = %s, prompt = %s, error = %s\n",
                       node->request.body.c_str(),
                       node->config["prompt"].string_value().c_str(),
                       node->error.c_str());
                writeJsonAndClose(
                    400, OpenAIHttpError(node->error,
                                         "invalid_request_error",
                                         "invalid_request"));
                return;
            }

            std::string output = "";
            bool rawPrompt = node->config["raw_prompt"].is_bool() && node->config["raw_prompt"].bool_value();
            std::string prompt;
            if (rawPrompt) {
                prompt = node->config["prompt"].string_value();
            } else {
                fastllm::ChatMessages messages;
                messages.push_back({"user", node->config["prompt"].string_value()});
                prompt = model->ApplyChatTemplate(messages);
            }
            fastllm::Data inputs = model->weight.tokenizer.Encode(prompt);
            std::vector<int> tokens;
            for (int i = 0; i < inputs.Count(0); i++) {
                tokens.push_back(((float *) inputs.cpuData)[i]);
            }
            fastllm::GenerationConfig config;
            if (!ResolveOutputTokenLimit(node->config["max_tokens"],
                                         ::config.defaultMaxTokens,
                                         config.output_token_limit,
                                         node->error)) {
                writeJsonAndClose(
                    400, OpenAIHttpError(node->error,
                                         "invalid_request_error",
                                         "invalid_max_tokens"));
                return;
            }
            // Admission control: a prompt beyond the configured token pool
            // can never be served — reject fast with a client-visible error
            // instead of letting it prefill into pool exhaustion and take
            // down concurrent requests.
            if (::config.tokens > 0 &&
                (long long)tokens.size() >= (long long)::config.tokens) {
                writeJsonAndClose(
                    400, OpenAIHttpError(
                             "prompt is " + std::to_string(tokens.size()) +
                                 " tokens, exceeds the server's " +
                                 std::to_string(::config.tokens) +
                                 "-token capacity",
                             "invalid_request_error",
                             "context_length_exceeded"));
                return;
            }
            int handleId = model->LaunchResponseTokens(tokens, config);
            std::vector<float> results;
            while (true) {
                int result = model->FetchResponseTokens(handleId);
                if (result == -1) {
                    break;
                } else {
                    results.clear();
                    results.push_back(result);
                    output += model->weight.tokenizer.Decode(fastllm::Data (fastllm::DataType::FLOAT32, {(int)results.size()}, results));

                }
            }

            WriteAllToSocket(
                node->client,
                BuildFixedHttpResponse(
                    200, output, "text/plain; charset=utf-8"));
            CloseNodeClient(node);
        } else if (chatRoute) {
            std::string message;

            fastllm::ChatMessages chatMessages;
            OpenAIParsedChatInput parsedChatInput;
            if (node->config["messages"].is_array()) {
                if (!ParseOpenAIChatInput(
                        node->config["messages"],
                        model->GetImagePlaceholder(),
                        model->GetVideoPlaceholder(),
                        parsedChatInput, node->error)) {
                    writeJsonAndClose(
                        400, OpenAIHttpError(node->error,
                                             "invalid_request_error",
                                             "invalid_messages"));
                    return;
                }
                chatMessages = parsedChatInput.messages;
            } else if (node->config["prompt"].is_string()) {
                chatMessages.push_back(
                    {"user", node->config["prompt"].string_value()});
            } else {
                node->error = "no input.\n";
            }

            if (node->error != "") {
                writeJsonAndClose(
                    400, OpenAIHttpError(node->error,
                                         "invalid_request_error",
                                         "invalid_request"));
                return;
            }
            if (node->config["model"].string_value() != ::config.modelName) {
                writeJsonAndClose(
                    404, OpenAIHttpError(
                        "The model `" + node->config["model"].string_value() + "` does not exist.",
                        "invalid_request_error", "model_not_found"));
                return;
            }

            bool rawPrompt = node->config["raw_prompt"].is_bool() && node->config["raw_prompt"].bool_value();
            std::string prompt;
            if (rawPrompt) {
                if (!node->config["prompt"].is_string()) {
                    node->error = "raw_prompt requires a string prompt.\n";
                } else {
                    prompt = node->config["prompt"].string_value();
                }
            } else {
                prompt = model->ApplyChatTemplate(chatMessages);
            }
            if (node->error != "") {
                writeJsonAndClose(
                    400, OpenAIHttpError(node->error,
                                         "invalid_request_error",
                                         "invalid_raw_prompt"));
                return;
            }
            MultimodalInputGuard multimodalGuard;
            if (!parsedChatInput.imageUrls.empty()) {
                std::vector<fastllm::MultimodalImage> images;
                images.reserve(parsedChatInput.imageUrls.size());
                for (const auto &url : parsedChatInput.imageUrls) {
                    OpenAIDecodedImage decoded;
                    if (!LoadOpenAIImageUrl(url, decoded, node->error)) {
                        writeJsonAndClose(
                            400, OpenAIHttpError(
                                node->error, "invalid_request_error",
                                "invalid_image_url"));
                        return;
                    }
                    fastllm::MultimodalImage image;
                    image.width = decoded.width;
                    image.height = decoded.height;
                    image.rgb = std::move(decoded.rgb);
                    images.push_back(std::move(image));
                }
                if (!model->PrepareMultimodalImageInputs(
                        prompt, images, multimodalGuard.inputs,
                        node->error)) {
                    writeJsonAndClose(
                        400, OpenAIHttpError(
                            node->error, "invalid_request_error",
                            "invalid_multimodal_input"));
                    return;
                }
            }
            if (!parsedChatInput.videoUrls.empty()) {
                std::vector<fastllm::MultimodalVideo> videos;
                videos.reserve(parsedChatInput.videoUrls.size());
                for (const auto &url : parsedChatInput.videoUrls) {
                    OpenAIDecodedVideo decoded;
                    printf("[video] LoadOpenAIVideoUrl begin (url len=%zu)\n", url.size());
                    fflush(stdout);
                    if (!LoadOpenAIVideoUrl(url, decoded, node->error)) {
                        writeJsonAndClose(
                            400, OpenAIHttpError(
                                node->error, "invalid_request_error",
                                "invalid_video_url"));
                        return;
                    }
                    printf("[video] decoded: %dx%d frames=%d rgb=%zu bytes\n",
                           decoded.width, decoded.height, decoded.frameCount,
                           decoded.rgb.size());
                    fflush(stdout);
                    fastllm::MultimodalVideo video;
                    video.width = decoded.width;
                    video.height = decoded.height;
                    video.frameCount = decoded.frameCount;
                    video.rgb = std::move(decoded.rgb);
                    videos.push_back(std::move(video));
                }
                if (!model->PrepareMultimodalVideoInputs(
                        prompt, videos, multimodalGuard.inputs,
                        node->error)) {
                    writeJsonAndClose(
                        400, OpenAIHttpError(
                            node->error, "invalid_request_error",
                            "invalid_multimodal_input"));
                    return;
                }
                printf("[video] PrepareMultimodalVideoInputs ok, prompt len=%zu\n",
                       prompt.size());
                fflush(stdout);
            }
            fastllm::Data inputs = model->weight.tokenizer.Encode(prompt);
            std::vector<int> tokens;
            for (int i = 0; i < inputs.Count(0); i++) {
                tokens.push_back(((float *) inputs.cpuData)[i]);
            }

            fastllm::GenerationConfig config;
            if (!ResolveOutputTokenLimit(node->config["max_tokens"],
                                         ::config.defaultMaxTokens,
                                         config.output_token_limit,
                                         node->error)) {
                writeJsonAndClose(
                    400, OpenAIHttpError(node->error,
                                         "invalid_request_error",
                                         "invalid_max_tokens"));
                return;
            }
            if (node->config["frequency_penalty"].is_number()) {
                config.repeat_penalty = node->config["frequency_penalty"].number_value();
            }
            if (node->config["temperature"].is_number()) {
                config.temperature = node->config["temperature"].number_value();
                // 【上游BUMP勿回退】这道钳制覆盖 CUDA 采样 kernel(它同样做
                // 1/temperature)。删掉它, 客户端发 temperature=0 就会让整行 logits
                // 变 inf/NaN, 输出退化成同一字符的长串。CPU 侧另有显式贪心分支,
                // 两处都要保留。
                if (!(config.temperature > 0.0f)) {
                    // 客户端发 temperature=0 表示"要确定性输出"。CUDA 采样
                    // kernel 同样会做 1/temperature, 除零后整行 logits 变
                    // inf/NaN, 输出退化成同一个字符的长串。这里钳到一个极小
                    // 正值, 数值上等价于 argmax; CPU 侧另有显式贪心分支。
                    config.temperature = 1e-6f;
                }
            }
            if (node->config["top_p"].is_number()) {
                config.top_p = node->config["top_p"].number_value();
            }
            if (node->config["top_k"].is_number()) {
                config.top_k = node->config["top_k"].number_value();
            }

            auto exactTokenLookup = [&](const std::string &stop,
                                        int &tokenId) {
                auto &tokenizer = model->weight.tokenizer;
                auto it = tokenizer.stringToTokenDict.find(stop);
                if (it == tokenizer.stringToTokenDict.end()) {
                    return false;
                }
                tokenId = it->second;
                return true;
            };
            auto fallbackEncode = [&](const std::string &stop) {
                fastllm::Data stopTokens = model->weight.tokenizer.Encode(stop);
                std::vector<int> tokenIds;
                tokenIds.reserve(stopTokens.Count(0));
                for (int i = 0; i < stopTokens.Count(0); i++) {
                    tokenIds.push_back(
                        static_cast<int>(((float *)stopTokens.cpuData)[i]));
                }
                return tokenIds;
            };
            auto encodeStop = [&](const std::string &stop) {
                return EncodeOpenAIStop(stop, exactTokenLookup,
                                        fallbackEncode);
            };
            if (!ParseOpenAIStop(node->config["stop"], encodeStop,
                                 config.stop_token_ids,
                                 config.stop_token_sequences,
                                 config.stop_strings, node->error)) {
                writeJsonAndClose(
                    400, OpenAIHttpError(node->error,
                                         "invalid_request_error",
                                         "invalid_stop"));
                return;
            }
            bool toolsEnabled = node->config["tools"].is_array();
            if (node->config["tool_choice"].is_string() &&
                node->config["tool_choice"].string_value() == "none") {
                toolsEnabled = false;
            }
            std::string selectedToolName;
            if (node->config["tool_choice"].is_object()) {
                selectedToolName =
                    node->config["tool_choice"]["function"]["name"].string_value();
            }
            if (toolsEnabled) {
                for (const auto &tool : node->config["tools"].array_items()) {
                    const std::string name = tool["function"]["name"].string_value();
                    if (!name.empty() &&
                        (selectedToolName.empty() || name == selectedToolName)) {
                        config.tool_call_allowed_names.push_back(name);
                        // 参数名约束: 从 JSON schema properties 提取该工具的合法参数名,
                        // 生成 <parameter=...> 时只放行 schema 内的名字 token, 堵死
                        // 量化损伤导致的参数名漂移(<path>/<prefix>/拼写变体)。
                        std::vector<std::string> paramNames;
                        const auto &params = tool["function"]["parameters"];
                        if (params.is_object()) {
                            const auto &props = params["properties"];
                            if (props.is_object()) {
                                for (const auto &kv : props.object_items()) {
                                    if (!kv.first.empty()) {
                                        paramNames.push_back(kv.first);
                                    }
                                }
                            }
                            // required 参数数: 用于强制 parameter 块约束,
                            // 防止模型跳过参数块直接 </function> 出空调用
                            const auto &required = params["required"];
                            if (required.is_array()) {
                                std::vector <std::string> requiredNames;
                                for (const auto &r : required.array_items()) {
                                    if (r.is_string() && !r.string_value().empty()) {
                                        requiredNames.push_back(r.string_value());
                                    }
                                }
                                if (!requiredNames.empty()) {
                                    config.tool_call_required_parameter_names[name] =
                                        std::move(requiredNames);
                                }
                            }
                        }
                        if (!paramNames.empty()) {
                            config.tool_call_allowed_parameter_names[name] =
                                std::move(paramNames);
                        }
                    }
                }
                toolsEnabled = !config.tool_call_allowed_names.empty();
            }
            if (toolsEnabled) {
                config.tool_call_name_constraint_enabled = true;
                config.tool_call_invoke_name_prefixes = {
                    "<function=", "<fuction="
                };
                config.tool_call_name_terminator = ">";
                if (!config.tool_call_allowed_parameter_names.empty()) {
                    config.tool_call_parameter_name_constraint_enabled = true;
                    config.tool_call_parameter_name_prefixes = {
                        "<parameter=", "<paramter="
                    };
                }
                if (!config.tool_call_required_parameter_names.empty() &&
                    config.tool_call_parameter_name_constraint_enabled) {
                    config.tool_call_required_parameter_constraint_enabled = true;
                }
            }

            // Admission control: reject prompts beyond the configured token
            // pool up front (see the /v1/completions path for rationale).
            if (::config.tokens > 0 &&
                (long long)tokens.size() >= (long long)::config.tokens) {
                writeJsonAndClose(
                    400, OpenAIHttpError(
                             "prompt is " + std::to_string(tokens.size()) +
                                 " tokens, exceeds the server's " +
                                 std::to_string(::config.tokens) +
                                 "-token capacity",
                             "invalid_request_error",
                             "context_length_exceeded"));
                return;
            }
            int handleId = model->LaunchResponseTokens(
                tokens, config, multimodalGuard.inputs);
            multimodalGuard.Release();
            const auto genStart = std::chrono::steady_clock::now();
            bool firstTokenSeen = false;
            double firstTokenMs = 0.0;
            printf("[req %d] start: prefill %d tok\n",
                   handleId, (int) tokens.size());
            fflush(stdout);
            auto noteFirstToken = [&]() {
                if (!firstTokenSeen) {
                    firstTokenSeen = true;
                    firstTokenMs = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - genStart).count();
                    // prefill 完成即入窗口(不等整个请求结束,长 decode 不遮罩)
                    NotePrefillDone((int) tokens.size(), firstTokenMs);
                    printf("[req %d] prefill done: %d tok in %.2fs (%.1f tok/s)\n",
                           handleId, (int) tokens.size(), firstTokenMs / 1000.0,
                           firstTokenMs > 0
                               ? tokens.size() * 1000.0 / firstTokenMs : 0.0);
                    fflush(stdout);
                }
            };
            auto recordMetrics = [&](int outputCount) {
                const double totalMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - genStart).count();
                const double decodeMs =
                    firstTokenSeen ? totalMs - firstTokenMs : 0.0;
                printf("[req %d] done: prefill %d tok %.2fs | decode %d tok in %.2fs (%.1f tok/s)\n",
                       handleId, (int) tokens.size(),
                       (firstTokenSeen ? firstTokenMs : totalMs) / 1000.0,
                       outputCount,
                       decodeMs / 1000.0,
                       decodeMs > 0 ? outputCount * 1000.0 / decodeMs : 0.0);
                fflush(stdout);
                gWinFinishedReqs.fetch_add(1, std::memory_order_relaxed);
            };
            const bool isStream = node->config["stream"].is_bool() &&
                                  node->config["stream"].bool_value();
            const std::string curId = "fastllm-" + GenerateRandomID();
            const auto createTime = _GetCurrentTime();
            OpenAIOutputParser outputParser(
                OpenAIReasoningParser::PromptEndsInReasoning(prompt),
                toolsEnabled);

            auto serializeToolCall = [&](const OpenAIParsedToolCall &call,
                                         const std::string &id,
                                         int index,
                                         bool includeIndex) {
                json11::Json::object toolCall = {
                    {"id", id},
                    {"type", "function"},
                    {"function", json11::Json::object {
                        {"name", call.name},
                        {"arguments", call.arguments}
                    }}
                };
                if (includeIndex) {
                    toolCall["index"] = index;
                }
                return json11::Json(toolCall);
            };

            if (isStream) {
                message = "HTTP/1.1 200 OK\r\n";
                message += "Content-Type:text/event-stream\r\n";
                message += "Cache-Control:no-cache\r\n";
                message += "server:fastllm api server\r\n";
                message += "Transfer-Encoding: chunked\r\n\r\n";

                auto abortDisconnectedStream = [&]() {
                    model->AbortResponse(handleId);
                    CloseNodeClient(node);
                };
                if (!WriteAllToSocket(node->client, message)) {
                    abortDisconnectedStream();
                    return;
                }

                json11::Json startResult = json11::Json::object {
                    {"id", curId},
                    {"object", "chat.completion.chunk"},
                    {"created", createTime},
                    {"model", ::config.modelName},
                    {"choices", json11::Json::array {
                        json11::Json::object {
                            {"index", 0},
                            {"delta", json11::Json::object {
                                {"role", "assistant"}
                            }},
                            {"logprobs", nullptr},
                            {"finish_reason", nullptr},
                            {"stop_reason", nullptr}
                        }
                    }}
                };
                if (!WriteHttpChunk(node->client,
                                    FormatSseData(compactJsonDump(startResult)))) {
                    abortDisconnectedStream();
                    return;
                }

                int outputTokens = 0;
                int toolCallIndex = 0;
                bool hasToolCalls = false;
                std::vector<float> results;
                std::string pendingStopText;
                bool matchedStopString = false;
                auto sendParsedDelta = [&](const OpenAIOutputDelta &parsed) {
                    if (parsed.Empty()) {
                        return true;
                    }
                    json11::Json::object delta;
                    if (!parsed.reasoningContent.empty()) {
                        delta["reasoning_content"] = parsed.reasoningContent;
                    }
                    if (!parsed.content.empty()) {
                        delta["content"] = parsed.content;
                    }
                    if (!parsed.toolCalls.empty()) {
                        json11::Json::array toolCalls;
                        for (const auto &call : parsed.toolCalls) {
                            toolCalls.push_back(serializeToolCall(
                                call, "call_" + GenerateRandomID(),
                                toolCallIndex++, true));
                        }
                        delta["tool_calls"] = toolCalls;
                        hasToolCalls = true;
                    }
                    json11::Json partResult = json11::Json::object {
                        {"id", curId},
                        {"object", "chat.completion.chunk"},
                        {"created", createTime},
                        {"model", ::config.modelName},
                        {"choices", json11::Json::array {
                            json11::Json::object {
                                {"index", 0},
                                {"delta", delta},
                                {"logprobs", nullptr},
                                {"finish_reason", nullptr},
                                {"stop_reason", nullptr}
                            }
                        }}
                    };
                    return WriteHttpChunk(
                        node->client, FormatSseData(compactJsonDump(partResult)));
                };

                while (true) {
                    while (!model->CanFetchResponse(handleId)) {
                        if (SocketPeerDisconnected(node->client)) {
                            abortDisconnectedStream();
                            return;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                    int result = model->FetchResponseTokens(handleId);
                    if (result == -1) {
                        recordMetrics(outputTokens);
                        std::string trailingText;
                        FlushPendingStopText(pendingStopText, trailingText);
                        if (!sendParsedDelta(outputParser.Push(trailingText)) ||
                            !sendParsedDelta(outputParser.Flush())) {
                            CloseNodeClient(node);
                            return;
                        }
                        json11::Json finishResult = json11::Json::object {
                            {"id", curId},
                            {"object", "chat.completion.chunk"},
                            {"created", createTime},
                            {"model", ::config.modelName},
                            {"choices", json11::Json::array {
                                json11::Json::object {
                                    {"index", 0},
                                    {"delta", json11::Json::object {
                                        {"content", ""}
                                    }},
                                    {"logprobs", nullptr},
                                    {"finish_reason", ResolveOpenAIFinishReason(
                                        hasToolCalls, matchedStopString,
                                        outputTokens,
                                        config.output_token_limit)},
                                    {"stop_reason", nullptr}
                                }
                            }},
                            {"usage", json11::Json::object {
                                {"prompt_tokens", (int)tokens.size()},
                                {"total_tokens", (int)tokens.size() + outputTokens},
                                {"completion_tokens", outputTokens}
                            }}
                        };
                        if (!WriteHttpChunk(node->client,
                                FormatSseData(compactJsonDump(finishResult)))) {
                            CloseNodeClient(node);
                            return;
                        }
                        break;
                    }

                    outputTokens++;
                    noteFirstToken();
                    // decode 实时入窗口(每 token,长生成也能反映到 tok/s)
                    gWinDecodeTokens.fetch_add(1, std::memory_order_relaxed);
                    if (outputTokens % 256 == 0) {
                        const double nowMs =
                            std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - genStart)
                                .count();
                        const double decMs = nowMs - firstTokenMs;
                        printf("[req %d] decode %d tok (%.1f tok/s)\n",
                               handleId, outputTokens,
                               decMs > 0 ? (outputTokens - 1) * 1000.0 / decMs
                                         : 0.0);
                        fflush(stdout);
                    }
                    results.assign(1, static_cast<float>(result));
                    std::string now = model->weight.tokenizer.Decode(
                        fastllm::Data(fastllm::DataType::FLOAT32,
                                      {(int)results.size()}, results));
                    std::string filtered;
                    bool matchedStop = PushStopText(
                        config.stop_strings, pendingStopText, now, filtered);
                    matchedStopString = matchedStopString || matchedStop;
                    if (matchedStop) {
                        model->AbortResponse(handleId);
                    }
                    if (!sendParsedDelta(outputParser.Push(filtered))) {
                        abortDisconnectedStream();
                        return;
                    }
                }

                if (!WriteHttpChunk(node->client, FormatSseData("[DONE]")) ||
                    !WriteAllToSocket(node->client, "0\r\n\r\n", 5)) {
                    CloseNodeClient(node);
                    return;
                }
                CloseNodeClient(node);
            } else {
                int outputTokens = 0;
                std::vector<float> results;
                std::string pendingStopText;
                bool matchedStopString = false;
                std::string reasoningOutput;
                std::string output;
                std::vector<OpenAIParsedToolCall> parsedToolCalls;
                auto appendParsedDelta = [&](const OpenAIOutputDelta &parsed) {
                    reasoningOutput += parsed.reasoningContent;
                    output += parsed.content;
                    parsedToolCalls.insert(parsedToolCalls.end(),
                                           parsed.toolCalls.begin(),
                                           parsed.toolCalls.end());
                };
                while (true) {
                    // 非流式路径原本直接调阻塞的 FetchResponseTokens, 于是
                    // **整个 prefill 期间都没有任何断连检测** —— 262K 的 prefill
                    // 要跑几分钟, 客户端早走了也照算到底, 白烧 GPU 和 KV 页。
                    // 流式路径已经是下面这个非阻塞轮询写法, 这里对齐它。
                    // (机制本来就有: basellm.h 里 isAbort 的注释写着"不会再有人
                    //  来 fetch 它了, 推理完就可以删除这个请求" —— 缺的只是
                    //  没人去触发。)
                    while (!model->CanFetchResponse(handleId)) {
                        if (SocketPeerDisconnected(node->client)) {
                            model->AbortResponse(handleId);
                            CloseNodeClient(node);
                            return;
                        }
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(10));
                    }
                    int result = model->FetchResponseTokens(handleId);
                    if (result == -1) {
                        recordMetrics(outputTokens);
                        break;
                    }
                    outputTokens++;
                    noteFirstToken();
                    // decode 实时入窗口(每 token,长生成也能反映到 tok/s)
                    gWinDecodeTokens.fetch_add(1, std::memory_order_relaxed);
                    results.assign(1, static_cast<float>(result));
                    std::string now = model->weight.tokenizer.Decode(
                        fastllm::Data(fastllm::DataType::FLOAT32,
                                      {(int)results.size()}, results));
                    std::string filtered;
                    bool matchedStop = PushStopText(
                        config.stop_strings, pendingStopText, now, filtered);
                    matchedStopString = matchedStopString || matchedStop;
                    appendParsedDelta(outputParser.Push(filtered));
                    if (matchedStop) {
                        model->AbortResponse(handleId);
                    }
                }
                std::string trailingText;
                FlushPendingStopText(pendingStopText, trailingText);
                appendParsedDelta(outputParser.Push(trailingText));
                appendParsedDelta(outputParser.Flush());

                json11::Json::object responseMessage = {
                    {"role", "assistant"},
                    {"content", output}
                };
                if (!reasoningOutput.empty()) {
                    responseMessage["reasoning_content"] = reasoningOutput;
                }
                if (!parsedToolCalls.empty()) {
                    json11::Json::array toolCalls;
                    for (const auto &call : parsedToolCalls) {
                        toolCalls.push_back(serializeToolCall(
                            call, "call_" + GenerateRandomID(), 0, false));
                    }
                    responseMessage["tool_calls"] = toolCalls;
                }
                json11::Json response = json11::Json::object {
                    {"id", curId},
                    {"object", "chat.completion"},
                    {"created", createTime},
                    {"model", ::config.modelName},
                    {"choices", json11::Json::array {
                        json11::Json::object {
                            {"index", 0},
                            {"message", responseMessage},
                            {"logprobs", nullptr},
                            {"finish_reason", ResolveOpenAIFinishReason(
                                !parsedToolCalls.empty(), matchedStopString,
                                outputTokens, config.output_token_limit)},
                            {"stop_reason", nullptr}
                        }
                    }},
                    {"usage", json11::Json::object {
                        {"prompt_tokens", (int)tokens.size()},
                        {"total_tokens", (int)tokens.size() + outputTokens},
                        {"completion_tokens", outputTokens}
                    }}
                };
                writeJsonAndClose(200, response);
            }
            return;
        } else {
            CloseNodeClient(node);
            return;
        }
    }
} workQueue;

void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                  显示帮助" << std::endl;
    std::cout << "<-p|--path> <args>:           模型文件的路径" << std::endl;
    std::cout << "<--mmproj> <args>:           Qwen3.5/3.6 vision projector GGUF path" << std::endl;
    std::cout << "<-t|--threads> <args>:        使用的线程数量" << std::endl;
    std::cout << "<-l|--low>:                   使用低内存模式" << std::endl;
    std::cout << "<--dtype> <args>:             设置权重类型(读取hf文件时生效)" << std::endl;
    std::cout << "<--atype> <args>:             设置推理使用的数据类型(float32/float16)" << std::endl;
    std::cout << "<--kv_cache_dtype> <args>:    设置KV Cache数据类型(auto/float32/float16/bfloat16/fp8_e4m3/turbo3/turbo4; Qwen3.5/3.6 turbo3/turbo4 uses q8_0 K + TurboQuant V)" << std::endl;
    std::cout << "<--batch> <args>:             最大batch数" << std::endl;
    std::cout << "<--tokens> <args>:            最大tokens容量" << std::endl;
    std::cout << "<--chunked_prefill_size> <args>: 分块prefill切片大小（调小可降低峰值显存）" << std::endl;
    std::cout << "<--default_max_tokens> <args>: 请求省略max_tokens时的默认输出上限（默认16384）" << std::endl;
    std::cout << "<--model_name> <args>:        模型名(openai api中使用)" << std::endl;
    std::cout << "<--port> <args>:              网页端口号" << std::endl;
    std::cout << "<--cuda_embedding>:           使用cuda来执行embedding" << std::endl;
    std::cout << "<--device>:                   执行设备" << std::endl;
}

void ParseArgs(int argc, char **argv, APIConfig &config) {
    std::vector<std::string> sargv;
    for (int i = 0; i < argc; i++) {
        sargv.push_back(std::string(argv[i]));
    }
    for (int i = 1; i < argc; i++) {
        if (sargv[i] == "-h" || sargv[i] == "--help") {
            Usage();
            exit(0);
        } else if (sargv[i] == "-p" || sargv[i] == "--path") {
            config.path = sargv[++i];
        } else if (sargv[i] == "--mmproj") {
            fastllm::AssertInFastLLM(
                i + 1 < argc, "--mmproj requires a value");
            config.multimodalProjectorPath = sargv[++i];
        } else if (sargv[i] == "-t" || sargv[i] == "--threads") {
            config.threads = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-l" || sargv[i] == "--low") {
            config.lowMemMode = true;
        } else if (sargv[i] == "--cuda_embedding"){
            config.cudaEmbedding = true;
        } else if (sargv[i] == "--port") {
            config.port = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--dtype") {
            std::string dtypeStr = sargv[++i];
            if (dtypeStr.size() > 5 && dtypeStr.substr(0, 5) == "int4g") {
                config.groupCnt = atoi(dtypeStr.substr(5).c_str());
                dtypeStr = dtypeStr.substr(0, 5);
            }
            fastllm::AssertInFastLLM(dataTypeDict.find(dtypeStr) != dataTypeDict.end(),
                                    "Unsupport data type: " + dtypeStr);
            config.dtype = dataTypeDict[dtypeStr];
        } else if (sargv[i] == "--tokens") {
            config.tokens = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--chunked_prefill_size" ||
                   sargv[i] == "--chunked-prefill-size") {
            fastllm::AssertInFastLLM(
                i + 1 < argc,
                "--chunked_prefill_size requires a value");
            std::string error;
            fastllm::AssertInFastLLM(
                ParsePositiveInt(
                    sargv[++i], config.chunkedPrefillSize, error),
                "Invalid --chunked_prefill_size: " + error);
        } else if (sargv[i] == "--default_max_tokens" || sargv[i] == "--default-max-tokens") {
            fastllm::AssertInFastLLM(i + 1 < argc,
                                    "--default_max_tokens requires a value");
            std::string error;
            fastllm::AssertInFastLLM(
                ParsePositiveInt(sargv[++i], config.defaultMaxTokens, error),
                "Invalid --default_max_tokens: " + error);
        } else if (sargv[i] == "--batch") {
            config.batch = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--atype") {
            std::string atypeStr = sargv[++i];
            fastllm::AssertInFastLLM(dataTypeDict.find(atypeStr) != dataTypeDict.end(),
                                    "Unsupport act type: " + atypeStr);
            config.atype = dataTypeDict[atypeStr];
        } else if (sargv[i] == "--kv_cache_dtype") {
            try {
                config.kvCacheDtype = fastllm::ParseKVCacheDataType(sargv[++i]);
            } catch (const std::invalid_argument &error) {
                fastllm::AssertInFastLLM(false, error.what());
            }
        } else if (sargv[i] == "--model_name") {
            config.modelName = sargv[++i];
        } else if (sargv[i] == "--device") {
            config.devices[sargv[++i]] = 1;
        } else {
            Usage();
            exit(-1);
        }
    }
}

// 64 MiB request buffer: the accept loop reads until Content-Length is
// satisfied (or the buffer is full), so a request whose body exceeds the
// buffer is dropped with a connection close. Long-context requests
// (>=262144 tokens) carry multi-MiB JSON bodies; 4K images re-encoded by
// the proxy to PNG reach ~10 MiB of base64 JSON, so 8 MiB rejected them
// with a silent connection close (ReadError/503 upstream). Must stay >=
// the caps in HttpRequest::Init/IsValid.
char buff[64 * 1024 * 1024] = {0};
std::string url = "generate";
std::mutex locker;

int main(int argc, char** argv) {
    ParseArgs(argc, argv, config);

    if (config.devices.size() != 0) {
        fastllm::SetDeviceMap(config.devices);
    }
    fastllm::SetThreads(config.threads);
    fastllm::SetLowMemMode(config.lowMemMode);
    fastllm::SetCudaEmbedding(config.cudaEmbedding);
    if (!fastllm::FileExists(config.path)) {
        printf("模型文件 %s 不存在！\n", config.path.c_str());
        exit(0);
    }
    if (!config.multimodalProjectorPath.empty() &&
        !fastllm::FileExists(config.multimodalProjectorPath)) {
        printf("多模态投影器文件 %s 不存在！\n",
               config.multimodalProjectorPath.c_str());
        exit(0);
    }
    bool isHFDir = fastllm::FileExists(config.path + "/config.json") || fastllm::FileExists(config.path + "config.json");
    fastllm::AssertInFastLLM(
        !isHFDir || config.multimodalProjectorPath.empty(),
        "--mmproj currently requires a GGUF text model.");
    // 分阶段加载进度：带时间戳打到 stdout，按 5% 桶节流避免刷屏。
    fastllm::SetModelLoadProgressCallback([](const fastllm::ModelLoadProgress &p) {
        static std::string lastStage;
        static int lastBucket = -1;
        static auto lastPrint =
            std::chrono::steady_clock::now() - std::chrono::seconds(10);
        const int pct = p.total > 0 ? (int)(p.current * 100 / p.total) : 100;
        const int bucket = pct / 5;
        const auto now = std::chrono::steady_clock::now();
        const bool stageChanged = p.stage != lastStage;
        const bool done = p.total > 0 && p.current >= p.total;
        if (!stageChanged && !done && bucket == lastBucket &&
            now - lastPrint < std::chrono::seconds(2)) {
            return;
        }
        lastStage = p.stage;
        lastBucket = bucket;
        lastPrint = now;
        if (p.totalBytes > 0) {
            LogTs("[Load] %-16s %3d%% (%.2f/%.2f GiB)\n", p.stage.c_str(), pct,
                  p.completedBytes / 1073741824.0, p.totalBytes / 1073741824.0);
        } else {
            LogTs("[Load] %-16s %3d%% (%llu/%llu)\n", p.stage.c_str(), pct,
                  (unsigned long long)p.current, (unsigned long long)p.total);
        }
    });
    // 先占端口, 再加载模型。
    //
    // 原顺序是"加载 16GB 权重 -> warmup -> 启工作线程 -> 建 socket -> bind"。
    // 2026-08-20 出过一次事故: 重启时上一个进程还占着 8002, 新进程把权重全部
    // 读完(RSS 20GB、约 3 分钟)才走到 bind, 然后失败。
    //
    // 更糟的是它**没能退出**。bind 失败时确实调了 exit(-1), 但那时工作线程已经
    // 起来、CUDA 上下文和 20GB 权重都在, exit() 要跑静态析构与 atexit ——
    // 其中任何一处阻塞就永远出不去。于是进程活着、握着 20GB 主机内存、GPU 上
    // 只有几百 MB(权重要到第一次请求才上卡), proxy 侧永远 backend=STARTING,
    // 客户端全部超时。僵持 23 分钟才被人工发现。
    //
    // 把 bind 提前: 端口冲突在毫秒级暴露, 此时既没有工作线程也没有 CUDA 上下文,
    // exit(-1) 干净利落; 顺带省掉那次白跑的 16GB 机械盘加载。
    // listen() 仍留在模型就绪之后 —— 端口已绑但未监听时连接会被拒绝, 这正是
    // 想要的语义: 上游立刻知道"还没准备好", 而不是连上以后干等。
    int local_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (local_fd == -1) {
        std::cout << "socket error!" << std::endl;
        exit(-1);
    }
    {
        // SO_REUSEADDR 只放行 TIME_WAIT 残留, **不会**掩盖"另一个进程正在监听"
        // 这种真冲突(那在 Linux 上需要 SO_REUSEPORT), 所以不削弱上面的保护。
        int reuseOne = 1;
        setsockopt(local_fd, SOL_SOCKET, SO_REUSEADDR, &reuseOne,
                   sizeof(reuseOne));
    }
    {
        struct sockaddr_in early_addr;
        memset(&early_addr, 0, sizeof(early_addr));
        early_addr.sin_family = AF_INET;
        early_addr.sin_port = htons(config.port);
        early_addr.sin_addr.s_addr = INADDR_ANY;
        if (bind(local_fd, (struct sockaddr *) &early_addr,
                 sizeof(early_addr)) == -1) {
            std::cout << "bind error! port " << config.port
                      << " already in use (nothing loaded yet, exiting)"
                      << std::endl;
            exit(-1);
        }
    }
    LogTs("[Server] port %d bound (model not loaded yet)\n", config.port);
    LogTs("[Load] start model load: %s\n", config.path.c_str());
    workQueue.model = isHFDir
        ? fastllm::CreateLLMModelFromHF(
              config.path, config.dtype, config.groupCnt)
        : fastllm::CreateLLMModelFromFile(
              config.path, config.multimodalProjectorPath);
    workQueue.model->SetTokenLimit(config.tokens);
    if (config.chunkedPrefillSize > 0) {
        workQueue.model->SetChunkedPrefillSize(
            config.chunkedPrefillSize);
    }
    workQueue.model->SetDataType(config.atype);
    if (config.kvCacheDtype != fastllm::DataType::DATA_AUTO_NONE) {
        workQueue.model->SetKVCacheDataType(config.kvCacheDtype);
    }
    // Paged KV managers are preallocated lazily on the first request (see
    // WorkQueue::Deal) so the ~5 GB allocation never stacks with the model
    // load peak or the persistent-prefix-cache restore.
    workQueue.maxActivateQueryNumber = std::max(1, std::min(256, config.batch));
    workQueue.model->maxBatch = workQueue.maxActivateQueryNumber;
    PrepareServerPersistentPrefixCache(workQueue.model.get());
    // Eagerly precreate paged-KV managers even on a cold start.  Prefix
    // queries run BEFORE the first prefill, but managers used to be created
    // lazily during that first prefill — so every generation's first
    // request found zero managers and reported miss=other, and under the
    // crash loop (1-2 requests per generation) the L1 trie could never
    // serve a single hit.  Lazy page growth keeps this cheap (~128 physical
    // pages per manager); the "~5 GB stacks with load peak" concern in the
    // comment above predates the on-demand page pool.
    {
        std::string pagedPrepareError;
        if (!workQueue.model->PreparePersistentPrefixCacheManagers(
                &pagedPrepareError)) {
            std::fprintf(
                stderr,
                "[Prefix-persist] eager paged manager precreate failed: %s\n",
                pagedPrepareError.empty() ? "unknown error"
                                          : pagedPrepareError.c_str());
        }
    }
    workQueue.ConfigureHostOffloadManager();
    // Warmup before accepting traffic: the load peak has passed, so the
    // H2D weight transfer + paged-KV sizing allocations here cannot stack
    // with it. Clients see READY only after VRAM reaches steady state.
    // Disable with FASTLLM_SKIP_WARMUP=1.
    if (!fastllm::GetFastllmEnv().skipWarmup) {
        LogTs("[Load] warmup start\n");
        workQueue.model->AutoWarmup();
        LogTs("[Load] warmup done\n");
    }
    workQueue.Start();
    LogTs("[Load] model loaded, workers started\n");

    LogTs("[Server] socket ready!\n");
#ifndef _WIN32
    serverSocketForSignal = local_fd;
    std::signal(SIGINT, HandleShutdownSignal);
    std::signal(SIGTERM, HandleShutdownSignal);
#endif

    // 端口已在加载模型之前 bind 过(见上面那段注释), 这里只需开始监听。
    LogTs("[Server] bind ready!\n");
    listen(local_fd, 2000);    
    LogTs("[Server] ready, listening on port %d\n", config.port);
    int queuePos = 0;
    while (true) { //循环接收客户端的请求
        //5.创建一个sockaddr_in结构体，用来存储客户机的地址
        struct sockaddr_in client_addr;
        socklen_t len = sizeof(client_addr);
        //6.accept()函数：阻塞运行，直到收到某一客户机的连接请求，并返回客户机的描述符
        int client = accept(local_fd, (struct sockaddr *) &client_addr, &len);
        if (client == -1) {
#ifndef _WIN32
            if (shutdownSignal != 0) {
                break;
            }
#endif
            std::perror("accept");
            workQueue.StopAndDrain();
            return -1;
        }

#ifdef _WIN32
        DWORD receiveTimeoutMs = 15000;
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO,
                   reinterpret_cast<const char *>(&receiveTimeoutMs),
                   sizeof(receiveTimeoutMs));
#else
        struct timeval receiveTimeout = {15, 0};
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO,
                   &receiveTimeout, sizeof(receiveTimeout));
#endif

        int size = 0;
        bool requestReady = false;
        while (size < (int)sizeof(buff) - 1) {
            int cur = read(client, buff + size, sizeof(buff) - 1 - size);
            if (cur <= 0) {
                break;
            }
            size += cur;
            buff[size] = 0;
            if (IsHttpRequestComplete(buff, static_cast<size_t>(size))) {
                requestReady = true;
                break;
            }
        }
        if (!requestReady) {
            close(client);
            continue;

        }

        workQueue.Push(buff, client);
    }

#ifndef _WIN32
    serverSocketForSignal = -1;
#endif
    workQueue.StopAndDrain();
    int shutdownStatus = 0;
#ifndef _WIN32
    if (shutdownSignal != 0 &&
        fastllm::GetPersistentPrefixCacheStatus().enabled &&
        !workQueue.suspended.load(std::memory_order_acquire) &&
        workQueue.cacheMutationEpoch.load(std::memory_order_relaxed) !=
            workQueue.checkpointedMutationEpoch.load(
                std::memory_order_relaxed)) {
        fastllm::PersistentPrefixCheckpointStats stats;
        std::string checkpointError;
        if (!fastllm::CheckpointPersistentPrefixCache(
                workQueue.model.get(), stats, &checkpointError)) {
            std::fprintf(
                stderr,
                "[Prefix-persist] shutdown checkpoint failed: %s\n",
                checkpointError.empty() ?
                    "unknown error" : checkpointError.c_str());
            shutdownStatus = 1;
        } else {
            std::fprintf(
                stderr,
                "[Prefix-persist] shutdown checkpoint generation %llu, "
                "%llu pages, %llu bytes, %.3f ms\n",
                (unsigned long long)stats.generation,
                (unsigned long long)stats.pages,
                (unsigned long long)stats.bytes,
                stats.durationMs);
        }
    }
#endif
    return shutdownStatus;
}
