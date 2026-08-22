// FastLLM apiserver 内嵌管理 WebUI。
//
// 设计(用户定的规格): 朴素单页, 无外部依赖; Bearer token 认证(与 .env 的
// AUTH_TOKEN 一致); 展示后端/代理日志、各级缓存(页池/L1 trie/L2 RAM/L3 磁盘)、
// 显存对账、内核路由普查; 在线切换/编辑 profile、启停服务。
//
// 为什么放 C++ 而不是外部轮询: 引擎状态(GetPagedCachePoolStats /
// GetPrefixCacheStatsSnapshot / GetVramBreakdown)是进程内数据, 这里直接读,
// 不需要任何中间层; 页面本身由 apiserver 直接服务(:8002/admin), 也可以经
// thinking_proxy 的 passthrough 访问(:8000/admin)。
//
// 切换/停止的实现约束: apiserver 无法"重启自己"。POST /admin/api/switch
// fork+setsid 一个 detached 脚本: sleep 2(让 HTTP 响应先落地) ->
// start_prod.sh <profile>(它负责杀旧 proxy/后端并按新 profile 拉起全套)。
// 浏览器侧页面在连接断开后持续重试 /admin/api/state, 新实例起来即恢复。
#include <string>
#include <unordered_map>
#include <vector>
namespace fastllm {
    namespace apiserver {

        // ---- 认证 ----

        // 从请求头集合里提取 "Authorization: Bearer <token>" 并与环境变量
        // AUTH_TOKEN 常量时间比较。AUTH_TOKEN 未设置或为空时一律拒绝
        // (fail closed): 管理面不允许无鉴权裸奔。
        bool AdminWebUiCheckAuth(
                const std::unordered_map<std::string, std::string> &headers);

        // ---- 日志 ----

        // 读日志文件尾部。maxBytes 兜底防止超大日志吃内存; 返回最后
        // maxLines 行(不足则全量)。文件不存在返回 false。
        bool AdminWebUiReadLogTail(const std::string &path,
                                   size_t maxLines, size_t maxBytes,
                                   std::string &out);

        // ---- profile 管理 ----

        struct AdminProfileInfo {
            std::string name;      // 文件名(xxx.env)
            std::string path;      // 绝对路径
            long long mtime = 0;   // 修改时间(epoch 秒)
            bool active = false;   // 是否当前运行实例所用
        };

        // 扫描 profiles 目录($PROJECT_DIR/runtime/fastllm-native-profiles,
        // 可被 FASTLLM_ADMIN_PROFILES_DIR 覆盖), 列出 *.env(跳过 .bak*),
        // 按 mtime 降序。activeProfileLogName 传入 PROXY_LOG_FILE 的完整值,
        // 用文件名匹配判断哪个 profile 是活的。
        std::vector<AdminProfileInfo> AdminWebUiListProfiles(
                const std::string &activeProxyLogFile);

        // ---- profile 编辑 ----

        struct AdminProfileKey {
            std::string key;
            std::string value;      // 已剥引号的原始值
            bool boolean = false;   // 值形如 0/1/true/false/yes/no/on/off
            std::string comment;    // 行内注释(若有)
            size_t lineIndex = 0;   // 在文件行数组里的下标
        };

        // 解析 profile 文件: 返回全部键(按出现顺序), 空文件/不存在返回空。
        bool AdminWebUiParseProfile(const std::string &path,
                                    std::vector<AdminProfileKey> &keys,
                                    std::string &error);

        // 写回 N 个键值对(值不含引号; 布尔传 "0"/"1")。原子写(临时文件 +
        // rename), 未改动的行逐字节保留, 行内注释保留。
        bool AdminWebUiWriteProfileKeys(
                const std::string &path,
                const std::vector<std::pair<std::string, std::string>> &updates,
                std::string &error);

        // ---- 进程动作(switch/stop) ----

        // fork + setsid + execl("/bin/sh", "-c", command)。
        // detached: 父进程退出不影响子进程。成功返回 true。
        bool AdminWebUiSpawnDetached(const std::string &command);
    }
}
