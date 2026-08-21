#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apiserver_tui.py — FastLLM apiserver 的 TUI 管理器 (stdlib curses, 无第三方依赖)

取代「手改 .env + 手跑 launch_proxy_tmux.sh」的流程:
  A. 浏览/选择 v100-perfs/runtime/fastllm-native-profiles/*.env profile
  B. 分组编辑加速参数 (枚举/布尔/整数步进控件, 不做数值自由文本输入)
  C. 回写 .env: FASTLLM_BACKEND_COMMAND 整串重建, 其它 FASTLLM_* 键值原地替换,
     未改动的行逐字节保留; 写盘前自动备份 *.bak-tui-<时间戳> 并要求二次确认
  D. 生命周期: launch / stop / restart (走 v100-perfs/scripts/launch_proxy_tmux.sh,
     profile 一律传绝对路径 —— 该脚本的 PROXY_SHELL pane cwd 是 v100-perfs)
  E. 只读状态面板: 8000/8002 端口、后端 /health、/props 关键计数器、GPU 显存

参数名全部来自真实文件:
  - profile 里的 FASTLLM_* 变量 (thinking_proxy.py 与 fastllm C++ 源码消费)
  - FASTLLM_BACKEND_COMMAND 里 apiserver 的命令行参数 (example/apiserver/apiserver.cpp)

用法:
  python3 apiserver_tui.py                 # 交互界面
  python3 apiserver_tui.py --dry-run       # 可以编辑但绝不写盘
  python3 apiserver_tui.py --check         # 只解析全部 profile 并自检 (无需 TTY)
  python3 apiserver_tui.py --dump <env>    # 打印单个 profile 的解析结果 (无需 TTY)
"""

import argparse
import curses
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from urllib.parse import urlparse

# ─── 路径与常量 ────────────────────────────────────────────────────
# 本文件位于 <项目根>/fastllm/tools/apiserver_tui.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES_DIR = (
    PROJECT_ROOT / "v100-perfs" / "runtime" / "fastllm-native-profiles"
)
LAUNCHER_SCRIPT = PROJECT_ROOT / "v100-perfs" / "scripts" / "launch_proxy_tmux.sh"
# launch_proxy_tmux.sh 的默认会话/窗口名 (生产即用这组)
TMUX_SESSION = "fastllm-prod"
TMUX_WINDOW = "proxy-8000"
TMUX_BIN = shutil.which("tmux") or "/usr/bin/tmux"

GiB = 1024 ** 3

# FASTLLM_BACKEND_COMMAND 里带值参数的白名单 (来自 apiserver.cpp ParseArgs)
CMD_VALUE_FLAGS = {
    "-p", "--path", "--mmproj", "-t", "--threads", "--port", "--dtype",
    "--tokens", "--chunked_prefill_size", "--chunked-prefill-size",
    "--default_max_tokens", "--default-max-tokens", "--batch", "--atype",
    "--kv_cache_dtype", "--model_name", "--device",
}


# ─── .env 解析 / 回写 ──────────────────────────────────────────────
# 需求: 除被编辑的键以外, 文件内容逐字节不变。策略:
#   每一行解析成 (key, value, quote, comment); 未改动行渲染时直接吐回原行文本。
# 兼容三种真实存在的写法:
#   KEY=裸值
#   KEY='带空格的整串'            (FASTLLM_BACKEND_COMMAND)
#   KEY=值  # 行尾中文注释        (bash 赋值后空格+# 即注释, 源文件里真实出现)
@dataclass
class EnvLine:
    raw: str                    # 原始行文本 (未改动时原样吐回)
    key: str = ""               # 变量名; 注释行/空行为空
    value: str = ""             # 去引号/去注释后的值
    quote: str = ""             # 原始引号: '' / "'" / '"'
    comment: str = ""           # 行尾注释 (含它前面的空白), 回写时保留
    modified: bool = False
    appended: bool = False      # 原文件没有、由编辑器新增的键


_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _split_value(rest: str):
    """把 '=' 之后的部分拆成 (值, 引号, 行尾注释)。"""
    if rest.startswith("'"):
        end = rest.find("'", 1)
        if end >= 0:
            return rest[1:end], "'", rest[end + 1:]
        return rest[1:], "'", ""
    if rest.startswith('"'):
        end = rest.find('"', 1)
        if end >= 0:
            return rest[1:end], '"', rest[end + 1:]
        return rest[1:], '"', ""
    # 裸值: 空白后跟 '#' 才是注释 (bash 语义)
    m = re.search(r"\s#", rest)
    if m:
        return rest[: m.start()].rstrip(), "", rest[m.start():]
    return rest.strip(), "", ""


class EnvFile:
    """一个 profile .env 的可回写内存表示。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.text = self.path.read_text(encoding="utf-8")
        self.lines = [self._parse(raw) for raw in self.text.split("\n")]
        self.index = {}  # key -> 行号 (第一次出现生效, 与 bash source 一致)
        for i, ln in enumerate(self.lines):
            if ln.key and ln.key not in self.index:
                self.index[ln.key] = i

    @staticmethod
    def _parse(raw: str) -> EnvLine:
        m = _ASSIGN_RE.match(raw)
        if not m:
            return EnvLine(raw=raw)
        value, quote, comment = _split_value(m.group(2))
        return EnvLine(raw=raw, key=m.group(1), value=value,
                       quote=quote, comment=comment)

    def get(self, key):
        i = self.index.get(key)
        return self.lines[i].value if i is not None else None

    def set(self, key, value, quote=""):
        """修改变量值; 原文件没有该键时追加到文件末尾。"""
        if key in self.index:
            ln = self.lines[self.index[key]]
        else:
            ln = EnvLine(raw="", key=key, appended=True)
            self.lines.append(ln)
            self.index[key] = len(self.lines) - 1
        ln.value = value
        if quote:
            ln.quote = quote
        ln.modified = True

    def render(self) -> str:
        out = []
        for ln in self.lines:
            if not ln.key:
                out.append(ln.raw)          # 空行 / 纯注释行: 原样
                continue
            if not ln.modified:
                out.append(ln.raw)          # 未改动: 原样 (保证逐字节一致)
                continue
            v = ln.value
            if ln.quote == "'":
                body = "'" + v.replace("'", "'\\''") + "'"
            elif ln.quote == '"':
                body = '"' + v + '"'
            else:
                body = v
            line = f"{ln.key}={body}"
            if ln.comment:
                line += ln.comment          # 保留行尾注释
            out.append(line)
        text = "\n".join(out)
        if self.text.endswith("\n") and not text.endswith("\n"):
            text += "\n"
        return text


# ─── FASTLLM_BACKEND_COMMAND 解析 / 重建 ──────────────────────────
def parse_command(cmd: str):
    """整串命令行 -> (二进制路径, [(flag, value|None), ...])。"""
    toks = shlex.split(cmd)
    if not toks:
        return "", []
    binary, pairs, i = toks[0], [], 1
    while i < len(toks):
        t = toks[i]
        if (t.startswith("--") or (t.startswith("-") and len(t) == 2)) \
                and t in CMD_VALUE_FLAGS and i + 1 < len(toks):
            pairs.append((t, toks[i + 1]))
            i += 2
        elif t.startswith("-"):
            pairs.append((t, None))          # 纯开关
            i += 1
        else:
            pairs.append((None, t))          # 游离 token, 原样保留
            i += 1
    return binary, pairs


def build_command(binary: str, pairs) -> str:
    parts = [binary]
    for flag, value in pairs:
        if flag is not None:
            parts.append(flag)
        if value is not None:
            parts.append(value)
    return " ".join(shlex.quote(p) for p in parts)


def cmd_value(pairs, flag):
    for f, v in pairs:
        if f == flag:
            return v
    return None


# ─── 字段定义 ──────────────────────────────────────────────────────
# kind: bool(0/1 开关) | int(整数步进) | float(小数步进) | enum(枚举循环) | path(路径选择)
# fmt:  bytes_gib(字节, 界面按 GiB) | mb(MB) | gib(GiB) | sec(秒) | ""
@dataclass
class FieldSpec:
    key: str                 # '--batch' 或 'FASTLLM_...'
    kind: str
    where: str               # 'cmd'(命令行参数) / 'env'(独立环境变量)
    help: str                # 中文说明 (含取值范围)
    lo: float = 0
    hi: float = 1
    step: float = 1
    options: list = dc_field(default_factory=list)   # enum 选项
    presets: list = dc_field(default_factory=list)   # Enter 弹出的快捷值
    fmt: str = ""
    start: float = 0         # 文件里未设置时的起始值


def _field_specs():
    """五组参数。全部变量名/参数名来自真实 profile 与源码, 未做任何臆造。"""
    G1 = ("① 模型/量化 (apiserver 命令行)", [
        FieldSpec("--path", "path", "cmd",
                  "GGUF 权重路径。换模型要同步换 --mmproj/--model_name"),
        FieldSpec("--mmproj", "path", "cmd",
                  "多模态 vision projector GGUF 路径"),
        FieldSpec("--kv_cache_dtype", "enum", "cmd",
                  "KV Cache 类型。turbo3/turbo4 = q8_0 K + TurboQuant V "
                  "(Qwen3.5/3.6/3.8), 显存最省; fp8_e4m3 需配套 kernel",
                  options=["auto", "float32", "float16", "bfloat16",
                           "fp8_e4m3", "turbo3", "turbo4"]),
        FieldSpec("--atype", "enum", "cmd",
                  "激活/推理数据类型。生产用 float16 (apiserver help 只列 "
                  "float32/float16)",
                  options=["float32", "float16", "half", "int8"]),
        FieldSpec("--batch", "int", "cmd",
                  "最大并发 batch (请求槽位数)。范围 1..64, 生产 1..4",
                  lo=1, hi=64, step=1, presets=[1, 2, 4, 8, 16], start=1),
        FieldSpec("--threads", "int", "cmd",
                  "CPU 线程数。范围 1..128, 本机生产用 2",
                  lo=1, hi=128, step=1, presets=[1, 2, 4, 8], start=2),
        FieldSpec("--tokens", "int", "cmd",
                  "KV token 池总容量 (所有请求共享)。262144 = 262K 上下文",
                  lo=8192, hi=1048576, step=16384, fmt="",
                  presets=[32768, 65536, 131072, 262144, 393216, 524288],
                  start=262144),
        FieldSpec("--default_max_tokens", "int", "cmd",
                  "请求省略 max_tokens 时的默认输出上限",
                  lo=1024, hi=262144, step=4096,
                  presets=[4096, 8192, 16384, 32768, 65536], start=16384),
        FieldSpec("--device", "enum", "cmd", "执行设备",
                  options=["cuda", "cpu"]),
    ])
    G2 = ("② MTP/加速 (FASTLLM_QWEN35_* 与 SM70 算子)", [
        FieldSpec("FASTLLM_QWEN35_ENABLE_MTP", "int", "env",
                  "MTP 投机解码: 每步 draft token 数。0=关, 1..9=草稿数 "
                  "(上限 QWEN35_MTP_MAX_DRAFTS=9), 生产用 2",
                  lo=0, hi=9, step=1, presets=[0, 1, 2], start=0),
        FieldSpec("FASTLLM_QWEN35_TURBO3_KV", "bool", "env",
                  "turbo3 KV kernel 开关 (q8_0 K + TurboQuant V), "
                  "需与 --kv_cache_dtype turbo3 配套",
                  hi=1, start=0),
        FieldSpec("FASTLLM_QWEN35_TURBO4_KV", "bool", "env",
                  "turbo4 KV kernel 开关, 需与 --kv_cache_dtype turbo4 配套",
                  hi=1, start=0),
        FieldSpec("FASTLLM_QWEN35_INTERLEAVE_LONG_PREFILL", "bool", "env",
                  "长 prefill 与 decode 交错执行, 避免流式输出被长 prefill "
                  "整段卡住",
                  hi=1, start=0),
        FieldSpec("FASTLLM_CUDA_SM70_TURBO_XQA", "bool", "env",
                  "V100(SM70) turbo KV 专用 XQA decode kernel, 对 "
                  "turbo3/turbo4 KV 有实测加速 (8K/32K/128K: 2.2x/3.4x/4.0x)",
                  hi=1, start=0),
        FieldSpec("FASTLLM_CUDA_PAGED_CUBLAS_BATCH_GQA", "bool", "env",
                  "paged attention 走 cuBLAS 批量 GQA 路径 "
                  "(chunked cublas attention)",
                  hi=1, start=0),
        FieldSpec("FASTLLM_CUDA_SM70_FLASH_ATTN", "bool", "env",
                  "SM70 flash attention prefill。要求分页 KV 都是 fp8_e4m3, "
                  "turbo3/turbo4 档位下是 no-op (取 0 取 1 一样)",
                  hi=1, start=0),
        FieldSpec("FASTLLM_CUDA_SM70_PAGED_XQA", "bool", "env",
                  "SM70 paged XQA。要求分页 KV 都是 float16, "
                  "turbo3/turbo4 档位下是 no-op",
                  hi=1, start=0),
    ])
    G3 = ("③ 显存/池", [
        FieldSpec("FASTLLM_PAGED_POOL_MAX_MB", "int", "env",
                  "KV 分页池显存上限 (MB)。生产 10600 ≈ 10.3GiB",
                  lo=0, hi=32768, step=256, fmt="mb",
                  presets=[4096, 6144, 8192, 10600, 12288], start=0),
        FieldSpec("FASTLLM_VRAM_MIN_FREE_GIB", "float", "env",
                  "显存空闲低于该值 (GiB) 时, owned 后端挂起/卸载让出 GPU",
                  lo=0, hi=32, step=0.25, fmt="gib",
                  presets=[0.5, 1.0, 1.5, 2.0, 3.0], start=1.5),
        FieldSpec("FASTLLM_VRAM_RESUME_FREE_GIB", "float", "env",
                  "显存空闲回到该值 (GiB) 以上时恢复挂起的后端",
                  lo=0, hi=32, step=0.25, fmt="gib",
                  presets=[1.0, 2.0, 2.5, 3.0, 4.0], start=2.5),
        FieldSpec("FASTLLM_VRAM_POOL_RESERVE_MB", "int", "env",
                  "为 KV 池预占的显存 (MB), 避免与其它进程抢满显存",
                  lo=0, hi=8192, step=128, fmt="mb",
                  presets=[0, 512, 1024, 1536, 2048], start=0),
    ])
    G4 = ("④ 前缀缓存 (三级: GPU/CPU/磁盘)", [
        FieldSpec("FASTLLM_PREFIX_CACHE_CPU_TIER", "bool", "env",
                  "CPU(主机内存)层开关。命中几乎白赚: 主机内存->GPU 带宽约 "
                  "550x 重算速度",
                  hi=1, start=0),
        FieldSpec("FASTLLM_PREFIX_CACHE_CPU_MAX_BYTES", "int", "env",
                  "CPU 层配额。界面按 GiB 编辑; 生产 16GiB≈470K token",
                  lo=0, hi=64 * GiB, step=GiB, fmt="bytes_gib",
                  presets=[0, 4 * GiB, 8 * GiB, 16 * GiB, 24 * GiB,
                           32 * GiB], start=0),
        FieldSpec("FASTLLM_PREFIX_CACHE_DISK_MAX_BYTES", "int", "env",
                  "磁盘层配额上限 (按 GiB 编辑)。太小会导致关机存档失败、"
                  "重启丢全部前缀缓存; 生产 32GiB",
                  lo=0, hi=2048 * GiB, step=GiB, fmt="bytes_gib",
                  presets=[2 * GiB, 8 * GiB, 16 * GiB, 32 * GiB, 64 * GiB],
                  start=0),
        FieldSpec("FASTLLM_PREFIX_CACHE_DISK_MIN_FREE_BYTES", "int", "env",
                  "磁盘层保留空闲下限 (按 GiB 编辑), 防止写满盘; 生产 8GiB",
                  lo=0, hi=1024 * GiB, step=GiB, fmt="bytes_gib",
                  presets=[1 * GiB, 4 * GiB, 8 * GiB, 16 * GiB], start=0),
        FieldSpec("FASTLLM_PREFIX_CACHE_PERSIST", "bool", "env",
                  "关机/启动时持久化前缀缓存 (磁盘存档)",
                  hi=1, start=0),
        FieldSpec("FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES", "int",
                  "env",
                  "GDN/linear 状态快照间隔 (页数)。生产 4",
                  lo=1, hi=64, step=1, presets=[1, 2, 4, 8], start=4),
    ])
    G5 = ("⑤ 生命周期", [
        FieldSpec("FASTLLM_IDLE_TIMEOUT", "int", "env",
                  "空闲多少秒后卸载后端。0=常驻不卸载 (生产值)",
                  lo=0, hi=7 * 86400, step=300, fmt="sec",
                  presets=[0, 600, 1800, 3600], start=0),
        FieldSpec("FASTLLM_START_TIMEOUT", "int", "env",
                  "后端冷启动 (加载权重) 超时秒数。机械盘加载 16GB 需数分钟",
                  lo=30, hi=7200, step=60, fmt="sec",
                  presets=[300, 900, 1800, 3600], start=900),
        FieldSpec("FASTLLM_SKIP_WARMUP", "bool", "env",
                  "跳过后端 warmup: 加载完直接就绪 (省启动时间, 首请求略慢)",
                  hi=1, start=0),
        FieldSpec("FASTLLM_OWNED", "bool", "env",
                  "proxy 托管后端子进程: 启动即拉起/空闲卸载。需要 "
                  "FASTLLM_BACKEND_COMMAND 与 FASTLLM_BACKEND_URL",
                  hi=1, start=1),
        FieldSpec("FASTLLM_TOOLCALL_GRAMMAR", "bool", "env",
                  "工具调用约束解码 (grammar) 开关",
                  hi=1, start=0),
    ])
    return [G1, G2, G3, G4, G5]


GROUPS = _field_specs()
ALL_SPECS = {s.key: (gi, s) for gi, (_, fs) in enumerate(GROUPS) for s in fs}


# ─── 值的显示 / 转换 ───────────────────────────────────────────────
def _to_bool(s):
    return str(s).strip().lower() in ("1", "true", "yes", "on")


def load_value(spec: FieldSpec, raw):
    """把 .env/命令行里的字符串值转成编辑器的内部值; None=未设置。"""
    if raw is None:
        return None
    if spec.kind == "bool":
        return _to_bool(raw)
    if spec.kind == "int":
        try:
            return int(float(raw))
        except ValueError:
            return None
    if spec.kind == "float":
        try:
            return float(raw)
        except ValueError:
            return None
    return str(raw)  # enum / path


def fmt_value(spec: FieldSpec, v):
    if v is None:
        return "(未设置)"
    if spec.kind == "bool":
        return "1 (开)" if v else "0 (关)"
    if spec.fmt == "bytes_gib":
        return f"{int(v) / GiB:.2f} GiB"
    if spec.fmt == "mb":
        return f"{int(v)} MB"
    if spec.fmt == "gib":
        return _trim_float(v) + " GiB"
    if spec.fmt == "sec":
        return f"{int(v)} 秒"
    return str(v)


def serialize_value(spec: FieldSpec, v) -> str:
    """内部值 -> 写回 .env 的字符串。"""
    if spec.kind == "bool":
        return "1" if v else "0"
    if spec.kind == "int":
        return str(int(v))
    if spec.kind == "float":
        return _trim_float(v)
    return str(v)


def _trim_float(v):
    v = float(v)
    if v == int(v):
        return str(int(v))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def human_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


# ─── Profile ───────────────────────────────────────────────────────
class Profile:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.name = self.path.stem
        self.env = EnvFile(self.path)
        self.cmd_binary, self.cmd_pairs = parse_command(
            self.env.get("FASTLLM_BACKEND_COMMAND") or "")
        self.error = ""

    def cmd(self, flag):
        return cmd_value(self.cmd_pairs, flag)

    @property
    def model_path(self):
        return self.cmd("--path") or ""

    def summary(self):
        kv = self.cmd("--kv_cache_dtype") or "?"
        mtp = self.env.get("FASTLLM_QWEN35_ENABLE_MTP") or "0"
        batch = self.cmd("--batch") or "?"
        tokens = self.cmd("--tokens") or "?"
        port = self.cmd("--port") or "?"
        return f"kv={kv} mtp={mtp} batch={batch} tokens={tokens} port={port}"

    def backend_port(self):
        url = self.env.get("FASTLLM_BACKEND_URL") or ""
        try:
            return urlparse(url).port or 8002
        except ValueError:
            return 8002

    def proxy_port(self):
        try:
            return int(self.env.get("PROXY_PORT") or 8000)
        except ValueError:
            return 8000


class BrokenProfile:
    """解析失败的 .env: 也列出来并标错, 但不可编辑。"""

    def __init__(self, path, err):
        self.path = Path(path)
        self.name = self.path.stem
        self.error = str(err)
        self.env = None
        self.cmd_binary, self.cmd_pairs = "", []
        self.model_path = ""

    def cmd(self, flag):
        return None

    def summary(self):
        return f"[解析失败: {self.error}]"

    def backend_port(self):
        return 8002

    def proxy_port(self):
        return 8000


def load_profiles(profiles_dir: Path):
    out = []
    for p in sorted(profiles_dir.glob("*.env")):
        # 排除备份文件 *.env.bak-* (glob *.env 本身不会匹配, 双保险)
        if ".bak" in p.name:
            continue
        try:
            out.append(Profile(p))
        except Exception as e:
            out.append(BrokenProfile(p, e))
    return out


# ─── 编辑状态 ──────────────────────────────────────────────────────
class EditState:
    """一次编辑会话: 在 profile 的独立副本上改, 确认后才落盘。"""

    def __init__(self, profile: Profile):
        self.profile_path = profile.path
        self.name = profile.name
        self.env = EnvFile(profile.path)       # 重新解析一份副本
        self.cmd_binary, self.cmd_pairs = parse_command(
            self.env.get("FASTLLM_BACKEND_COMMAND") or "")
        # 每个字段的 (原始值, 当前值); None = 文件里未设置
        self.original = {}
        self.values = {}
        for _, specs in GROUPS:
            for spec in specs:
                raw = (cmd_value(self.cmd_pairs, spec.key)
                       if spec.where == "cmd" else self.env.get(spec.key))
                v = load_value(spec, raw)
                self.original[spec.key] = v
                self.values[spec.key] = v

    def dirty_keys(self):
        return [k for k in self.values if self.values[k] != self.original[k]]

    def step(self, spec: FieldSpec, direction: int, big: bool):
        """+/- 步进。布尔翻转, 枚举循环, 数值按 step(大步 x8) 并夹取。"""
        v = self.values[spec.key]
        if spec.kind == "bool":
            if v is None:
                v = False
            self.values[spec.key] = not v
            return
        if spec.kind == "enum":
            opts = spec.options
            if not opts:
                return
            if v not in opts:
                self.values[spec.key] = opts[0]
                return
            self.values[spec.key] = opts[(opts.index(v) + direction) % len(opts)]
            return
        if spec.kind == "path":
            return                              # 路径只通过选择器/文本修改
        mult = 8 if big else 1
        if v is None:
            v = spec.start
        self.values[spec.key] = max(spec.lo, min(spec.hi, v + direction * spec.step * mult))

    def set_value(self, spec: FieldSpec, v):
        if spec.kind == "int":
            v = max(int(spec.lo), min(int(spec.hi), int(v)))
        elif spec.kind == "float":
            v = max(spec.lo, min(spec.hi, float(v)))
        self.values[spec.key] = v

    def reset(self, spec: FieldSpec):
        self.values[spec.key] = self.original[spec.key]

    def save(self, dry_run=False):
        """把编辑结果应用到 EnvFile 副本并返回 (新文本, 变更列表)。

        关键: 命令行参数改动 -> 重建整条 FASTLLM_BACKEND_COMMAND
        (保留二进制路径/--model_name/--port 等其余全部参数);
        独立 FASTLLM_* 变量 -> 只替换键值。未改动的行不动。
        """
        changes = []
        cmd_dirty = False
        pairs = list(self.cmd_pairs)
        for gi, (_, specs) in enumerate(GROUPS):
            for spec in specs:
                new, old = self.values[spec.key], self.original[spec.key]
                if new == old:
                    continue
                changes.append((spec.key, fmt_value(spec, old),
                                fmt_value(spec, new)))
                if spec.where == "cmd":
                    new_s = None if new is None else serialize_value(spec, new)
                    for i, (f, _v) in enumerate(pairs):
                        if f == spec.key:
                            pairs[i] = (f, new_s)
                            break
                    else:
                        pairs.append((spec.key, new_s))   # 原命令行缺该参数则补上
                    cmd_dirty = True
                else:
                    if new is None:
                        continue    # 未设置 -> 仍不写入
                    self.env.set(spec.key, serialize_value(spec, new))
        if cmd_dirty:
            self.cmd_pairs = pairs
            self.env.set("FASTLLM_BACKEND_COMMAND",
                         build_command(self.cmd_binary, pairs), quote="'")
        return self.env.render(), changes

    def write(self, text: str):
        """备份原文件后原子替换写回。"""
        ts = time.strftime("%Y%m%d-%H%M%S")
        backup = self.profile_path.with_name(
            self.profile_path.name + f".bak-tui-{ts}")
        shutil.copy2(self.profile_path, backup)
        tmp = self.profile_path.with_name(self.profile_path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        try:
            shutil.copystat(self.profile_path, tmp)   # 保留权限位
        except OSError:
            pass
        os.replace(tmp, self.profile_path)
        return backup


# ─── 状态探测 (后台线程, 只读) ────────────────────────────────────
# /props 关键字段 (example/apiserver/apiserver.cpp 的 /props 响应)
PROP_KEYS = [
    "kv_cache_dtype", "max_batch", "token_pool", "default_max_tokens",
    "prefix_cache_stats_requests", "prefix_cache_stats_hit_requests",
    "prefix_cache_stats_query_tokens", "prefix_cache_stats_hit_tokens",
    "prefix_cache_cpu_tier_bytes", "prefix_cache_disk_live_bytes",
    "cpu_request_swap_disk_spills", "cpu_request_swap_disk_restores",
]


def _port_open(port, host="127.0.0.1", timeout=0.25):
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def _http_json(url, timeout=1.2):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return None


def _nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.used,memory.free,memory.total,"
                  "utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        first = out.stdout.strip().splitlines()[0]
        used, free, total, util = [x.strip() for x in first.split(",")]
        return {"used": float(used), "free": float(free),
                "total": float(total), "util": float(util)}
    except Exception:
        return None


def _tmux_session_alive(session=TMUX_SESSION):
    try:
        return subprocess.run([TMUX_BIN, "has-session", "-t", session],
                              capture_output=True, timeout=2).returncode == 0
    except Exception:
        return False


def _pgrep_f(pattern):
    try:
        out = subprocess.run(["pgrep", "-f", pattern],
                             capture_output=True, text=True, timeout=3)
        return [int(x) for x in out.stdout.split() if x.isdigit()]
    except Exception:
        return []


class StatusProbe(threading.Thread):
    """每 3 秒刷一次端口/健康/计数器/显存; /props 较大, 每 3 轮取一次。"""

    def __init__(self, app):
        super().__init__(daemon=True)
        self.app = app
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        tick = 0
        while not self._stop_evt.wait(3.0):
            snap = {"ts": time.time()}
            prof = self.app.current_profile()
            pport = prof.proxy_port() if prof else 8000
            bport = prof.backend_port() if prof else 8002
            snap["ports"] = {pport: _port_open(pport),
                             bport: _port_open(bport)}
            snap["health"] = _http_json(f"http://127.0.0.1:{bport}/health")
            tick += 1
            if tick % 3 == 1 or "props" not in self.app.status:
                props = _http_json(f"http://127.0.0.1:{bport}/props",
                                   timeout=2.0)
                if props is not None:
                    snap["props"] = {k: props.get(k) for k in PROP_KEYS}
            else:
                snap["props"] = self.app.status.get("props")
            snap["gpu"] = _nvidia_smi()
            snap["tmux"] = _tmux_session_alive()
            self.app.status = snap


# ─── 生命周期操作 (launch / stop / restart) ───────────────────────
def tmux_cmd(*args):
    return subprocess.run([TMUX_BIN, *args], capture_output=True,
                          text=True, timeout=5)


class Lifecycle:
    """镜像 launch_proxy_tmux.sh 的 stop_session/launch 语义。

    注意: launcher 末尾会 `tmux attach-session`, 直接 subprocess 调用会因没有
    TTY 而失败并触发它的 EXIT trap 把刚建的会话杀掉。所以用 `script` 分配一个
    pty 在后台跑; 没有 script 时退回独立 tmux 会话。
    """

    def __init__(self, flash):
        self.flash = flash

    def launch(self, profile: Profile):
        prof_path = str(profile.path.resolve())     # 必须绝对路径
        inner = (f"bash {shlex.quote(str(LAUNCHER_SCRIPT))} "
                 f"{shlex.quote(prof_path)} {TMUX_SESSION} {TMUX_WINDOW}")
        log_dir = profile.path.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "tui-launch.log"
        script_bin = shutil.which("script")
        if script_bin:
            # `script` 给 launcher 一个 pty, 让它末尾的 tmux attach 能成功
            cmd = [script_bin, "-qec", inner, "/dev/null"]
        else:
            # 本机没有 script: 放进一次性 tmux 会话跑 (名字唯一, 可重复启动);
            # launcher 内部 unset TMUX 后其 attach 会把该 pane 带到目标会话
            uniq = f"fastllm-tui-launch-{os.getpid()}-{int(time.time())}"
            cmd = [TMUX_BIN, "new-session", "-d", "-s", uniq,
                   "/bin/bash", "-lc", inner]
        with open(log_path, "a", encoding="utf-8") as logfh:
            logfh.write(f"\n[{time.strftime('%F %T')}] TUI 启动 "
                        f"{profile.name}\n")
            subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=logfh,
                stderr=subprocess.STDOUT, start_new_session=True,
                cwd=str(PROJECT_ROOT))
        self.flash(f"已发起启动 {profile.name}: tmux a -t {TMUX_SESSION} "
                   f"查看; 日志 {log_path.name}", 8)

    def stop(self, profile: Profile):
        """发 C-c -> 等退出 -> kill 会话 -> 清孤儿进程 (与脚本一致)。"""
        self.flash("停止中: 向代理发 C-c ...", 5)
        alive = _tmux_session_alive()
        if alive:
            try:
                tmux_cmd("send-keys", "-t", f"{TMUX_SESSION}:{TMUX_WINDOW}",
                         "C-c")
            except Exception:
                pass
            # 最多等 45 秒; 两个端口都关掉即视为代理+后端已退
            deadline = time.time() + 45
            closed_since = None
            prof = profile
            ports = (prof.proxy_port(), prof.backend_port())
            while time.time() < deadline:
                time.sleep(1)
                panes_dead = True
                try:
                    out = tmux_cmd("list-panes", "-t",
                                   f"{TMUX_SESSION}:{TMUX_WINDOW}",
                                   "-F", "#{pane_dead}")
                    lines = [l.strip() for l in out.stdout.splitlines()
                             if l.strip()]
                    panes_dead = bool(lines) and all(l == "1" for l in lines)
                except Exception:
                    pass
                ports_closed = all(not _port_open(p) for p in ports)
                if panes_dead or ports_closed:
                    closed_since = closed_since or time.time()
                    if time.time() - closed_since >= 2:
                        break
                else:
                    closed_since = None
            try:
                tmux_cmd("kill-session", "-t", TMUX_SESSION)
            except Exception:
                pass
        # 孤儿清扫: 与脚本相同, 按 "apiserver --path <模型路径>" 匹配
        model = profile.model_path
        if model:
            pids = _pgrep_f(f"apiserver --path {model}")
            if pids:
                self.flash(f"清理孤儿 apiserver: {pids}", 5)
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        pass
                time.sleep(3)
                for pid in _pgrep_f(f"apiserver --path {model}"):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
        self.flash("停止完成", 5)

    def restart(self, profile: Profile):
        self.stop(profile)
        self.launch(profile)


# ─── TUI ───────────────────────────────────────────────────────────
HELP_TEXT = """
列表界面            编辑界面               全局
─────────          ─────────              ─────
↑/↓ 或 j/k 选择     ↑/↓ 选择字段            Tab   状态面板(再按返回)
Enter / e 编辑      ←/→ 或 -/+ 调整         r     强制刷新状态
Tab      状态面板    * / Shift+→ 大步进      ?     本帮助
L 启动 (确认)       Enter 快捷值选择器       q     返回/退出
S 停止 (确认)       r 重置该字段为原值
R 重启 (确认)       1..5 切换参数组
                   s 保存(差异预览+确认)
                   Esc/q 返回(有改动需确认)

数值调整均为步进/枚举/布尔控件, 不接受自由文本数字;
--path / --mmproj 可在选择器里选已有值或手动输入路径。
""".strip("\n")


class App:
    def __init__(self, args):
        self.args = args
        self.profiles_dir = Path(args.profiles_dir).expanduser()
        if not self.profiles_dir.is_dir():
            raise SystemExit(f"profile 目录不存在: {self.profiles_dir}")
        self.profiles = load_profiles(self.profiles_dir)
        if not self.profiles:
            raise SystemExit(f"目录里没有 *.env profile: {self.profiles_dir}")
        self.sel = 0
        self.top = 0
        self.screen = "list"          # list / edit / confirm / status / help
        self.edit = None              # EditState
        self.group = 0
        self.cursor = 0
        self.pending_changes = []     # 确认界面用
        self.pending_text = ""
        self.status = {}
        self.message = ""
        self.msg_until = 0.0
        self.lifecycle = Lifecycle(self.flash)
        self.probe = None
        if not args.no_status:
            self.probe = StatusProbe(self)
            self.probe.start()

    # ── 工具 ──
    def flash(self, text, secs=5):
        self.message = text
        self.msg_until = time.time() + secs

    def current_profile(self):
        if 0 <= self.sel < len(self.profiles):
            return self.profiles[self.sel]
        return None

    # ── 主循环 ──
    def run(self, stdscr):
        self.scr = stdscr
        curses.curs_set(0)
        stdscr.timeout(400)           # 0.4s 轮询: 刷新状态/消息
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_GREEN, -1)
            curses.init_pair(4, curses.COLOR_RED, -1)
            curses.init_pair(5, curses.COLOR_CYAN, -1)
            curses.init_pair(6, curses.COLOR_MAGENTA, -1)
        self.C = lambda n: curses.color_pair(n) if curses.has_colors() else 0
        self.A_SEL = self.C(1) | curses.A_BOLD
        while True:
            try:
                self.draw()
            except curses.error:
                pass
            ch = stdscr.getch()
            if ch == -1:
                continue
            if ch == curses.KEY_RESIZE:
                continue
            if not self.handle_key(ch):
                break
        if self.probe:
            self.probe.stop()

    def _addstr(self, y, x, text, attr=0):
        h, w = self.scr.getmaxyx()
        if y < 0 or y >= h or x >= w:
            return
        try:
            self.scr.addstr(y, x, text[: max(0, w - x - 1)], attr)
        except curses.error:
            pass

    def _clr_line(self, y):
        h, w = self.scr.getmaxyx()
        self._addstr(y, 0, " " * (w - 1))

    # ── 绘制 ──
    def draw(self):
        self.scr.erase()
        h, w = self.scr.getmaxyx()
        if self.screen == "list":
            self._draw_list(h, w)
        elif self.screen == "edit":
            self._draw_edit(h, w)
        elif self.screen == "confirm":
            self._draw_confirm(h, w)
        elif self.screen == "status":
            self._draw_status(h, w)
        elif self.screen == "help":
            self._draw_help(h, w)
        self.scr.refresh()

    def _draw_title(self, h, w, title):
        self._clr_line(0)
        bar = f" FastLLM apiserver TUI — {title} "
        self._addstr(0, 0, bar.ljust(w - 1)[: w - 1], self.C(1) | curses.A_BOLD)
        if self.args.dry_run:
            self._addstr(0, max(0, w - 12), " DRY-RUN ", self.C(4) | curses.A_BOLD)

    def _draw_footer(self, h, w, keys):
        # 消息行
        self._clr_line(h - 2)
        if self.message and time.time() < self.msg_until:
            self._addstr(h - 2, 0, " " + self.message, self.C(2))
        self._clr_line(h - 1)
        self._addstr(h - 1, 0, keys[: w - 1], self.C(5))

    def _status_strip(self, row, w):
        """列表界面底部的 3 行只读状态条。"""
        st = self.status
        age = time.time() - st.get("ts", 0) if st else None
        stale = "" if (age is not None and age < 10) else " (刷新中…)"
        # 行1: 端口 / tmux / GPU
        ports = st.get("ports", {})
        seg = []
        for p, ok in sorted(ports.items()):
            seg.append(f":{p} [{'监听' if ok else '未监听'}]")
        seg = " ".join(seg) + f"  tmux {TMUX_SESSION}: " + \
              ("运行中" if st.get("tmux") else "无")
        gpu = st.get("gpu")
        if gpu:
            seg += (f"  GPU {gpu['used']/1024:.1f}/{gpu['total']/1024:.1f} GiB "
                    f"占用, 空闲 {gpu['free']/1024:.1f} GiB, util {gpu['util']:.0f}%")
        self._addstr(row, 0, (" " + seg + stale)[: w - 1], self.C(5))
        # 行2: /health
        health = st.get("health")
        if health:
            line = (f" 后端 /health: {'ready' if health.get('ready') else '未就绪'}"
                    f" (status={health.get('status')}, "
                    f"tier={health.get('tier_state')}, "
                    f"active={health.get('active_requests')}, "
                    f"queued={health.get('queued_requests')}, "
                    f"model={health.get('model')})")
            attr = self.C(3) if health.get("ready") else self.C(4)
        else:
            line, attr = " 后端 /health: 未响应", self.C(4)
        self._addstr(row + 1, 0, line[: w - 1], attr)
        # 行3: /props 摘要
        props = st.get("props") or {}
        if props:
            req = props.get("prefix_cache_stats_requests") or 0
            hit = props.get("prefix_cache_stats_hit_requests") or 0
            qt = props.get("prefix_cache_stats_query_tokens") or 0
            ht = props.get("prefix_cache_stats_hit_tokens") or 0
            rate = f"{hit/req*100:.0f}%" if req else "-"
            tr = f"{ht/qt*100:.0f}%" if qt else "-"
            line = (f" /props: kv={props.get('kv_cache_dtype')} "
                    f"batch={props.get('max_batch')} "
                    f"pool={props.get('token_pool')} | "
                    f"前缀缓存命中: 请求 {rate}, token {tr} "
                    f"(CPU层 {human_bytes(props.get('prefix_cache_cpu_tier_bytes') or 0)}, "
                    f"磁盘存活 {human_bytes(props.get('prefix_cache_disk_live_bytes') or 0)})")
        else:
            line = " /props: 未响应"
        self._addstr(row + 2, 0, line[: w - 1], self.C(5))

    def _draw_list(self, h, w):
        self._draw_title(h, w, str(self.profiles_dir))
        hdr = f" {'profile 名':<46} {'模型 (--path 文件名)':<44} 关键参数"
        self._addstr(1, 0, hdr[: w - 1], curses.A_BOLD)
        body_h = max(1, h - 7)              # 标题1 + 表头1 + 状态3 + 底部2
        n = len(self.profiles)
        if self.sel < self.top:
            self.top = self.sel
        if self.sel >= self.top + body_h:
            self.top = self.sel - body_h + 1
        self.top = max(0, self.top)
        for i in range(self.top, min(n, self.top + body_h)):
            p = self.profiles[i]
            if p.error:
                model = "-"
            else:
                model = os.path.basename(p.model_path) or "(无命令行)"
            line = f" {p.name:<46} {model:<44} {p.summary()}"
            attr = self.A_SEL if i == self.sel else 0
            self._clr_line(2 + i - self.top)
            self._addstr(2 + i - self.top, 0, line[: w - 1], attr)
        self._status_strip(h - 5, w)
        self._draw_footer(
            h, w,
            " ↑↓选择  Enter编辑  Tab状态  L启动 S停止 R重启  ?帮助  q退出")

    def _draw_edit(self, h, w):
        ed = self.edit
        self._draw_title(h, w, f"编辑 {ed.name}")
        # 组标签行
        tabs = []
        for gi, (title, _) in enumerate(GROUPS):
            t = f" {gi+1}:{title.split(' ')[0]} "
            tabs.append((t, gi == self.group))
        x = 0
        for t, act in tabs:
            self._addstr(1, x, t, self.A_SEL if act else self.C(5))
            x += len(t) + 1
        title, specs = GROUPS[self.group]
        self._addstr(2, 0, (" " + title)[: w - 1], curses.A_BOLD)
        body_h = max(1, h - 7)
        self.cursor = max(0, min(self.cursor, len(specs) - 1))
        for i, spec in enumerate(specs[:body_h]):
            v = ed.values[spec.key]
            dirty = v != ed.original[spec.key]
            val = fmt_value(spec, v)
            if dirty:
                val += f"   ← 原: {fmt_value(spec, ed.original[spec.key])}"
            mark = "*" if dirty else " "
            line = f"{mark} {spec.key:<44} {val}"
            attr = self.A_SEL if i == self.cursor else \
                (self.C(6) if dirty else 0)
            self._clr_line(3 + i)
            self._addstr(3 + i, 0, line[: w - 1], attr)
        # 选中字段的说明
        spec = specs[self.cursor]
        rng = ""
        if spec.kind in ("int", "float"):
            rng = (f"  [范围 {spec.lo:.0f}..{spec.hi:.0f}, "
                   f"步长 {spec.step:g}, 大步 x8]")
        help1 = f" 说明: {spec.help}{rng}"
        self._clr_line(h - 4)
        self._addstr(h - 4, 0, help1[: w - 1], self.C(5))
        self._clr_line(h - 3)
        dirty_n = len(ed.dirty_keys())
        dirty_txt = f" 已修改 {dirty_n} 项 → s 保存预览" if dirty_n else " (无改动)"
        self._addstr(h - 3, 0, dirty_txt[: w - 1],
                     self.C(6) if dirty_n else self.C(5))
        self._draw_footer(
            h, w,
            " ↑↓字段  ←→/-+调整  *大步  Enter快捷值  r重置  1-5换组  s保存  q返回")

    def _draw_confirm(self, h, w):
        self._draw_title(h, w, "保存确认 — 写回 " + self.edit.name)
        self._addstr(2, 0, " 以下参数将被写回 (FASTLLM_BACKEND_COMMAND 整串重建,"
                            " 其余行不动):", curses.A_BOLD)
        y = 4
        for key, old, new in self.pending_changes[: h - 9]:
            self._addstr(y, 1, f"{key:<44} {old}  →  {new}", 0)
            y += 1
        if not self.pending_changes:
            self._addstr(y, 1, "(没有改动)", self.C(5))
        note_y = min(y + 2, h - 4)
        if self.args.dry_run:
            self._addstr(note_y, 1, "--dry-run: 只预览, 不写盘", self.C(4) | curses.A_BOLD)
        else:
            self._addstr(note_y, 1,
                         "写盘前自动备份为 *.bak-tui-<时间戳>", self.C(5))
        self._draw_footer(h, w, " y 确认写回    其它键取消")

    def _draw_status(self, h, w):
        self._draw_title(h, w, f"只读状态 — {self.current_profile().name}")
        st = self.status
        y = 2
        prof = self.current_profile()
        backend_url = prof.env.get("FASTLLM_BACKEND_URL") if prof.env else "?"
        self._addstr(y, 1, f"profile: {prof.name}   后端URL: {backend_url}   "
                           f"代理端口: {prof.proxy_port()}", 0); y += 1
        ports = st.get("ports", {})
        self._addstr(y, 1, "端口: " + "  ".join(
            f":{p}={'监听' if ok else '未监听'}" for p, ok in sorted(ports.items())),
            0); y += 1
        self._addstr(y, 1, f"tmux 会话 {TMUX_SESSION}: " +
                     ("运行中" if st.get("tmux") else "不存在"), 0); y += 1
        gpu = st.get("gpu")
        if gpu:
            self._addstr(y, 1,
                         f"GPU 显存: {gpu['used']:.0f}/{gpu['total']:.0f} MiB 占用, "
                         f"空闲 {gpu['free']:.0f} MiB, 利用率 {gpu['util']:.0f}%",
                         0); y += 1
        else:
            self._addstr(y, 1, "GPU: nvidia-smi 不可用", self.C(4)); y += 1
        health = st.get("health")
        if health:
            self._addstr(y, 1, "后端 /health:", curses.A_BOLD); y += 1
            for k in ("status", "ready", "suspended", "tier_state",
                      "accepting", "active_requests", "queued_requests",
                      "model"):
                if k in health:
                    self._addstr(y, 3, f"{k}: {health[k]}", 0); y += 1
        else:
            self._addstr(y, 1, "后端 /health: 未响应", self.C(4)); y += 1
        props = st.get("props")
        if props:
            self._addstr(y, 1, "后端 /props 关键计数器:", curses.A_BOLD); y += 1
            for k, v in props.items():
                if y >= h - 3:
                    break
                if isinstance(v, float) and k.endswith("_bytes"):
                    v = human_bytes(v)
                self._addstr(y, 3, f"{k}: {v}", 0); y += 1
        self._draw_footer(h, w, " Tab/q/Esc 返回   r 立即刷新")

    def _draw_help(self, h, w):
        self._draw_title(h, w, "快捷键帮助")
        for i, line in enumerate(HELP_TEXT.splitlines()[: h - 4]):
            self._addstr(2 + i, 1, line[: w - 2], 0)
        self._draw_footer(h, w, " 任意键返回")

    # ── 按键处理 ──
    def handle_key(self, ch):
        if self.screen == "list":
            return self._key_list(ch)
        if self.screen == "edit":
            return self._key_edit(ch)
        if self.screen == "confirm":
            return self._key_confirm(ch)
        if self.screen == "status":
            if ch in (ord("q"), 27, 9, ord("\t")):
                self.screen = "list"
            elif ch in (ord("r"), ord("g")):
                self.flash("已请求刷新", 2)
            return True
        if self.screen == "help":
            self.screen = "list"
            return True
        return True

    def _key_list(self, ch):
        n = len(self.profiles)
        if ch in (ord("q"), ord("Q")):
            return False
        elif ch in (curses.KEY_DOWN, ord("j")):
            self.sel = min(n - 1, self.sel + 1)
        elif ch in (curses.KEY_UP, ord("k")):
            self.sel = max(0, self.sel - 1)
        elif ch == curses.KEY_NPAGE:
            self.sel = min(n - 1, self.sel + 10)
        elif ch == curses.KEY_PPAGE:
            self.sel = max(0, self.sel - 10)
        elif ch in (ord("g"), curses.KEY_HOME):
            self.sel = 0
        elif ch in (ord("G"), curses.KEY_END):
            self.sel = n - 1
        elif ch in (curses.KEY_ENTER, 10, 13, ord("e")):
            prof = self.current_profile()
            if prof.error:
                self.flash(f"该文件解析失败: {prof.error}", 6)
            elif not prof.cmd_pairs:
                self.flash("该 profile 没有 FASTLLM_BACKEND_COMMAND, 无法编辑命令行参数", 6)
            else:
                self.edit = EditState(prof)
                self.group = 0
                self.cursor = 0
                self.screen = "edit"
        elif ch == 9:                       # Tab
            self.screen = "status"
        elif ch == ord("?"):
            self.screen = "help"
        elif ch in (ord("r"),):
            self.flash("状态每 3 秒自动刷新", 3)
        elif ch == ord("L"):
            self._op_launch()
        elif ch == ord("S"):
            self._op_stop()
        elif ch == ord("R"):
            self._op_restart()
        return True

    def _op_launch(self):
        prof = self.current_profile()
        if not self.confirm(
                f"用 {prof.name} 启动? 会先停掉现有 {TMUX_SESSION} 会话! [y/n]"):
            return
        self._async(lambda: self.lifecycle.launch(prof))

    def _op_stop(self):
        prof = self.current_profile()
        if not self.confirm(
                f"停止 {TMUX_SESSION} 并按 {prof.name} 的模型路径清孤儿进程? [y/n]"):
            return
        self._async(lambda: self.lifecycle.stop(prof))

    def _op_restart(self):
        prof = self.current_profile()
        if not self.confirm(
                f"重启为 {prof.name}? 先停止再启动 (生产服务会中断)! [y/n]"):
            return
        self._async(lambda: self.lifecycle.restart(prof))

    def _async(self, fn):
        def wrap():
            try:
                fn()
            except Exception as e:
                self.flash(f"操作失败: {e}", 8)
        threading.Thread(target=wrap, daemon=True).start()

    def _key_edit(self, ch):
        ed = self.edit
        _, specs = GROUPS[self.group]
        spec = specs[self.cursor]
        if ch in (ord("q"), 27):
            if ed.dirty_keys():
                if not self.confirm("有未保存的改动, 确定放弃? [y/n]"):
                    return True
            self.screen = "list"
            self.edit = None
        elif ch == curses.KEY_DOWN:
            self.cursor = min(len(specs) - 1, self.cursor + 1)
        elif ch == curses.KEY_UP:
            self.cursor = max(0, self.cursor - 1)
        elif ord("1") <= ch <= ord("5"):
            self.group = ch - ord("1")
            self.cursor = 0
        elif ch in (curses.KEY_RIGHT, ord("+"), ord("="), ord("l")):
            ed.step(spec, +1, big=False)
        elif ch in (curses.KEY_LEFT, ord("-"), ord("h")):
            ed.step(spec, -1, big=False)
        elif ch in (ord("*"), curses.KEY_SRIGHT):
            ed.step(spec, +1, big=True)
        elif ch in (ord("/"), curses.KEY_SLEFT):
            ed.step(spec, -1, big=True)
        elif ch == ord("r"):
            ed.reset(spec)
            self.flash(f"{spec.key} 已重置为原值", 3)
        elif ch in (curses.KEY_ENTER, 10, 13):
            self._edit_picker(spec)
        elif ch == ord("s"):
            self._prepare_save()
        return True

    def _edit_picker(self, spec: FieldSpec):
        """Enter: 弹出快捷值选择器 (枚举/预设/路径), 避免自由文本输数字。"""
        ed = self.edit
        if spec.kind == "enum":
            opts = [(str(o), o) for o in spec.options]
        elif spec.kind == "bool":
            opts = [("0 (关)", False), ("1 (开)", True)]
        elif spec.kind in ("int", "float"):
            opts = []
            cur = ed.values[spec.key]
            for p in spec.presets:
                opts.append((fmt_value(spec, p), p))
            if cur is not None and cur not in spec.presets:
                opts.append((fmt_value(spec, cur) + " (当前)", cur))
        elif spec.kind == "path":
            # 汇集所有 profile 里同一参数的已知取值
            known = []
            for p in self.profiles:
                v = p.cmd(spec.key) if spec.where == "cmd" else \
                    p.env.get(spec.key)
                if v and v not in known:
                    known.append(v)
            opts = [(v, v) for v in known]
            opts.append(("— 手动输入路径 —", "__input__"))
        else:
            return
        idx = self.pick(f"{spec.key} 选择新值", [o[0] for o in opts])
        if idx < 0:
            return
        label, val = opts[idx]
        if val == "__input__":
            cur = ed.values[spec.key] or ""
            text = self.text_input(f"{spec.key} =", cur)
            if text is None:
                return
            val = text.strip()
            if not val:
                return
        ed.set_value(spec, val)

    def _prepare_save(self):
        ed = self.edit
        text, changes = ed.save(dry_run=self.args.dry_run)
        if not changes:
            self.flash("没有改动", 3)
            return
        self.pending_text = text
        self.pending_changes = changes
        self.screen = "confirm"

    def _key_confirm(self, ch):
        if ch in (ord("y"), ord("Y")) and not self.args.dry_run:
            try:
                backup = self.edit.write(self.pending_text)
                self.flash(f"已写回 {self.edit.name}; 备份 {backup.name}", 8)
                self.profiles = load_profiles(self.profiles_dir)
                self.sel = max(0, min(self.sel, len(self.profiles) - 1))
            except Exception as e:
                self.flash(f"写回失败: {e}", 8)
        elif ch in (ord("y"), ord("Y")):
            self.flash("dry-run: 未写盘 (改动丢弃)", 6)
        else:
            self.flash("已取消, 编辑状态保留", 3)
            self.screen = "edit"
            return True
        self.screen = "list"
        self.edit = None
        return True

    # ── 交互小部件 ──
    def confirm(self, question):
        while True:
            self.draw()
            h, w = self.scr.getmaxyx()
            self._clr_line(h - 2)
            self._addstr(h - 2, 0, " " + question, self.C(2) | curses.A_BOLD)
            self.scr.refresh()
            ch = self.scr.getch()
            if ch in (ord("y"), ord("Y")):
                return True
            if ch in (ord("n"), ord("N"), 27, 3):
                return False

    def pick(self, title, options):
        """居中列表选择器; 返回下标, -1 表示取消。"""
        if not options:
            return -1
        h, w = self.scr.getmaxyx()
        width = min(w - 4, max(len(title) + 4,
                               max(len(o) for o in options) + 6))
        vis = min(len(options), max(3, h - 8))
        win_h = vis + 2
        win = curses.newwin(win_h, width, max(0, (h - win_h) // 2),
                            max(0, (w - width) // 2))
        cur, top = 0, 0
        while True:
            win.erase()
            win.attron(self.C(1) | curses.A_BOLD)
            win.addstr(0, 0, (" " + title)[: width - 1].ljust(width - 1))
            win.attroff(self.C(1) | curses.A_BOLD)
            top = max(0, min(top, cur - vis + 1, len(options) - vis))
            if cur < top:
                top = cur
            for i in range(top, min(len(options), top + vis)):
                line = (" ▸ " if i == cur else "   ") + options[i]
                try:
                    win.addstr(1 + i - top, 0, line[: width - 1],
                               self.A_SEL if i == cur else 0)
                except curses.error:
                    pass
            win.refresh()
            ch = self.scr.getch()
            if ch in (curses.KEY_DOWN,):
                cur = min(len(options) - 1, cur + 1)
            elif ch in (curses.KEY_UP,):
                cur = max(0, cur - 1)
            elif ch == curses.KEY_NPAGE:
                cur = min(len(options) - 1, cur + vis)
            elif ch == curses.KEY_PPAGE:
                cur = max(0, cur - vis)
            elif ch in (curses.KEY_ENTER, 10, 13):
                return cur
            elif ch in (27, ord("q"), 3):
                return -1

    def text_input(self, prompt, initial=""):
        """底部单行文本输入 (仅用于路径类字段); 返回 None 表示取消。"""
        s = initial
        while True:
            self.draw()
            h, w = self.scr.getmaxyx()
            self._clr_line(h - 2)
            shown = f" {prompt} {s}"
            self._addstr(h - 2, 0, shown[: w - 1], self.C(2))
            self.scr.refresh()
            ch = self.scr.getch()
            if ch in (curses.KEY_ENTER, 10, 13):
                return s
            if ch in (27, 3):
                return None
            if ch in (curses.KEY_BACKSPACE, 127, 8):
                s = s[:-1]
            elif 32 <= ch < 127:
                s += chr(ch)


# ─── 非交互模式 ────────────────────────────────────────────────────
def do_check(profiles_dir):
    """解析全部 profile 并自检: 汇总 + 渲染回读必须逐字节一致。"""
    profiles = load_profiles(Path(profiles_dir))
    print(f"profile 目录: {profiles_dir}  (共 {len(profiles)} 个)\n")
    ok = True
    for p in profiles:
        if p.error:
            print(f"[错误] {p.name}: {p.error}")
            ok = False
            continue
        # 回读自检: 未改动渲染必须与原文件一致
        if p.env.render() != p.env.text:
            print(f"[错误] {p.name}: 解析-渲染不回环!")
            ok = False
        print(f"{p.name:<48} {os.path.basename(p.model_path) or '-':<46} "
              f"{p.summary()}")
    print("\n自检:", "通过 (全部解析且回读一致)" if ok else "存在问题")
    return 0 if ok else 1


def do_dump(profile, group_filter=None):
    ed = EditState(Profile(Path(profile)))
    print(f"profile: {ed.name}  ({profile})")
    print(f"apiserver: {ed.cmd_binary}\n")
    for gi, (title, specs) in enumerate(GROUPS):
        print(f"── {title}")
        for spec in specs:
            v = ed.values[spec.key]
            print(f"   {spec.key:<46} = {fmt_value(spec, v):<18} # {spec.help}")
        print()


# ─── 入口 ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="FastLLM apiserver TUI 管理器 (profile 浏览/编辑/启停)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="生产安全: 写盘前自动备份 *.bak-tui-* 并要求二次确认; "
               "--dry-run 完全不写盘。\n"
               "生命周期操作针对 tmux 会话 "
               f"{TMUX_SESSION}/{TMUX_WINDOW} (launch_proxy_tmux.sh 默认值)。")
    ap.add_argument("--profiles-dir", default=str(DEFAULT_PROFILES_DIR),
                    help="profile *.env 目录")
    ap.add_argument("--dry-run", action="store_true",
                    help="可以编辑预览, 但绝不写盘")
    ap.add_argument("--no-status", action="store_true",
                    help="不启动后台状态探测线程")
    ap.add_argument("--check", action="store_true",
                    help="非交互: 解析全部 profile 并自检后退出")
    ap.add_argument("--dump", metavar="PROFILE_ENV",
                    help="非交互: 打印单个 profile 的分组解析结果后退出")
    args = ap.parse_args()

    if args.check:
        sys.exit(do_check(args.profiles_dir))
    if args.dump:
        do_dump(args.dump)
        sys.exit(0)

    if not sys.stdout.isatty():
        sys.exit("需要 TTY 终端才能运行 TUI (试试 --check / --dump)")
    app = App(args)
    try:
        curses.wrapper(app.run)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
