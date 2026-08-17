#pragma once

#include <string>
#include <vector>

// 一段解码+采样完成的视频:所有帧同尺寸,rgb 按帧连续存放(0-255 浮点)。
// 采样策略(fps / 最大帧数 / 帧长边)由 video_loader 依据环境变量决定:
//   FASTLLM_VIDEO_FPS        采样帧率,默认 2.0(Qwen3-VL 官方默认)
//   FASTLLM_VIDEO_MAX_FRAMES 帧数上限,默认 32
//   FASTLLM_VIDEO_MAX_EDGE   帧长边上限(像素),默认 768;引擎再做精确对齐
struct OpenAIDecodedVideo {
    int width = 0;
    int height = 0;
    int frameCount = 0;
    std::vector<float> rgb;  // frameCount * height * width * 3
};

// 支持 data:video/...;base64, / http(s):// / 本地路径。
// 解码依赖系统 ffmpeg/ffprobe 子进程。
bool LoadOpenAIVideoUrl(const std::string &url,
                        OpenAIDecodedVideo &video,
                        std::string &error);
