# Audio Visual Fusion (C++ 实时音视频说话人识别与融合定位系统)

本项目是基于 C++17 实现的高性能、低延迟多模态说话人识别与定位系统。通过融合人脸检测、人体/人脸追踪、主动说话人检测（ASD）和自动语音识别（ASR）等多模态信号，回答：“**视频中是谁在说当前这句话？**”

---

## 目录结构

本仓库是一个独立的 C++ 构建项目，规范目录结构如下：

```text
├── CMakeLists.txt         # 顶层 CMake 构建配置文件
├── 3rdparty/              # 第三方依赖库（如头文件版的 cpp-httplib）
│   └── httplib.h
├── include/               # 项目公共头文件目录
│   └── speaker_id/
│       ├── api/
│       ├── capture/       # [新增] 视频流捕获接口（libav_video_capture.hpp）
│       ├── core/          # 管道线（Pipeline）、事件与配置接口（包含新增的 audio_clock.hpp）
│       └── modules/       # 视觉、音频、ASR、ASD 等子模块接口
├── src/                   # 源代码实现目录
│   ├── config.cpp         # YAML 配置加载器
│   ├── pipeline.cpp       # 多线程流式管道线主逻辑
│   ├── capture/           # [新增] 基于 FFmpeg/libav 的硬件/本地设备视频捕获实现
│   ├── gateway/           # Gateway 服务器与 WebSocket 通信实现
│   └── modules/           # 各模型推理模块实现 (ASD、ASR、Face、Diarization)
├── tests/                 # 单元测试与集成测试
│   └── core_tests.cpp     # C++ 单元测试回归集
├── ui/                    # [新增] 前端 Web 界面代码 (HTML/CSS/JS)
│   ├── index.html         # UI 结构主页面
│   ├── styles.css         # UI 美化样式
│   └── app.js             # UI 交互与 WebSocket 推送渲染逻辑
├── configs/               # 运行时配置文件
│   ├── live_mac.yaml      # 本地 Mac 摄像头实时体验配置
│   └── orin_nx.yaml       # Jetson Orin NX 边缘部署配置
└── scripts/               # 辅助脚本
    └── download_models.py # 一键模型权重下载与管理脚本
```

---

## 依赖要求

编译和运行本系统需要安装以下依赖：

1. **编译器**：支持 C++17 的编译器（如 GCC 9+、Clang 10+ 或 AppleClang）。
2. **OpenCV**：用于图像解码、画幅裁剪与嘴部图像预处理。
3. **ONNX Runtime**：用于运行 LR-ASD 主动说话人检测模型、人脸检测/识别模型等。
4. **Sherpa ONNX**：用作 ASR（SenseVoice / Zipformer）推理后端与声纹聚类引擎。
5. **yaml-cpp**：自动通过 CMake FetchContent 获取（也可使用系统级安装包）。
6. **FFmpeg/Libav（可选）**：用于硬件及摄像头流的原生视频捕获（包括 `libavformat`, `libavcodec`, `libavdevice`, `libavutil`, `libswscale`）。在 macOS/Linux 上安装后，CMake 将通过 `PkgConfig` 自动检测并编译 `src/capture/libav_video_capture.cpp` 模块，同时定义宏 `SPEAKER_ID_HAS_LIBAV=1`。

### macOS 安装依赖示例
```bash
brew install opencv onnxruntime ffmpeg pkg-config
```

---

## 模型权重配置

根据项目规则，本仓库**不直接携带**庞大的二进制模型权重文件。克隆代码后，可通过运行内置脚本自动下载：

```bash
# 安装必要的 Python 库（如 urllib、tarfile 等已内置，可直接运行）
python3 scripts/download_models.py
```

该脚本将下载以下模型并存放在项目根目录下的 `models/` 目录中：
- **YOLO**：人体框检测器 (`yolov8n.pt`)
- **Silero VAD**：静音检测/语音活动检测器 (`silero_vad_v6.onnx`)
- **SenseVoice ONNX**：离线 ASR 模型 (`model.int8.onnx` & `tokens.txt`)
- **Zipformer CTC**：流式 ASR 模型 (`model.int8.onnx` & `tokens.txt`)
- **CAM++声纹特征提取**：用于 diarization 的声纹识别模型
- **LR-ASD**：主动说话人检测模型 (`lr_asd_talkset.onnx`)
- **InsightFace**：`buffalo_l` 人脸特征和对齐模型包

---

## 构建与编译

使用标准 CMake 构建流程：

```bash
# 1. 配置项目（如默认 Python 找不到 sherpa-onnx，可通过 -DPython3_EXECUTABLE 指定 Python 解释器）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 2. 编译
cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

编译完成后，会在 `build/` 目录下生成以下两个可执行文件：
* `speaker_id_core_tests`：单元测试程序。
* `speaker_id_gateway`：C++ 网关与流式推送主程序。

---

## 运行测试

在根目录下运行 `ctest` 即可启动自动化测试：

```bash
cd build
ctest --output-on-failure
```

---

## 运行 Gateway 服务器

```bash
# 默认加载 configs/live_mac.yaml 配置文件
./build/speaker_id_gateway
```

---

## 边缘就绪声明与运行提示

- **边缘端意向目标**：本 C++ 仓是为 Jetson Orin NX 等计算边缘端量身定制的高性能运行方案。
- **音频/视频时钟同步**：本次更新引入了 `audio_clock.hpp`，提供更精准的高精度单调音频参考时钟；若检测到 PkgConfig 包含 `libav` 系列库，则原生编译 `libav_video_capture.cpp` 用于底层摄像头硬件直采。
- **现状说明**：当前项目主干的实时可运行网关依然是 **Python Gateway (`gateway/main.py`)**。在 C++ 仓完全完成基于 TensorRT 加速器、硬解码、以及 Orin NX 开发板上的完整 Latency/Accuracy 测试前，请勿将其直接作为默认生产就绪版本。
