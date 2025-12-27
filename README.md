# Immich 机器学习识别服务

这是一个为 Immich 照片管理系统开发的独立机器学习服务，支持中文文本理解（Chinese-CLIP）、人脸识别（Buffalo_L）以及多语言 CLIP 模型。

## 主要特性

- **多模态理解**: 支持 Chinese-CLIP 和 Multilingual CLIP (SigLIP) 进行图片-文本跨模态搜索。
- **人脸识别**: 支持人脸检测和识别 (Buffalo_L / SCRFD + ArcFace)。
- **高性能**: 基于 Rust 和 ONNX Runtime 开发。
- **硬件加速**: 支持 CUDA (NVIDIA GPU)、DirectX 12 (DirectML) 和 TensorRT。
- **灵活配置**: 支持端口配置和模型预加载。

## 快速开始

### 1. 下载模型文件

请将模型文件下载并放置在 `models` 目录下。目录结构如下：

```
models/
├── clip_cn_tokenizer.json             # Chinese-CLIP 分词器
├── ViT-B-16.img.fp32.onnx             # Chinese-CLIP ViT-B-16 图像模型
├── ViT-B-16.txt.fp32.onnx             # Chinese-CLIP ViT-B-16 文本模型
├── buffalo_l/
│   ├── det_10g.onnx                   # 人脸检测模型 (SCRFD)
│   └── w600k_r50.onnx                 # 人脸识别模型 (ArcFace)
└── nllb-clip-large-siglip__v1/
    ├── vision.onnx                    # SigLIP 图像模型
    ├── text.onnx                      # SigLIP 文本模型
    └── tokenizer.json                 # SigLIP 分词器
```

**模型下载地址：**

- **Chinese-CLIP**: [ModelScope - Chinese-CLIP](https://www.modelscope.cn/models/deadash/chinese_clip_ViT-B-16/summary) (需下载对应的 ONNX 文件)
- **Buffalo_L (人脸识别)**: 通常包含在 Immich 的默认模型库中，也可从 [InsightFace](https://github.com/deepinsight/insightface) 或 Immich 提供的 HuggingFace 镜像下载。
- **Multilingual CLIP (SigLIP)**: 可从 [Immich HuggingFace](https://huggingface.co/immich-app/nllb-clip-large-siglip__v1) 下载。

### 2. 运行服务

默认监听端口为 3003。

```bash
# 默认运行
cargo run --release

# 指定端口
cargo run --release -- --port 3004
```

或者构建后运行：

```bash
./target/release/immich_ml --port 3004
```

### 3. 环境变量配置

支持通过环境变量进行配置：

- `RUST_LOG`: 日志级别 (默认 `debug,actix_web=debug`)。
- `MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL`: 指定启动时预加载的文本模型名称 (例如 `ViT-B-16__openai`)。

## 支持的模型列表

| 功能 | 模型名称 (Immich ID) | 对应文件 |
| --- | --- | --- |
| 中文搜索 | `ViT-B-16__openai` | `ViT-B-16.img.fp32.onnx`, `ViT-B-16.txt.fp32.onnx` |
| 中文搜索 (大模型) | `ViT-L-14__openai` | `ViT-L-14.img.fp32.onnx`, `ViT-L-14.txt.fp32.onnx` |
| 人脸识别 | `buffalo_l` | `buffalo_l/det_10g.onnx` (检测), `buffalo_l/w600k_r50.onnx` (识别) |
| 多语言搜索 | `nllb-clip-large-siglip__v1` | `nllb-clip-large-siglip__v1/vision.onnx`, `text.onnx` |

## 开发与构建

**依赖要求:**
- Rust (最新稳定版)
- CUDA / TensorRT / DirectML (根据所需的硬件加速后端)

**构建:**

```bash
cargo build --release
```

**测试:**

```bash
cargo test
```
