<div align="center">

# 🚢 Ship Hull Agent

**基于 LangChain + FAISS + YOLO + Qwen3.5 VLM 的智能船弦号识别系统**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![YOLO](https://img.shields.io/badge/YOLO-v8-orange?logo=ultralytics)](https://github.com/ultralytics/ultralytics)
[![LangChain](https://img.shields.io/badge/LangChain-Agent-teal)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-purple)](https://github.com/facebookresearch/faiss)

<br/>

> **两大核心模块**：Agent 对话检索（精确匹配 + RAG 语义搜索）+ Pipeline 视频处理（实时检测 + 跟踪 + 弦号框定位）

<img src="https://img.shields.io/badge/弦号识别-精确匹配-brightgreen" /> <img src="https://img.shields.io/badge/弦号框定位-VLM坐标-yellow" /> <img src="https://img.shields.io/badge/语义检索-FAISS-blue" /> <img src="https://img.shields.io/badge/实时检测-YOLO8-red" />

</div>

---

## 📑 目录

- [功能概览](#-功能概览)
  - [Agent 对话检索](#agent-对话检索)
  - [Pipeline 视频处理](#pipeline-视频处理)
- [项目结构](#-项目结构)
- [快速开始](#-快速开始)
  - [1. 克隆项目](#1-克隆项目)
  - [2. 启动视觉模型服务](#2-启动视觉模型服务)
  - [3. 安装依赖](#3-安装依赖)
  - [4. 配置](#4-配置)
  - [5. 运行](#5-运行)
- [Agent 使用示例](#-agent-使用示例)
  - [精确匹配](#精确匹配)
  - [语义检索](#语义检索)
  - [详细模式](#详细模式)
  - [Agent 工作流程](#agent-工作流程)
- [Pipeline 视频处理](#-pipeline-视频处理)
  - [架构](#架构)
  - [快速使用](#快速使用)
  - [命令行参数](#命令行参数)
  - [级联 vs 并发模式](#级联-vs-并发模式)
  - [弦号框定位](#-弦号框定位)
  - [跟踪与识别流程](#跟踪与识别流程)
  - [交互按键](#交互按键)
- [批量建库](#-批量建库)
- [配置说明](#-配置说明)
- [本地部署 Embedding 模型](#-本地部署-embedding-模型)
- [测试](#-测试)
- [技术栈](#-技术栈)
- [开发指南](#-开发指南)

---

## ✨ 功能概览

### Agent 对话检索

| 功能 | 说明 |
|:-----|:-----|
| 🎯 精确弦号匹配 | 输入弦号直接查字典，O(1) 响应 |
| 🔍 RAG 语义检索 | FAISS 向量库对船描述做语义相似度匹配 |
| 🧠 智能路由 | 有弦号先精确查，查不到再语义检索；纯描述直接检索 |
| 🚦 阈值过滤 | 语义检索结果低于置信度阈值自动过滤 |
| 💬 Few-shot 引导 | 内置示例对话，确保 Agent 严格遵循识别流程 |
| 💾 向量库持久化 | FAISS 索引首次构建后缓存磁盘，后续直接加载 |
| 🔄 自动变更检测 | MD5 哈希比对 CSV，数据变更自动重建向量库 |
| 📦 批量建库 | 视觉模型自动识别图片生成弦号+描述，支持查重和原子写入 |

### Pipeline 视频处理

| 功能 | 说明 |
|:-----|:-----|
| 📦 YOLO 船只检测 | 基于 ultralytics YOLO，支持 ByteTrack / BoTSORT 追踪 |
| 🔗 跟踪 ID 绑定 | track ID 绑定唯一弦号，跟踪持续则沿用，无需重复调用 Agent |
| 🔄 定时刷新 | `enable_refresh` 开启后，每隔 `gap_num` 帧自动重新识别已跟踪船只 |
| 🤖 Qwen3.5 VLM 识别 | 视觉大模型对裁剪图像进行弦号识别与描述生成 |
| 📍 **弦号框定位** | VLM 输出弦号文字的相对坐标，映射到 frame 坐标绘制虚线框 |
| 🔀 Agent/硬编码双模式 | `use_agent` 开关：Agent 模式用 LangChain 三步工具链；硬编码模式直接调用 |
| ⚡ 级联/并发双模式 | 级联同步等待；并发双层架构（帧级队列 + crop 级并发） |
| 📊 FPS 统计 | 10 秒滑动窗口统计码流和处理帧率 |
| 🎨 Demo 可视化 | 实时显示 YOLO 检测框 + 弦号虚线框 + 跟踪 ID + 识别结果 + FPS HUD |

---

## 🏗️ 项目结构

```
ship-hull-agent/
├── config.py                  # 配置读取：config.yaml + 内置默认值
├── config.yaml                # 全局配置文件（LLM / Embedding / Pipeline）
├── build_db.py                # 批量建库脚本（图片 → 弦号+描述 → CSV）
│
├── database/__init__.py       # ShipDatabase：CSV + FAISS 向量库 + 自动变更检测
├── tools/__init__.py          # LangChain @tool：lookup + retrieve + _vlm_infer
├── agent/__init__.py          # ShipHullAgent：ReAct Agent + Few-shot 示例
├── cli/                       # Rich CLI：单次查询 / 交互 REPL / --verbose
│
├── pipeline/                  # 🎬 视频处理流水线
│   ├── __main__.py            # python -m pipeline 入口
│   ├── cli.py                 # 命令行参数解析
│   ├── pipeline.py            # 主流水线编排（ShipPipeline）
│   ├── detector.py            # YOLO 船只检测 + ByteTrack 跟踪
│   ├── agent_inference.py     # Qwen3.5 VLM 弦号识别 + 弦号框定位
│   ├── tracker.py             # 跟踪状态管理（track ID ↔ 弦号 + hull_box 绑定）
│   ├── fps.py                 # 10 秒滑动窗口 FPS 统计
│   ├── video_input.py         # 视频/相机/视频流统一输入
│   └── demo.py                # Demo 可视化（检测框 + 弦号虚线框 + HUD）
│
├── data/ships.csv             # 船只数据库
├── tests/                     # 单元测试 + 并发压力测试
├── .env.example               # 环境变量模板
└── pyproject.toml             # 项目元数据 + 依赖声明
```

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/hyshhh/boatdetect2.git
cd boatdetect2
```

### 2. 启动视觉模型服务

使用 vLLM 部署 Qwen3.5 VLM（兼容 OpenAI API 格式）：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /media/ddc/新加卷/hys/hysnew/Qwen3.5-2B-AWQ \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 1024 \
  --port 7890 \
  --gpu-memory-utilization 0.15 \
  --max-num-seqs 10 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen
```

<details>
<summary>📋 参数说明</summary>

| 参数 | 值 | 说明 |
|:-----|:---|:-----|
| `--served-model-name` | `Qwen/Qwen3-VL-4B-AWQ` | 对外暴露的模型名称 |
| `--api-key` | `abc123` | API 密钥 |
| `--port` | `7890` | 服务端口 |
| `--max-model-len` | `1024` | 最大上下文长度 |
| `--gpu-memory-utilization` | `0.15` | GPU 显存占用比例 |
| `--max-num-seqs` | `10` | 最大并发序列数 |
| `--enable-auto-tool-choice` | — | 启用自动工具调用 |
| `--tool-call-parser` | `qwen` | 工具调用解析器 |

</details>

### 3. 安装依赖

```bash
pip install -e .
# 开发模式（含测试依赖）
pip install -e ".[dev]"
```

### 4. 配置

```bash
cp .env.example .env
```

编辑 `config.yaml`：

```yaml
# LLM 对话模型
llm:
  model: "Qwen/Qwen3-VL-4B-AWQ"
  api_key: "abc123"
  base_url: "http://localhost:7890/v1"
  temperature: 0.0

# Embedding 模型
embed:
  model: "text-embedding-v4"
  api_key: "your-embed-api-key"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

# RAG 检索
retrieval:
  top_k: 3
  score_threshold: 0.5
```

### 5. 运行

```bash
# Agent 单次查询
ship-hull "帮我查一下弦号0014是什么船"

# Agent 交互模式
ship-hull --interactive

# Pipeline 处理视频
python -m pipeline.cli video.mp4 --demo --output result.mp4
```

---

## 💬 Agent 使用示例

### 精确匹配

```
$ ship-hull "帮我查一下弦号0014是什么船"

识别结果：弦号 0014，描述：白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪
```

### 语义检索

```
$ ship-hull "弦号9999，这是一艘大型白色邮轮，船身有蓝色条纹装饰，有三个烟囱"

未找到对应弦号，根据描述检索到最相似的船：
1. 弦号 0123，描述：白色邮轮，船身有红蓝条纹装饰，三座烟囱（相似度：0.9234）
2. 弦号 0014，描述：白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪（相似度：0.7521）
```

### 详细模式

```
$ ship-hull --verbose "我看到一艘灰色的军舰，外形很隐身"

┌─────────────────── 🔧 Agent 调用链 ───────────────────┐
│ # │ 类型     │ 内容                                   │
├───┼──────────┼────────────────────────────────────────┤
│ 0 │ human    │ 我看到一艘灰色的军舰，外形很隐身         │
│ 1 │ ai       │ → lookup_by_hull_number({"hull_number… │
│ 2 │ tool     │ ← {"found": false, "hull_number": ""}  │
│ 3 │ ai       │ → retrieve_by_description({"target_de… │
│ 4 │ tool     │ ← {"results": [{"hull_number": "0256"… │
│ 5 │ ai       │ 未找到对应弦号，根据描述检索到最相似的船… │
└─────────────────────────────────────────────────────────┘
```

### Agent 工作流程

```
用户输入
  │
  ├─ 包含弦号？
  │    → lookup_by_hull_number 精确查找
  │         ├─ found=true  → 直接返回结果
  │         └─ found=false → retrieve_by_description 语义检索
  │
  └─ 只有描述？
       → retrieve_by_description 语义检索
            ├─ FAISS 向量相似度匹配（top_k + 阈值过滤）
            └─ 返回最匹配的弦号 + 描述 + 相似度
```

---

## 🎬 Pipeline 视频处理

### 架构

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  视频输入     │────▶│  YOLO 检测+跟踪   │────▶│  裁剪船只区域 (crop) │
│ 文件/相机/流  │     │  ByteTrack       │     └────────┬──────────┘
└─────────────┘     └──────────────────┘              │
                                                      ▼
                                           ┌───────────────────┐
                                           │  use_agent 开关    │
                                           └───┬───────────┬───┘
                                               │           │
                                         false ▼     true  ▼
                                     ┌──────────┐  ┌───────────────────┐
                                     │ 硬编码链路 │  │ LangChain Agent   │
                                     │ VLM→查库  │  │ 三步工具链编排     │
                                     │ →语义检索 │  │ →lookup→retrieve  │
                                     └────┬─────┘  └────────┬──────────┘
                                          │                 │
                                          └────────┬────────┘
                                                   ▼
                                        ┌───────────────────┐
                                        │  绑定结果 + hull_box │
                                        └────────┬──────────┘
                                                   ▼
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  输出视频     │◀────│  渲染检测框        │◀────│  Agent 数据库匹配   │
│ + Demo 窗口  │     │  + 弦号虚线框     │     │  精确匹配 / 语义检索 │
│ + FPS HUD    │     │  + HUD           │     └───────────────────┘
└─────────────┘     └──────────────────┘
```

### 快速使用

```bash
# 处理视频文件
python -m pipeline.cli video.mp4

# USB 相机
python -m pipeline.cli 0

# RTSP 视频流
python -m pipeline.cli rtsp://192.168.1.100/stream

# Demo 可视化 + 输出结果视频
python -m pipeline.cli video.mp4 --demo --output result.mp4

# Agent 模式（LangChain ReAct Agent 三步工具链）
python -m pipeline.cli video.mp4 --agent --demo --output result.mp4

# 硬编码模式（默认，直接调用 VLM + 查库 + 语义检索）
python -m pipeline.cli video.mp4 --no-agent --demo --output result.mp4

# 并发模式（高吞吐）
python -m pipeline.cli video.mp4 --concurrent --max-concurrent 8

# 简略提示词 + 每 5 帧处理一次（提速）
python -m pipeline.cli video.mp4 --prompt-mode brief --process-every 5
```

### 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|:-----|:-----|:-----|:-------|
| `source` | — | 输入源：文件路径 / 相机号 / RTSP URL | **必填** |
| `--output` | `-o` | 输出视频路径 | 无 |
| `--demo` | — | 叠加检测框 + 弦号虚线框 + 识别结果 | 关闭 |
| `--display` | — | 实时显示窗口 | 关闭 |
| `--concurrent` | `-c` | 并发模式 | 关闭 |
| `--agent` | — | 使用 LangChain Agent 模式 | 关闭 |
| `--no-agent` | — | 使用硬编码模式 | — |
| `--enable-refresh` | — | 开启定时刷新 | 关闭 |
| `--gap-num` | — | 刷新间隔帧数 | `150` |
| `--max-concurrent` | — | 最大并发推理数 | `4` |
| `--process-every` | — | 每 N 帧处理一次 | `1` |
| `--prompt-mode` | — | `detailed` / `brief` | `detailed` |
| `--max-frames` | — | 最大处理帧数（0=不限） | `0` |
| `--yolo-model` | — | YOLO 模型路径 | `yolov8n.pt` |
| `--device` | — | 推理设备 | 自动 |
| `--conf` | — | 检测置信度阈值 | `0.25` |
| `--verbose` | `-v` | 详细日志 | 关闭 |

### 级联 vs 并发模式

<table>
<tr>
<th>级联模式（默认）</th>
<th>并发模式（--concurrent）</th>
</tr>
<tr>
<td>

```
帧 N → YOLO → 同步等待 Agent → 渲染帧 N → 帧 N+1
```

✅ 简单可靠，帧间严格有序
❌ 速度受 Agent 延迟影响

</td>
<td>

```
帧 N → YOLO → crop 送入队列 → 渲染帧 N
                  ↓
          Agent 线程池异步推理 → 结果回填
```

✅ 高吞吐，帧率不受 Agent 延迟影响
❌ 需要更多显存，结果可能有 1-2 帧延迟

</td>
</tr>
</table>

### 📍 弦号框定位

**新增功能**：VLM 在识别弦号的同时输出弦号文字的精确边界框坐标。

#### 数据流

```
VLM 输出 hull_box (crop 内相对坐标 0.0~1.0)
        │
        ▼
Pipeline 传递 hull_box → AgentResult → TrackInfo
        │
        ▼
Demo 坐标映射：相对坐标 → crop 像素 → frame 坐标
        │
        ▼
绘制弦号虚线框（与 YOLO 检测框叠加显示）
```

#### 坐标映射

```
crop 创建：pad=20px 扩展 YOLO bbox
crop[cy1:cy2, cx1:cx2] → resize 到 256~512px → VLM

VLM 返回：hull_box = [rx1, ry1, rx2, ry2]（相对坐标）

映射公式：
  frame_x = cx1 + rx × (cx2 - cx1)
  frame_y = cy1 + ry × (cy2 - cy1)
```

#### Demo 可视化效果

```
┌─────────────────────────────────────────┐
│  ┌──────────────────────────┐           │
│  │ ID:3 (YOLO 检测框-实线)   │           │
│  │                          │           │
│  │   ┌ ─ ─ ─ ─ ┐           │           │
│  │   │ 0014    │ ← 弦号虚线框│           │
│  │   └ ─ ─ ─ ─ ┘           │           │
│  │                          │           │
│  └──────────────────────────┘           │
│  (库内确定id：0014)                       │
│                                          │
│  Frame: 142 | stream: 25.3 FPS          │
└─────────────────────────────────────────┘
```

### 跟踪与识别流程

```
新 track 出现（YOLO 分配 ID）
  │
  ▼
标记 pending → Agent 推理（VLM 识别弦号 + 描述 + hull_box）
                    │
        ┌───────────┴───────────┐
        │                       │
   识别到弦号              未识别到弦号
        │                       │
  数据库精确匹配           数据库语义检索
        │                       │
   ┌────┴────┐             ┌────┴────┐
   │         │             │         │
 匹配成功  未匹配        匹配成功  未匹配
   │         │             │         │
(绿色-库内) (红色-未知)  (黄色-候选) (红色-无)
```

### 交互按键

| 按键 | 动作 |
|:-----|:-----|
| `q` | 退出 |
| `d` | 切换提示词模式（详细 ↔ 简略） |
| `p` | 暂停 / 继续 |
| `s` | 截图 |

---

## 📸 批量建库

通过图片自动识别船只，构建 CSV 数据库。

```bash
python3 build_db.py ./images
```

支持格式：`.jpg` `.jpeg` `.png` `.bmp` `.webp` `.gif`

<details>
<summary>📋 处理流程与查重逻辑</summary>

### 处理流程

对每张图片：
1. **调用视觉模型**识别船只 → 弦号 + 描述
2. **查重检查**：弦号已存在 → 提示跳过 / 覆盖 / 手动输入
3. **弦号确认**：新弦号 → 提示确认 / 手动修正
4. **立即写入 CSV**（中断不丢数据）

### 交互示例

```
$ python3 build_db.py ./my_ship_photos

📦 已有数据库: ./data/ships.csv（4 条记录）
🖼️  找到 12 张图片，开始识别...

============================================================
[1/12] 处理: ship_001.jpg
============================================================

  📝 识别结果:
     弦号: 0014
     描述: 白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪

  ⚠️  弦号 [0014] 已存在于数据库中
  按 1 跳过 / 按 2 覆盖 / 按 3 手动输入新弦号
  请选择 [1/2/3] (1): 1
  ⏭️  已跳过

============================================================
📊 处理完成
   总计: 12 张图片 | 成功: 10 条 | 跳过: 2 条
   数据库: ./data/ships.csv（共 14 条记录）
============================================================
```

### 查重逻辑

| 场景 | 处理方式 |
|:-----|:---------|
| 模型识别出弦号，数据库已存在 | 提示：跳过 / 覆盖 / 手动输入新弦号 |
| 用户手动输入弦号，数据库已存在 | 提示：跳过 / 覆盖 |
| 无弦号，用文件名作 fallback | 自动加后缀 `_2`、`_3` 避免覆盖 |

</details>

---

## ⚙️ 配置说明

<details>
<summary>📄 config.yaml 完整参考</summary>

```yaml
# ── 对话模型 ──
llm:
  model: "Qwen/Qwen3-VL-4B-AWQ"
  api_key: "abc123"
  base_url: "http://localhost:7890/v1"
  temperature: 0.0

# ── Embedding 模型 ──
embed:
  model: "text-embedding-v4"
  api_key: "your-embed-api-key"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ── RAG 检索 ──
retrieval:
  top_k: 3
  score_threshold: 0.5

# ── 向量库 ──
vector_store:
  persist_path: "./vector_store"
  auto_rebuild: false

# ── 应用 ──
app:
  log_level: "INFO"
  ship_db_path: "./data/ships.csv"

# ── Pipeline ──
pipeline:
  use_agent: false           # false=硬编码模式 / true=Agent模式
  enable_refresh: false      # 定时刷新开关
  gap_num: 150               # 刷新间隔帧数
  concurrent_mode: false     # false=级联 / true=并发
  max_concurrent: 4          # 最大并发推理线程数
  max_queued_frames: 30      # 最大队列深度（防 OOM）
  process_every_n_frames: 1  # 每 N 帧处理一次
  prompt_mode: "detailed"    # detailed / brief
  demo: false                # Demo 可视化开关
  yolo_model: "yolov8n.pt"
  device: ""                 # "" 自动 / "cpu" / "0"
  conf_threshold: 0.25
  tracker: "bytetrack"       # bytetrack / botsort
  detect_classes: [8]        # COCO: 8=boat
  max_stale_frames: 300
```

</details>

---

## 🧠 本地部署 Embedding 模型

<details>
<summary>📋 推荐模型与部署方式</summary>

### 推荐模型

| 模型 | 参数量 | 显存 | 特点 |
|:-----|:-------|:-----|:-----|
| **Qwen3-Embedding-0.6B** ⭐ | 0.6B | ~1.5GB | 极轻量，中文效果好 |
| BGE-M3 | ~2.2GB | ~3GB | 中英双语强 |
| bge-large-zh-v1.5 | ~1.3GB | ~2GB | 纯中文优化 |

### 部署

```bash
# 下载模型
pip install modelscope
modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir ./models/Qwen3-Embedding-0.6B

# vLLM 部署
python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen3-Embedding-0.6B \
  --api-key abc123 \
  --served-model-name Qwen3-Embedding-0.6B \
  --convert embed \
  --gpu-memory-utilization 0.15 \
  --max-model-len 2048 \
  --port 7891
```

### 本地 vs 云端

| | DashScope 云端 | 本地 Qwen3-Embedding-0.6B |
|:---|:---|:---|
| 延迟 | ~200-500ms | ~20-50ms |
| 费用 | 按 token 收费 | 免费 |
| batch 限制 | 10 条/次 | 无限制 |
| 网络依赖 | 需联网 | 完全离线 |

</details>

---

## 🧪 测试

```bash
# 全部测试
pytest -v

# 数据库测试
pytest tests/test_database.py -v

# Pipeline 测试
pytest tests/test_pipeline.py -v
```

---

## 🛠️ 技术栈

| 组件 | 技术 | 用途 |
|:-----|:-----|:-----|
| LLM 编排 | LangChain + LangGraph | ReAct Agent 模式 |
| 向量库 | FAISS (faiss-cpu) | 语义检索索引 |
| Embedding | OpenAI Embeddings API | 文本向量化 |
| 视觉模型 | Qwen3.5 VLM | 弦号识别 + 弦号框定位 |
| 目标检测 | ultralytics YOLO | 船只检测 |
| 跟踪 | ByteTrack | 多目标跟踪 |
| 视频处理 | OpenCV (cv2) | 视频读写、图像处理 |
| 并发 | threading + queue.Queue | 级联/并发双模式 |
| CLI | Rich | 终端美化输出 |

---

## 📝 开发指南

### 换 LLM Provider

任何兼容 OpenAI API 格式的服务：

```yaml
llm:
  model: "gpt-4o"
  api_key: "sk-xxx"
  base_url: "https://api.openai.com/v1"
```

### 添加新工具

在 `tools/__init__.py` 添加 `@tool` 函数，`build_tools()` 返回列表中加上即可。

### 数据变更后重建索引

```bash
# 方式一：自动重建
VECTOR_STORE_AUTO_REBUILD=true ship-hull "触发重建"

# 方式二：删除缓存
rm -rf ./vector_store  # 下次启动自动重建
```

---

<div align="center">

**[⬆ 回到顶部](#-ship-hull-agent)**

Made with ❤️ for maritime intelligence

</div>
