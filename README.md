# 🏋️ Workout Coach（VLM 健身动作纠正）

> 基于 **Qwen2-VL-7B-Instruct** 的本地端侧多模态项目：支持摄像头实时采集或本地视频文件输入，异步推理动作问题，并将反馈叠加到画面中。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/Transformers-4.37+-yellow)
![Platform](https://img.shields.io/badge/Platform-Windows%2011-lightgrey)
![GPU](https://img.shields.io/badge/GPU-RTX%204060%208GB-green)

## 项目概览

这是一个面向多模态岗位的工程化 Demo，重点解决三个问题：

1. **8GB 显存下的 VLM 部署**（4-bit 量化 + low_cpu_mem_usage）
2. **实时视频循环不卡顿**（异步推理 + 多线程）
3. **输出可执行动作建议**（结构化 prompt + 可视化反馈）

---

## ✨ 核心亮点

- **端侧部署**：Qwen2-VL-7B-Instruct + BitsAndBytes 4-bit，~5.5GB VRAM
- **双输入模式**：摄像头实时拍摄 / 本地视频文件（mp4、mov、mkv 等）
- **任意动作支持**：不依赖预设动作库，输入任意动作名称，VLM 自动核验并分析
- **动作核验**：VLM 对比用户声称动作与画面内容，不符时自动识别真实动作
- **系统设计**：采样缓冲队列 + 后台推理线程 + 回调渲染
- **工程可维护**：`.env` 配置、模块化结构、离线模式支持

---

## 📁 文件结构

```
Workout-Coach/
├── input/                    # 放置待分析的本地视频文件
├── action_profiles.py        # 动作分析提示词构建（动态，支持任意动作）
├── download_model.py         # 模型预下载脚本（支持 HuggingFace / ModelScope）
├── vlm_inference.py          # VLM 推理核心（同步/异步）
├── video_streamer.py         # 视频采样、缓冲与触发
├── main.py                   # 主程序入口
├── config.env.example        # 环境变量模板
├── requirements.txt          # 依赖列表
├── .gitignore
└── README.md
```

---

## ⚙️ 环境配置（Conda）

### 1) 创建环境

```bash
conda create -n workout-coach python=3.10
conda activate workout-coach
```

### 2) 安装 PyTorch（CUDA）

```bash
conda install pytorch::pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3) 安装项目依赖

```bash
pip install -r requirements.txt
```

### 4) 配置 `.env`

```powershell
Copy-Item config.env.example .env
```

打开 `.env`，填写模型路径（下载完成后脚本会打印）：

```
VLM_MODEL_NAME=<模型本地路径或 HF model id>
HF_HOME=C:/Users/<你的用户名>/Desktop/Workout-Coach/hf_cache
VLM_LOCAL_FILES_ONLY=1
HF_HUB_OFFLINE=1
```

### 5) 下载模型

**方式 A：ModelScope（国内推荐，速度快）**

```bash
pip install modelscope
python download_model.py --model Qwen/Qwen2-VL-7B-Instruct --source modelscope
```

**方式 B：HuggingFace 镜像**

```bash
python download_model.py --model Qwen/Qwen2-VL-7B-Instruct
```

下载完成后，脚本会输出建议写入 `.env` 的配置，复制粘贴即可。

---

## 🚀 快速启动

```bash
conda activate workout-coach
python main.py
```

### 启动流程

1. **VLM 模型加载**（约 30 秒）
   - 4-bit 量化加载，稳态显存占用约 5.5GB

2. **选择视频输入源**
   - 将视频文件放入 `input/` 目录，程序启动时自动扫描并列出
   - 选择文件编号使用文件模式；回车或选 `0` 使用摄像头

3. **输入动作名称**
   - 支持任意动作，如：深蹲、硬拉、哑铃飞鸟、引体向上、推肩……
   - 直接回车使用默认：深蹲

4. **就绪**
   - 文件模式：视频自动播放，播放结束后自动触发分析
   - 摄像头模式：按 **S** 键触发分析

---

## 🎮 快捷键

| 模式 | 快捷键 | 功能 |
|------|--------|------|
| 摄像头 | **S** | 倒计时 5 秒后录制 10 秒，发送给 VLM 分析 |
| 文件 | **S** | 立即触发分析（使用已采样的帧） |
| 通用 | **Q** | 退出程序 |

---

## 📊 工作流说明

```
输入源（摄像头/文件）
    → 按短边缩放（横竖屏自适应，默认 336px）
    → 每 10 帧采 1 帧，存入缓冲队列
    → 触发分析（按 S / 文件播放结束）
    → 均匀抽取最多 8 帧 → VLM 异步推理
    → 结果叠加到画面左上角（显示 4 秒）
```

### VLM 分析逻辑

每次推理时，VLM 会：

1. **核验动作**：对比用户声称的动作与画面内容是否一致
   - 一致 → `【画面动作】深蹲`
   - 不一致 → `【画面动作】硬拉（用户声称：深蹲）`
2. **评判质量**：给出总体结论、3 个关键问题（问题/原因/修正）、下一组执行口令

**输出示例**：

```
【画面动作】杠铃深蹲
【总体结论】动作整体较为标准，但膝盖稍微内扣，需注意向外打开。
【关键问题1】问题：膝盖内扣；原因：下降过程中膝盖未保持向外；修正：确保大腿与脚尖方向一致。
【关键问题2】问题：背部略微前倾；原因：核心收紧不足；修正：保持脊柱中立位。
【关键问题3】问题：杠铃位置偏高；原因：影响稳定性；修正：调整杠铃至合适背部位置。
【下一组执行口令】1. 膝盖向外打开；2. 挺胸收核心；3. 调整杠铃位置。
```

---

## ⚠️ 常见问题

### Q1: 加载模型时 OOM（显存溢出）

**原因**：BitsAndBytes 量化过程中的峰值显存高于稳态

**解决**：已内置 `low_cpu_mem_usage=True`，权重先在 CPU 处理再搬到 GPU，消除加载峰值。若仍 OOM，关闭其他占用 GPU 的程序后重试。

### Q2: 推理时间较长（约 2 分钟）

**原因**：Windows 上 FlashAttention2 不可用，回退到 SDPA，7B 模型 + 8 帧正常耗时

**说明**：可将 `VLM_MAX_FRAMES` 从 8 降到 6 缩短推理时间，代价是减少分析帧数。

### Q3: 摄像头无法打开

**解决**：关闭其他占用摄像头的程序（Teams、浏览器等）；或在 `.env` 中修改 `CAMERA_ID` / `CAMERA_BACKEND`

### Q4: 下载失败（`HF_HUB_OFFLINE` 报错）

**原因**：环境变量中 `HF_HUB_OFFLINE=1` 阻止了网络访问

**解决**：`download_model.py` 已自动强制覆盖此开关，直接运行脚本即可，无需手动修改环境变量

### Q5: ModelScope 下载后 `VLM_MODEL_NAME` 怎么填

下载完成后脚本会输出类似：

```
VLM_MODEL_NAME=C:\...\hf_cache\modelscope\Qwen--Qwen2-VL-7B-Instruct
```

将该行复制到 `.env` 替换原有 `VLM_MODEL_NAME` 即可（本地路径，不是 HF model id）。

---

## 🔧 主要配置项（`.env`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VLM_MODEL_NAME` | — | 模型名称或本地路径 |
| `VLM_MAX_FRAMES` | `8` | 发给 VLM 的最大帧数（均匀抽取） |
| `VLM_MAX_TOKENS` | `256` | 生成最大 token 数 |
| `TARGET_HEIGHT` | `336` | 短边缩放目标像素（横竖屏自适应） |
| `SAMPLE_RATE` | `10` | 每 N 帧采 1 帧 |
| `PRE_RECORD_DELAY` | `5.0` | 摄像头模式按 S 后的倒计时（秒） |
| `RECORD_DURATION` | `10.0` | 摄像头模式录制时长（秒） |
| `FEEDBACK_DURATION` | `4.0` | 结果在画面上显示时长（秒） |
| `VLM_LOCAL_FILES_ONLY` | `1` | 离线模式（仅用本地缓存） |
| `USE_FLASH_ATTENTION_2` | `True` | FlashAttention2（Windows 通常不可用，自动回退） |

---

## 📚 技术栈

| 组件 | 用途 |
|------|------|
| PyTorch 2.x | 深度学习框架 |
| Transformers 4.37+ | Qwen2-VL 模型加载 |
| BitsAndBytes | 4-bit 量化 |
| OpenCV | 视频采集与显示 |
| Pillow | 图像处理 |
| Accelerate | device_map 推理加速 |
| ModelScope（可选） | 国内模型下载 |

---

## 📞 故障排查清单

- [ ] Conda 环境已激活（`conda activate workout-coach`）
- [ ] `torch.cuda.is_available()` 返回 `True`
- [ ] `.env` 已创建且 `VLM_MODEL_NAME` / `HF_HOME` 填写正确
- [ ] 模型文件完整（`hf_cache/` 下存在 safetensors 权重文件）
- [ ] 无其他进程占用 GPU 显存
- [ ] 摄像头模式下硬件已连接且未被占用

---

## 🌐 GitHub 上传说明

`.gitignore` 已配置，以下内容不会被提交：

- `.env`（含本地路径）
- `hf_cache/`（模型权重，体积过大）
- `input/`（本地视频文件）
- `__pycache__/`

仅保留 `config.env.example` 作为配置模板。

### 项目描述建议

> 在 8GB 显存受限场景下完成 Qwen2-VL-7B 端侧部署，设计双输入模式（摄像头/视频文件）的异步推理链路，支持任意健身动作的自动核验与实时纠正反馈。

---

## 📄 许可证

本项目为教学 / 演示用途。

---

**祝你健身愉快！💪**
