# 🏋️ Workout Coach（VLM 健身动作纠正）

> 基于 **Qwen2-VL-2B-Instruct** 的本地端侧多模态项目：实时采集视频流，异步推理动作问题，并将反馈叠加到画面中。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/Transformers-4.37+-yellow)
![Platform](https://img.shields.io/badge/Platform-Windows%2011-lightgrey)
![GPU](https://img.shields.io/badge/GPU-RTX%204060%208GB-green)

## 项目概览

这是一个面向多模态岗位的工程化 Demo，重点解决三个问题：

1. **8GB 显存下的 VLM 部署**（4-bit 量化）
2. **实时视频循环不卡顿**（异步推理）
3. **输出可执行动作建议**（可视化反馈）

---

## ✨ 核心亮点

- **端侧部署**：Qwen2-VL-2B-Instruct + BitsAndBytes 4-bit
- **实时交互**：OpenCV 摄像头流 + 键盘触发分析
- **系统设计**：采样缓冲队列 + 后台推理线程 + 回调渲染
- **工程可维护**：`.env` 配置、模块化结构、离线模式支持

---

## 📁 文件结构

```
Workout-Coach/
├── action_profiles.py        # 动作类型与提示词模板
├── download_model.py         # 模型预下载脚本
├── vlm_inference.py          # VLM 推理核心（同步/异步）
├── video_streamer.py         # 视频采样、缓冲与按键触发
├── main.py                   # 主程序入口
├── config.env.example        # 环境变量模板
├── requirements.txt          # 依赖列表
├── .gitignore                # Git 忽略规则（模型缓存/.env 等）
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

**推荐：CUDA 11.8（RTX 4060 稳定）**

```bash
conda install pytorch::pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3) 安装项目依赖

```bash
pip install -r requirements.txt
```

### 4) 配置 `.env`

复制 `config.env.example` 为 `.env`，按需修改：

```powershell
Copy-Item config.env.example .env
```

（Git Bash / macOS / Linux 可用：`cp config.env.example .env`）

- `VLM_MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct`
- `VLM_LOCAL_FILES_ONLY=0`（首次下载）

若在中国大陆，可设置：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HUGGINGFACE_HUB_ENDPOINT = "https://hf-mirror.com"
```

### 5) 可选：预下载模型

```powershell
python download_model.py --model Qwen/Qwen2-VL-2B-Instruct --cache-dir .\hf_cache
```

---

## 🚀 快速启动

```bash
conda activate workout-coach
python main.py
```

> 离线模式下（`VLM_LOCAL_FILES_ONLY=1`），程序仅使用本地缓存。

### 启动流程说明

1. **VLM 模型加载**（15-30 秒）
   - 首次运行会从 Hugging Face 下载 Qwen2-VL-2B-Instruct 模型
   - 显存会从 0GB 升到 ~4-5GB（4-bit 量化）

2. **摄像头初始化**（1-2 秒）
   - 打开本地摄像头，显示实时视频

3. **就绪**
   - 继续做动作，完成后按 **'S'** 键触发分析

---

## 🎮 使用快捷键

| 快捷键 | 功能 |
|--------|------|
| 启动后 CLI 输入 | 选择动作类型（支持自定义，如：哑铃飞鸟） |
| **S** | 触发定时分析（先倒计时 5 秒，再录制 10 秒发送给 VLM） |
| **Q** | 退出程序 |

---

## 📊 工作流说明

### 动作分析流程

```
摄像头 → 帧采样缓冲（15帧） → 用户按S键 → VLM异步推理 → 建议显示在视频上
```

1. **连续采样**
   - 每 10 帧采 1 帧（降低推理显存压力）
   - 缓存 15 帧为一个"窗口"
   - 每帧短边缩放到 336 像素

2. **按 'S' 键触发分析**
   - 先倒计时 5 秒（用于走位准备）
   - 再自动录制 10 秒动作序列并发送给 VLM
   - VLM 在后台异步推理，不阻塞视频显示

3. **VLM 推理**（5-10 秒）
   - System Prompt：角色定义 + 任务说明
   - User Query：具体分析指令
   - 输出：专业的动作纠正建议

4. **反馈显示**
   - 建议文字以绿色显示在视频左上角
   - 持续显示 4 秒后消失

---

## 📝 核心代码架构

### 1. `vlm_inference.py` - VLM 推理类

```python
from vlm_inference import FitnessVLM

vlm = FitnessVLM(model_name="Qwen/Qwen2-VL-2B-Instruct", device="cuda")

# 同步推理
result = vlm.analyze_fitness_frames(
    frames=[PIL_image_1, PIL_image_2, ...],
    system_prompt="你是专业教练...",
    user_query="分析这些帧中的问题"
)

# 或异步推理（推荐，不阻塞UI）
vlm.analyze_fitness_frames_async(
    frames=[...],
    system_prompt="...",
    callback=lambda result: print(result)
)
```

### 2. `video_streamer.py` - 视频流处理

```python
from video_streamer import VideoStreamer

vs = VideoStreamer(
    camera_id=0,
    buffer_size=15,        # 缓冲 15 帧
    sample_rate=10,        # 每 10 帧采 1 帧
    target_height=336      # 缩放目标高度
)

# 设置触发回调
vs.on_analysis_trigger = lambda frames: analyze(frames)

# 启动（阻塞直到用户按 Q）
vs.start()
```

### 3. `main.py` - 完整集成

整合 VLM 和视频流，处理数据流向和 UI 反馈。

---

## ⚠️ 常见问题

### Q1: 首次运行很慢/卡住

**原因**: 模型首次从 Hugging Face 下载（1-2GB）

**解决**:
- 检查网络连接
- 或提前下载: `huggingface-cli download Qwen/Qwen2-VL-2B-Instruct`
- 中国大陆网络建议先设置镜像：`$env:HF_ENDPOINT="https://hf-mirror.com"`

### Q1.1: `download_model.py` 出现 WinError 1314（权限不足）

**原因**: Windows 未开启符号链接权限（Developer Mode/管理员权限）。

**解决**:
- 已内置自动回退：脚本会自动切换为“无符号链接模式”下载到 `hf_cache/local_models/...`
- 重新执行：`python download_model.py --model Qwen/Qwen2-VL-2B-Instruct --cache-dir .\hf_cache`
- 下载后将 `.env` 中 `VLM_MODEL_NAME` 设为本地目录路径（脚本会打印）

### Q2: GPU 显存溢出 (OOM)

**原因**: 8GB 显存不足或有其他 GPU 占用进程

**解决**:
- 减少 `buffer_size`（默认 15，改为 10）
- 或增加 `sample_rate`（默认 10，改为 15，即减少采样帧数）
- 关闭其他 GPU 程序

### Q3: 摄像头无法打开

**原因**: 
- 摄像头被其他程序占用
- 或硬件未连接

**解决**:
- 关闭其他视频程序（如 Teams、Chrome 摄像头等）
- 检查设备管理器中摄像头是否正常

### Q4: VLM 推理很慢（>20 秒）

**原因**: 正常行为（2B 模型在 RTX 4060 上）

**说明**: 
- 4-bit 量化 + 多帧推理约需 5-10 秒
- 首次推理会多花 1-2 秒（模型编译）

### Q4.1: 日志提示 `FlashAttention2 不可用`，Windows 不支持吗？

**结论**:
- `flash-attn` 在 Linux 上支持最完整，Windows 不是一等公民平台。
- Windows 并非绝对不能用，但通常需要本地编译，安装复杂且兼容性要求严格（CUDA / PyTorch / 编译工具链版本需匹配）。

**建议**:
- 日常使用可直接接受自动回退到 `SDPA/Eager`（程序可正常运行）。
- 若不想每次看到回退提示，可在 `.env` 中设置：`USE_FLASH_ATTENTION_2=0`。
- 若必须稳定使用 FA2，建议迁移到 WSL2 / Linux 环境。

### Q5: 结果显示不清楚/被遮挡

**解决**:
- 摄像头视角应包含全身（或至少关键关节）
- 光线充足有利于 VLM 理解

---

## 🔧 调优参数

可在 `main.py` 的 `initialize_video_streamer()` 中修改：

```python
video_streamer = VideoStreamer(
    camera_id=0,           # 摄像头 ID（如果多个摄像头，试 0, 1, 2...）
    buffer_size=15,        # 缓冲帧数（↓ 减少显存，↑ 更多上下文）
    sample_rate=10,        # 采样率（↑ 减少帧数，↓ 更高采样密度）
    target_height=336,     # 缩放高度（↓ 显存更低，↑ 细节更多）
    fps=30,                # 摄像头帧率
)
```

---

## 🐛 调试模式

如果需要更详细的日志：

```python
# 在 main.py 中修改
vlm_model = FitnessVLM(..., verbose=True)  # 已启用
video_streamer = VideoStreamer(..., verbose=True)  # 已启用
```

---

## 📚 技术栈

| 组件 | 版本 | 用途 |
|------|------|------|
| PyTorch | 2.0+ | 深度学习框架 |
| Transformers | 4.37+ | HuggingFace 模型加载 |
| BitsAndBytes | 0.41.3+ | 4-bit 量化 |
| OpenCV | 4.8+ | 视频采集 |
| Pillow | 10.0+ | 图像处理 |
| Accelerate | 0.25+ | 推理加速 |

---

## 📖 进阶用法

### 自定义 System Prompt

在 `main.py` 中修改 `SYSTEM_PROMPT`:

```python
SYSTEM_PROMPT = """你是一个...（自定义角色定义）"""
```

### 修改分析查询

```python
ANALYSIS_QUERY = "请分析...（自定义问题）"
```

### 批量分析模式

```python
# 连续按 S 多次，每次都会触发新的推理
# 建议等待上一次推理完成后再按 S（UI 会提示）
```

---

## 🎯 预期效果

**输入**: 15 帧连续视频（深蹲/硬拉）

**输出示例**:
```
【问题】躯干过度前倾，腰部圆背风险
【原因】髋关节灵活性不足，核心肌群激活不充分
【修正】想象向后坐，保持脊柱中立，下蹲时挺胸
```

---

## 💡 提示

1. **最佳实践**
   - 每次动作完成后立即按 S（缓冲区最新数据最准）
   - 光线充足，摄像头距离 1-3 米

2. **显存管理**
   - VLM 加载后占用 ~4.5GB（4-bit 量化）
   - 单次推理最多额外占用 1-2GB（临时）
   - 推理完成后自动释放

3. **精度优化**
   - 增加 `buffer_size` 获得更多上下文（但显存升高）
   - 减少 `sample_rate` 增加采样密度（更细致分析）

---

## 📞 故障排查清单

- [ ] Conda 环境已激活
- [ ] PyTorch CUDA 版本正确 (`torch.cuda.is_available() == True`)
- [ ] 所有依赖已安装 (`pip check` 无错误)
- [ ] 摄像头硬件正常连接
- [ ] 网络可访问 Hugging Face（模型下载）
- [ ] GPU 显存足够 (8GB+)
- [ ] 无其他 GPU 进程占用

---

## 🌐 GitHub 上传建议（已处理）

- 已移除本地模型与缓存目录（避免仓库过大）
- 已添加 `.gitignore`，默认忽略 `.env`、`hf_cache/`、`__pycache__/` 等本地文件
- 建议仅保留 `config.env.example` 作为配置模板
- 不要提交 `.env`（可能包含本地路径、镜像端点或私有配置）

### 作为实习项目的推荐描述

> 在 8GB 显存受限场景下，完成 Qwen2-VL 端侧部署，设计异步视频推理链路，支持实时动作纠正反馈，兼顾性能、可用性与工程可维护性。

---

## 📄 许可证

本项目为教学/演示用途。

---

**祝你健身愉快！💪**
