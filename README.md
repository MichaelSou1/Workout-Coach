# Workout Coach — Qwen2-VL 健身动作质量评估

> 基于 **Qwen2-VL-7B-Instruct** 的端到端健身动作质量评估系统。
> 训练端覆盖 Fitness-AQA 数据解析、LoRA SFT 与 GRPO 奖励对齐全流程；
> 推理端支持摄像头 / 本地视频文件输入，实时输出结构化纠错建议。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/Transformers-5.x-yellow)
![Train](https://img.shields.io/badge/Train-A100%2080GB%20BF16-purple)
![Infer](https://img.shields.io/badge/Infer-RTX%204060%208GB-green)

---

## 项目概览

| 阶段 | 内容 | 硬件 |
|------|------|------|
| 数据准备 | Fitness-AQA 解析 → 统一 JSON | 本地 CPU |
| SFT 微调 | LoRA (r=32) + Gradient Checkpointing + DeepSpeed Zero-2 | A100 80GB |
| GRPO 对齐 | 自定义奖励函数（错误识别 + 时间精度） + num_generations=16 | A100 80GB |
| 端侧推理 | 4-bit 量化，摄像头 / 视频文件，异步推理 + 实时反馈叠加 | RTX 4060 8GB |

### 为什么用 GRPO 而不是 PPO

GRPO 无需 Value Model（Critic），将同一 prompt 的多个生成结果在组内相对归一化作为优势估计，在小数据集（Fitness-AQA ~4400 条）下更稳定，且显存占用更低。

---

## 核心亮点

**训练侧**
- Fitness-AQA 三动作子集（深蹲/过头推举/杠铃划船）完整解析，含时间区间与帧级二值两种标注格式
- 奖励函数四分量：格式合规 × 错误识别 × 时间精度 × 修正质量
- LoRA 同时注入语言解码器 Attention 层与视觉-语言对齐层（`visual.merger.mlp`）

**推理侧**
- 任意动作支持：不依赖预设动作库，VLM 自动核验用户输入与画面内容是否一致
- 双输入模式：摄像头实时拍摄 / 本地 mp4 文件
- 采样缓冲队列 + 后台推理线程 + 回调渲染，主线程不阻塞

---

## 文件结构

```
Workout-Coach/
├── training/                       # 训练管线
│   ├── prepare_dataset.py          # Fitness-AQA 原始格式 → 统一 JSON
│   ├── data_builder.py             # Dataset / Collator / Tokenization
│   ├── train_sft.py                # SFT 训练（SFTTrainer + LoRA）
│   ├── train_grpo.py               # GRPO 对齐训练（GRPOTrainer）
│   ├── ds_config_zero2.json        # DeepSpeed ZeRO-2 配置
│   ├── annotations/                # 生成的 train/val/test JSON（gitignore）
│   └── training_README.md          # 训练启动文档
│
├── input/                          # 放置待分析的本地视频文件
├── action_profiles.py              # 动作分析提示词构建
├── vlm_inference.py                # VLM 推理核心（同步/异步）
├── video_streamer.py               # 视频采样、缓冲与触发
├── main.py                         # 推理主程序入口
├── download_model.py               # 模型预下载脚本
├── config.env.example              # 环境变量模板
└── requirements.txt                # 推理端依赖
```

---

## 快速上手

### 推理（本地，RTX 4060）

```bash
conda create -n workout-coach python=3.10
conda activate workout-coach
conda install pytorch::pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

# 下载模型（国内用 ModelScope）
python download_model.py --model Qwen/Qwen2-VL-7B-Instruct --source modelscope

# 配置 .env
Copy-Item config.env.example .env   # Windows PowerShell
# 填写 VLM_MODEL_NAME、HF_HOME

python main.py
```

**操作流程**：启动后选择视频源 → 输入动作名称 → 文件模式自动触发 / 摄像头模式按 **S** 触发分析 → 按 **Q** 退出。

### 训练（A100 80GB）

```bash
# 0. 安装训练依赖
pip install trl>=0.13 deepspeed>=0.14 flash-attn>=2.5 qwen-vl-utils --no-build-isolation

# 1. 解析 Fitness-AQA 数据集
python training/prepare_dataset.py \
    --dataset_root dataset/Fitness-AQA_dataset_release \
    --output_dir   training/annotations

# 2. SFT 微调
deepspeed --num_gpus=1 training/train_sft.py \
    --model_name_or_path hf_cache/modelscope/Qwen--Qwen2-VL-7B-Instruct \
    --train_ann_file training/annotations/train.json \
    --val_ann_file   training/annotations/val.json \
    --dataset_root   dataset/Fitness-AQA_dataset_release \
    --output_dir     checkpoints/sft

# 3. GRPO 对齐
deepspeed --num_gpus=1 training/train_grpo.py \
    --model_name_or_path checkpoints/sft/final \
    --train_ann_file training/annotations/train.json \
    --dataset_root   dataset/Fitness-AQA_dataset_release \
    --output_dir     checkpoints/grpo
```

详细参数说明见 [training/training_README.md](training/training_README.md)。

---

## Fitness-AQA 数据集

| 子集 | 动作 | 媒体 | 标注格式 | 训练样本 |
|------|------|------|---------|---------|
| Squat | 深蹲 | 视频 | 时间区间（膝盖前伸/内扣） | 1136 |
| OHP | 过头推举 | 视频 | 时间区间（手肘/膝盖） | 1582 |
| ShallowSquat | 深蹲深度检测 | 图片帧 | 二值（0/1） | 440 |
| BarbellRow | 杠铃划船 | 图片帧 | 二值（腰椎/躯干角度） | 1284 |

---

## VLM 输出格式

```
【动作识别】杠铃深蹲
【总体结论】动作整体较为标准，但膝盖稍微内扣。
【关键问题1】问题：膝盖内扣（约 2.3s）；原因：臀中肌激活不足；修正：下蹲时膝盖跟随脚尖方向向外推
【关键问题2】问题：重心略偏前；原因：踝关节灵活性不足；修正：可在脚跟下垫小板辅助训练
【关键问题3】暂无明显问题
【下一组口令】收紧核心；膝盖向外；缓慢下降控制离心
```

---

## 技术栈

| 组件 | 用途 |
|------|------|
| Qwen2-VL-7B-Instruct | 视觉语言基础模型 |
| PyTorch 2.x + BF16 | 训练框架 |
| TRL (SFTTrainer / GRPOTrainer) | 训练算法 |
| PEFT / LoRA | 参数高效微调 |
| DeepSpeed ZeRO-2 | 显存优化 |
| FlashAttention-2 | 注意力加速 |
| BitsAndBytes | 推理端 4-bit 量化 |
| OpenCV + Pillow | 视频 / 图像处理 |

---

## 常见问题

**Q: 推理时显存 OOM（RTX 4060 8GB）**  
已内置 `low_cpu_mem_usage=True`，权重先在 CPU 处理再搬 GPU。若仍 OOM，关闭其他占用 GPU 的进程。

**Q: 摄像头无法打开**  
关闭 Teams、浏览器等占用摄像头的程序；或在 `.env` 中修改 `CAMERA_ID` / `CAMERA_BACKEND`。

**Q: 训练时 `ModuleNotFoundError: qwen_vl_utils`**  
```bash
pip install qwen-vl-utils
```

**Q: A100 上 flash-attn 编译失败**  
将 `train_sft.py` 和 `train_grpo.py` 中的 `attn_implementation="flash_attention_2"` 改为 `"sdpa"`，性能略降约 15%。

---

## 引用

本项目使用的 Fitness-AQA 数据集：

```bibtex
@article{parmar2022domain,
  title   = {Domain Knowledge-Informed Self-Supervised Representations for Workout Form Assessment},
  author  = {Parmar, Paritosh and Gharat, Amol and Rhodin, Helge},
  journal = {arXiv preprint arXiv:2202.14019},
  year    = {2022}
}
```

---

## 许可证

本项目为教学 / 研究用途。
