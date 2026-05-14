# Workout-Coach 训练管线文档

## 概览

两阶段对 **Qwen2-VL-7B-Instruct** 进行健身动作分析领域的后训练：

| 阶段 | 脚本 | 算法 | 目标 |
|------|------|------|------|
| 1 | `train_sft.py` | SFT + LoRA | 让模型学会输出格式化的动作分析报告 |
| 2 | `train_grpo.py` | GRPO + 纯 RLAIF | 用 LLM 裁判信号对齐错误识别精度与修正质量 |

**支持硬件**：

| 配置 | DeepSpeed | SFT | GRPO |
|------|-----------|-----|------|
| A100 80GB × 1 | ZeRO-2 | ✓ | G=8, 256 tokens |
| RTX 3060 20GB × 4 | ZeRO-3 + CPU offload | ✓ | G=4, 128 tokens |

---

## 1. 环境安装

```bash
conda activate workout-coach   # 与推理端共用同一环境

# 训练专用依赖
pip install trl>=0.13.0
pip install deepspeed>=0.14.0
pip install openai                                         # RLAIF 裁判 API 调用
pip install flash-attn>=2.5.0 --no-build-isolation        # A100/RTX30系列 必装
pip install qwen-vl-utils opencv-python pillow
```

> `flash-attn` 编译需要与 PyTorch 版本匹配的 CUDA Toolkit。若编译失败，
> 将两个训练脚本中的 `attn_implementation="flash_attention_2"` 改为 `"sdpa"`，
> 性能略降约 15% 但功能完全正常。

---

## 2. 数据集准备

### 2.1 放置数据集

将 `Fitness-AQA_dataset_release` 放到 `dataset/` 目录下，最终结构：

```
dataset/Fitness-AQA_dataset_release/
├── Squat/
│   └── Labeled_Dataset/
│       ├── videos/videos/          # *.mp4
│       ├── Labels/                 # error_knees_forward.json, error_knees_inward.json
│       ├── Splits/                 # train_keys.json, val_keys.json, test_keys.json
│       └── Shallow_Squat_Error_Dataset/
│           ├── images/crops_unaligned/   # *.jpg
│           ├── labels_shallow_depth.json
│           └── splits/             # train_ids.json, val_ids.json, test_ids.json
├── OHP/
│   └── Labeled_Dataset/
│       ├── videos/videos/          # *.mp4
│       ├── Labels/                 # error_elbows.json, error_knees.json
│       └── Splits/
└── BarbellRow/
    └── Labeled_Dataset/
        ├── barbellrow_images_raw/barbellrow_images_raw/   # *.jpg
        ├── Labels/                 # labels_lumbar_error.json, labels_torso_angle_error.json
        └── Splits/Splits_Lumbar_Error/   # train_ids.json, val_ids.json, test_ids.json
```

### 2.2 运行解析脚本

```bash
python training/prepare_dataset.py \
    --dataset_root dataset/Fitness-AQA_dataset_release \
    --output_dir   training/annotations
```

输出：

```
training/annotations/
├── train.json   # ~3400 条
├── val.json     # ~500 条
└── test.json    # ~500 条
```

每条样本结构：

```json
{
  "video_id":    "32903_8",
  "action_class": "深蹲",
  "video_path":  "Squat/Labeled_Dataset/videos/videos/32903_8.mp4",
  "image_paths": null,
  "error_annotations": [
    {
      "start_time":  3.2,
      "end_time":    5.8,
      "error_type":  "knees_forward",
      "error_cn":    "膝盖过度前伸",
      "correction":  "保持小腿尽量垂直地面，重心落在足跟"
    }
  ],
  "reference_response": null
}
```

> `reference_response` 为 null 时，`data_builder.py` 会根据 `error_annotations` 自动合成 GT 回答。

### 2.3 验证数据集

```bash
python training/data_builder.py \
    --ann_file     training/annotations/train.json \
    --dataset_root dataset/Fitness-AQA_dataset_release \
    --model_path   hf_cache/modelscope/Qwen--Qwen2-VL-7B-Instruct \
    --mode         sft
```

正常输出示例：

```
[FitnessAQADataset] 3412 条样本 (mode=sft, n_frames=16)
keys: ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'labels']
input_ids: torch.Size([1842])  masked: 1651/1842
```

---

## 3. 阶段一：SFT 微调

### 3.1 启动命令

**A100 80GB 单卡：**

```bash
deepspeed --num_gpus=1 training/train_sft.py \
    --model_name_or_path hf_cache/modelscope/Qwen--Qwen2-VL-7B-Instruct \
    --train_ann_file     training/annotations/train.json \
    --val_ann_file       training/annotations/val.json \
    --data_root          dataset/Fitness-AQA_dataset_release \
    --output_dir         checkpoints/sft \
    --deepspeed          training/ds_config_zero2.json \
    --num_train_epochs   3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate      2e-4 \
    --lora_r             32 \
    --lora_alpha         64 \
    --n_frames           16 \
    --max_seq_length     2048 \
    --frame_mode         uniform \
    --report_to          tensorboard
```

**4×RTX3060 20GB：**

```bash
deepspeed --num_gpus=4 training/train_sft.py \
    --model_name_or_path hf_cache/modelscope/Qwen--Qwen2-VL-7B-Instruct \
    --train_ann_file     training/annotations/train.json \
    --val_ann_file       training/annotations/val.json \
    --data_root          dataset/Fitness-AQA_dataset_release \
    --output_dir         checkpoints/sft \
    --deepspeed          training/ds_config_zero3_4gpu.json \
    --num_train_epochs   3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate      2e-4 \
    --lora_r             32 \
    --lora_alpha         64 \
    --n_frames           16 \
    --max_seq_length     2048
```

### 3.2 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora_r` | 32 | LoRA rank；r↑ 参数量↑ 拟合能力↑ |
| `--lora_alpha` | 64 | LoRA scaling = alpha/r；通常 alpha = 2×r |
| `--n_frames` | 16 | 每视频抽帧数；减小可降显存（最低 4） |
| `--frame_mode` | uniform | `uniform` 均匀抽帧；`timestamp` 在错误时刻附近密集采样 |
| `--gradient_accumulation_steps` | 8 | 等效 batch_size = per_device × n_gpu × accum |
| `--max_seq_length` | 2048 | 含图像 token，16 帧时约占 1200-1800 tokens |
| `--use_grid` | False | 将多帧拼为单张 Grid Image，减少 image token 数但损失细节 |

### 3.3 显存估算（SFT）

```
A100 80GB 单卡（ZeRO-2）:
  模型参数 (7B BF16)         : ~14 GB
  LoRA 梯度                  :  ~1 GB
  ZeRO-2 优化器状态          :  ~8 GB
  Gradient Checkpointing 激活:  ~6 GB
  图像特征 + KV-cache         : ~10 GB
  合计 ~39 GB / 80 GB  ← 充裕
```

### 3.4 监控

```bash
tensorboard --logdir checkpoints/sft/runs
```

关注：`train/loss` 应在前 200 steps 内从 ~3.0 降到 ~1.5，之后缓慢收敛。

---

## 4. 阶段二：GRPO 对齐（纯 RLAIF）

GRPO 的奖励信号**完全来自 LLM 裁判 API**，训练前必须先完成裁判服务部署。

### 4.1 部署裁判模型

裁判服务是任意实现了 `/v1/chat/completions` 接口的服务。推荐用 vLLM 本地部署：

```bash
pip install vllm

# 单卡部署 Qwen2.5-7B-Instruct（~16GB 显存，速度快，用于资源受限场景）
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 --gpu-memory-utilization 0.9

# 双卡部署 Qwen2.5-72B-Instruct（~80GB 显存，判断质量更高）
vllm serve Qwen/Qwen2.5-72B-Instruct \
    --port 8000 --tensor-parallel-size 2

# 验证服务正常
curl http://localhost:8000/v1/models
```

**其他兼容方案（任选一种）：**

| 方案 | 适用场景 | base_url |
|------|---------|----------|
| Ollama + qwen2.5:7b | 本机轻量部署 | `http://localhost:11434/v1` |
| LM Studio | Windows 本机 GUI 部署 | `http://localhost:1234/v1` |
| DeepSeek API | 无本地 GPU | `https://api.deepseek.com/v1` |
| OpenAI API | 精度最高 | `https://api.openai.com/v1` |

> 裁判模型质量越高，奖励信号越准确，但 API 延迟也越大。每个 training step 需要
> `G × per_device_batch_size` 次 API 调用，建议本地 vLLM 部署以减少等待。

### 4.2 裁判打分机制

裁判从四个维度打 0-10 分，归一化为 [0,1] 奖励：

| 维度 | 分值 | 评估内容 |
|------|------|---------|
| 动作识别 | 2 分 | 动作类别识别是否正确 |
| 错误识别 | 3 分 | GT 错误是否被准确找出，有无误报 |
| 修正建议 | 3 分 | 建议是否具体可执行，方向是否与 GT 一致 |
| 格式规范 | 2 分 | 是否包含全部四个必须节区 |

API 调用失败时自动重试（指数退避），最终回退到中性分 0.5（不影响梯度方向）。

### 4.3 启动命令

**A100 80GB 单卡：**

```bash
deepspeed --num_gpus=1 training/train_grpo.py \
    --model_name_or_path checkpoints/sft/final \
    --train_ann_file     training/annotations/train.json \
    --val_ann_file       training/annotations/val.json \
    --data_root          dataset/Fitness-AQA_dataset_release \
    --output_dir         checkpoints/grpo \
    --deepspeed          training/ds_config_zero2.json \
    --judge_base_url     http://localhost:8000/v1 \
    --judge_model        Qwen2.5-72B-Instruct \
    --judge_api_key      EMPTY \
    --num_train_epochs   2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate      5e-6 \
    --num_generations    8 \
    --max_new_tokens     256 \
    --kl_coef            0.04 \
    --temperature        0.9 \
    --lora_r             16 \
    --lora_alpha         32 \
    --report_to          tensorboard
```

**4×RTX3060 20GB（裁判部署在其他机器或同机另一进程）：**

```bash
deepspeed --num_gpus=4 training/train_grpo.py \
    --model_name_or_path checkpoints/sft/final \
    --train_ann_file     training/annotations/train.json \
    --val_ann_file       training/annotations/val.json \
    --data_root          dataset/Fitness-AQA_dataset_release \
    --output_dir         checkpoints/grpo \
    --deepspeed          training/ds_config_zero3_4gpu.json \
    --judge_base_url     http://JUDGE_SERVER_IP:8000/v1 \
    --judge_model        Qwen2.5-7B-Instruct \
    --judge_api_key      EMPTY \
    --num_train_epochs   2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate      5e-6 \
    --num_generations    4 \
    --max_new_tokens     128 \
    --kl_coef            0.04 \
    --temperature        0.9 \
    --lora_r             16 \
    --lora_alpha         32
```

### 4.4 GRPO 核心参数说明

| 参数 | A100 默认 | 4×3060 默认 | 说明 |
|------|-----------|-------------|------|
| `--num_generations` | 8 | 4 | 组内候选数 G；G↑ 方差估计更准，但显存和 API 调用数 ∝ G |
| `--max_new_tokens` | 256 | 128 | Rollout 生成长度上限；256 tokens ≈ 完整 4-section 回答 |
| `--kl_coef` | 0.04 | 0.04 | KL 惩罚系数；数据 <5k 时建议 0.02~0.08，防止遗忘 |
| `--temperature` | 0.9 | 0.9 | Rollout 采样温度；>1.0 多样性增加但质量下降 |
| `--learning_rate` | 5e-6 | 5e-6 | GRPO 推荐比 SFT 小 10~40× |
| `--judge_timeout` | 30.0 | 30.0 | 单次裁判 API 超时秒数 |
| `--judge_max_retries` | 2 | 2 | API 失败最大重试次数（指数退避） |

### 4.5 显存估算（GRPO）

```
A100 80GB 单卡（ZeRO-2）:
  Actor (7B BF16)              : ~14 GB
  Reference Policy (frozen 7B) : ~14 GB
  LoRA 梯度 + 优化器 (ZeRO-2)  :  ~8 GB
  Rollout buffer (G=8, 256tok) : ~10 GB
  合计 ~46 GB / 80 GB  ← 安全

4×RTX3060 20GB（ZeRO-3 + CPU offload）, per-GPU:
  Actor 分片 (7B/4 BF16)       :  ~3.5 GB
  Reference 分片 (7B/4 BF16)   :  ~3.5 GB
  Rollout buffer (G=4, 128tok) :  ~4   GB
  通信缓冲区                    :  ~2   GB
  合计 ~13-16 GB / 20 GB  ← 安全
```

---

## 5. 训练后使用微调模型

### 5.1 加载 LoRA Adapter

```python
import torch
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

base = Qwen2VLForConditionalGeneration.from_pretrained(
    "hf_cache/modelscope/Qwen--Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "checkpoints/grpo/final")
model = model.merge_and_unload()   # 合并 LoRA 权重，消除推理开销
processor = Qwen2VLProcessor.from_pretrained("checkpoints/grpo/final")
```

### 5.2 接入推理端

在 `.env` 中修改模型路径：

```env
VLM_MODEL_NAME=checkpoints/grpo/final
```

然后正常启动 `python main.py`。

---

## 6. 常见问题

**Q: GRPO 训练中 reward 全为 0.5**  
裁判 API 调用失败时会回退到中性分 0.5。排查步骤：
```bash
curl http://localhost:8000/v1/models         # 确认服务在线
curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"你好"}],"max_tokens":5}'
```

**Q: ZeRO-3 训练时报 `RuntimeError: Expected all tensors on same device`**  
确认 `device_map=None`（两个训练脚本均已设置），让 DeepSpeed 接管设备分配，不要与 `accelerate` 的 auto device_map 混用。

**Q: 4×RTX3060 上 GRPO 训练 OOM**  
按优先级依次尝试：
1. `--num_generations 2`（最有效）
2. `--max_new_tokens 64`
3. `--n_frames 8`
4. 在 `ds_config_zero3_4gpu.json` 中启用 `"offload_param": {"device": "cpu"}`

**Q: GRPO reward NaN**  
检查 `error_annotations` 字段是否为空列表（`[]`）而非 `null`，并确认视频 / 图片文件路径存在。

**Q: flash-attn 安装失败**  
改用 SDPA：将两个训练脚本中 `attn_implementation="flash_attention_2"` 改为 `"sdpa"`。

**Q: 训练速度很慢（4×RTX3060）**  
RTX3060 无 NVLink，all-reduce 走 PCIe（~30 GB/s vs NVLink 600 GB/s），比 A100 单卡慢 3-5 倍属正常。另外，裁判 API 延迟也会拖慢每个 step，建议本机或局域网部署裁判。

---

## 7. 文件清单

```
training/
├── prepare_dataset.py        # Fitness-AQA 原始格式 → 统一 JSON
├── data_builder.py           # Dataset / Collator / Tokenization
├── train_sft.py              # SFT 训练管线
├── train_grpo.py             # GRPO 对齐训练管线（纯 RLAIF）
├── ds_config_zero2.json      # DeepSpeed ZeRO-2（A100 单卡）
├── ds_config_zero3_4gpu.json # DeepSpeed ZeRO-3（4×RTX3060 20GB）
├── annotations/              # 生成的 JSON 标注（gitignore）
└── training_README.md        # 本文档
```
