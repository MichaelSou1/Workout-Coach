# Workout-Coach 训练管线文档

## 概览

本训练管线分两阶段对 **Qwen2-VL-7B-Instruct** 进行健身动作质量分析领域的微调：

| 阶段 | 脚本 | 算法 | 目标 |
|------|------|------|------|
| 1    | `train_sft.py` | SFT + LoRA | 让模型学会输出格式化的动作分析报告 |
| 2    | `train_grpo.py` | GRPO + LoRA | 用奖励信号对齐错误识别精度与时间精度 |

**硬件要求**：单卡 NVIDIA A100 80GB，BF16，无量化。

---

## 1. 环境安装

```bash
# 激活 conda 环境（与推理环境相同）
conda activate workout-coach

# 训练专用依赖
pip install trl>=0.13.0          # GRPOTrainer + SFTTrainer
pip install deepspeed>=0.14.0    # ZeRO-2 分片优化
pip install flash-attn>=2.5.0 --no-build-isolation  # FlashAttention-2（A100 必装）
pip install qwen-vl-utils        # process_vision_info 官方工具
pip install opencv-python pillow # 视频帧提取
```

> **注意**：`flash-attn` 编译需要与 PyTorch 版本匹配的 CUDA Toolkit。
> 若编译失败，可在模型加载时将 `attn_implementation="flash_attention_2"` 改为 `"sdpa"`。

---

## 2. 数据集准备

### 2.1 目录结构

```
data/Fitness-AQA/
├── annotations/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── videos/
    ├── squat/
    │   ├── squat_001.mp4
    │   └── ...
    ├── deadlift/
    └── bench_press/
```

### 2.2 标注 JSON 格式

`train.json` 为 JSON 数组，每条样本结构如下：

```json
[
  {
    "video_id":    "squat_001",
    "action_class": "深蹲",
    "video_path":  "videos/squat/squat_001.mp4",
    "duration":    30.5,
    "fps":         30,
    "error_annotations": [
      {
        "start_time":   5.2,
        "end_time":     7.8,
        "error_type":   "knee_valgus",
        "error_cn":     "膝关节外翻",
        "correction":   "保持膝盖与脚尖同向，下蹲时膝盖不超过脚尖"
      }
    ],
    "overall_quality": "poor",
    "reference_response": "【动作识别】深蹲\n【总体结论】..."
  }
]
```

**`reference_response` 字段说明**：
- SFT 阶段的 Ground Truth 回答，必须包含 `【动作识别】【总体结论】【关键问题N】【下一组口令】` 四个节区。
- 若留空，`data_builder.py` 会根据 `error_annotations` 自动合成。

### 2.3 验证数据集

```bash
python training/data_builder.py \
    --ann_file  data/Fitness-AQA/annotations/train.json \
    --data_root data/Fitness-AQA \
    --model_path hf_cache/modelscope/Qwen--Qwen2-VL-7B-Instruct \
    --mode sft
```

输出示例：
```
[FitnessAQADataset] 加载 1200 条标注 (mode=sft, frame_mode=uniform, n_frames=16)
── Sample keys: ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'labels']
  input_ids shape : torch.Size([1842])
  labels shape    : torch.Size([1842])
  masked tokens   : 1651 / 1842   ← ~90% 为 prompt，~10% 为 GT 回答
```

---

## 3. 阶段一：SFT 训练

### 3.1 启动命令（DeepSpeed ZeRO-2，单卡）

```bash
deepspeed --num_gpus=1 training/train_sft.py \
    --model_name_or_path hf_cache/modelscope/Qwen--Qwen2-VL-7B-Instruct \
    --train_ann_file     data/Fitness-AQA/annotations/train.json \
    --val_ann_file       data/Fitness-AQA/annotations/val.json \
    --data_root          data/Fitness-AQA \
    --output_dir         checkpoints/sft \
    --deepspeed          training/ds_config_zero2.json \
    --num_train_epochs   3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate      2e-4 \
    --n_frames           16 \
    --lora_r             32 \
    --lora_alpha         64 \
    --max_seq_length     2048 \
    --frame_mode         uniform \
    --report_to          tensorboard
```

### 3.2 关键超参说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora_r` | 32 | LoRA rank；r↑ → 参数量↑ → 拟合能力↑ |
| `--lora_alpha` | 64 | LoRA scaling = alpha/r = 2；通常 alpha = 2×r |
| `--n_frames` | 16 | 每视频抽帧数；A100 80GB 上 16 帧 + seq_len=2048 安全 |
| `--gradient_accumulation_steps` | 8 | 等效 batch size = 1×8 = 8 |
| `--max_seq_length` | 2048 | 含图像 token；Qwen2-VL-7B 图像 token 数 ≈ 1~4K |

### 3.3 预期显存占用（SFT）

```
模型参数 (7B BF16)      : ~14 GB
LoRA 梯度               :  ~2 GB
ZeRO-2 优化器状态       :  ~8 GB  (ZeRO-2 分片后约减半)
Activation checkpointing :  ~6 GB
KV-cache + 图像特征      : ~10 GB
总计约 40 GB / 80 GB    ← 充裕
```

### 3.4 监控训练

```bash
tensorboard --logdir checkpoints/sft/runs
```

---

## 4. 阶段二：GRPO 对齐训练

### 4.1 启动命令（从 SFT checkpoint 继续）

```bash
deepspeed --num_gpus=1 training/train_grpo.py \
    --model_name_or_path checkpoints/sft/final \
    --train_ann_file     data/Fitness-AQA/annotations/train.json \
    --val_ann_file       data/Fitness-AQA/annotations/val.json \
    --data_root          data/Fitness-AQA \
    --output_dir         checkpoints/grpo \
    --deepspeed          training/ds_config_zero2.json \
    --num_train_epochs   2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate      5e-6 \
    --num_generations    8 \
    --max_new_tokens     256 \
    --kl_coef            0.04 \
    --temperature        0.9 \
    --top_p              0.95 \
    --lora_r             16 \
    --lora_alpha         32 \
    --report_to          tensorboard
```

### 4.2 GRPO 核心超参说明

| 参数 | 默认值 | 调参建议 |
|------|--------|---------|
| `--num_generations` | 8 | 单卡最大安全值；G↑显存↑，建议 G=4/8/16 |
| `--max_new_tokens` | 256 | 控制 rollout 长度；>512 时需减小 G |
| `--kl_coef` | 0.04 | KL 惩罚强度；数据<5k 时建议 0.02~0.06 |
| `--temperature` | 0.9 | rollout 采样温度；>1.0 多样性增加但质量下降 |

### 4.3 奖励函数权重调整

默认权重分配：

```
format_reward     (α=0.20)  — 输出格式合规性
error_id_reward   (β=0.45)  — 错误类型识别准确率（最重要）
temporal_reward   (γ=0.20)  — 时间戳定位精度
correction_reward (δ=0.15)  — 修正建议质量
```

若数据集中时间标注较稀疏，可调整：
```bash
--reward_weight_error_id   0.55 \
--reward_weight_temporal   0.10
```

### 4.4 预期显存占用（GRPO）

```
Actor 模型 (7B BF16)        : ~14 GB
Reference Policy (frozen)   : ~14 GB  ← GRPOTrainer 自动管理
LoRA 梯度                   :  ~2 GB
ZeRO-2 优化器               :  ~8 GB
Rollout buffer (G=8, 256tok) : ~10 GB
总计约 48 GB / 80 GB        ← 安全余量 32 GB
```

---

## 5. 模型推理（训练完成后）

### 5.1 加载 SFT/GRPO LoRA Adapter

```python
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "hf_cache/modelscope/Qwen--Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "checkpoints/grpo/final")
model = model.merge_and_unload()  # 合并 LoRA 权重，消除推理开销
```

### 5.2 更新 main.py 中的模型路径

在 `.env` 文件中修改：
```env
VLM_MODEL_NAME=checkpoints/grpo/final
```

---

## 6. 常见问题

**Q: `flash_attn` 安装失败怎么办？**  
A: 在 `train_sft.py` 和 `train_grpo.py` 中将 `attn_implementation="flash_attention_2"` 改为 `"sdpa"`，性能略降但功能正常。

**Q: GRPO 训练中出现 reward NaN？**  
A: 检查 `error_annotations` 字段是否为空列表（非 null），并确认视频文件路径正确。

**Q: OOM（Out of Memory）如何解决？**  
A: 优先尝试：1) 减小 `--n_frames` 到 8；2) 减小 `--num_generations` 到 4；3) 减小 `--max_new_tokens` 到 128。

**Q: DeepSpeed 与 PEFT 报 `RuntimeError: Expected all tensors to be on the same device`？**  
A: 确认 `device_map=None`（不使用 accelerate 自动分配），让 DeepSpeed 接管设备管理。

---

## 7. 文件清单

```
training/
├── data_builder.py          # 数据集解析 + 多模态样本构建
├── train_sft.py             # SFT 训练管线
├── train_grpo.py            # GRPO 对齐训练管线
├── ds_config_zero2.json     # DeepSpeed ZeRO-2 配置
└── training_README.md       # 本文档
```
