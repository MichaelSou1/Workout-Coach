"""
Supervised Fine-Tuning (SFT) 管线 — Qwen2-VL-7B-Instruct + LoRA + DeepSpeed Zero-2
======================================================================================

硬件目标：单卡 A100 80GB，BF16 全精度，无量化。

训练策略：
  - LoRA (r=32, alpha=64) 注入到语言模型 Attention 层 + 视觉-语言对齐层
  - Gradient Checkpointing 大幅降低激活显存（约 60% 节省）
  - DeepSpeed Zero-2 分片优化器状态与梯度（适合单机多卡或单卡大 batch）
  - SFTTrainer from TRL，原生支持 completion-only loss 掩码

启动命令（见 training_README.md）：
  deepspeed --num_gpus=1 training/train_sft.py --config training/sft_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from datasets import Dataset as HFDataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

from data_builder import FitnessAQACollator, FitnessAQADataset

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Qwen2-VL 架构中需要注入 LoRA 的关键模块
# ──────────────────────────────────────────────────────────────────────────────

# 语言解码器注意力层
LM_ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# MLP 投影层（可选，提升 LoRA 表达力，但会增加参数量约 2×）
LM_MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]

# 视觉-语言对齐层（merger/projection）
# Qwen2-VL 中视觉特征通过 visual.merger 进入语言模型，这是关键跨模态通路
VL_ALIGNMENT_MODULES = [
    "visual.merger.mlp.0",  # 视觉合并 MLP 第 0 层
    "visual.merger.mlp.2",  # 视觉合并 MLP 第 2 层
]

# 合并所有目标模块（SFT 推荐：LM 注意力 + VL 对齐；
# 若显存充足可加入 LM_MLP_MODULES）
TARGET_MODULES = LM_ATTN_MODULES + VL_ALIGNMENT_MODULES


# ──────────────────────────────────────────────────────────────────────────────
# 参数配置
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SFTArguments:
    """命令行参数与默认值。"""

    # 模型
    model_name_or_path: str = field(
        default="Qwen/Qwen2-VL-7B-Instruct",
        metadata={"help": "预训练模型路径或 HuggingFace 模型名"},
    )
    # 数据
    train_ann_file: str = field(
        default="data/Fitness-AQA/annotations/train.json",
        metadata={"help": "训练集标注 JSON"},
    )
    val_ann_file: Optional[str] = field(
        default="data/Fitness-AQA/annotations/val.json",
        metadata={"help": "验证集标注 JSON（可选）"},
    )
    data_root: str = field(
        default="data/Fitness-AQA",
        metadata={"help": "数据集根目录"},
    )
    frame_mode: str = field(
        default="uniform",
        metadata={"help": "帧提取模式: uniform | timestamp"},
    )
    n_frames: int = field(default=16, metadata={"help": "每个视频提取的帧数"})
    use_grid: bool = field(default=False, metadata={"help": "是否合并为 Grid Image"})
    max_seq_length: int = field(default=2048, metadata={"help": "最大 token 序列长度"})

    # LoRA
    lora_r: int          = field(default=32,   metadata={"help": "LoRA rank"})
    lora_alpha: int      = field(default=64,   metadata={"help": "LoRA scaling alpha"})
    lora_dropout: float  = field(default=0.05, metadata={"help": "LoRA dropout"})
    use_lora: bool       = field(default=True,  metadata={"help": "是否使用 LoRA"})

    # 训练超参
    output_dir: str        = field(default="checkpoints/sft")
    num_train_epochs: int  = field(default=3)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size:  int = field(default=1)
    gradient_accumulation_steps: int = field(default=8,  metadata={"help": "等效 batch = 1×8 = 8"})
    learning_rate: float = field(default=2e-4)
    weight_decay:  float = field(default=0.01)
    warmup_ratio:  float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps:  int = field(default=10)
    save_steps:     int = field(default=200)
    eval_steps:     int = field(default=200)
    save_total_limit: int = field(default=3)
    dataloader_num_workers: int = field(default=4)

    # DeepSpeed
    deepspeed: Optional[str] = field(
        default="training/ds_config_zero2.json",
        metadata={"help": "DeepSpeed 配置文件路径，None 则不使用 DeepSpeed"},
    )

    # 其他
    seed: int = field(default=42)
    report_to: str = field(default="tensorboard")


# ──────────────────────────────────────────────────────────────────────────────
# 模型加载 & LoRA 注入
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_processor(args: SFTArguments):
    """
    加载 Qwen2-VL 模型与 Processor，注入 LoRA。

    关键细节：
    - `torch_dtype=torch.bfloat16` 让模型权重直接以 BF16 存储
    - `device_map=None` 配合 DeepSpeed，不使用 accelerate 的自动 device_map
    - Gradient Checkpointing 在 model.enable_input_require_grads() 后启用，
      这是 PEFT + gradient checkpointing 共用时的必要步骤
    """
    logger.info(f"[SFT] 加载模型: {args.model_name_or_path}")

    processor = Qwen2VLProcessor.from_pretrained(
        args.model_name_or_path,
        # min/max_pixels 控制图片被 resize 到的像素范围，
        # 256*28*28 ~ 1280*28*28 是官方推荐的合理区间
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=None,       # DeepSpeed 接管设备分配
        attn_implementation="flash_attention_2",  # A100 支持 FA2，显著加速长序列
    )

    # ── Gradient Checkpointing ──
    # 以重计算激活值换显存，训练速度约降低 20%，但显存节省 ~60%
    model.gradient_checkpointing_enable()
    # PEFT 需要此调用以确保输入 requires_grad=True，否则 LoRA 梯度传不到 base model
    model.enable_input_require_grads()

    if args.use_lora:
        logger.info(
            f"[SFT] 注入 LoRA: r={args.lora_r}, alpha={args.lora_alpha}, "
            f"target_modules={TARGET_MODULES}"
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=TARGET_MODULES,
            # bias="none" 不训练 bias，减少参数量
            bias="none",
            # modules_to_save: 除 LoRA 之外，这些层的全部参数也会被训练
            # embed_tokens & lm_head 保持与 LoRA 层一致的词表对齐
            modules_to_save=["embed_tokens", "lm_head"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, processor


# ──────────────────────────────────────────────────────────────────────────────
# 构建 TrainingArguments
# ──────────────────────────────────────────────────────────────────────────────

def build_training_args(args: SFTArguments) -> SFTConfig:
    """
    将 SFTArguments 映射到 TRL SFTConfig（继承自 TrainingArguments）。

    SFTConfig 在 TrainingArguments 基础上增加了 max_seq_length 等 SFT 专属参数。
    """
    return SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps" if args.val_ann_file else "no",
        save_total_limit=args.save_total_limit,
        deepspeed=args.deepspeed,
        bf16=True,                   # A100 原生支持 BF16，比 FP16 数值更稳定
        fp16=False,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        report_to=args.report_to,
        # SFT 专属：数据集中 input_ids/labels 已由 data_builder 处理，
        # 关闭 SFTTrainer 的自动 tokenization 流程
        dataset_text_field=None,
        max_seq_length=args.max_seq_length,
        # 关闭 SFTTrainer 的 packing，因为多模态 padding 需要自定义处理
        packing=False,
        remove_unused_columns=False,  # 保留 pixel_values 等自定义字段
    )


# ──────────────────────────────────────────────────────────────────────────────
# 主训练函数
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # ── 解析参数 ──
    parser = transformers.HfArgumentParser(SFTArguments)
    (sft_args,) = parser.parse_args_into_dataclasses()

    logger.info("[SFT] ========== 开始 SFT 训练 ==========")
    logger.info(f"[SFT] 配置: {sft_args}")

    # ── 加载模型 ──
    model, processor = load_model_and_processor(sft_args)

    # ── 构建数据集 ──
    logger.info("[SFT] 构建训练数据集...")
    train_dataset = FitnessAQADataset(
        ann_file=sft_args.train_ann_file,
        data_root=sft_args.data_root,
        processor=processor,
        mode="sft",
        frame_mode=sft_args.frame_mode,
        n_frames=sft_args.n_frames,
        use_grid=sft_args.use_grid,
        max_length=sft_args.max_seq_length,
    )

    eval_dataset = None
    if sft_args.val_ann_file and os.path.exists(sft_args.val_ann_file):
        logger.info("[SFT] 构建验证数据集...")
        eval_dataset = FitnessAQADataset(
            ann_file=sft_args.val_ann_file,
            data_root=sft_args.data_root,
            processor=processor,
            mode="sft",
            frame_mode=sft_args.frame_mode,
            n_frames=sft_args.n_frames,
            use_grid=sft_args.use_grid,
            max_length=sft_args.max_seq_length,
        )

    # ── 数据 Collator ──
    data_collator = FitnessAQACollator(processor=processor)

    # ── TrainingArguments ──
    training_args = build_training_args(sft_args)

    # ── SFTTrainer ──
    # SFTTrainer 是 Trainer 的子类，额外提供：
    #   - completion-only loss（已由 data_builder 的 labels 掩码实现）
    #   - 自动 packing（已关闭）
    #   - 与 PEFT LoRA 的无缝集成
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,  # TRL >= 0.13 使用 processing_class 替代 tokenizer
    )

    logger.info("[SFT] 开始训练...")
    trainer.train()

    # ── 保存最终权重 ──
    logger.info(f"[SFT] 保存模型到 {sft_args.output_dir}/final")
    trainer.save_model(os.path.join(sft_args.output_dir, "final"))
    processor.save_pretrained(os.path.join(sft_args.output_dir, "final"))

    logger.info("[SFT] ========== 训练完成 ==========")


if __name__ == "__main__":
    main()
