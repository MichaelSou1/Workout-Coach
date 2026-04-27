"""
GRPO (Group Relative Policy Optimization) 训练管线
===================================================
基于 trl.GRPOTrainer 对 SFT 后的 Qwen2-VL-7B 做偏好对齐。

GRPO 与 PPO 的核心区别：
  - GRPO 无需 Value Model（Critic），将同一 prompt 的 G 个生成结果在组内做
    相对归一化，极大降低显存占用与超参敏感性。
  - 奖励信号直接来自外部 Reward Function，无需奖励模型网络。

显存估算（单卡 A100 80GB，BF16 无量化）：
  Actor (7B BF16)         : ~14 GB
  Reference (frozen 7B)   : ~14 GB  ← GRPOTrainer 内置，用于 KL 约束
  LoRA 梯度 + 优化器(ZeRO2): ~20 GB
  Rollout buffer (8 gen)  : ~10 GB（max_new_tokens=256 时）
  总计约 58 GB，安全余量 22 GB。

训练目标：
  最大化期望奖励 E[r(y)] 同时用 KL 惩罚防止偏离 Reference Policy 过远。
  r(y) = α·format_reward + β·error_id_reward + γ·temporal_reward + δ·correction_reward

启动命令：
  deepspeed --num_gpus=1 training/train_grpo.py
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from trl import GRPOConfig, GRPOTrainer

from data_builder import FitnessAQADataset

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 与 train_sft.py 一致的 LoRA 目标模块
# ──────────────────────────────────────────────────────────────────────────────
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "visual.merger.mlp.0",
    "visual.merger.mlp.2",
]

# ──────────────────────────────────────────────────────────────────────────────
# 参数配置
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GRPOArguments:
    """GRPO 训练参数。"""

    # 从 SFT checkpoint 继续训练（推荐），也可从 base 模型从头 GRPO
    model_name_or_path: str = field(
        default="checkpoints/sft/final",
        metadata={"help": "SFT checkpoint 路径或基础模型名"},
    )
    train_ann_file: str  = field(default="data/Fitness-AQA/annotations/train.json")
    val_ann_file: Optional[str] = field(default="data/Fitness-AQA/annotations/val.json")
    data_root: str       = field(default="data/Fitness-AQA")
    frame_mode: str      = field(default="uniform")
    n_frames: int        = field(default=16)
    use_grid: bool       = field(default=False)

    # LoRA（GRPO 阶段通常用更小的 r 以稳定训练）
    lora_r: int         = field(default=16)
    lora_alpha: int     = field(default=32)
    lora_dropout: float = field(default=0.0)   # GRPO 阶段去掉 dropout 更稳定

    # GRPO 核心超参
    num_generations: int = field(
        default=8,
        metadata={
            "help": (
                "每个 prompt 采样的候选回答数 G。G=8 在 80GB 卡上安全。"
                "G 越大，组内方差估计越准确，但显存占用线性增长。"
            )
        },
    )
    max_new_tokens: int = field(
        default=256,
        metadata={
            "help": (
                "Rollout 阶段每个生成的最大 token 数。"
                "256 tokens 足够表达完整的 4-section 健身分析，"
                "同时避免单卡 OOM。"
            )
        },
    )
    temperature: float = field(default=0.9,  metadata={"help": "采样温度，控制生成多样性"})
    top_p: float       = field(default=0.95, metadata={"help": "Nucleus sampling top-p"})
    kl_coef: float     = field(
        default=0.04,
        metadata={"help": "KL 惩罚系数 β；越大越保守，越小越激进"},
    )

    # 奖励权重（α + β + γ + δ = 1）
    reward_weight_format:     float = field(default=0.20)
    reward_weight_error_id:   float = field(default=0.45)
    reward_weight_temporal:   float = field(default=0.20)
    reward_weight_correction: float = field(default=0.15)

    # 训练超参
    output_dir: str      = field(default="checkpoints/grpo")
    num_train_epochs: int = field(default=2)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=5e-6,  metadata={"help": "GRPO 推荐更小 LR"})
    weight_decay:  float = field(default=0.01)
    warmup_ratio:  float = field(default=0.05)
    logging_steps: int   = field(default=5)
    save_steps:    int   = field(default=100)
    eval_steps:    int   = field(default=100)
    save_total_limit: int = field(default=2)
    deepspeed: Optional[str] = field(default="training/ds_config_zero2.json")
    seed: int = field(default=42)
    report_to: str = field(default="tensorboard")


# ──────────────────────────────────────────────────────────────────────────────
# Reward 函数：核心设计
# ──────────────────────────────────────────────────────────────────────────────

# 必须存在的输出节区（格式合规检测）
REQUIRED_SECTIONS = ["【动作识别】", "【总体结论】", "【关键问题", "【下一组口令】"]

# 错误类型关键词映射（英文 error_type → 中文匹配词列表）
ERROR_TYPE_KEYWORDS: dict[str, list[str]] = {
    "knee_valgus":          ["膝关节外翻", "膝外翻", "膝盖内扣"],
    "forward_lean":         ["前倾", "躯干前倾", "重心前移"],
    "butt_wink":            ["骨盆后倾", "屁股眨眼", "腰椎弯曲"],
    "hip_shift":            ["髋部偏移", "重心偏移", "骨盆侧移"],
    "rounded_back":         ["圆背", "驼背", "脊柱弯曲"],
    "bar_path_deviation":   ["杠铃路径", "杠铃偏移", "轨迹偏离"],
    "lockout_incomplete":   ["未锁定", "锁定不完全", "髋部未伸展"],
    "depth_insufficient":   ["深度不足", "蹲得不够深", "平行以上"],
}


class FitnessRewardCalculator:
    """
    健身动作质量奖励计算器。

    将 Fitness-AQA GT 标注转化为可微分的标量奖励信号。

    奖励由四个子分数组成：

    1. format_reward (α=0.20)
       ── 检查输出是否包含所有必须节区
       ── 全有 → 1.0；每缺一个扣 0.25

    2. error_id_reward (β=0.45)
       ── GT 中标注了哪些错误类型，模型是否正确识别
       ── 基于关键词匹配，F1-style scoring

    3. temporal_reward (γ=0.20)
       ── 若模型输出中提到了时间戳（如 "5.2s" 或 "第5秒"），
          判断其与 GT 时间戳的接近程度
       ── 误差 ≤ 1s → 1.0；误差 ≤ 3s → 0.5；> 3s → 0
       ── 若模型未提及时间，该子分数为中性值 0.5（不奖不罚）

    4. correction_reward (δ=0.15)
       ── 模型给出的修正建议是否与 GT 关键词一致
       ── 简单词重叠 (Jaccard) 评分
    """

    def __init__(
        self,
        weight_format:     float = 0.20,
        weight_error_id:   float = 0.45,
        weight_temporal:   float = 0.20,
        weight_correction: float = 0.15,
    ) -> None:
        self.w_fmt  = weight_format
        self.w_err  = weight_error_id
        self.w_tmp  = weight_temporal
        self.w_cor  = weight_correction

    # ── 子奖励 1：格式合规 ──────────────────────────────────────────────────

    def format_reward(self, generated: str) -> float:
        score = 1.0
        for sec in REQUIRED_SECTIONS:
            if sec not in generated:
                score -= 0.25
        return max(0.0, score)

    # ── 子奖励 2：错误类型识别 ──────────────────────────────────────────────

    def error_id_reward(
        self,
        generated: str,
        gt_error_annotations: list[dict],
    ) -> float:
        """
        计算模型识别出的错误类型与 GT 的 F1 得分。

        Args:
            generated          : 模型生成的文本
            gt_error_annotations: GT 错误标注列表
        """
        if not gt_error_annotations:
            return 0.8  # 无错误动作：模型不提错误则给较高分

        gt_types = {e["error_type"] for e in gt_error_annotations}
        tp = 0
        for et in gt_types:
            keywords = ERROR_TYPE_KEYWORDS.get(et, [et])
            if any(kw in generated for kw in keywords):
                tp += 1

        # Precision：模型提到的错误类型中有多少是正确的
        # 这里简化为 Recall（GT 中有多少被模型找到）
        recall = tp / len(gt_types) if gt_types else 0.0

        # 额外检查：若模型输出了本不应有的"严重错误"描述（false positive 惩罚）
        # 当 GT 为空（完美动作）时，模型应输出"无明显问题"而非虚假错误
        return recall

    # ── 子奖励 3：时间精度（Temporal Accuracy）──────────────────────────────

    def temporal_reward(
        self,
        generated: str,
        gt_error_timestamps: list[float],
    ) -> float:
        """
        提取生成文本中的时间表达，与 GT 时间戳比对。

        支持格式：
          - "5.2s" / "5.2秒"
          - "第5秒" / "第5.2s"
          - "(5.2s)" / "[5.2s]"
        """
        if not gt_error_timestamps:
            return 0.8  # 无时间标注，中性高分

        # 正则提取数字时间戳
        time_pattern = re.compile(
            r"(?:第\s*)?(\d+(?:\.\d+)?)\s*[sS秒]", re.UNICODE
        )
        matches = time_pattern.findall(generated)

        if not matches:
            return 0.5  # 模型未提及时间，中性分（既不奖励也不惩罚）

        predicted_ts = [float(m) for m in matches]

        total_score = 0.0
        for gt_t in gt_error_timestamps:
            # 找最近的预测时间戳
            min_diff = min(abs(p - gt_t) for p in predicted_ts)
            if min_diff <= 1.0:
                total_score += 1.0
            elif min_diff <= 3.0:
                # 线性衰减：1s ~ 3s 区间从 1.0 降到 0.0
                total_score += 1.0 - (min_diff - 1.0) / 2.0
            else:
                total_score += 0.0  # 误差 > 3s，完全没有时效性

        return total_score / len(gt_error_timestamps)

    # ── 子奖励 4：修正建议质量 ──────────────────────────────────────────────

    def correction_reward(
        self,
        generated: str,
        gt_error_annotations: list[dict],
    ) -> float:
        """
        使用词重叠 (Jaccard similarity) 评估修正建议质量。
        GT 修正文本 vs 生成文本中【修正】后的部分。
        """
        if not gt_error_annotations:
            return 0.8

        gt_corrections = " ".join(
            e.get("correction", "") for e in gt_error_annotations
        )
        if not gt_corrections.strip():
            return 0.7

        # 提取生成文本中"修正："之后的内容
        correction_pattern = re.compile(r"修正[：:](.*?)(?:【|$)", re.DOTALL)
        pred_corrections = " ".join(correction_pattern.findall(generated))

        if not pred_corrections.strip():
            return 0.3

        # Jaccard similarity（字符级 n-gram 而非词级，更适合中文）
        def char_ngrams(text: str, n: int = 2) -> set[str]:
            return {text[i:i+n] for i in range(len(text) - n + 1)}

        gt_ngrams   = char_ngrams(gt_corrections)
        pred_ngrams = char_ngrams(pred_corrections)

        if not gt_ngrams:
            return 0.5

        intersection = gt_ngrams & pred_ngrams
        union        = gt_ngrams | pred_ngrams
        return len(intersection) / len(union) if union else 0.0

    # ── 综合奖励 ────────────────────────────────────────────────────────────

    def __call__(
        self,
        generated: str,
        gt_error_annotations: list[dict],
    ) -> float:
        """
        计算最终奖励分数 ∈ [0, 1]。

        Args:
            generated            : 模型生成的完整文本
            gt_error_annotations : GT 错误标注列表

        Returns:
            float: 综合奖励分数
        """
        gt_timestamps = [e["start_time"] for e in gt_error_annotations]

        r_fmt = self.format_reward(generated)
        r_err = self.error_id_reward(generated, gt_error_annotations)
        r_tmp = self.temporal_reward(generated, gt_timestamps)
        r_cor = self.correction_reward(generated, gt_error_annotations)

        combined = (
            self.w_fmt * r_fmt +
            self.w_err * r_err +
            self.w_tmp * r_tmp +
            self.w_cor * r_cor
        )

        return float(combined)


# ──────────────────────────────────────────────────────────────────────────────
# TRL GRPOTrainer 兼容的 Reward Function 包装
# ──────────────────────────────────────────────────────────────────────────────

def make_reward_fn(calculator: FitnessRewardCalculator):
    """
    生成符合 GRPOTrainer 签名的奖励函数。

    GRPOTrainer 要求 reward_func 签名为：
        fn(completions: list[str], **kwargs) -> list[float]

    其中 **kwargs 包含数据集的所有额外字段（在 dataset.__getitem__ 中返回的键），
    例如 "error_annotations", "gt_response" 等。
    """

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        """
        Args:
            completions          : 当前 batch 所有生成的文本列表，
                                   长度 = batch_size × num_generations
            **kwargs             : 数据集字段（重复 num_generations 次以对齐）
        """
        rewards: list[float] = []

        # kwargs 中的每个字段长度 = batch_size × num_generations
        error_annotations_list = kwargs.get("error_annotations", [[] for _ in completions])

        for generated, gt_anns in zip(completions, error_annotations_list):
            reward = calculator(generated, gt_anns)
            rewards.append(reward)
            logger.debug(
                f"[Reward] 生成前50字: {generated[:50]!r} → reward={reward:.4f}"
            )

        return rewards

    return reward_fn


# ──────────────────────────────────────────────────────────────────────────────
# 模型加载
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_processor(args: GRPOArguments):
    """加载模型并注入 LoRA（参见 train_sft.py 的详细注释）。"""
    logger.info(f"[GRPO] 加载模型: {args.model_name_or_path}")

    processor = Qwen2VLProcessor.from_pretrained(
        args.model_name_or_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        attn_implementation="flash_attention_2",
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


# ──────────────────────────────────────────────────────────────────────────────
# GRPO Dataset Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class GRPOFitnessDataset(Dataset):
    """
    将 FitnessAQADataset(mode="grpo") 包装为 GRPOTrainer 期望的格式。

    GRPOTrainer 期望数据集的每条样本包含：
      - "prompt"     : 完整的 prompt 文本（已应用 chat template）
      - 额外字段     : 奖励函数需要的 GT 标签

    对于多模态输入，image 信息通过 processor 在 collate 时处理。
    """

    def __init__(self, base_dataset: FitnessAQADataset) -> None:
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        return {
            "prompt":            item["prompt"],
            "error_annotations": item["error_annotations"],
            "gt_response":       item["gt_response"],
            "action_class":      item["action_class"],
            # GRPOTrainer 通过 processing_class 重新处理图像，
            # 这里保留 images 字段以便 collator 使用
            "images":            item["images"],
        }


# ──────────────────────────────────────────────────────────────────────────────
# 主训练函数
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = transformers.HfArgumentParser(GRPOArguments)
    (grpo_args,) = parser.parse_args_into_dataclasses()

    logger.info("[GRPO] ========== 开始 GRPO 训练 ==========")

    # ── 加载模型 ──
    model, processor = load_model_and_processor(grpo_args)

    # ── 构建奖励函数 ──
    calculator = FitnessRewardCalculator(
        weight_format=grpo_args.reward_weight_format,
        weight_error_id=grpo_args.reward_weight_error_id,
        weight_temporal=grpo_args.reward_weight_temporal,
        weight_correction=grpo_args.reward_weight_correction,
    )
    reward_fn = make_reward_fn(calculator)

    # ── 构建数据集 ──
    base_train = FitnessAQADataset(
        ann_file=grpo_args.train_ann_file,
        data_root=grpo_args.data_root,
        processor=processor,
        mode="grpo",
        frame_mode=grpo_args.frame_mode,
        n_frames=grpo_args.n_frames,
        use_grid=grpo_args.use_grid,
    )
    train_dataset = GRPOFitnessDataset(base_train)

    eval_dataset = None
    if grpo_args.val_ann_file and os.path.exists(grpo_args.val_ann_file):
        base_eval = FitnessAQADataset(
            ann_file=grpo_args.val_ann_file,
            data_root=grpo_args.data_root,
            processor=processor,
            mode="grpo",
            frame_mode=grpo_args.frame_mode,
            n_frames=grpo_args.n_frames,
            use_grid=grpo_args.use_grid,
        )
        eval_dataset = GRPOFitnessDataset(base_eval)

    # ── GRPOConfig ──
    grpo_config = GRPOConfig(
        output_dir=grpo_args.output_dir,
        num_train_epochs=grpo_args.num_train_epochs,
        per_device_train_batch_size=grpo_args.per_device_train_batch_size,
        gradient_accumulation_steps=grpo_args.gradient_accumulation_steps,
        learning_rate=grpo_args.learning_rate,
        weight_decay=grpo_args.weight_decay,
        warmup_ratio=grpo_args.warmup_ratio,
        logging_steps=grpo_args.logging_steps,
        save_steps=grpo_args.save_steps,
        eval_steps=grpo_args.eval_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_total_limit=grpo_args.save_total_limit,
        deepspeed=grpo_args.deepspeed,
        bf16=True,
        fp16=False,
        seed=grpo_args.seed,
        report_to=grpo_args.report_to,
        remove_unused_columns=False,

        # ── GRPO 专属参数 ──────────────────────────────────────────────────
        # num_generations: 每个 prompt 的组内候选数 G
        # 关键权衡：G↑ → 方差估计更准 → 奖励信号更稳定，但显存 ∝ G
        # A100 80GB 单卡：G=8, max_new_tokens=256 安全上限
        num_generations=grpo_args.num_generations,

        # max_new_tokens: Rollout 阶段生成长度上限
        # 256 tokens ≈ 健身分析 4-section 完整回答的平均长度
        # 设置过大（>512）会导致 KV-cache 爆显存
        max_new_tokens=grpo_args.max_new_tokens,

        temperature=grpo_args.temperature,
        top_p=grpo_args.top_p,

        # kl_coef: KL 散度惩罚系数 β，防止 actor 偏离 reference policy 过远
        # 健身数据较少（<10k 样本）时建议 0.02~0.08，避免 catastrophic forgetting
        kl_coef=grpo_args.kl_coef,

        # use_vllm: 若安装了 vllm 可设为 True 加速 Rollout 生成
        # 注意：vllm 需要额外显存加载第二份权重，80GB 单卡谨慎开启
        # use_vllm=False,
    )

    # ── GRPOTrainer ──
    # GRPOTrainer 内置 Reference Policy（frozen actor copy），
    # 负责在每个 training step 计算 KL(actor || ref)。
    # reward_funcs 接受一个列表，可以组合多个奖励函数（这里合并为单一函数）。
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=[reward_fn],
        processing_class=processor,
    )

    logger.info("[GRPO] 开始训练...")
    logger.info(
        f"[GRPO] num_generations={grpo_args.num_generations}, "
        f"max_new_tokens={grpo_args.max_new_tokens}, "
        f"kl_coef={grpo_args.kl_coef}"
    )

    trainer.train()

    logger.info(f"[GRPO] 保存模型到 {grpo_args.output_dir}/final")
    trainer.save_model(os.path.join(grpo_args.output_dir, "final"))
    processor.save_pretrained(os.path.join(grpo_args.output_dir, "final"))

    logger.info("[GRPO] ========== 训练完成 ==========")


if __name__ == "__main__":
    main()
