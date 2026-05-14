"""
GRPO (Group Relative Policy Optimization) 训练管线 —— 纯 RLAIF 奖励
=====================================================================
基于 trl.GRPOTrainer 对 SFT 后的 Qwen2-VL-7B 做偏好对齐。
奖励信号完全来自 LLM-as-Judge（OpenAI-compatible API），无规则奖励。

GRPO 与 PPO 的核心区别：
  - 无需 Value Model（Critic），同一 prompt 的 G 个生成结果在组内相对归一化。
  - 奖励信号来自外部 Reward Function（这里是 LLM 裁判 API）。

显存估算：

  A100 80GB 单卡（ZeRO-2）：
    Actor + Reference (7B BF16 × 2) : ~28 GB
    LoRA 梯度 + 优化器               : ~2 GB
    Rollout buffer (G=8, 256 tokens) : ~10 GB
    合计 ~40 GB，安全余量 40 GB。
    启动：deepspeed --num_gpus=1 training/train_grpo.py

  4× RTX 3060 20GB（ZeRO-3 + CPU optimizer offload，无 NVLink）：
    per-GPU 参数分片 (Actor+Ref)     : ~7 GB
    Rollout buffer (G=4, 128 tokens) : ~4 GB
    通信缓冲区                        : ~2 GB
    合计 per-GPU ~13-16 GB，20 GB 卡安全。
    启动：deepspeed --num_gpus=4 training/train_grpo.py \\
            --deepspeed training/ds_config_zero3_4gpu.json \\
            --num_generations 4 --max_new_tokens 128

训练目标：
  最大化 LLM 裁判给出的期望奖励 E[r(y)]，同时用 KL 惩罚防止偏离 Reference Policy。

注意事项：
  - 每个 training step 的奖励计算需要 G × batch_size 次 API 调用。
  - API 延迟是训练速度的主要瓶颈，建议本地部署裁判模型（vLLM）。
  - API 失败时回退到中性分 0.5，不影响梯度方向但会降低信号质量。
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from openai import OpenAI
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from trl import GRPOConfig, GRPOTrainer

from data_builder import FitnessAQADataset

logger = logging.getLogger(__name__)

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

    model_name_or_path: str = field(
        default="checkpoints/sft/final",
        metadata={"help": "SFT checkpoint 路径或基础模型名"},
    )
    train_ann_file: str      = field(default="data/Fitness-AQA/annotations/train.json")
    val_ann_file: Optional[str] = field(default="data/Fitness-AQA/annotations/val.json")
    data_root: str           = field(default="data/Fitness-AQA")
    frame_mode: str          = field(default="uniform")
    n_frames: int            = field(default=16)
    use_grid: bool           = field(default=False)

    # LoRA
    lora_r: int         = field(default=16)
    lora_alpha: int     = field(default=32)
    lora_dropout: float = field(default=0.0)

    # GRPO 核心超参
    num_generations: int = field(default=8,   metadata={"help": "组内候选数 G"})
    max_new_tokens: int  = field(default=256, metadata={"help": "Rollout 生成长度上限"})
    temperature: float   = field(default=0.9)
    top_p: float         = field(default=0.95)
    kl_coef: float       = field(default=0.04, metadata={"help": "KL 惩罚系数 β"})

    # RLAIF：LLM-as-Judge
    judge_base_url: str  = field(default="http://localhost:8000/v1", metadata={"help": "OpenAI-compatible API base URL"})
    judge_model:    str  = field(default="Qwen2.5-72B-Instruct",     metadata={"help": "裁判模型名称"})
    judge_api_key:  str  = field(default="EMPTY",                     metadata={"help": "API Key，本地部署填 EMPTY"})
    judge_timeout:  float = field(default=30.0, metadata={"help": "单次 API 请求超时秒数"})
    judge_max_retries: int = field(default=2,   metadata={"help": "API 失败最大重试次数"})

    # 训练超参
    output_dir: str      = field(default="checkpoints/grpo")
    num_train_epochs: int = field(default=2)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=5e-6)
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
# RLAIF：LLM-as-Judge
# ──────────────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
你是专业力量训练裁判，负责评估 AI 健身教练的回答质量。

评分标准（满分 10 分）：
  【动作识别】（2 分）
    2分：动作类别识别完全正确
    1分：识别模糊或部分正确
    0分：识别错误或缺失

  【错误识别】（3 分）
    3分：GT 中所有错误均被准确识别，无虚假错误
    2分：识别出大部分错误，或有少量漏检
    1分：仅识别出部分错误，或有明显误报
    0分：完全未识别出错误，或 GT 无错误却虚报错误

  【修正建议】（3 分）
    3分：建议具体、可执行，与 GT 修正方向一致
    2分：建议较笼统但方向正确
    1分：建议与 GT 偏差较大或不可操作
    0分：无修正建议或完全错误

  【格式规范】（2 分）
    2分：包含全部四个节区：【动作识别】【总体结论】【关键问题】【下一组口令】
    1分：缺少 1 个节区
    0分：缺少 2 个及以上节区

只输出一个 0-10 的整数，不要任何解释。"""

_JUDGE_USER_TMPL = """\
【GT 错误标注】
{gt_errors}

【模型回答】
{response}

评分（0-10）："""


class AIJudgeReward:
    """
    调用 OpenAI-compatible API 的 LLM 裁判，输出 [0, 1] 奖励分。

    兼容任意实现了 /chat/completions 的服务：
    vLLM、Ollama、LM Studio、SGLang、OpenAI、DeepSeek、智谱 GLM 等。
    """

    def __init__(
        self,
        base_url:    str   = "http://localhost:8000/v1",
        model:       str   = "Qwen2.5-72B-Instruct",
        api_key:     str   = "EMPTY",
        timeout:     float = 30.0,
        max_retries: int   = 2,
    ) -> None:
        self.model       = model
        self.timeout     = timeout
        self.max_retries = max_retries
        self.client      = OpenAI(base_url=base_url, api_key=api_key)
        logger.info(f"[AIJudge] base_url={base_url}  model={model}")

    def _format_gt(self, gt_error_annotations: list[dict]) -> str:
        if not gt_error_annotations:
            return "无错误（本组动作完美，模型应输出无明显问题）"
        lines = []
        for e in gt_error_annotations:
            ts = e.get("start_time")
            ts_str = f"  @{ts:.1f}s" if ts is not None and ts < 200 else ""
            lines.append(
                f"- {e.get('error_cn', e.get('error_type', '未知'))}{ts_str}\n"
                f"  修正：{e.get('correction', '无')}"
            )
        return "\n".join(lines)

    def __call__(self, generated: str, gt_error_annotations: list[dict]) -> float:
        user_msg = _JUDGE_USER_TMPL.format(
            gt_errors=self._format_gt(gt_error_annotations),
            response=generated[:1500],
        )
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _JUDGE_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    max_tokens=4,
                    temperature=0.0,
                    timeout=self.timeout,
                )
                text  = resp.choices[0].message.content.strip()
                score = float(re.search(r"\d+", text).group()) / 10.0
                return min(max(score, 0.0), 1.0)
            except Exception as exc:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                logger.warning(f"[AIJudge] API 失败（回退 0.5）: {exc}")
                return 0.5


# ──────────────────────────────────────────────────────────────────────────────
# GRPOTrainer 兼容的奖励函数
# ──────────────────────────────────────────────────────────────────────────────

def make_reward_fn(judge: AIJudgeReward):
    """
    返回符合 GRPOTrainer 签名的奖励函数：
        fn(completions: list[str], **kwargs) -> list[float]
    """

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        error_annotations_list = kwargs.get("error_annotations", [[] for _ in completions])
        rewards = []
        for generated, gt_anns in zip(completions, error_annotations_list):
            score = judge(generated, gt_anns)
            rewards.append(score)
            logger.debug(f"[Reward] score={score:.3f} | {generated[:40]!r}")
        return rewards

    return reward_fn


# ──────────────────────────────────────────────────────────────────────────────
# 模型加载
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_processor(args: GRPOArguments):
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

    logger.info("[GRPO] ========== 开始 GRPO 训练（纯 RLAIF）==========")

    model, processor = load_model_and_processor(grpo_args)

    judge = AIJudgeReward(
        base_url=grpo_args.judge_base_url,
        model=grpo_args.judge_model,
        api_key=grpo_args.judge_api_key,
        timeout=grpo_args.judge_timeout,
        max_retries=grpo_args.judge_max_retries,
    )
    reward_fn = make_reward_fn(judge)

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
        num_generations=grpo_args.num_generations,
        max_new_tokens=grpo_args.max_new_tokens,
        temperature=grpo_args.temperature,
        top_p=grpo_args.top_p,
        kl_coef=grpo_args.kl_coef,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=[reward_fn],
        processing_class=processor,
    )

    logger.info(
        f"[GRPO] 裁判={grpo_args.judge_model}  "
        f"G={grpo_args.num_generations}  "
        f"max_new_tokens={grpo_args.max_new_tokens}  "
        f"kl_coef={grpo_args.kl_coef}"
    )
    trainer.train()

    logger.info(f"[GRPO] 保存到 {grpo_args.output_dir}/final")
    trainer.save_model(os.path.join(grpo_args.output_dir, "final"))
    processor.save_pretrained(os.path.join(grpo_args.output_dir, "final"))
    logger.info("[GRPO] ========== 训练完成 ==========")


if __name__ == "__main__":
    main()
