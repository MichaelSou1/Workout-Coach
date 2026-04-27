"""
Fitness-AQA 数据集解析与多模态样本构建
=======================================

支持两种媒体来源（由 prepare_dataset.py 生成的统一 JSON 驱动）：
  - video_path  : 直接从 .mp4 文件抽帧（Squat、OHP）
  - image_paths : 加载预提取图片帧列表（ShallowSquat、BarbellRow）

帧采样策略：
  - uniform   : 均匀抽取 n_frames 帧
  - timestamp : 在错误时间戳附近密集采样（仅对 video_path 有效）

输出两种样本格式：
  - SFT  : {input_ids, attention_mask, pixel_values, image_grid_thw, labels}
  - GRPO : {prompt, images, error_annotations, ...}

前置步骤：
  python training/prepare_dataset.py \
      --dataset_root dataset/Fitness-AQA_dataset_release \
      --output_dir   training/annotations
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Literal, Optional

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset

# Qwen2VLProcessor 和 process_vision_info 在训练服务器上才需要；
# 帧提取/数据解析部分可在本地（无 qwen-vl-utils）独立运行。
try:
    from transformers import Qwen2VLProcessor
    from qwen_vl_utils import process_vision_info
    _VLM_AVAILABLE = True
except ImportError:
    Qwen2VLProcessor = None   # type: ignore
    process_vision_info = None  # type: ignore
    _VLM_AVAILABLE = False

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

SYSTEM_PROMPT = (
    "你是专业力量训练教练，通过连续视频帧分析健身动作并提供技术指导。"
    "请按以下格式输出分析结果（只输出中文，不输出代码或坐标）：\n"
    "【动作识别】识别到的动作类别\n"
    "【总体结论】整体质量一句话总结\n"
    "【关键问题1】问题：...；原因：...；修正：...\n"
    "【关键问题2】问题：...；原因：...；修正：...\n"
    "【关键问题3】问题：...；原因：...；修正：...\n"
    "【下一组口令】2-3 条简短的执行口令"
)

USER_TEMPLATE = (
    "以下是一组【{action_class}】动作的连续帧截图。\n"
    "{error_hint}"
    "请按指定格式完成分析。"
)


# ──────────────────────────────────────────────────────────────────────────────
# 帧提取：视频文件
# ──────────────────────────────────────────────────────────────────────────────

def extract_frames_uniform(
    video_path: str,
    n_frames: int = 16,
    target_size: tuple[int, int] = (448, 448),
) -> list[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    indices = [min(int(i * total / n_frames), total - 1) for i in range(n_frames)]

    frames: list[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frames.append(frames[-1] if frames else Image.new("RGB", target_size))
            continue
        frames.append(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(
                target_size, Image.LANCZOS
            )
        )
    cap.release()
    return frames


def extract_frames_by_timestamps(
    video_path: str,
    error_timestamps: list[float],
    context_window: float = 1.0,
    frames_per_segment: int = 4,
    n_global: int = 4,
    target_size: tuple[int, int] = (448, 448),
) -> list[Image.Image]:
    """在错误时间戳附近密集采样，同时保留全局上下文帧。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    target_secs: list[float] = [i * duration / (n_global + 1) for i in range(n_global)]
    for ts in error_timestamps:
        start_t = max(0.0, ts - context_window)
        end_t   = min(duration, ts + context_window)
        for j in range(frames_per_segment):
            t = start_t + j * (end_t - start_t) / max(frames_per_segment - 1, 1)
            target_secs.append(t)

    target_secs = sorted(set(round(t, 3) for t in target_secs))
    frame_ids   = sorted(set(min(int(t * fps), total - 1) for t in target_secs))

    frames: list[Image.Image] = []
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(
                target_size, Image.LANCZOS
            )
        )
    cap.release()
    return frames or [Image.new("RGB", target_size)]


# ──────────────────────────────────────────────────────────────────────────────
# 帧提取：预提取图片列表（ShallowSquat、BarbellRow）
# ──────────────────────────────────────────────────────────────────────────────

def load_image_frames(
    image_paths: list[str],
    dataset_root: str,
    n_frames: int = 16,
    target_size: tuple[int, int] = (448, 448),
) -> list[Image.Image]:
    """
    从预提取图片列表中均匀选取 n_frames 张，并 resize。

    Args:
        image_paths  : 相对于 dataset_root 的图片路径列表（已按帧号排序）
        dataset_root : 数据集根目录（绝对路径）
        n_frames     : 目标帧数
        target_size  : 输出尺寸 (W, H)
    """
    if not image_paths:
        return [Image.new("RGB", target_size)]

    total = len(image_paths)
    indices = [min(int(i * total / n_frames), total - 1) for i in range(n_frames)]
    indices = sorted(set(indices))

    frames: list[Image.Image] = []
    for idx in indices:
        rel_path = image_paths[idx]
        full_path = os.path.join(dataset_root, rel_path)
        try:
            img = Image.open(full_path).convert("RGB").resize(target_size, Image.LANCZOS)
            frames.append(img)
        except Exception as e:
            logger.warning(f"无法加载图片 {full_path}: {e}")
            if frames:
                frames.append(frames[-1])

    return frames or [Image.new("RGB", target_size)]


# ──────────────────────────────────────────────────────────────────────────────
# Grid Image 拼合（可选，减少 image token 数量）
# ──────────────────────────────────────────────────────────────────────────────

def build_grid_image(
    frames: list[Image.Image],
    grid_cols: int = 4,
    cell_size: tuple[int, int] = (224, 224),
) -> Image.Image:
    n = len(frames)
    rows = math.ceil(n / grid_cols)
    W, H = cell_size
    grid = Image.new("RGB", (grid_cols * W, rows * H), (0, 0, 0))
    for i, f in enumerate(frames):
        r, c = divmod(i, grid_cols)
        grid.paste(f.resize(cell_size, Image.LANCZOS), (c * W, r * H))
    return grid


# ──────────────────────────────────────────────────────────────────────────────
# 构建 Qwen2-VL Chat Messages
# ──────────────────────────────────────────────────────────────────────────────

def build_messages(
    frames: list[Image.Image],
    action_class: str,
    error_annotations: list[dict],
    assistant_response: Optional[str] = None,
    use_grid: bool = False,
    grid_cols: int = 4,
) -> list[dict]:
    """
    生成符合 Qwen2-VL Chat Template 的 messages 列表。

    error_annotations 中若有时间戳（start_time 为秒数而非帧号），
    则在 User prompt 中提示模型关注该时间段。
    对于图片帧数据集（ShallowSquat/BarbellRow），start_time 为帧号，不作时间提示。
    """
    # 判断是否有实际时间信息（视频来源的 start_time 通常 < 100s，帧号可能很大）
    has_time = any(
        ann.get("start_time", 9999) < 200
        for ann in error_annotations
    )

    if has_time and error_annotations:
        ts_list = sorted(set(ann["start_time"] for ann in error_annotations))
        ts_str  = "、".join(f"{t:.1f}s" for t in ts_list)
        error_hint = f"请重点关注如下时间节点（单位：秒）：{ts_str}\n"
    else:
        error_hint = ""

    user_text = USER_TEMPLATE.format(
        action_class=action_class,
        error_hint=error_hint,
    )

    if use_grid:
        user_content = [
            {"type": "image", "image": build_grid_image(frames, grid_cols)},
            {"type": "text",  "text":  user_text},
        ]
    else:
        user_content = [{"type": "image", "image": f} for f in frames]
        user_content.append({"type": "text", "text": user_text})

    messages: list[dict] = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
    ]
    if assistant_response is not None:
        messages.append({"role": "assistant", "content": assistant_response})

    return messages


# ──────────────────────────────────────────────────────────────────────────────
# SFT Tokenization + Label 掩码
# ──────────────────────────────────────────────────────────────────────────────

def tokenize_and_mask_labels(
    messages: list[dict],
    processor,
    max_length: int = 2048,
) -> dict:
    """
    对完整对话（含 assistant 回答）tokenize，对 System/User 部分应用 -100 掩码。

    实现：
      1. process_vision_info 提取图像输入
      2. 完整对话 tokenize → full_ids
      3. 仅 prompt tokenize → prompt_len
      4. labels = [-100]*prompt_len + full_ids[prompt_len:]
    """
    if not _VLM_AVAILABLE:
        raise ImportError("需要 qwen-vl-utils：pip install qwen-vl-utils")
    image_inputs, video_inputs = process_vision_info(messages)

    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_msgs = [m for m in messages if m["role"] != "assistant"]
    prompt_text = processor.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )

    full_inputs = processor(
        text=[full_text],
        images=image_inputs or None,
        videos=video_inputs or None,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    full_ids: list[int] = full_inputs["input_ids"][0].tolist()

    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs or None,
        videos=video_inputs or None,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    prompt_len: int = prompt_inputs["input_ids"].shape[1]

    labels = [IGNORE_INDEX] * prompt_len + full_ids[prompt_len:]
    labels = labels[: len(full_ids)]

    result = {
        "input_ids":      full_inputs["input_ids"][0],
        "attention_mask": full_inputs["attention_mask"][0],
        "labels":         torch.tensor(labels, dtype=torch.long),
    }
    if "pixel_values" in full_inputs:
        result["pixel_values"]   = full_inputs["pixel_values"]
    if "image_grid_thw" in full_inputs:
        result["image_grid_thw"] = full_inputs["image_grid_thw"]

    return result


# ──────────────────────────────────────────────────────────────────────────────
# GT 回答合成（当 reference_response 为 None 时使用）
# ──────────────────────────────────────────────────────────────────────────────

def synthesize_response(action_class: str, error_annotations: list[dict]) -> str:
    lines = [
        f"【动作识别】{action_class}",
        "【总体结论】本组动作存在以下关键技术问题，需针对性改正。",
    ]
    for i, err in enumerate(error_annotations[:3], 1):
        ts = err.get("start_time")
        ts_note = f"（约 {ts:.1f}s）" if ts is not None and ts < 200 else ""
        lines.append(
            f"【关键问题{i}】"
            f"问题：{err.get('error_cn', err.get('error_type', '动作错误'))}{ts_note}；"
            f"原因：肌肉代偿或活动度不足；"
            f"修正：{err.get('correction', '请咨询专业教练')}"
        )
    for i in range(len(error_annotations) + 1, 4):
        lines.append(f"【关键问题{i}】暂无明显问题")
    lines.append("【下一组口令】收紧核心；保持中立脊柱；缓慢下降控制离心")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset 类
# ──────────────────────────────────────────────────────────────────────────────

class FitnessAQADataset(Dataset):
    """
    统一 Fitness-AQA 多模态 Dataset。

    读取 prepare_dataset.py 生成的 JSON，支持视频（Squat/OHP）
    和预提取图片帧（ShallowSquat/BarbellRow）两种媒体来源。

    Args:
        ann_file     : JSON 标注文件路径（由 prepare_dataset.py 生成）
        dataset_root : Fitness-AQA 数据集根目录（video_path/image_paths 相对此目录）
        processor    : Qwen2VLProcessor
        mode         : "sft" | "grpo"
        frame_mode   : "uniform" | "timestamp"（仅对视频有效）
        n_frames     : 每样本抽取帧数
        use_grid     : 是否合并为 Grid Image
        max_length   : SFT 序列最大长度
    """

    def __init__(
        self,
        ann_file:     str,
        dataset_root: str,
        processor:    Qwen2VLProcessor,
        mode:         Literal["sft", "grpo"] = "sft",
        frame_mode:   Literal["uniform", "timestamp"] = "uniform",
        n_frames:     int  = 16,
        use_grid:     bool = False,
        max_length:   int  = 2048,
    ) -> None:
        super().__init__()
        self.dataset_root = dataset_root
        self.processor    = processor
        self.mode         = mode
        self.frame_mode   = frame_mode
        self.n_frames     = n_frames
        self.use_grid     = use_grid
        self.max_length   = max_length

        with open(ann_file, "r", encoding="utf-8") as f:
            self.annotations: list[dict] = json.load(f)

        logger.info(
            f"[FitnessAQADataset] {len(self.annotations)} 条样本 "
            f"(mode={mode}, n_frames={n_frames})"
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def _load_frames(self, ann: dict) -> list[Image.Image]:
        """根据样本类型加载帧列表。"""
        error_anns = ann.get("error_annotations", [])

        if ann.get("video_path"):
            video_path = os.path.join(self.dataset_root, ann["video_path"])
            if self.frame_mode == "timestamp" and error_anns:
                ts = [e["start_time"] for e in error_anns if e.get("start_time", 9999) < 200]
                if ts:
                    return extract_frames_by_timestamps(
                        video_path, ts,
                        n_global=max(4, self.n_frames - len(ts) * 4),
                    )
            return extract_frames_uniform(video_path, self.n_frames)

        elif ann.get("image_paths"):
            return load_image_frames(
                ann["image_paths"], self.dataset_root, self.n_frames
            )

        else:
            logger.warning(f"[FitnessAQADataset] 样本无视频/图片路径: {ann.get('video_id')}")
            return [Image.new("RGB", (448, 448))]

    def __getitem__(self, idx: int) -> dict:
        ann        = self.annotations[idx]
        frames     = self._load_frames(ann)
        action_cls = ann.get("action_class", "未知动作")
        error_anns = ann.get("error_annotations", [])
        gt_resp    = ann.get("reference_response") or synthesize_response(action_cls, error_anns)

        if self.mode == "sft":
            messages = build_messages(
                frames, action_cls, error_anns,
                assistant_response=gt_resp,
                use_grid=self.use_grid,
            )
            return tokenize_and_mask_labels(messages, self.processor, self.max_length)

        else:   # grpo
            if not _VLM_AVAILABLE:
                raise ImportError("需要 qwen-vl-utils：pip install qwen-vl-utils")
            messages = build_messages(
                frames, action_cls, error_anns,
                assistant_response=None,
                use_grid=self.use_grid,
            )
            image_inputs, _ = process_vision_info(messages)
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return {
                "prompt":            prompt_text,
                "images":            image_inputs,
                "gt_response":       gt_resp,
                "action_class":      action_cls,
                "error_annotations": error_anns,
                "video_id":          ann.get("video_id", str(idx)),
            }


# ──────────────────────────────────────────────────────────────────────────────
# Data Collator（SFT BatchLoader）
# ──────────────────────────────────────────────────────────────────────────────

class FitnessAQACollator:
    def __init__(self, processor: Qwen2VLProcessor):
        self.pad_id = processor.tokenizer.pad_token_id or 0

    def __call__(self, samples: list[dict]) -> dict:
        from torch.nn.utils.rnn import pad_sequence

        batch = {
            "input_ids":      pad_sequence([s["input_ids"]      for s in samples], batch_first=True, padding_value=self.pad_id),
            "attention_mask": pad_sequence([s["attention_mask"]  for s in samples], batch_first=True, padding_value=0),
            "labels":         pad_sequence([s["labels"]          for s in samples], batch_first=True, padding_value=IGNORE_INDEX),
        }
        if "pixel_values" in samples[0]:
            try:
                batch["pixel_values"]   = torch.cat([s["pixel_values"]   for s in samples], dim=0)
                batch["image_grid_thw"] = torch.cat([s["image_grid_thw"] for s in samples], dim=0)
            except Exception:
                batch["pixel_values"]   = samples[0]["pixel_values"]
                batch["image_grid_thw"] = samples[0]["image_grid_thw"]
        return batch


# ──────────────────────────────────────────────────────────────────────────────
# 快速验证入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    p = argparse.ArgumentParser()
    p.add_argument("--ann_file",     required=True)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--model_path",   default="hf_cache/modelscope/Qwen--Qwen2-VL-7B-Instruct")
    p.add_argument("--mode",         default="sft", choices=["sft", "grpo"])
    args = p.parse_args()

    proc = Qwen2VLProcessor.from_pretrained(args.model_path)
    ds   = FitnessAQADataset(
        ann_file=args.ann_file,
        dataset_root=args.dataset_root,
        processor=proc,
        mode=args.mode,
    )
    sample = ds[0]
    print("keys:", list(sample.keys()))
    if "input_ids" in sample:
        n_mask = (sample["labels"] == IGNORE_INDEX).sum().item()
        print(f"input_ids: {sample['input_ids'].shape}  masked: {n_mask}/{len(sample['labels'])}")
    else:
        print("prompt[:200]:", sample["prompt"][:200])
