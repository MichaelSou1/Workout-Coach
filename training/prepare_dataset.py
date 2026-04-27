"""
Fitness-AQA 原始格式 → 统一训练 JSON 转换脚本
================================================

数据集真实结构（经 README 确认）：

  Squat (视频 + 时间区间标注)
    Labels/error_knees_forward.json  -> {video_id: [[start, end], ...]}
    Labels/error_knees_inward.json   -> {video_id: [[start, end], ...]}
    Splits/train_keys.json           -> [video_id, ...]

  ShallowSquat (图片帧 + 二值标注，独立子数据集)
    Shallow_Squat_Error_Dataset/labels_shallow_depth.json
                                     -> {vid_frame_id: 0/1}
    Shallow_Squat_Error_Dataset/images/crops_unaligned/  -> *.jpg

  OHP (视频 + 时间区间标注)
    Labels/error_elbows.json         -> {video_id: [[start, end], ...]}
    Labels/error_knees.json          -> {video_id: [[start, end], ...]}

  BarbellRow (图片帧 + 二值标注)
    Labels/labels_lumbar_error.json      -> {vid_frame_id: 0/1}
    Labels/labels_torso_angle_error.json -> {vid_frame_id: 0/1}
    barbellrow_images_raw/barbellrow_images_raw/ -> *.jpg

用法：
  cd c:/Users/20968/Desktop/Workout-Coach
  python training/prepare_dataset.py \
      --dataset_root dataset/Fitness-AQA_dataset_release \
      --output_dir   training/annotations
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 错误类型中文描述及纠正建议
# ──────────────────────────────────────────────────────────────────────────────

ERROR_META = {
    "knees_forward": {
        "error_cn":   "膝盖过度前伸",
        "correction": "保持小腿尽量垂直地面，重心落在足跟，膝盖不过度超过脚尖",
    },
    "knees_inward": {
        "error_cn":   "膝关节内扣",
        "correction": "下蹲时膝盖跟随脚尖方向向外推，激活臀中肌防止内扣",
    },
    "shallow_depth": {
        "error_cn":   "深蹲深度不足",
        "correction": "髋关节下降至膝关节水平以下（大腿平行地面），加强踝关节灵活性",
    },
    "elbows_flared": {
        "error_cn":   "手肘外展过度",
        "correction": "推举时保持手肘向前约45°，不要完全外展，保护肩关节",
    },
    "knees_locked": {
        "error_cn":   "膝关节过度锁定",
        "correction": "站立推举时膝盖保持微弯（约5°），避免锁死关节造成压迫",
    },
    "lumbar_flexion": {
        "error_cn":   "腰椎屈曲",
        "correction": "保持腰背挺直，通过髋铰链启动动作，核心收紧防止脊柱屈曲",
    },
    "torso_angle": {
        "error_cn":   "躯干角度错误",
        "correction": "杠铃划船时躯干与地面保持约45°，不要过于直立或过于水平",
    },
}


def load_json(path: str | Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# 解析器 1：视频 + 时间区间标注（Squat、OHP）
# ──────────────────────────────────────────────────────────────────────────────

def parse_video_dataset(
    dataset_root: Path,
    action_cn: str,
    video_dir: str,
    split_dir: str,
    split_file_map: dict[str, str],       # {"train": "train_keys.json", ...}
    error_label_files: list[dict],        # [{"file": ..., "error_type": ...}, ...]
) -> dict[str, list[dict]]:
    """
    解析视频动作数据集，返回 {split: [sample, ...]}。
    """
    # 读取 split
    splits: dict[str, list[str]] = {}
    for split_name, fname in split_file_map.items():
        p = dataset_root / split_dir / fname
        if p.exists():
            splits[split_name] = load_json(p)
        else:
            logger.warning(f"split 文件不存在: {p}")

    all_video_ids = {vid for ids in splits.values() for vid in ids}
    video_errors: dict[str, list[dict]] = defaultdict(list)

    # 合并各错误类型
    for err in error_label_files:
        label_path = dataset_root / err["file"]
        if not label_path.exists():
            logger.warning(f"标注文件不存在: {label_path}")
            continue
        labels: dict = load_json(label_path)
        meta = ERROR_META.get(err["error_type"], {})
        for vid, intervals in labels.items():
            if vid not in all_video_ids:
                continue
            for itv in intervals:
                if len(itv) == 2:
                    video_errors[vid].append({
                        "start_time": float(itv[0]),
                        "end_time":   float(itv[1]),
                        "error_type": err["error_type"],
                        "error_cn":   meta.get("error_cn", err["error_type"]),
                        "correction": meta.get("correction", ""),
                    })

    result: dict[str, list[dict]] = {}
    for split_name, video_ids in splits.items():
        samples = []
        for vid in video_ids:
            vpath = Path(video_dir) / f"{vid}.mp4"
            full = dataset_root / vpath
            if not full.exists():
                logger.debug(f"视频不存在，跳过: {full}")
                continue
            samples.append({
                "video_id":          vid,
                "action_class":      action_cn,
                "video_path":        str(vpath),     # 相对于 dataset_root
                "image_paths":       None,
                "error_annotations": sorted(
                    video_errors.get(vid, []),
                    key=lambda x: x["start_time"],
                ),
                "reference_response": None,
            })
        result[split_name] = samples
        logger.info(f"{action_cn}/{split_name}: {len(samples)} 条")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 解析器 2：图片帧 + 二值标注（ShallowSquat、BarbellRow）
# ──────────────────────────────────────────────────────────────────────────────

def parse_image_dataset(
    dataset_root: Path,
    action_cn: str,
    image_dir: str,
    split_dir: str,
    split_file_map: dict[str, str],   # {"train": "train_ids.json", ...}
    error_label_files: list[dict],    # [{"file": ..., "error_type": ...}, ...]
) -> dict[str, list[dict]]:
    """
    解析图片帧二值标注数据集。

    key 格式: "{video_id}_{frame_number}"，value: 0/1
    按 video_id 聚合，每个 video_id 生成一条 VLM 训练样本。
    """
    img_dir_full = dataset_root / image_dir

    # ── 按 video_id 聚合图片路径 ──
    video_images: dict[str, list[tuple[int, str]]] = defaultdict(list)
    if img_dir_full.exists():
        for img_file in img_dir_full.iterdir():
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stem = img_file.stem               # e.g., "52701_11_40"
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                vid, frame_id = parts[0], int(parts[1])
                rel = str(Path(image_dir) / img_file.name)
                video_images[vid].append((frame_id, rel))

    # 按帧号排序
    for vid in video_images:
        video_images[vid].sort(key=lambda x: x[0])

    # ── 聚合二值帧标注 ──
    # video_frame_errors[vid][error_type] = [error_frame_id, ...]
    video_frame_errors: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    for err in error_label_files:
        label_path = dataset_root / err["file"]
        if not label_path.exists():
            logger.warning(f"标注文件不存在: {label_path}")
            continue
        labels: dict = load_json(label_path)
        for key, val in labels.items():
            parts = key.rsplit("_", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            vid, frame_id = parts[0], int(parts[1])
            if int(val) == 1:
                video_frame_errors[vid][err["error_type"]].append(frame_id)

    # ── 读取 split（split IDs 为 frame_id 级别，需提取 video_id）──
    splits: dict[str, set[str]] = {}
    for split_name, fname in split_file_map.items():
        p = dataset_root / split_dir / fname
        if not p.exists():
            logger.warning(f"split 文件不存在: {p}")
            continue
        ids = load_json(p)
        # 从 frame-level ID 中提取 video_id
        video_set: set[str] = set()
        for item_id in ids:
            parts = item_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                video_set.add(parts[0])
            else:
                video_set.add(item_id)
        splits[split_name] = video_set

    result: dict[str, list[dict]] = {}
    for split_name, video_ids in splits.items():
        samples = []
        for vid in video_ids:
            img_list = [rel for _, rel in video_images.get(vid, [])]
            if not img_list:
                continue

            error_annotations = []
            for err in error_label_files:
                et = err["error_type"]
                meta = ERROR_META.get(et, {})
                error_frames = sorted(video_frame_errors[vid].get(et, []))
                if error_frames:
                    # 用帧号占位 start_time/end_time（data_builder 中区别于时间秒数）
                    error_annotations.append({
                        "start_time":  float(error_frames[0]),
                        "end_time":    float(error_frames[-1]),
                        "error_type":  et,
                        "error_cn":    meta.get("error_cn", et),
                        "correction":  meta.get("correction", ""),
                        "frame_ids":   error_frames,   # 额外保留帧号列表
                    })

            samples.append({
                "video_id":          vid,
                "action_class":      action_cn,
                "video_path":        None,
                "image_paths":       img_list,    # 相对于 dataset_root
                "error_annotations": error_annotations,
                "reference_response": None,
            })
        result[split_name] = samples
        logger.info(f"{action_cn}/{split_name}: {len(samples)} 条")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="dataset/Fitness-AQA_dataset_release")
    parser.add_argument("--output_dir",   default="training/annotations")
    parser.add_argument(
        "--actions", nargs="+",
        default=["Squat", "OHP", "ShallowSquat", "BarbellRow"],
        choices=["Squat", "OHP", "ShallowSquat", "BarbellRow"],
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_splits: dict[str, list] = {"train": [], "val": [], "test": []}

    # ── Squat ──────────────────────────────────────────────────────────────────
    if "Squat" in args.actions:
        logger.info("== 处理 Squat ==")
        data = parse_video_dataset(
            dataset_root=dataset_root,
            action_cn="深蹲",
            video_dir="Squat/Labeled_Dataset/videos/videos",
            split_dir="Squat/Labeled_Dataset/Splits",
            split_file_map={"train": "train_keys.json", "val": "val_keys.json", "test": "test_keys.json"},
            error_label_files=[
                {"file": "Squat/Labeled_Dataset/Labels/error_knees_forward.json", "error_type": "knees_forward"},
                {"file": "Squat/Labeled_Dataset/Labels/error_knees_inward.json",  "error_type": "knees_inward"},
            ],
        )
        for split, samples in data.items():
            all_splits[split].extend(samples)

    # ── OHP ────────────────────────────────────────────────────────────────────
    if "OHP" in args.actions:
        logger.info("== 处理 OHP ==")
        data = parse_video_dataset(
            dataset_root=dataset_root,
            action_cn="过头推举",
            video_dir="OHP/Labeled_Dataset/videos/videos",
            split_dir="OHP/Labeled_Dataset/Splits",
            split_file_map={"train": "train_keys.json", "val": "val_keys.json", "test": "test_keys.json"},
            error_label_files=[
                {"file": "OHP/Labeled_Dataset/Labels/error_elbows.json", "error_type": "elbows_flared"},
                {"file": "OHP/Labeled_Dataset/Labels/error_knees.json",  "error_type": "knees_locked"},
            ],
        )
        for split, samples in data.items():
            all_splits[split].extend(samples)

    # ── ShallowSquat（图片帧，二值标注）────────────────────────────────────────
    if "ShallowSquat" in args.actions:
        logger.info("== 处理 ShallowSquat ==")
        data = parse_image_dataset(
            dataset_root=dataset_root,
            action_cn="深蹲（深度检测）",
            image_dir="Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/images/crops_unaligned",
            split_dir="Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/splits",
            split_file_map={"train": "train_ids.json", "val": "val_ids.json", "test": "test_ids.json"},
            error_label_files=[
                {
                    "file":       "Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/labels_shallow_depth.json",
                    "error_type": "shallow_depth",
                },
            ],
        )
        for split, samples in data.items():
            all_splits[split].extend(samples)

    # ── BarbellRow（图片帧，二值标注）─────────────────────────────────────────
    if "BarbellRow" in args.actions:
        logger.info("== 处理 BarbellRow ==")
        data = parse_image_dataset(
            dataset_root=dataset_root,
            action_cn="杠铃划船",
            image_dir="BarbellRow/Labeled_Dataset/barbellrow_images_raw/barbellrow_images_raw",
            split_dir="BarbellRow/Labeled_Dataset/Splits/Splits_Lumbar_Error",
            split_file_map={"train": "train_ids.json", "val": "val_ids.json", "test": "test_ids.json"},
            error_label_files=[
                {"file": "BarbellRow/Labeled_Dataset/Labels/labels_lumbar_error.json",      "error_type": "lumbar_flexion"},
                {"file": "BarbellRow/Labeled_Dataset/Labels/labels_torso_angle_error.json", "error_type": "torso_angle"},
            ],
        )
        for split, samples in data.items():
            all_splits[split].extend(samples)

    # ── 写出 JSON ──────────────────────────────────────────────────────────────
    for split_name, samples in all_splits.items():
        out = output_dir / f"{split_name}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        logger.info(f"写出 {out}  ({len(samples)} 条)")

    total = sum(len(v) for v in all_splits.values())
    logger.info(f"\n完成！总计 {total} 条")
    logger.info(f"  train={len(all_splits['train'])}, val={len(all_splits['val'])}, test={len(all_splits['test'])}")


if __name__ == "__main__":
    main()
