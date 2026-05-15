"""
Agentic vs 非 agentic 对比评测脚本。

用法:
    # 用 input/ 下所有视频，每个跑 3 次，每次跑两种模式
    python benchmark.py --video-dir input/ --runs 3

    # 用 manifest 指定视频和声明的动作名（支持故意错标动作测试）
    python benchmark.py --manifest bench_manifest.json --runs 3 --output-dir bench_results/

    # 贪心解码（消除采样方差，便于看模式差异）
    python benchmark.py --video-dir input/ --runs 1 --greedy

manifest 格式 (JSON 数组)：
    [
        {"file": "squat_01.mp4", "action": "深蹲", "notes": "标准动作（无标注）"},
        {"file": "squat_bad.mp4", "action": "深蹲", "notes": "膝内扣"},
        {"file": "squat_mislabel.mp4", "action": "硬拉", "notes": "故意错标，测试动作识别"}
    ]

输出：
    output_dir/
        per_run.csv           # 每次运行一行
        summary.csv           # 按 (video, mode) 聚合
        outputs/              # 每次运行的原始 VLM 文本输出
            squat_01__agentic__run0.txt
            squat_01__non_agentic__run0.txt
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# 重依赖（cv2 / torch / PIL）延迟到 main() 中导入，保证 --help 在未装依赖时也可用


# ----------------------------------------------------------------------
# 日志
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("benchmark")


# ----------------------------------------------------------------------
# 环境加载（与 main.py 一致）
# ----------------------------------------------------------------------

def _load_local_env(env_file: str = ".env"):
    if not os.path.exists(env_file):
        return
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as e:
        logger.warning(f"读取 .env 失败: {e}")


_load_local_env()


# ----------------------------------------------------------------------
# 视频加载
# ----------------------------------------------------------------------

def load_video_frames(
    video_path: str,
    max_frames: int = 8,
    target_height: int = 336,
):
    """读取整个视频 → 均匀抽帧 → 等比缩放 → PIL RGB。"""
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    all_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        all_frames.append(frame)
    cap.release()

    if not all_frames:
        raise RuntimeError(f"视频无可读帧: {video_path}")

    # 均匀抽帧
    if len(all_frames) > max_frames:
        step = len(all_frames) / max_frames
        sampled = [all_frames[int(i * step)] for i in range(max_frames)]
    else:
        sampled = all_frames

    pil_frames = []
    for f in sampled:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        short = min(h, w)
        if short > target_height:
            scale = target_height / short
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            rgb = cv2.resize(rgb, (new_w, new_h))
        pil_frames.append(Image.fromarray(rgb))
    return pil_frames


def _peak_mem_gb_all_devices() -> float:
    """SGLang 模式下 GPU 在独立 server 进程内，本进程无法直接测峰值。
    需要时用外部 `nvidia-smi` 或 SGLang `/get_server_info` 监控。
    """
    return 0.0


def _reset_peak_mem_all_devices():
    pass


# ----------------------------------------------------------------------
# Manifest 解析
# ----------------------------------------------------------------------

_DEFAULT_ACTION_HEURISTICS = [
    ("squat", "深蹲"),
    ("深蹲", "深蹲"),
    ("deadlift", "硬拉"),
    ("硬拉", "硬拉"),
    ("bench", "卧推"),
    ("卧推", "卧推"),
    ("press", "推举"),
    ("推举", "推举"),
]


def _infer_action_from_filename(filename: str) -> str:
    """文件名启发式映射动作；找不到返回 '未知动作'。"""
    low = filename.lower()
    for key, action in _DEFAULT_ACTION_HEURISTICS:
        if key in low:
            return action
    return "未知动作"


def load_manifest(manifest_path: Optional[str], video_dir: Optional[str]) -> List[dict]:
    """返回 [{file, action, notes}, ...]"""
    if manifest_path:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("manifest 必须是 JSON 数组")
        for item in data:
            if "file" not in item or "action" not in item:
                raise ValueError(f"manifest 条目缺少 file/action: {item}")
            item.setdefault("notes", "")
        return data

    if not video_dir:
        raise ValueError("必须提供 --manifest 或 --video-dir 之一")

    videos = []
    for path in sorted(Path(video_dir).glob("*")):
        if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
            videos.append({
                "file": path.name,
                "action": _infer_action_from_filename(path.name),
                "notes": "（按文件名推断动作）",
            })
    if not videos:
        raise RuntimeError(f"目录 {video_dir} 下未发现视频文件")
    return videos


# ----------------------------------------------------------------------
# 输出文本评分
# ----------------------------------------------------------------------

_REQUIRED_SECTIONS = [
    "【画面动作】",
    "【总体结论】",
    "【关键问题1】",
    "【关键问题2】",
    "【关键问题3】",
    "【下一组执行口令】",
]


def _extract_section(text: str, header: str, next_headers: List[str]) -> str:
    if header not in text:
        return ""
    start = text.index(header) + len(header)
    end = len(text)
    for nh in next_headers:
        if nh in text:
            pos = text.index(nh, start) if text.find(nh, start) != -1 else -1
            if pos != -1 and pos < end:
                end = pos
    return text[start:end].strip()


def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", s).strip().lower()


def score_output(text: str, declared_action: str) -> dict:
    """对单次输出文本评分（不依赖人工标注）。"""
    if not text:
        return {
            "format_score": 0.0,
            "action_recognized": "",
            "action_match": 0,
            "mismatch_flagged": 0,
            "populated_issues": 0,
            "issue_redundancy": 0,
            "output_chars": 0,
            "had_warning_prefix": 0,
        }

    had_warning = 1 if text.lstrip().startswith("（已达推理上限") else 0

    present = [s for s in _REQUIRED_SECTIONS if s in text]
    format_score = round(len(present) / len(_REQUIRED_SECTIONS), 3)

    section_action = _extract_section(text, "【画面动作】", _REQUIRED_SECTIONS[1:])
    action_recognized = section_action.split("（")[0].strip()  # 去掉用户声称注释

    declared_norm = _normalize(declared_action)
    recognized_norm = _normalize(action_recognized)
    action_match = 1 if (declared_norm and declared_norm in recognized_norm) else 0
    mismatch_flagged = 1 if ("（" in section_action or "(" in section_action or "用户声称" in section_action) else 0

    issues = []
    for n in (1, 2, 3):
        sec = _extract_section(text, f"【关键问题{n}】", _REQUIRED_SECTIONS)
        issues.append(sec)
    populated_issues = sum(1 for s in issues if s and "暂无" not in s and "无明显" not in s)

    issue_redundancy = 0
    normalized_issues = [_normalize(s) for s in issues if s]
    for i in range(len(normalized_issues)):
        for j in range(i + 1, len(normalized_issues)):
            if normalized_issues[i] and normalized_issues[i] == normalized_issues[j]:
                issue_redundancy = 1

    return {
        "format_score": format_score,
        "action_recognized": action_recognized[:40],
        "action_match": action_match,
        "mismatch_flagged": mismatch_flagged,
        "populated_issues": populated_issues,
        "issue_redundancy": issue_redundancy,
        "output_chars": len(text),
        "had_warning_prefix": had_warning,
    }


# ----------------------------------------------------------------------
# 推理执行
# ----------------------------------------------------------------------

def run_non_agentic(vlm_client, frames, action_type: str) -> Tuple[str, dict]:
    from action_profiles import build_prompts
    system_prompt, user_query = build_prompts(action_type)
    t0 = time.time()
    result = vlm_client.analyze_fitness_frames(
        frames=frames,
        system_prompt=system_prompt,
        user_query=user_query,
        max_new_tokens=int(os.getenv("BENCH_NON_AGENTIC_MAX_TOKENS", "512")),
    )
    elapsed = time.time() - t0
    meta = {
        "total_seconds": round(elapsed, 3),
        "turns": 1,
        "tools_called": "",
        "hit_max_iter": 0,
        "per_turn_seconds": json.dumps([round(elapsed, 3)]),
    }
    return result, meta


def run_agentic(agent, frames, action_type: str) -> Tuple[str, dict]:
    result, stats = agent.analyze_sync(frames=frames, action_type=action_type)
    meta = {
        "total_seconds": round(stats.total_seconds, 3),
        "turns": stats.turns,
        "tools_called": ",".join(stats.tools_called),
        "hit_max_iter": 1 if stats.hit_max_iter else 0,
        "per_turn_seconds": json.dumps([round(s, 3) for s in stats.per_turn_seconds]),
    }
    return result, meta


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------

PER_RUN_COLUMNS = [
    "video", "declared_action", "mode", "run_idx",
    "total_seconds", "turns", "tools_called", "hit_max_iter", "per_turn_seconds",
    "peak_gpu_gb",
    "format_score", "action_recognized", "action_match", "mismatch_flagged",
    "populated_issues", "issue_redundancy", "output_chars", "had_warning_prefix",
    "error",
]


def main():
    parser = argparse.ArgumentParser(description="Agentic vs 非 agentic 对比评测")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=str, help="JSON manifest 路径")
    src.add_argument("--video-dir", type=str, help="视频目录（按文件名推断动作）")

    parser.add_argument("--runs", type=int, default=3, help="每个 (video, mode) 跑几次")
    parser.add_argument("--output-dir", type=str, default="bench_results", help="输出目录")
    parser.add_argument("--max-frames", type=int, default=int(os.getenv("VLM_MAX_FRAMES", "8")))
    parser.add_argument("--target-height", type=int, default=int(os.getenv("TARGET_HEIGHT", "336")))
    parser.add_argument("--modes", nargs="+", default=["non_agentic", "agentic"], choices=["non_agentic", "agentic"])
    parser.add_argument("--greedy", action="store_true", help="强制贪心解码（消除采样方差）")
    parser.add_argument("--limit", type=int, default=0, help="限制视频数量（调试用）")
    args = parser.parse_args()

    if args.greedy:
        os.environ["VLM_DO_SAMPLE"] = "0"
        logger.info("[Bench] 启用贪心解码 (VLM_DO_SAMPLE=0)")

    manifest = load_manifest(args.manifest, args.video_dir)
    if args.limit > 0:
        manifest = manifest[:args.limit]
    logger.info(f"[Bench] 准备评测 {len(manifest)} 个视频 × {len(args.modes)} 模式 × {args.runs} 次")

    # ------------------------------------------------------------------
    # 初始化（延迟导入，避免不需要时拉起模型）
    # ------------------------------------------------------------------

    from vlm_inference import FitnessVLMClient
    from agent_loop import AgenticAnalyzer

    logger.info("[Bench] 连接 SGLang server...")
    client = FitnessVLMClient(
        endpoint=os.getenv("SGLANG_ENDPOINT", "http://127.0.0.1:30000"),
        model_name=os.getenv("VLM_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct"),
        timeout=float(os.getenv("SGLANG_TIMEOUT", "300")),
        verbose=False,
    )

    agent = AgenticAnalyzer(
        vlm_client=client,
        max_iterations=int(os.getenv("AGENT_MAX_ITER", "3")),
        max_new_tokens=int(os.getenv("VLM_MAX_TOKENS", "512")),
    )

    # ------------------------------------------------------------------
    # 输出准备
    # ------------------------------------------------------------------

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = out_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    per_run_csv = out_dir / "per_run.csv"
    summary_csv = out_dir / "summary.csv"

    rows: List[dict] = []

    # 视频目录（manifest 中 file 是相对路径）
    video_base = Path(args.video_dir) if args.video_dir else Path(args.manifest).parent

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    for video_idx, entry in enumerate(manifest):
        video_file = entry["file"]
        declared_action = entry["action"]
        video_path = video_base / video_file
        if not video_path.exists():
            # 兼容绝对路径
            if Path(video_file).exists():
                video_path = Path(video_file)
            else:
                logger.error(f"[Bench] 视频不存在: {video_path}，跳过")
                continue

        logger.info(f"\n[Bench] === Video {video_idx + 1}/{len(manifest)}: {video_file} (action={declared_action}) ===")

        try:
            frames = load_video_frames(
                str(video_path),
                max_frames=args.max_frames,
                target_height=args.target_height,
            )
            logger.info(f"[Bench] 加载 {len(frames)} 帧")
        except Exception as e:
            logger.error(f"[Bench] 视频加载失败: {e}")
            continue

        for mode in args.modes:
            for run_idx in range(args.runs):
                logger.info(f"[Bench] -- mode={mode} run={run_idx} --")
                _reset_peak_mem_all_devices()
                error = ""
                try:
                    if mode == "non_agentic":
                        result_text, meta = run_non_agentic(client, frames, declared_action)
                    else:
                        result_text, meta = run_agentic(agent, frames, declared_action)
                except Exception as e:
                    logger.exception("[Bench] 推理异常")
                    result_text = ""
                    meta = {"total_seconds": 0, "turns": 0, "tools_called": "",
                            "hit_max_iter": 0, "per_turn_seconds": "[]"}
                    error = str(e)[:200]

                peak_mem = _peak_mem_gb_all_devices()
                scores = score_output(result_text, declared_action)

                # 保存原始输出
                base_name = Path(video_file).stem
                out_file = outputs_dir / f"{base_name}__{mode}__run{run_idx}.txt"
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(f"# video: {video_file}\n# declared_action: {declared_action}\n")
                    f.write(f"# mode: {mode}\n# run: {run_idx}\n")
                    f.write(f"# notes: {entry.get('notes', '')}\n")
                    f.write(f"# meta: {json.dumps(meta, ensure_ascii=False)}\n")
                    f.write(f"# scores: {json.dumps(scores, ensure_ascii=False)}\n")
                    f.write(f"# peak_gpu_gb: {peak_mem}\n")
                    f.write("---\n")
                    f.write(result_text)

                row = {
                    "video": video_file,
                    "declared_action": declared_action,
                    "mode": mode,
                    "run_idx": run_idx,
                    **meta,
                    "peak_gpu_gb": peak_mem,
                    **scores,
                    "error": error,
                }
                rows.append(row)

                logger.info(
                    f"[Bench]   t={meta['total_seconds']}s | turns={meta['turns']} "
                    f"| tools=[{meta['tools_called']}] | format={scores['format_score']} "
                    f"| issues={scores['populated_issues']} | match={scores['action_match']}"
                )

    # ------------------------------------------------------------------
    # 写 CSV
    # ------------------------------------------------------------------

    with open(per_run_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_RUN_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in PER_RUN_COLUMNS})
    logger.info(f"[Bench] 每次运行明细: {per_run_csv}")

    # ------------------------------------------------------------------
    # 聚合
    # ------------------------------------------------------------------

    write_summary(rows, summary_csv)
    logger.info(f"[Bench] 聚合摘要: {summary_csv}")

    print_summary_table(rows)


def _agg(values: List[float]) -> dict:
    finite = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not finite:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": round(statistics.mean(finite), 3),
        "median": round(statistics.median(finite), 3),
        "std": round(statistics.pstdev(finite), 3) if len(finite) > 1 else 0.0,
        "min": round(min(finite), 3),
        "max": round(max(finite), 3),
        "n": len(finite),
    }


SUMMARY_COLUMNS = [
    "video", "declared_action", "mode", "n_runs",
    "latency_mean", "latency_median", "latency_std", "latency_min", "latency_max",
    "turns_mean", "tool_call_total",
    "format_score_mean", "action_match_rate", "mismatch_flag_rate",
    "populated_issues_mean", "issue_redundancy_rate",
    "output_chars_mean", "warning_prefix_rate", "peak_gpu_gb_max",
    "error_rate",
]


def write_summary(rows: List[dict], path: Path):
    # 按 (video, mode) 聚合
    keys = sorted({(r["video"], r["declared_action"], r["mode"]) for r in rows})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for video, action, mode in keys:
            subset = [r for r in rows if r["video"] == video and r["mode"] == mode]
            lat = _agg([float(r["total_seconds"]) for r in subset])
            turns = _agg([float(r["turns"]) for r in subset])
            tools_all = []
            for r in subset:
                tc = r.get("tools_called", "") or ""
                if tc:
                    tools_all.extend([t for t in tc.split(",") if t])
            fmt = _agg([float(r["format_score"]) for r in subset])
            issues = _agg([float(r["populated_issues"]) for r in subset])
            chars = _agg([float(r["output_chars"]) for r in subset])
            peak = max((float(r.get("peak_gpu_gb", 0) or 0) for r in subset), default=0.0)
            n = len(subset)
            writer.writerow({
                "video": video,
                "declared_action": action,
                "mode": mode,
                "n_runs": n,
                "latency_mean": lat["mean"],
                "latency_median": lat["median"],
                "latency_std": lat["std"],
                "latency_min": lat["min"],
                "latency_max": lat["max"],
                "turns_mean": turns["mean"],
                "tool_call_total": len(tools_all),
                "format_score_mean": fmt["mean"],
                "action_match_rate": round(sum(int(r["action_match"]) for r in subset) / n, 3) if n else 0,
                "mismatch_flag_rate": round(sum(int(r["mismatch_flagged"]) for r in subset) / n, 3) if n else 0,
                "populated_issues_mean": issues["mean"],
                "issue_redundancy_rate": round(sum(int(r["issue_redundancy"]) for r in subset) / n, 3) if n else 0,
                "output_chars_mean": chars["mean"],
                "warning_prefix_rate": round(sum(int(r["had_warning_prefix"]) for r in subset) / n, 3) if n else 0,
                "peak_gpu_gb_max": round(peak, 2),
                "error_rate": round(sum(1 for r in subset if r.get("error")) / n, 3) if n else 0,
            })


def print_summary_table(rows: List[dict]):
    """终端打印模式间对比。"""
    print("\n" + "=" * 78)
    print("MODE-LEVEL AGGREGATE (across all videos × runs)")
    print("=" * 78)
    modes = sorted({r["mode"] for r in rows})
    for mode in modes:
        subset = [r for r in rows if r["mode"] == mode]
        if not subset:
            continue
        n = len(subset)
        lat = _agg([float(r["total_seconds"]) for r in subset])
        turns = _agg([float(r["turns"]) for r in subset])
        fmt_mean = round(statistics.mean(float(r["format_score"]) for r in subset), 3)
        match_rate = round(sum(int(r["action_match"]) for r in subset) / n, 3)
        issues_mean = round(statistics.mean(float(r["populated_issues"]) for r in subset), 2)
        chars_mean = round(statistics.mean(float(r["output_chars"]) for r in subset), 1)
        warning_rate = round(sum(int(r["had_warning_prefix"]) for r in subset) / n, 3)
        error_rate = round(sum(1 for r in subset if r.get("error")) / n, 3)
        total_tools = sum(len((r.get("tools_called") or "").split(",")) - (1 if not r.get("tools_called") else 0)
                          for r in subset)

        print(f"\n[{mode}]  n_runs={n}")
        print(f"  latency_s          mean={lat['mean']}  median={lat['median']}  std={lat['std']}  range=[{lat['min']},{lat['max']}]")
        print(f"  turns_mean         {turns['mean']}  (only meaningful for agentic)")
        print(f"  tool_calls_total   {total_tools}")
        print(f"  format_score_mean  {fmt_mean}  (1.0 = 全部 6 段齐全)")
        print(f"  action_match_rate  {match_rate}")
        print(f"  populated_issues   {issues_mean} / 3")
        print(f"  output_chars_mean  {chars_mean}")
        print(f"  warning_prefix_rate {warning_rate}  (agentic 达到 max_iter 的占比)")
        print(f"  error_rate         {error_rate}")
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
