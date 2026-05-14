"""
用 LLM 为 Fitness-AQA 标注批量生成 SFT 参考回答
================================================

Fitness-AQA 只提供错误类型 + 时间戳 + 修正关键词，没有自然语言教练评语。
本脚本调用 OpenAI-compatible API，根据每条样本的标注生成符合项目输出格式的
reference_response，写回 JSON 后供 train_sft.py 使用。

用法：
  # 在 GRPO 裁判服务（vLLM）启动后运行，使用同一个服务即可
  python training/generate_references.py \
      --input_file  training/annotations/train.json \
      --output_file training/annotations/train_with_ref.json \
      --base_url    http://localhost:8000/v1 \
      --model       Qwen2.5-72B-Instruct \
      --workers     4

  # 同时处理 train + val
  python training/generate_references.py \
      --input_file  training/annotations/train.json \
      --output_file training/annotations/train_with_ref.json
  python training/generate_references.py \
      --input_file  training/annotations/val.json \
      --output_file training/annotations/val_with_ref.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt 设计
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM = """\
你是经验丰富的力量训练教练，语言风格简洁专业，直接给出可执行的建议。
请根据提供的动作标注，生成一段完整的教练分析报告。

输出必须严格按照以下格式，不要添加任何额外内容：
【动作识别】<识别到的动作名称>
【总体结论】<一句话总结本组动作的整体质量>
【关键问题1】问题：<问题描述>（<时间，如"约3.2s"，无时间信息则省略括号>）；原因：<成因分析>；修正：<具体可执行的修正方法>
【关键问题2】问题：...；原因：...；修正：...
【关键问题3】问题：...；原因：...；修正：...
【下一组口令】<2-3条简短执行口令，用分号分隔>

如果错误数量少于3个，剩余关键问题填写"暂无明显问题"。
如果没有任何错误，所有关键问题均填写"暂无明显问题"，总体结论写动作质量良好。"""

_USER_TMPL = """\
动作类别：{action_class}

错误标注（来自运动科学专家）：
{error_block}

请生成教练分析报告："""


def _build_error_block(error_annotations: list[dict]) -> str:
    if not error_annotations:
        return "无错误（本组动作完美）"
    lines = []
    for e in error_annotations:
        ts = e.get("start_time")
        ts_str = f"（{ts:.1f}s）" if ts is not None and ts < 200 else ""
        lines.append(
            f"- 错误类型：{e.get('error_cn', e.get('error_type', '未知'))}{ts_str}\n"
            f"  修正要点：{e.get('correction', '无')}"
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 生成单条参考回答
# ──────────────────────────────────────────────────────────────────────────────

def generate_one(
    sample: dict,
    client: OpenAI,
    model: str,
    max_retries: int = 3,
    timeout: float = 60.0,
) -> str | None:
    """
    为单条样本生成 reference_response。
    返回生成文本，失败返回 None（调用方决定是否保留旧值）。
    """
    user_msg = _USER_TMPL.format(
        action_class=sample.get("action_class", "未知动作"),
        error_block=_build_error_block(sample.get("error_annotations", [])),
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=400,
                temperature=0.7,   # 适当多样性，避免所有样本输出相同
                timeout=timeout,
            )
            text = resp.choices[0].message.content.strip()
            # 基本格式校验：必须包含四个必须节区
            if all(sec in text for sec in ["【动作识别】", "【总体结论】", "【关键问题", "【下一组口令】"]):
                return text
            logger.warning(f"[{sample.get('video_id')}] 格式不合规，重试 {attempt+1}/{max_retries}")
        except Exception as e:
            logger.warning(f"[{sample.get('video_id')}] API 失败（{attempt+1}/{max_retries}）: {e}")
            time.sleep(2 ** attempt)

    return None


# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",  required=True,  help="输入 JSON（prepare_dataset.py 生成）")
    parser.add_argument("--output_file", required=True,  help="输出 JSON（写回 reference_response）")
    parser.add_argument("--base_url",    default="http://localhost:8000/v1")
    parser.add_argument("--model",       default="Qwen2.5-72B-Instruct")
    parser.add_argument("--api_key",     default="EMPTY")
    parser.add_argument("--workers",     type=int, default=4,   help="并发线程数（受 API QPS 限制）")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--timeout",     type=float, default=60.0)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="重新生成已有 reference_response 的样本（默认跳过）"
    )
    args = parser.parse_args()

    input_path  = Path(args.input_file)
    output_path = Path(args.output_file)

    with open(input_path, "r", encoding="utf-8") as f:
        samples: list[dict] = json.load(f)

    logger.info(f"读取 {len(samples)} 条样本：{input_path}")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # 过滤需要生成的样本
    todo = [
        (i, s) for i, s in enumerate(samples)
        if args.overwrite or not s.get("reference_response")
    ]
    skip = len(samples) - len(todo)
    logger.info(f"需要生成：{len(todo)} 条  跳过（已有）：{skip} 条")

    success = 0
    fail    = 0

    def _task(args_tuple):
        idx, sample = args_tuple
        ref = generate_one(sample, client, args.model, args.max_retries, args.timeout)
        return idx, ref

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_task, item): item for item in todo}
        for i, future in enumerate(as_completed(futures), 1):
            idx, ref = future.result()
            vid = samples[idx].get("video_id", str(idx))
            if ref:
                samples[idx]["reference_response"] = ref
                success += 1
                if success % 50 == 0 or success == 1:
                    logger.info(f"进度 {i}/{len(todo)}  成功={success}  失败={fail}")
            else:
                fail += 1
                logger.warning(f"[{vid}] 生成失败，保留 None")

    # 中间保存，避免长时间运行后崩溃丢失
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    logger.info(
        f"\n完成！成功={success}  失败={fail}  总计={len(samples)}\n"
        f"输出文件：{output_path}"
    )

    if fail > 0:
        logger.warning(
            f"{fail} 条样本未能生成 reference_response，SFT 时将由 data_builder.py 的 "
            f"synthesize_response() 兜底合成（质量较低）。"
            f"可加 --overwrite 重新运行只针对这些样本。"
        )


if __name__ == "__main__":
    main()
