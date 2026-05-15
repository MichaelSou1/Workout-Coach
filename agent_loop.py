"""
Agentic 推理循环：VLM 输出 ReAct 风格的 Action / Action_Input，
runtime 执行工具并把 Observation 回灌给模型，直至模型不再输出 Action 行。
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from threading import Thread
from typing import Callable, List, Tuple

from PIL import Image

from action_profiles import build_agentic_system_prompt, build_agentic_user_query
from tools import execute_tool
from vlm_inference import FitnessVLMClient

logger = logging.getLogger(__name__)


_ACTION_LINE_RE = re.compile(r"^\s*Action\s*:\s*(.+?)\s*$", re.MULTILINE)
_ACTION_INPUT_LINE_RE = re.compile(r"^\s*Action_Input\s*:\s*(.+?)\s*$", re.MULTILINE)


@dataclass
class LoopStats:
    """单次 agentic 调用的执行指标。"""
    turns: int = 0
    tools_called: List[str] = field(default_factory=list)
    per_turn_seconds: List[float] = field(default_factory=list)
    hit_max_iter: bool = False
    total_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "turns": self.turns,
            "tools_called": list(self.tools_called),
            "per_turn_seconds": list(self.per_turn_seconds),
            "hit_max_iter": self.hit_max_iter,
            "total_seconds": self.total_seconds,
        }


class AgenticAnalyzer:
    """
    持有 FitnessVLMClient，提供 agentic 异步/同步分析接口。
    """

    def __init__(self, vlm_client: FitnessVLMClient, max_iterations: int = 3, max_new_tokens: int = 512):
        self.vlm_client = vlm_client
        self.max_iterations = max(1, max_iterations)
        self.max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_async(
        self,
        frames: List[Image.Image],
        action_type: str,
        callback: Callable[[str], None],
    ):
        """异步运行 agentic loop，完成后 callback(final_text)。"""
        def _worker():
            try:
                result, _ = self._run_loop(frames, action_type)
                callback(result)
            except Exception as e:
                logger.exception("[Agent] 循环异常")
                callback(f"推理错误: {e}")

        Thread(target=_worker, daemon=True).start()

    def analyze_sync(
        self,
        frames: List[Image.Image],
        action_type: str,
    ) -> Tuple[str, LoopStats]:
        """同步运行 agentic loop，返回 (最终文本, 执行指标)。用于评测脚本。"""
        return self._run_loop(frames, action_type)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _run_loop(self, frames: List[Image.Image], action_type: str) -> Tuple[str, LoopStats]:
        stats = LoopStats()
        loop_start = time.time()

        if not frames:
            stats.total_seconds = time.time() - loop_start
            return "无帧可分析", stats

        logger.info(
            f"[Agent] 开始 loop，max_iter={self.max_iterations}，frames={len(frames)}"
        )
        # SGLang RadixAttention 会自动复用 system + 历史前缀的 KV cache，
        # 我们只需每轮把完整 messages 重新发过去即可。

        messages = self._build_initial_messages(frames, action_type)
        last_response = ""

        for turn in range(self.max_iterations):
            logger.info(f"[Agent] === Turn {turn} ===")
            turn_start = time.time()
            response = self.vlm_client.chat(messages, max_new_tokens=self.max_new_tokens)
            turn_elapsed = time.time() - turn_start
            stats.per_turn_seconds.append(turn_elapsed)
            last_response = response
            logger.info(f"[Agent] 模型输出 (前 200 字)：{response[:200]}")

            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            })

            parsed = self._parse_response(response)

            if parsed["type"] == "final":
                logger.info("[Agent] ✓ 收到最终回答")
                stats.turns = turn + 1
                stats.total_seconds = time.time() - loop_start
                return parsed["text"], stats

            tool_name = parsed["name"]
            tool_args = parsed["args"]
            stats.tools_called.append(tool_name)
            logger.info(f"[Agent] 调用工具: {tool_name} args={tool_args}")
            observation = execute_tool(tool_name, tool_args, frames, action_type=action_type)
            logger.info(f"[Agent] Observation (前 200 字)：{observation[:200]}")

            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": f"Observation:\n{observation}"}],
            })

        logger.warning("[Agent] 已达推理上限，返回最后一轮输出")
        stats.turns = self.max_iterations
        stats.hit_max_iter = True
        stats.total_seconds = time.time() - loop_start
        return f"（已达推理上限）\n{last_response}", stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_initial_messages(self, frames: List[Image.Image], action_type: str) -> List[dict]:
        system_prompt = build_agentic_system_prompt(action_type)
        user_query = build_agentic_user_query(action_type, num_frames=len(frames))

        user_content: List[dict] = []
        for idx, frame in enumerate(frames):
            user_content.append({"type": "text", "text": f"第{idx}帧:"})
            user_content.append({"type": "image", "image": frame})
        user_content.append({"type": "text", "text": user_query})

        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

    def _parse_response(self, text: str) -> dict:
        """
        识别响应：含 Action 行视为工具调用，否则视为最终回答。
        返回：{"type":"final","text":...} 或 {"type":"tool_call","name":...,"args":...}
        """
        action_match = _ACTION_LINE_RE.search(text)
        if action_match is None:
            return {"type": "final", "text": text.strip()}

        name = action_match.group(1).strip()
        # 去掉可能的括号 / 引号包装
        name = name.strip("`'\"")

        # 解析 Action_Input
        args = {}
        input_match = _ACTION_INPUT_LINE_RE.search(text)
        if input_match:
            raw = input_match.group(1).strip()
            args = self._parse_action_input(raw)

        if name not in {"get_pose_angles", "detect_phase_boundaries"}:
            logger.warning(f"[Agent] 未知工具 '{name}'，按最终回答处理")
            return {"type": "final", "text": text.strip()}

        return {"type": "tool_call", "name": name, "args": args}

    @staticmethod
    def _parse_action_input(raw: str) -> dict:
        """容错 JSON 解析：尝试 json.loads，失败时尝试常见修复。"""
        raw = raw.strip()
        if not raw or raw in {"{}", "null", "None"}:
            return {}

        # 直接解析
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

        # 单引号 → 双引号
        try:
            parsed = json.loads(raw.replace("'", '"'))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

        # 提取 frame_indices 中的数字
        nums = re.findall(r"\d+", raw)
        if nums:
            return {"frame_indices": [int(n) for n in nums]}

        logger.warning(f"[Agent] 无法解析 Action_Input: {raw[:100]}")
        return {}
