"""
SGLang OpenAI 兼容 API 客户端。

设计：
- 不再在本进程内加载模型；模型由独立的 sglang.launch_server 进程承载
- 多卡并行 / 量化 / dtype / FlashAttention 等全部下沉到 server 启动参数
- 本类只负责构造 messages、把 PIL 图像编码为 base64 data URL、调用 chat completions
- 同进程内多请求并发由 server 端自动批处理 + RadixAttention 前缀缓存
"""

import base64
import io
import logging
import os
from threading import Thread
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "需要安装 openai 包：pip install openai>=1.40.0"
    ) from e


# ---------------------------------------------------------------------------
# PIL <-> base64 data URL
# ---------------------------------------------------------------------------

def _pil_to_data_url(image: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    """PIL Image → 'data:image/jpeg;base64,...'"""
    buf = io.BytesIO()
    img = image.convert("RGB") if image.mode != "RGB" else image
    save_kwargs = {"format": fmt}
    if fmt.upper() == "JPEG":
        save_kwargs["quality"] = quality
    img.save(buf, **save_kwargs)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def _convert_messages_to_openai(messages: List[dict]) -> List[dict]:
    """
    将内部 message 格式（含 {"type":"image","image":PIL}）转为 OpenAI 兼容格式
    （image_url 带 base64 data URL）。其余 type 直通。
    """
    converted = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            converted.append({"role": role, "content": str(content)})
            continue

        new_content = []
        for item in content:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "image":
                pil = item.get("image")
                if pil is None:
                    continue
                new_content.append({
                    "type": "image_url",
                    "image_url": {"url": _pil_to_data_url(pil)},
                })
            elif itype == "image_url":
                new_content.append(item)
            elif itype == "text":
                new_content.append({"type": "text", "text": item.get("text", "")})
            else:
                # 透传未知 type，便于扩展
                new_content.append(item)
        converted.append({"role": role, "content": new_content})
    return converted


# ---------------------------------------------------------------------------
# SGLang 客户端
# ---------------------------------------------------------------------------

class FitnessVLMClient:
    """
    SGLang server 的薄客户端。

    保留 chat() / analyze_fitness_frames() / analyze_fitness_frames_async() 接口，
    兼容 agent_loop 和 benchmark 的调用方式。
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: str = "EMPTY",
        timeout: float = 300.0,
        verbose: bool = True,
    ):
        """
        Args:
            endpoint: 形如 http://127.0.0.1:30000/v1 (默认从 SGLANG_ENDPOINT 读)
            model_name: 用于 chat.completions 请求的 model 字段
            api_key: SGLang 默认不校验，传任意非空即可
            timeout: 单次请求超时（秒）
        """
        endpoint = endpoint or os.getenv("SGLANG_ENDPOINT", "http://127.0.0.1:30000")
        if not endpoint.endswith("/v1"):
            endpoint = endpoint.rstrip("/") + "/v1"

        self.endpoint = endpoint
        self.model_name = model_name or os.getenv("VLM_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
        self.timeout = timeout
        self.verbose = verbose
        self.client = OpenAI(base_url=endpoint, api_key=api_key, timeout=timeout)

        if verbose:
            logger.setLevel(logging.DEBUG)

        logger.info(f"[VLM] SGLang endpoint: {endpoint} | model={self.model_name}")
        self._check_server()

    def _check_server(self):
        """启动时探测 server 可达性；失败仅警告，不抛异常（容许后启动 server）。"""
        try:
            models = self.client.models.list()
            ids = [m.id for m in models.data]
            logger.info(f"[VLM] ✓ SGLang server 可达，已加载模型: {ids}")
        except Exception as e:
            logger.warning(
                f"[VLM] 无法连接到 SGLang server ({self.endpoint}): {e}\n"
                f"      请先启动: bash scripts/launch_sglang.sh"
            )

    # ------------------------------------------------------------------
    # 核心：多轮 chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[dict],
        max_new_tokens: int = 512,
    ) -> str:
        """通用多轮 chat completion。messages 可含 {"type":"image","image":PIL}。"""
        if not messages:
            logger.warning("[VLM] 空 messages，跳过推理")
            return ""

        do_sample = os.getenv("VLM_DO_SAMPLE", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
        temperature = float(os.getenv("VLM_TEMPERATURE", "0.6")) if do_sample else 0.0
        top_p = float(os.getenv("VLM_TOP_P", "0.9"))
        repetition_penalty = float(os.getenv("VLM_REPETITION_PENALTY", "1.05"))

        openai_messages = _convert_messages_to_openai(messages)

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body={
                    "repetition_penalty": repetition_penalty,
                    "skip_special_tokens": True,
                },
            )
            text = resp.choices[0].message.content or ""
            logger.info(f"[VLM] ✓ 推理完成，输出长度: {len(text)} 字符")
            return text.strip()

        except Exception as e:
            logger.exception("[VLM] SGLang 调用失败")
            return f"推理错误: {e}"

    # ------------------------------------------------------------------
    # 单次便捷接口（用于非 agentic 模式和评测）
    # ------------------------------------------------------------------

    def analyze_fitness_frames(
        self,
        frames: List[Image.Image],
        system_prompt: str,
        user_query: str = "请分析这些帧中的动作问题。",
        max_new_tokens: int = 512,
    ) -> str:
        """单轮：组装 messages 后转发 chat()。"""
        if not frames:
            return ""

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [{"type": "image", "image": f} for f in frames]
                + [{"type": "text", "text": user_query}],
            },
        ]
        return self.chat(messages, max_new_tokens=max_new_tokens)

    def analyze_fitness_frames_async(
        self,
        frames: List[Image.Image],
        system_prompt: str,
        callback,
        user_query: str = "请分析这些帧中的动作问题。",
        max_new_tokens: int = 512,
    ):
        """异步包装：后台线程调用 analyze_fitness_frames 后触发 callback。"""
        def _worker():
            try:
                result = self.analyze_fitness_frames(
                    frames=frames,
                    system_prompt=system_prompt,
                    user_query=user_query,
                    max_new_tokens=max_new_tokens,
                )
                callback(result)
            except Exception as e:
                logger.exception("[VLM] 异步推理异常")
                callback(f"推理错误: {e}")

        Thread(target=_worker, daemon=True).start()
