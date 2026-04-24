"""
主程序入口 - Fitness Coach 多模态健身动作指导系统
整合 VLM 推理、视频流处理、UI 反馈的完整工作流
"""

import logging
import sys
import os
from typing import List

from PIL import Image

from action_profiles import build_prompts
from vlm_inference import FitnessVLM
from video_streamer import VideoStreamer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _load_local_env(env_file: str = ".env"):
    """轻量加载 .env（不依赖 python-dotenv）。"""
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
        logger.warning(f"[Main] 读取 .env 失败: {e}")


_load_local_env()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"[Main] 环境变量 {name}={value} 非法，使用默认值 {default}")
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"[Main] 环境变量 {name}={value} 非法，使用默认值 {default}")
        return default


# ============================================================================
# 全局状态
# ============================================================================

vlm_model = None
video_streamer = None
is_analyzing = False


# ============================================================================
# 回调函数：处理 VLM 推理结果
# ============================================================================

def on_vlm_result(result_text: str):
    """VLM 推理完成后的回调函数，将摘要显示在视频画面上。"""
    global is_analyzing, video_streamer

    logger.info(f"[Main] VLM 返回结果:\n{result_text}")

    feedback_duration = _env_float("FEEDBACK_DURATION", 4.0)
    if video_streamer:
        display_text = result_text[:50] if result_text else "分析完成"

        # 优先显示总体结论；若有画面动作与声称不符则一并显示
        if "【总体结论】" in result_text:
            try:
                start = result_text.index("【总体结论】") + len("【总体结论】")
                end = result_text.index("【关键问题1】") if "【关键问题1】" in result_text else len(result_text)
                display_text = result_text[start:end].strip()[:45]
            except Exception:
                pass
        if "【画面动作】" in result_text:
            try:
                start = result_text.index("【画面动作】") + len("【画面动作】")
                end = result_text.index("【总体结论】") if "【总体结论】" in result_text else len(result_text)
                action_line = result_text[start:end].strip()
                # 只有当动作行包含括号说明（即不符时）才加入显示
                if "（" in action_line:
                    display_text = action_line[:40] + " | " + display_text[:20]
            except Exception:
                pass

        video_streamer.set_feedback(display_text, duration=feedback_duration)

    is_analyzing = False


def on_analysis_request(frames: List[Image.Image], action_type: str):
    """
    视频流触发分析请求时的回调函数。
    启动异步 VLM 推理。
    
    Args:
        frames: PIL Image 列表
        action_type: 动作类型（squat/deadlift/bench_press）
    """
    global is_analyzing, vlm_model
    
    if is_analyzing:
        logger.warning("[Main] 已有分析在进行中，忽略此请求")
        video_streamer.set_feedback("分析中，请稍候...", duration=1.0)
        return

    if not vlm_model:
        logger.error("[Main] VLM 模型未初始化")
        video_streamer.set_feedback("模型未就绪，请检查", duration=2.0)
        return

    is_analyzing = True
    system_prompt, user_query = build_prompts(action_type)

    # 均匀抽帧，避免帧数过多导致 2B 模型注意力分散
    max_frames = _env_int("VLM_MAX_FRAMES", 8)
    if len(frames) > max_frames:
        step = len(frames) / max_frames
        frames = [frames[int(i * step)] for i in range(max_frames)]

    logger.info(f"[Main] 开始异步推理 {len(frames)} 帧，动作类型: {action_type}")
    video_streamer.set_feedback(f"分析中: {action_type}", duration=2.0)

    max_new_tokens = _env_int("VLM_MAX_TOKENS", 256)

    vlm_model.analyze_fitness_frames_async(
        frames=frames,
        system_prompt=system_prompt,
        user_query=user_query,
        callback=on_vlm_result,
        max_new_tokens=max_new_tokens,
    )


# ============================================================================
# 初始化函数
# ============================================================================

def initialize_vlm():
    """初始化 VLM 模型。"""
    global vlm_model
    
    logger.info("[Main] ========== 初始化 VLM 模型 ==========")
    
    try:
        model_name = os.getenv("VLM_MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")
        device = os.getenv("VLM_DEVICE", "cuda")
        verbose = _env_bool("VERBOSE", True)
        local_files_only = _env_bool("VLM_LOCAL_FILES_ONLY", False)
        use_flash_attention_2 = _env_bool("USE_FLASH_ATTENTION_2", True)
        cache_dir = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")

        hf_endpoint = os.getenv("HF_ENDPOINT") or os.getenv("HUGGINGFACE_HUB_ENDPOINT")
        if hf_endpoint:
            logger.info(f"[Main] 使用 Hugging Face 镜像端点: {hf_endpoint}")
        if local_files_only:
            logger.info("[Main] 离线模式已启用（仅使用本地缓存）")
        if cache_dir:
            logger.info(f"[Main] 模型缓存目录: {cache_dir}")

        vlm_model = FitnessVLM(
            model_name=model_name,
            device=device,
            verbose=verbose,
            hf_endpoint=hf_endpoint,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            use_flash_attention_2=use_flash_attention_2,
        )
        logger.info("[Main] ✓ VLM 模型初始化完成")
        return True
    
    except Exception as e:
        logger.error(f"[Main] ❌ VLM 模型初始化失败: {e}")
        logger.error("[Main] 请检查：")
        logger.error("  1. CUDA 是否可用（torch.cuda.is_available()）")
        logger.error("  2. 显存是否足够（至少 8GB）")
        logger.error("  3. Hugging Face 模型是否可下载")
        return False


_INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".flv", ".wmv", ".webm"}


def _scan_input_dir() -> List[str]:
    """扫描 input/ 目录，返回所有视频文件的完整路径列表（按文件名排序）。"""
    if not os.path.isdir(_INPUT_DIR):
        return []
    files = [
        os.path.join(_INPUT_DIR, f)
        for f in sorted(os.listdir(_INPUT_DIR))
        if os.path.splitext(f)[1].lower() in _VIDEO_EXTS
    ]
    return files


def _select_input_source() -> str:
    """扫描 input/ 文件夹，让用户选择视频文件或摄像头。返回文件路径（摄像头返回空字符串）。"""
    logger.info("[Main] ========== 选择视频输入源 ==========")
    logger.info(f"[Main] 扫描目录: {_INPUT_DIR}")

    video_files = _scan_input_dir()

    if not video_files:
        logger.info("[Main] input/ 目录为空，自动使用摄像头")
        logger.info(f"[Main] 提示：将视频文件放入 {_INPUT_DIR} 即可使用文件模式")
        return ""

    # 列出所有文件供用户选择
    logger.info(f"[Main] 发现 {len(video_files)} 个视频文件:")
    for i, path in enumerate(video_files, 1):
        size_mb = os.path.getsize(path) / 1024 / 1024
        logger.info(f"  [{i}] {os.path.basename(path)}  ({size_mb:.1f} MB)")
    logger.info(f"  [0] 摄像头实时拍摄")

    while True:
        try:
            choice = input(f"请选择 [0-{len(video_files)}]（直接回车使用摄像头）: ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("\n[Main] 使用默认：摄像头")
            return ""

        if choice == "" or choice == "0":
            logger.info("[Main] 已选择：摄像头")
            return ""

        if choice.isdigit() and 1 <= int(choice) <= len(video_files):
            selected = video_files[int(choice) - 1]
            logger.info(f"[Main] 已选择文件: {os.path.basename(selected)}")
            return selected

        logger.warning(f"[Main] 请输入 0 到 {len(video_files)} 之间的数字")


def initialize_video_streamer(file_path: str = ""):
    """初始化视频流处理模块。file_path 非空时使用文件模式。"""
    global video_streamer

    logger.info("[Main] ========== 初始化视频流 ==========")

    try:
        camera_id = _env_int("CAMERA_ID", 0)
        camera_backend = os.getenv("CAMERA_BACKEND", "auto")
        buffer_size = _env_int("BUFFER_SIZE", 15)
        sample_rate = _env_int("SAMPLE_RATE", 10)
        target_height = _env_int("TARGET_HEIGHT", 336)
        video_fps = _env_int("VIDEO_FPS", 30)
        pre_record_delay = _env_float("PRE_RECORD_DELAY", 5.0)
        record_duration = _env_float("RECORD_DURATION", 10.0)
        verbose = _env_bool("VERBOSE", True)

        video_streamer = VideoStreamer(
            camera_id=camera_id,
            buffer_size=buffer_size,
            sample_rate=sample_rate,
            target_height=target_height,
            fps=video_fps,
            camera_backend=camera_backend,
            pre_record_delay=pre_record_delay,
            record_duration=record_duration,
            verbose=verbose,
            file_path=file_path or None,
        )

        # 绑定回调函数
        video_streamer.on_analysis_trigger = on_analysis_request

        mode = f"文件模式: {file_path}" if file_path else "摄像头模式"
        logger.info(f"[Main] ✓ 视频流初始化完成（{mode}）")
        return True

    except Exception as e:
        logger.error(f"[Main] ❌ 视频流初始化失败: {e}")
        if not file_path:
            logger.error("[Main] 请检查摄像头是否连接并可用")
        return False


# ============================================================================
# 清理函数
# ============================================================================

def cleanup():
    """清理资源。"""
    global vlm_model, video_streamer
    
    logger.info("[Main] ========== 清理资源 ==========")
    
    if video_streamer:
        try:
            video_streamer.stop()
        except Exception as e:
            logger.error(f"[Main] 停止视频流失败: {e}")
    
    if vlm_model:
        try:
            vlm_model.unload_model()
        except Exception as e:
            logger.error(f"[Main] 卸载 VLM 失败: {e}")
    
    logger.info("[Main] ✓ 资源清理完成")


def _select_action_from_cli() -> str:
    """通过 CLI 交互让用户输入动作名称，支持任意动作。"""
    logger.info("[Main] 请输入本轮要做的动作名称（任意动作均可）")
    logger.info("[Main] 直接回车使用默认：深蹲")

    try:
        user_input = input("请输入动作名称（例如：深蹲 / 硬拉 / 哑铃飞鸟 / 引体向上）: ").strip()
    except EOFError:
        user_input = ""
    except KeyboardInterrupt:
        logger.info("\n[Main] 已取消输入，使用默认：深蹲")
        return "深蹲"

    action = user_input if user_input else "深蹲"
    logger.info(f"[Main] 已选择动作：{action}")
    return action


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主程序入口。"""
    model_name = os.getenv("VLM_MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")
    pre_record_delay = _env_float("PRE_RECORD_DELAY", 5.0)
    record_duration = _env_float("RECORD_DURATION", 10.0)
    feedback_duration = _env_float("FEEDBACK_DURATION", 4.0)
    
    logger.info("=" * 70)
    logger.info("   🏋️ Fitness Coach - 多模态健身动作指导系统 🏋️")
    logger.info("=" * 70)
    logger.info("")
    logger.info("[Main] 硬件信息:")
    logger.info("  - GPU: NVIDIA GeForce RTX 4060 (8GB VRAM)")
    logger.info(f"  - Model: {model_name} (4-bit quantization)")
    logger.info("  - OS: Windows 11")
    logger.info("")
    
    # 初始化 VLM
    if not initialize_vlm():
        logger.error("[Main] VLM 初始化失败，程序退出")
        return 1
    
    logger.info("")
    
    # 选择视频输入源
    input_file = _select_input_source()

    # 初始化视频流
    if not initialize_video_streamer(file_path=input_file):
        logger.error("[Main] 视频流初始化失败，程序退出")
        cleanup()
        return 1

    selected_action = _select_action_from_cli()
    video_streamer.current_action_type = selected_action
    video_streamer.set_feedback(f"动作类型: {selected_action}", duration=2.0)

    logger.info("")
    logger.info("=" * 70)
    if input_file:
        logger.info("[Main] ✓ 系统就绪！开始文件回放...")
    else:
        logger.info("[Main] ✓ 系统就绪！开始视频直播...")
    logger.info("=" * 70)
    logger.info("")
    logger.info("【快捷键说明】")
    if input_file:
        logger.info("  - 按 'S' 键: 立即触发分析（使用已播放部分的采样帧）")
        logger.info("  - 文件播放结束时自动触发分析")
    else:
        logger.info(f"  - 按 'S' 键: 先倒计时 {pre_record_delay:.1f} 秒，再录制 {record_duration:.1f} 秒并触发分析")
    logger.info("  - 按 'Q' 键: 退出程序")
    logger.info("")
    logger.info("【使用流程】")
    logger.info(f"  1. 已通过 CLI 选择动作类型：{selected_action}")
    if input_file:
        logger.info(f"  2. 视频文件将自动播放，结束后发送采样帧给 VLM")
        logger.info(f"  3. 分析建议将显示在画面上（{feedback_duration:.1f} 秒）")
    else:
        logger.info(f"  2. 按 'S' 后有 {pre_record_delay:.1f} 秒倒计时（走位准备）")
        logger.info(f"  3. 系统自动录制 {record_duration:.1f} 秒动作并发送给 VLM")
        logger.info(f"  4. 建议将显示在视频画面上（{feedback_duration:.1f} 秒）")
    logger.info("")
    
    try:
        # 启动视频流（阻塞，直到用户按 'Q' 退出）
        video_streamer.start()
    
    except KeyboardInterrupt:
        logger.info("[Main] 收到中断信号 (Ctrl+C)")
    
    except Exception as e:
        logger.error(f"[Main] 程序异常: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        cleanup()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("[Main] 程序已退出，感谢使用 Fitness Coach！")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
