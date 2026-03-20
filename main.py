"""
主程序入口 - Fitness Coach 多模态健身动作指导系统
整合 VLM 推理、视频流处理、UI 反馈的完整工作流
"""

import logging
import sys
import os
from typing import List

from PIL import Image

from action_profiles import build_prompts, get_action_profile
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
    """
    VLM 推理完成后的回调函数。
    将建议显示在视频画面上。
    
    Args:
        result_text: VLM 生成的文本建议
    """
    global is_analyzing, video_streamer
    
    logger.info(f"[Main] VLM 返回结果:\n{result_text}")
    
    # 在视频画面上显示反馈（提取关键信息）
    feedback_duration = _env_float("FEEDBACK_DURATION", 4.0)
    if video_streamer:
        # 简化显示：取前 50 字符
        display_text = result_text[:50] if result_text else "分析中..."
        if "【总体结论】" in result_text:
            # 提取总体结论
            try:
                summary_start = result_text.index("【总体结论】") + len("【总体结论】")
                summary_end = result_text.index("【关键问题1】") if "【关键问题1】" in result_text else len(result_text)
                display_text = result_text[summary_start:summary_end].strip()[:45]
            except Exception:
                pass
        elif "【问题】" in result_text:
            # 兼容旧格式：提取问题部分
            try:
                problem_start = result_text.index("【问题】") + len("【问题】")
                problem_end = result_text.index("【原因】") if "【原因】" in result_text else len(result_text)
                display_text = result_text[problem_start:problem_end].strip()[:45]
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
    profile = get_action_profile(action_type)
    system_prompt, user_query = build_prompts(action_type)

    logger.info(f"[Main] 开始异步推理 {len(frames)} 帧，动作类型: {profile.label}")
    video_streamer.set_feedback(f"分析中: {profile.label}", duration=2.0)
    
    max_new_tokens = _env_int("VLM_MAX_TOKENS", 256)

    # 异步推理
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


def initialize_video_streamer():
    """初始化视频流处理模块。"""
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
        )
        
        # 绑定回调函数
        video_streamer.on_analysis_trigger = on_analysis_request
        
        logger.info("[Main] ✓ 视频流初始化完成")
        return True
    
    except Exception as e:
        logger.error(f"[Main] ❌ 视频流初始化失败: {e}")
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
    """通过 CLI 交互让用户输入动作名称。"""
    logger.info("[Main] 请选择本轮分析动作（支持自定义，例如：哑铃飞鸟）")
    logger.info("[Main] 直接回车使用默认动作：深蹲")

    while True:
        try:
            user_input = input("请输入动作名称（示例：深蹲/硬拉/卧推/哑铃飞鸟）: ").strip()
        except EOFError:
            user_input = ""
        except KeyboardInterrupt:
            logger.info("\n[Main] 已取消输入，使用默认动作：深蹲")
            return "squat"

        if not user_input:
            profile = get_action_profile("squat")
            logger.info(f"[Main] 使用默认动作：{profile.label}")
            return profile.label

        profile = get_action_profile(user_input)
        logger.info(f"[Main] 已选择动作：{profile.label}")
        return profile.label


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
    
    # 初始化视频流
    if not initialize_video_streamer():
        logger.error("[Main] 视频流初始化失败，程序退出")
        cleanup()
        return 1

    selected_action = _select_action_from_cli()
    video_streamer.current_action_type = selected_action
    video_streamer.set_feedback(f"动作类型: {selected_action}", duration=2.0)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("[Main] ✓ 系统就绪！开始视频直播...")
    logger.info("=" * 70)
    logger.info("")
    logger.info("【快捷键说明】")
    logger.info(f"  - 按 'S' 键: 先倒计时 {pre_record_delay:.1f} 秒，再录制 {record_duration:.1f} 秒并触发分析")
    logger.info("  - 按 'Q' 键: 退出程序")
    logger.info("")
    logger.info("【使用流程】")
    logger.info(f"  1. 已通过 CLI 选择动作类型：{selected_action}")
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
