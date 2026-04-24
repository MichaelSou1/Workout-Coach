"""
视频流处理模块 - 使用 OpenCV 捕获摄像头并进行帧采样缓存
支持多线程事件驱动的动作分析触发机制
"""

import cv2
import logging
import time
import os
from collections import deque
from threading import Thread, Event, Lock
from typing import Callable, Optional, List
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class VideoStreamer:
    """
    多线程视频流采样和缓存管理器。

    核心功能：
    - 实时捕获摄像头视频，或从本地视频文件读取
    - 维护固定长度的帧缓存队列（降低显存压力）
    - 监听键盘事件（按 'S' 触发分析）
    - 不阻塞主画面显示循环
    """

    def __init__(
        self,
        camera_id: int = 0,
        buffer_size: int = 15,
        sample_rate: int = 10,
        target_height: int = 336,
        fps: int = 30,
        camera_backend: Optional[str] = None,
        pre_record_delay: float = 5.0,
        record_duration: float = 10.0,
        verbose: bool = True,
        file_path: Optional[str] = None,
    ):
        """
        初始化视频流处理器。

        Args:
            camera_id: 摄像头 ID（0 为默认前置摄像头），file_path 存在时忽略
            buffer_size: 帧缓存队列的最大长度
            sample_rate: 每多少帧采样一次（如 10 表示每 10 帧采 1 帧）
            target_height: 缩放后图像的目标高度（像素），降低推理显存压力
            fps: 摄像头帧率目标（文件模式下忽略，使用文件自身 FPS）
            camera_backend: 摄像头后端（auto/dshow/msmf/ffmpeg/gstreamer）
            pre_record_delay: 按下 S 后开始录制前的倒计时秒数（仅摄像头模式）
            record_duration: 实际录制时长（秒）（仅摄像头模式）
            verbose: 是否输出详细日志
            file_path: 本地视频文件路径（mp4/avi/mov/mkv 等），若提供则切换为文件模式
        """
        self.camera_id = camera_id
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.target_height = target_height
        self.fps = fps
        self.camera_backend = (camera_backend or "auto").strip().lower()
        self.pre_record_delay = pre_record_delay
        self.record_duration = record_duration
        self.verbose = verbose
        self.file_path = file_path
        self.is_file_mode = bool(file_path)

        # 视频流状态
        self.cap = None
        self.is_running = False
        self.frame_buffer = deque(maxlen=buffer_size)
        self.frame_lock = Lock()
        self.frame_count = 0
        self.last_read_fail_warn_time = 0.0
        self.read_fail_warn_interval = float(os.getenv("CAMERA_READ_FAIL_WARN_INTERVAL", "1.0"))

        # 定时录制状态（仅摄像头模式使用）
        self.record_state = "idle"  # idle | countdown | recording
        self.countdown_end_time = 0.0
        self.recording_end_time = 0.0
        self.recorded_frames: List[np.ndarray] = []

        # 文件模式：全量采样帧存储
        self._file_sampled_frames: List[np.ndarray] = []
        self._file_eof = False  # 文件播放结束标志

        # 事件标志
        self.stop_event = Event()
        self.analysis_triggered = Event()

        # 回调函数
        self.on_analysis_trigger: Optional[Callable[[List[Image.Image], str], None]] = None

        # 动作类型状态（用户输入的原始动作名称）
        self.current_action_type = "深蹲"

        # 用户界面反馈
        self.feedback_text = ""
        self.feedback_time = 0.0
        self.feedback_duration = 3.0  # 显示 3 秒

        if verbose:
            logger.setLevel(logging.DEBUG)

        if self.is_file_mode:
            self._initialize_file()
        else:
            self._initialize_camera()

    def _initialize_file(self):
        """初始化本地视频文件源。"""
        logger.info(f"[Video] 打开视频文件: {self.file_path}")
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"视频文件不存在: {self.file_path}")

        cap = cv2.VideoCapture(self.file_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {self.file_path}")

        self.cap = cap
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        file_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / file_fps if file_fps > 0 else 0
        self._file_native_fps = file_fps
        logger.info(
            f"[Video] ✓ 视频文件已加载 - {width}x{height} @ {file_fps:.1f} FPS"
            f"  共 {total_frames} 帧 ({duration:.1f} 秒)"
        )

    def _resolve_backend_candidates(self) -> List[int]:
        """根据配置与平台生成后端候选列表。"""
        backend_map = {
            "auto": cv2.CAP_ANY,
            "any": cv2.CAP_ANY,
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
            "ffmpeg": cv2.CAP_FFMPEG,
            "gstreamer": cv2.CAP_GSTREAMER,
        }

        # 用户显式指定
        if self.camera_backend in backend_map and self.camera_backend not in {"auto", "any"}:
            return [backend_map[self.camera_backend]]

        # auto：Windows 优先 DSHOW，规避 MSMF 常见 grabFrame 问题
        if os.name == "nt":
            return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

        return [cv2.CAP_ANY]
    
    def _initialize_camera(self):
        """初始化摄像头。"""
        logger.info(f"[Video] 初始化摄像头 (ID: {self.camera_id})...")

        backend_names = {
            cv2.CAP_ANY: "ANY",
            cv2.CAP_DSHOW: "DSHOW",
            cv2.CAP_MSMF: "MSMF",
            cv2.CAP_FFMPEG: "FFMPEG",
            cv2.CAP_GSTREAMER: "GSTREAMER",
        }

        last_error = None
        for backend in self._resolve_backend_candidates():
            backend_name = backend_names.get(backend, str(backend))
            try:
                logger.info(f"[Video] 尝试后端: {backend_name}")
                cap = cv2.VideoCapture(self.camera_id, backend)

                if not cap.isOpened():
                    cap.release()
                    continue

                # 设置摄像头属性
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲以降低延迟

                # 预热读取，确保该后端可稳定出帧
                warmup_ok = False
                for _ in range(5):
                    ret, _ = cap.read()
                    if ret:
                        warmup_ok = True
                        break
                    time.sleep(0.03)

                if not warmup_ok:
                    logger.warning(f"[Video] 后端 {backend_name} 已打开但无法稳定取帧，尝试下一个后端")
                    cap.release()
                    continue

                self.cap = cap
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(self.cap.get(cv2.CAP_PROP_FPS))

                logger.info(f"[Video] ✓ 摄像头初始化成功 ({backend_name}) - {width}x{height} @ {fps} FPS")
                return

            except Exception as e:
                last_error = e
                logger.warning(f"[Video] 后端 {backend_name} 初始化失败: {e}")

        msg = f"无法打开摄像头 {self.camera_id}，请检查 CAMERA_ID / CAMERA_BACKEND 配置"
        if last_error:
            msg = f"{msg}: {last_error}"
        logger.error(f"[Video] 摄像头初始化失败: {msg}")
        raise RuntimeError(msg)
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """按短边缩放图像，保持宽高比，横竖屏均适用。"""
        h, w = frame.shape[:2]
        short = min(h, w)
        if short == self.target_height:
            return frame
        scale = self.target_height / short
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    def _frame_to_pil(self, frame: np.ndarray) -> Image.Image:
        """
        将 OpenCV BGR 图像转换为 PIL RGB 图像。
        
        Args:
            frame: OpenCV 图像数组
        
        Returns:
            PIL 图像
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)
    
    def _capture_loop(self):
        """
        后台循环：捕获、采样、缓存视频帧。
        在独立线程中运行，不阻塞主显示循环。
        """
        if self.is_file_mode:
            self._capture_loop_file()
        else:
            self._capture_loop_camera()

    def _capture_loop_camera(self):
        """摄像头模式的帧捕获循环。"""
        logger.info("[Video] 帧捕获循环启动（摄像头模式）...")

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()

            if not ret:
                now = time.time()
                if now - self.last_read_fail_warn_time >= self.read_fail_warn_interval:
                    logger.warning("[Video] 无法读取帧，跳过（将限频打印）")
                    self.last_read_fail_warn_time = now
                time.sleep(0.03)
                continue

            # 采样：每 sample_rate 帧采 1 帧
            self.frame_count += 1
            if self.frame_count % self.sample_rate != 0:
                continue

            resized_frame = self._resize_frame(frame)

            with self.frame_lock:
                self.frame_buffer.append(resized_frame)

            self._recording_state_step(resized_frame)

        logger.info("[Video] 帧捕获循环停止（摄像头模式）")

    def _capture_loop_file(self):
        """文件模式的帧捕获循环：按视频原始 FPS 逐帧送入缓冲区，并采样存储所有帧。"""
        logger.info("[Video] 帧捕获循环启动（文件模式）...")
        frame_interval = 1.0 / max(self._file_native_fps, 1.0)

        while not self.stop_event.is_set():
            t0 = time.time()
            ret, frame = self.cap.read()

            if not ret:
                # 文件播放结束
                logger.info("[Video] 视频文件播放完毕")
                self._file_eof = True
                break

            self.frame_count += 1
            resized_frame = self._resize_frame(frame)

            # 更新显示缓冲区（取最新帧用于预览）
            with self.frame_lock:
                self.frame_buffer.append(resized_frame)

            # 按 sample_rate 采样存储，供 VLM 分析
            if self.frame_count % self.sample_rate == 0:
                self._file_sampled_frames.append(resized_frame.copy())

            # 按原始 FPS 节奏播放
            elapsed = time.time() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("[Video] 帧捕获循环停止（文件模式）")
    
    def _display_loop(self):
        """
        主显示循环：实时展示视频并处理键盘输入。
        运行在主线程，响应用户交互。
        """
        logger.info("[Video] 显示循环启动...")
        if self.is_file_mode:
            logger.info("[Video] 快捷键: 按 'S' 立即触发分析 | 按 'Q' 退出（视频结束时自动触发分析）")
        else:
            logger.info("[Video] 快捷键: 按 'S' 触发分析 | 按 'Q' 退出")

        window_title = "Fitness Coach - File Playback" if self.is_file_mode else "Fitness Coach - Video Stream"

        while not self.stop_event.is_set():
            # 文件模式：检测播放结束后自动触发分析
            if self.is_file_mode and self._file_eof and not self.analysis_triggered.is_set():
                self._trigger_file_analysis()

            with self.frame_lock:
                if len(self.frame_buffer) == 0:
                    time.sleep(0.01)
                    continue
                display_frame = self.frame_buffer[-1].copy()

            # 在画面上绘制反馈文本
            if self.feedback_text and time.time() - self.feedback_time < self.feedback_duration:
                cv2.putText(
                    display_frame,
                    self.feedback_text,
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # 摄像头模式：显示录制倒计时/录制状态
            if not self.is_file_mode:
                now = time.time()
                if self.record_state == "countdown":
                    left = max(0.0, self.countdown_end_time - now)
                    status_text = f"倒计时: {left:.1f}s 后开始录制"
                    color = (0, 255, 255)
                elif self.record_state == "recording":
                    left = max(0.0, self.recording_end_time - now)
                    status_text = f"录制中: 剩余 {left:.1f}s"
                    color = (0, 165, 255)
                else:
                    status_text = ""
                    color = (255, 255, 255)

                if status_text:
                    cv2.putText(
                        display_frame,
                        status_text,
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            # 文件模式：显示文件名和已采样帧数
            if self.is_file_mode:
                file_name = os.path.basename(self.file_path)
                sampled_count = len(self._file_sampled_frames)
                file_info = f"文件: {file_name}  已采样: {sampled_count} 帧"
                cv2.putText(
                    display_frame,
                    file_info,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 220, 100),
                    1,
                    cv2.LINE_AA,
                )
                if self._file_eof:
                    cv2.putText(
                        display_frame,
                        "播放结束，分析中...",
                        (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )

            # 显示缓冲区状态
            buffer_status = f"缓冲帧: {len(self.frame_buffer)}/{self.buffer_size}"
            cv2.putText(
                display_frame,
                buffer_status,
                (10, display_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # 显示当前动作类型
            action_status = f"动作类型: {self.current_action_type}"
            cv2.putText(
                display_frame,
                action_status,
                (10, display_frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 220, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_title, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                logger.info("[Video] 用户按下 'Q'，准备退出...")
                self.stop()
                break

            elif key == ord('s') or key == ord('S'):
                logger.info("[Video] 用户按下 'S'，触发分析...")
                if self.is_file_mode:
                    self._trigger_file_analysis()
                else:
                    self._trigger_analysis()

        cv2.destroyAllWindows()
        logger.info("[Video] 显示循环停止")
    
    def _trigger_analysis(self):
        """
        触发动作分析：按下 S 后先倒计时，再录制固定时长并送入 VLM 推理。
        """
        if self.record_state in {"countdown", "recording"}:
            logger.warning("[Video] 已有录制任务在进行中，忽略本次 S")
            self.set_feedback("录制任务进行中，请稍候...", duration=1.5)
            return

        with self.frame_lock:
            if len(self.frame_buffer) == 0:
                logger.warning("[Video] 缓冲区为空，无法启动录制")
                self.set_feedback("缓冲区为空，请稍候...", duration=1.5)
                return

        self.record_state = "countdown"
        self.countdown_end_time = time.time() + self.pre_record_delay
        self.recorded_frames = []

        logger.info(f"[Video] 将在 {self.pre_record_delay:.1f} 秒后开始录制，录制时长 {self.record_duration:.1f} 秒")
        self.set_feedback(f"{self.pre_record_delay:.0f}秒后开始录制", duration=min(self.pre_record_delay, 3.0))

    def _trigger_file_analysis(self):
        """文件模式：将所有采样帧发送给 VLM 进行分析（仅触发一次）。"""
        if self.analysis_triggered.is_set():
            logger.warning("[Video] 文件分析已触发过，忽略重复请求")
            return

        frames_np = self._file_sampled_frames
        if not frames_np:
            logger.warning("[Video] 文件模式：没有采样到有效帧，无法分析")
            self.set_feedback("未采到有效帧，无法分析", duration=3.0)
            return

        self.analysis_triggered.set()
        frames_pil = [self._frame_to_pil(f) for f in frames_np]
        logger.info(f"[Video] 文件模式：发送 {len(frames_pil)} 帧给 VLM 推理...")

        if self.on_analysis_trigger:
            try:
                self.on_analysis_trigger(frames_pil, self.current_action_type)
                logger.info("[Video] ✓ 文件分析请求已提交")
            except Exception as e:
                logger.error(f"[Video] 文件分析触发失败: {e}")
                self.set_feedback(f"分析失败: {str(e)[:30]}")

    def _recording_state_step(self, sampled_frame: np.ndarray):
        """在采样循环中推进定时录制状态机。"""
        now = time.time()

        if self.record_state == "countdown":
            if now >= self.countdown_end_time:
                self.record_state = "recording"
                self.recording_end_time = now + self.record_duration
                self.recorded_frames = []
                logger.info(f"[Video] 开始录制，持续 {self.record_duration:.1f} 秒")
                self.set_feedback("开始录制...", duration=1.2)
            return

        if self.record_state == "recording":
            self.recorded_frames.append(sampled_frame.copy())

            if now >= self.recording_end_time:
                frames_np = self.recorded_frames
                self.recorded_frames = []
                self.record_state = "idle"

                if len(frames_np) == 0:
                    logger.warning("[Video] 录制结束但未采到有效帧")
                    self.set_feedback("录制失败：未采到有效帧", duration=2.0)
                    return

                frames_to_analyze = [self._frame_to_pil(frame) for frame in frames_np]
                logger.info(f"[Video] 录制完成，发送 {len(frames_to_analyze)} 帧给 VLM 推理...")
                logger.info(f"[Video] 当前动作类型: {self.current_action_type}")

                if self.on_analysis_trigger:
                    try:
                        self.on_analysis_trigger(frames_to_analyze, self.current_action_type)
                        logger.info("[Video] ✓ 分析请求已提交")
                    except Exception as e:
                        logger.error(f"[Video] 分析触发失败: {e}")
                        self.set_feedback(f"分析失败: {str(e)[:30]}")
    
    def set_feedback(self, text: str, duration: float = 3.0):
        """
        设置在视频画面上显示的反馈文本。
        
        Args:
            text: 要显示的文本（建议长度 < 40 字）
            duration: 显示时长（秒）
        """
        self.feedback_text = text
        self.feedback_time = time.time()
        self.feedback_duration = duration
        logger.info(f"[Video] 反馈: {text}")
    
    def start(self):
        """启动视频捕获和显示循环。"""
        if self.is_running:
            logger.warning("[Video] 视频流已在运行")
            return
        
        logger.info("[Video] 启动视频流...")
        self.is_running = True
        self.stop_event.clear()
        
        # 启动后台捕获线程
        capture_thread = Thread(target=self._capture_loop, daemon=True)
        capture_thread.start()
        
        # 在主线程启动显示循环
        try:
            self._display_loop()
        except KeyboardInterrupt:
            logger.info("[Video] 收到中断信号")
            self.stop()
    
    def stop(self):
        """停止视频流处理。"""
        logger.info("[Video] 停止视频流...")
        self.is_running = False
        self.stop_event.set()
        
        # 释放资源
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        logger.info("[Video] ✓ 视频流已停止")
    
    def get_buffered_frames(self) -> List[Image.Image]:
        """
        获取当前缓冲区中的所有帧（转换为 PIL 图像）。
        
        Returns:
            PIL 图像列表
        """
        with self.frame_lock:
            return [self._frame_to_pil(frame) for frame in self.frame_buffer]
