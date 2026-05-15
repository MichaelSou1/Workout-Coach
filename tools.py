"""
Agentic loop 可调用的工具集。

每个工具签名：fn(frames: List[PIL.Image], **kwargs) -> str
返回纯文本，agent_loop 会把文本拼到下一轮 user message 的 Observation 中。
"""

import logging
import math
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    _mp_pose = mp.solutions.pose
except ImportError:
    _mp_pose = None
    logger.warning("[Tools] mediapipe 未安装，姿态相关工具将不可用")


# MediaPipe Pose 33 landmark 索引
_LM = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


def _get_pose_landmarks(pil_image: Image.Image) -> Optional[Dict[str, tuple]]:
    """返回归一化坐标的 landmark 字典；未检测到时返回 None。"""
    if _mp_pose is None:
        return None
    try:
        arr = np.asarray(pil_image.convert("RGB"))
        with _mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
            result = pose.process(arr)
        if not result.pose_landmarks:
            return None
        lms = result.pose_landmarks.landmark
        return {name: (lms[idx].x, lms[idx].y, lms[idx].visibility) for name, idx in _LM.items()}
    except Exception as e:
        logger.debug(f"[Tools] pose landmark 提取失败: {e}")
        return None


def _angle_at(a: tuple, b: tuple, c: tuple) -> float:
    """计算 b 点处由 a-b-c 形成的角度（度）。"""
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    cx, cy = c[0], c[1]
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos_t = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_t))


def _torso_lean(lm: Dict[str, tuple]) -> float:
    """躯干前倾角（相对垂直方向）。肩-髋中点连线与竖直轴夹角。"""
    ls, rs = lm["left_shoulder"], lm["right_shoulder"]
    lh, rh = lm["left_hip"], lm["right_hip"]
    shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    if dx == 0 and dy == 0:
        return float("nan")
    # 与垂直方向（向上）的夹角
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def _compute_angles(lm: Dict[str, tuple]) -> Dict[str, float]:
    return {
        "左膝": _angle_at(lm["left_hip"], lm["left_knee"], lm["left_ankle"]),
        "右膝": _angle_at(lm["right_hip"], lm["right_knee"], lm["right_ankle"]),
        "左髋": _angle_at(lm["left_shoulder"], lm["left_hip"], lm["left_knee"]),
        "右髋": _angle_at(lm["right_shoulder"], lm["right_hip"], lm["right_knee"]),
        "左肘": _angle_at(lm["left_shoulder"], lm["left_elbow"], lm["left_wrist"]),
        "右肘": _angle_at(lm["right_shoulder"], lm["right_elbow"], lm["right_wrist"]),
        "躯干前倾": _torso_lean(lm),
    }


# ---------------------------------------------------------------------------
# 对外工具：get_pose_angles
# ---------------------------------------------------------------------------

def get_pose_angles(frames: List[Image.Image], frame_indices: List[int]) -> str:
    """提取指定帧的关节角度，返回多行文本。"""
    if not frame_indices:
        return "未提供 frame_indices"

    lines = []
    for idx in frame_indices:
        if not isinstance(idx, int) or idx < 0 or idx >= len(frames):
            lines.append(f"第{idx}帧: 索引越界（有效范围 0-{len(frames) - 1}）")
            continue
        lm = _get_pose_landmarks(frames[idx])
        if lm is None:
            lines.append(f"第{idx}帧: 未检测到姿态")
            continue
        angles = _compute_angles(lm)
        parts = " ".join(
            f"{name}={int(round(val))}°" if not math.isnan(val) else f"{name}=未知"
            for name, val in angles.items()
        )
        lines.append(f"第{idx}帧: {parts}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 对外工具：detect_phase_boundaries
# ---------------------------------------------------------------------------

def _key_metric_for_action(action_type: str) -> str:
    """根据动作类型选择主导关节角度。"""
    action = (action_type or "").lower()
    if any(k in action for k in ["bench", "卧推", "推举", "肩推", "press"]):
        return "elbow"
    return "knee"


def detect_phase_boundaries(frames: List[Image.Image], action_type: str = "") -> str:
    """扫描全部帧，找出动作的关键相位边界。返回文本。"""
    if not frames:
        return "无帧可分析"

    metric = _key_metric_for_action(action_type)

    angle_series: List[Optional[float]] = []
    for frame in frames:
        lm = _get_pose_landmarks(frame)
        if lm is None:
            angle_series.append(None)
            continue
        angles = _compute_angles(lm)
        if metric == "elbow":
            val = _avg_finite(angles["左肘"], angles["右肘"])
        else:
            val = _avg_finite(angles["左膝"], angles["右膝"])
        angle_series.append(val if not (val is None or math.isnan(val)) else None)

    valid_indices = [i for i, v in enumerate(angle_series) if v is not None]
    if len(valid_indices) < 3:
        return f"姿态检测覆盖率不足({len(valid_indices)}/{len(frames)} 帧)，无法识别相位边界"

    valid_values = [angle_series[i] for i in valid_indices]
    min_pos = valid_indices[int(np.argmin(valid_values))]
    max_pos = valid_indices[int(np.argmax(valid_values))]
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1]

    bottom_label = "蹲底/底部位置" if metric == "knee" else "推起底部"
    top_label = "站立顶点/伸直位" if metric == "knee" else "推起顶点"
    metric_label = "膝关节" if metric == "knee" else "肘关节"

    events = []
    events.append((start_idx, f"起始姿态({metric_label}角度={int(round(angle_series[start_idx]))}°)"))
    events.append((min_pos, f"{bottom_label}({metric_label}最小角度={int(round(angle_series[min_pos]))}°)"))
    events.append((max_pos, f"{top_label}({metric_label}最大角度={int(round(angle_series[max_pos]))}°)"))
    if end_idx not in {start_idx, min_pos, max_pos}:
        events.append((end_idx, f"结束姿态({metric_label}角度={int(round(angle_series[end_idx]))}°)"))

    events.sort(key=lambda x: x[0])
    lines = ["动作相位边界:"]
    for idx, desc in events:
        lines.append(f"- 第{idx}帧: {desc}")
    return "\n".join(lines)


def _avg_finite(*vals) -> Optional[float]:
    finite = [v for v in vals if v is not None and not math.isnan(v)]
    if not finite:
        return None
    return sum(finite) / len(finite)


# ---------------------------------------------------------------------------
# 工具注册表 + 描述（用于注入 system prompt）
# ---------------------------------------------------------------------------

TOOLS: Dict[str, Callable[..., str]] = {
    "get_pose_angles": get_pose_angles,
    "detect_phase_boundaries": detect_phase_boundaries,
}


TOOL_DESCRIPTIONS = """可用工具：

1. get_pose_angles
   说明：测量指定帧的关节角度（膝、髋、肘、躯干前倾）。
   参数：{"frame_indices": [0, 3, 5]}  // 帧序号列表，从 0 开始
   返回：每帧的角度数值文本
   适用：需要量化某些时刻的关节角度时

2. detect_phase_boundaries
   说明：扫描全部帧，识别动作的关键相位边界（起始/底部/顶点/结束）。
   参数：{}  // 无参数
   返回：关键帧序号 + 该帧含义
   适用：需要定位动作哪一刻最值得细看时"""


def execute_tool(name: str, args: dict, frames: List[Image.Image], action_type: str = "") -> str:
    """统一工具调度入口，捕获异常并以文本形式返回。"""
    fn = TOOLS.get(name)
    if fn is None:
        return f"未知工具: {name}。可用工具: {', '.join(TOOLS.keys())}"

    try:
        if name == "get_pose_angles":
            frame_indices = args.get("frame_indices", [])
            if not isinstance(frame_indices, list):
                return "参数错误：frame_indices 必须是列表"
            return fn(frames, frame_indices=frame_indices)
        if name == "detect_phase_boundaries":
            return fn(frames, action_type=action_type)
        return fn(frames, **args)
    except Exception as e:
        logger.exception(f"[Tools] 工具 {name} 执行失败")
        return f"工具 {name} 执行错误: {e}"
