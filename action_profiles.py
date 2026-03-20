"""
动作分析显式模块：手动动作类型 + 动作专属提示词模板。

支持动作：
- squat（深蹲）
- deadlift（硬拉）
- bench_press（卧推）
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ActionProfile:
    key: str
    label: str
    system_prompt: str
    user_query_template: str


def _common_output_format() -> str:
    return (
        "回复格式：\n"
        "【动作】{action_label}\n"
        "【总体结论】一句话总结本组动作质量\n"
        "【关键问题1】问题 + 原因 + 修正（按`问题：...；原因：...；修正：...`）\n"
        "【关键问题2】问题 + 原因 + 修正（同上）\n"
        "【关键问题3】问题 + 原因 + 修正（同上）\n"
        "【下一组执行口令】给 2-3 条可直接照做的短口令\n"
        "约束：\n"
        "- 只输出上述 6 段，不要添加其他标题\n"
        "- 禁止输出坐标、边界框、关键点、JSON、代码\n"
        "- 结论要基于画面，不要复述检查清单原文"
    )


ACTION_PROFILES: Dict[str, ActionProfile] = {
    "squat": ActionProfile(
        key="squat",
        label="深蹲",
        system_prompt=(
            "你是专业力量训练教练。当前用户动作类型是【深蹲】。\n"
            "请重点检查：\n"
            "1) 下蹲深度与重心（髋部是否充分下沉，是否前倾过度）\n"
            "2) 膝盖轨迹（是否内扣，是否与脚尖方向一致）\n"
            "3) 脊柱中立与核心稳定（是否圆背/塌腰）\n"
            "4) 起身阶段发力顺序（髋膝协同，避免臀部先抬）\n"
            + _common_output_format().format(action_label="深蹲")
        ),
        user_query_template=(
            "请把这组连续视频帧视为一次【深蹲】动作，"
            "请给出至少 3 个关键问题，并按输出格式给出原因和可执行修正。"
        ),
    ),
    "deadlift": ActionProfile(
        key="deadlift",
        label="硬拉",
        system_prompt=(
            "你是专业力量训练教练。当前用户动作类型是【硬拉】。\n"
            "请重点检查：\n"
            "1) 起拉姿势（髋位、肩杠关系、脚中部受力）\n"
            "2) 脊柱中立与背部张力（是否圆背、耸肩、松背）\n"
            "3) 杠铃路径（是否贴腿直线上升）\n"
            "4) 锁定阶段（髋伸展是否完成，是否过度后仰）\n"
            + _common_output_format().format(action_label="硬拉")
        ),
        user_query_template=(
            "请把这组连续视频帧视为一次【硬拉】动作，"
            "请给出至少 3 个关键问题，并按输出格式给出原因和可执行修正。"
        ),
    ),
    "bench_press": ActionProfile(
        key="bench_press",
        label="卧推",
        system_prompt=(
            "你是专业力量训练教练。当前用户动作类型是【卧推】。\n"
            "请重点检查：\n"
            "1) 肩胛稳定与胸椎姿势（是否耸肩、肩前顶）\n"
            "2) 杠铃落点与轨迹（是否落在合理胸线，推起路径是否稳定）\n"
            "3) 肘腕对齐与前臂垂直（是否手腕后折、肘外展过大）\n"
            "4) 下肢与核心稳定（脚跟驱动与全身张力）\n"
            + _common_output_format().format(action_label="卧推")
        ),
        user_query_template=(
            "请把这组连续视频帧视为一次【卧推】动作，"
            "请给出至少 3 个关键问题，并按输出格式给出原因和可执行修正。"
        ),
    ),
}


ACTION_ALIASES: Dict[str, str] = {
    "深蹲": "squat",
    "squat": "squat",
    "硬拉": "deadlift",
    "deadlift": "deadlift",
    "卧推": "bench_press",
    "bench press": "bench_press",
    "bench_press": "bench_press",
}


def normalize_action_type(action_type: str) -> str:
    """将用户输入动作归一化为内置 key 或原始自定义动作名。"""
    text = (action_type or "").strip()
    if not text:
        return "squat"

    lowered = text.lower()
    return ACTION_ALIASES.get(text, ACTION_ALIASES.get(lowered, text))


def _build_custom_profile(action_label: str) -> ActionProfile:
    """为自定义动作动态构建通用分析模板。"""
    label = (action_label or "自定义动作").strip() or "自定义动作"
    return ActionProfile(
        key=f"custom:{label}",
        label=label,
        system_prompt=(
            f"你是专业力量训练教练。当前用户动作类型是【{label}】。\n"
            "请重点检查：\n"
            "1) 起始姿势是否稳定（站姿/躯干/关节对齐）\n"
            "2) 动作轨迹是否可控（是否偏移、晃动、代偿）\n"
            "3) 关键关节是否安全（避免塌腰、耸肩、膝内扣等）\n"
            "4) 发力节奏是否合理（离心与向心控制、呼吸配合）\n"
            + _common_output_format().format(action_label=label)
        ),
        user_query_template=(
            f"请把这组连续视频帧视为一次【{label}】动作，"
            "请给出至少 3 个关键问题，并按输出格式给出原因和可执行修正。"
        ),
    )


def get_action_profile(action_type: str) -> ActionProfile:
    """获取动作模板；未知动作将动态构建自定义模板。"""
    normalized = normalize_action_type(action_type)
    if normalized in ACTION_PROFILES:
        return ACTION_PROFILES[normalized]
    return _build_custom_profile(normalized)


def build_prompts(action_type: str):
    """根据动作类型生成 system_prompt 与 user_query。"""
    profile = get_action_profile(action_type)
    return profile.system_prompt, profile.user_query_template
