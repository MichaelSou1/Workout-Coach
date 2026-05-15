"""
动作分析提示词构建模块。

提供两套 prompt 构建器：
- build_prompts: 单轮推理（兼容旧调用路径）
- build_agentic_system_prompt / build_agentic_user_query: 用于 agent_loop 的 ReAct 提示
"""

from tools import TOOL_DESCRIPTIONS


def build_prompts(action_name: str):
    """
    根据用户声称的动作名称生成 system_prompt 与 user_query。

    Args:
        action_name: 用户输入的动作名称

    Returns:
        (system_prompt, user_query)
    """
    action_name = (action_name or "未知动作").strip() or "未知动作"

    system_prompt = (
        "你是专业力量训练教练，通过连续视频帧分析健身动作并给出技术指导。"
        "请只输出中文，禁止输出坐标、边界框、JSON 或代码。"
    )

    user_query = (
        f"用户声称正在做的动作是【{action_name}】。\n"
        "请按以下顺序完成分析，严格按格式逐段输出，不要添加其他内容：\n\n"
        "【画面动作】写出你在视频帧中观察到的实际动作名称。"
        f"若与【{action_name}】不符，在括号内注明，例如：硬拉（用户声称：{action_name}）\n"
        "【总体结论】一句话总结本组动作的整体质量\n"
        "【关键问题1】问题：...；原因：...；修正：...\n"
        "【关键问题2】问题：...；原因：...；修正：...\n"
        "【关键问题3】问题：...；原因：...；修正：...\n"
        "【下一组执行口令】给出 2-3 条可直接照做的简短口令"
    )

    return system_prompt, user_query


_FINAL_FORMAT = (
    "【画面动作】实际观察到的动作名称；若与用户声称不符，括号注明\n"
    "【总体结论】一句话总结本组动作整体质量\n"
    "【关键问题1】问题：...；原因：...；修正：...\n"
    "【关键问题2】问题：...；原因：...；修正：...\n"
    "【关键问题3】问题：...；原因：...；修正：...\n"
    "【下一组执行口令】2-3 条可直接照做的简短口令"
)


def build_agentic_system_prompt(action_name: str) -> str:
    """构建 agentic loop 用的 system prompt（含工具说明 + ReAct 协议 + 最终格式）。"""
    action_name = (action_name or "未知动作").strip() or "未知动作"

    return (
        "你是专业力量训练教练，通过连续视频帧分析健身动作并给出技术指导。"
        f"用户声称正在做的动作是【{action_name}】。\n\n"
        f"{TOOL_DESCRIPTIONS}\n\n"
        "你可以选择调用工具来获取量化数据，也可以仅凭画面直接给出最终回答。\n\n"
        "## 输出协议\n"
        "当你想调用工具时，必须严格按以下三行格式输出（且仅输出这三行，不要附加其他内容）：\n"
        "Thought: <你为什么需要这个工具，一句话>\n"
        "Action: <工具名称，必须是上方列出的之一>\n"
        "Action_Input: <一行 JSON，包含工具参数>\n\n"
        "示例：\n"
        "Thought: 需要确认蹲底时的膝关节角度\n"
        "Action: get_pose_angles\n"
        "Action_Input: {\"frame_indices\": [3, 4, 5]}\n\n"
        "当你已经掌握足够信息，给出最终回答时，不要再写 Action 行，"
        "直接按下方格式严格逐段输出最终结论：\n\n"
        f"{_FINAL_FORMAT}\n\n"
        "约束：只输出中文，禁止输出坐标、边界框或代码块。"
    )


def build_agentic_user_query(action_name: str, num_frames: int) -> str:
    """构建 agentic loop 用的初始 user query。"""
    action_name = (action_name or "未知动作").strip() or "未知动作"
    return (
        f"以下是按时间顺序排列的 {num_frames} 帧画面（每帧前已标注帧序号，从 0 开始）。\n"
        f"请分析用户的【{action_name}】动作。如需量化数据可调用工具；准备好后给出最终结论。"
    )
