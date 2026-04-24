"""
动作分析提示词构建模块。

让 VLM 先识别画面中的实际动作，再与用户声称的动作对比，最后评判质量。
顺序指令结构，避免条件分支，适配 2B 小模型。
"""


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
