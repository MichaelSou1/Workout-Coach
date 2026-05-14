# Workout-Coach 开发方向

## 背景

当前推理侧是单次 inference：视频帧 → `model.generate()` → 文本输出，没有多步推理或工具调用。

---

## 方向一：推理侧 Agentic 化

### 短期（直接改 `main.py` + `vlm_inference.py`）

**两阶段迭代分析**
- 第一次推理：粗扫全部帧（4帧），输出"哪个阶段/时间段最有问题"
- 根据输出选出关键帧区间（真正的 agentic 决策）
- 第二次推理：只看关键帧，做深度分析 + 口令生成
- 代价：延迟 +2-3s，质量提升明显

**自我反思 + 格式验证循环**（改 `main.py:on_vlm_result()`）
- 生成完成后自动检查【关键问题1/2/3】【下一组口令】是否齐全
- 格式不合格 → 重新生成（最多 N 次），带 constraint prompt
- 成本极低，解决格式缺失问题

**工具调用**（给模型配置轻量工具集）
- `get_exercise_standards(action)` — 查动作标准参数
- `get_session_history()` — 读取历史分析记录（本地 JSON）
- `sample_critical_frames(timestamps)` — 按需重采帧

### 中期

**跨 Set 记忆**（本地 session JSON）
- 记录 recurring_errors、每 set 问题、已改善项
- 注入 system prompt："你已提醒含胸 3 次，这次重点膝关节"
- 模型从单次分析员 → 有记忆的私人教练

---

## 方向二：Agentic SFT（训练侧）

### 核心洞察

Fitness-AQA 的时间戳 = **oracle 中间行动**，普通 SFT 把它丢掉了，Agentic SFT 把它变成中间步骤的监督信号。

### 训练数据格式转换

把每条数据从单步：
```
输入: [均匀采样帧]
输出: "【关键问题1】膝关节内扣..."
```

改造成多步轨迹：
```
输入: [粗采 4 帧]

<think>观察整体动作节奏，识别运动阶段</think>
<action>identify_phases</action>
<observation>下降 0-2.1s，底部 2.1-2.8s，上升 2.8-4.5s</observation>

<think>底部转换最容易出问题</think>
<action>sample_frames(timestamps=[1.8, 2.1, 2.4])</action>  ← 时间戳作 GT
<observation>[3张关键帧]</observation>

<think>在2.1s膝关节明显内扣</think>
<action>generate_coaching</action>

最终输出: "【关键问题】膝关节内扣..."
```

推理时**没有时间戳**，但模型学会了主动决定"去哪里看"。

### 轨迹数据生成方案

1. **规则模板 + timestamps 自动生成**（最快，bootstrap 用）
   - 用 `prepare_dataset.py` 扩展，自动填充 `<action>sample_frames(timestamps=[...])</action>`
2. **大模型扩写**（质量更高）
   - 用 Qwen2.5-72B 把 `(video, timestamps, errors)` 扩写成自然语言轨迹
   - 用 timestamps 验证中间步骤正确性

### 与现有训练流程的关系

```
Agentic SFT（时间戳构造轨迹）
    ↓  学会"分步推理"基础能力
GRPO（AI Judge 对完整轨迹打分）
    ↓  升级为 Process-level GRPO
推理时的 agentic 行为（无需时间戳）
```

关键文件：
- 轨迹构造逻辑加在 `training/prepare_dataset.py`
- 训练格式改动影响 `training/data_builder.py:build_messages()`
- GRPO 打分对象从单步输出变为完整轨迹

---

## 实施优先级

| 优先级 | 任务 | 涉及文件 |
|--------|------|---------|
| ⭐⭐⭐ | 两阶段推理 + 格式验证循环 | `main.py`, `vlm_inference.py` |
| ⭐⭐⭐ | Agentic SFT 轨迹数据构造 | `training/prepare_dataset.py`, `training/data_builder.py` |
| ⭐⭐ | Session 记忆 + 工具调用 | `main.py`, `action_profiles.py` |
| ⭐ | Process-level GRPO | `training/train_grpo.py` |
