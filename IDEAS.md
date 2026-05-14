# Workout-Coach 开发方向

## 背景

当前推理侧是单次 inference：视频帧 → `model.generate()` → 文本输出，没有多步推理或工具调用。

核心策略：**推理轻、训练深**
- 推理侧只加姿态估计做关键帧选择，不堆额外模型
- 训练侧用 Agentic SFT + Process-level GRPO，这是真正的差异化

---

## 改造 Plan

### Phase 1：推理侧轻量 Agentic 化

**目标：** 单步推理 → 两阶段推理，推理时间控制在 +3s 以内

#### 1.1 新增 `pose_selector.py`

封装 MediaPipe Pose，提供关键帧选择能力。

```python
# pose_selector.py（新建）
def select_key_frames(frames: List[np.ndarray], action_type: str) -> List[int]:
    """
    返回关键帧索引列表。
    原理：提取关节角度曲线 → 找极值点（动作相位边界）
    """
```

- 依赖：`mediapipe`（加入 requirements.txt）
- 无需 GPU，CPU 约 5-10ms/帧
- 输出帧索引，由 `vlm_inference.py` 取对应 PIL Image

#### 1.2 修改 `action_profiles.py`

拆分现有 `build_prompts()` 为两阶段：

```python
# 现有（保留）
build_prompts(action_name) -> (system_prompt, user_query)

# 新增：第一阶段粗扫描
build_first_pass_prompt(action_name) -> (system_prompt, user_query)
# prompt 内容：识别动作阶段 + 指出哪个区间最有问题 + 返回帧编号列表

# 修改：第二阶段精分析（接收粗扫描结论）
build_second_pass_prompt(action_name, phase_summary: str) -> (system_prompt, user_query)
# phase_summary 注入 system prompt："第一阶段发现底部转换有问题"
```

#### 1.3 修改 `vlm_inference.py`

在现有 `analyze_fitness_frames()` 基础上新增两阶段方法：

```python
# vlm_inference.py 新增方法（约 +60 行）
def analyze_fitness_frames_two_pass(
    self,
    all_frames: List[Image],       # 全部帧（粗采）
    key_frames: List[Image],       # pose_selector 选出的关键帧
    system_prompt_1: str,
    user_query_1: str,
    system_prompt_2: str,
    user_query_2: str,
    callback: Callable,
) -> None:
    # 第一次推理：all_frames + 粗扫 prompt
    # 解析输出，提取 phase_summary
    # 第二次推理：key_frames + 精分析 prompt（含 phase_summary）
    # 完成后触发 callback
```

不改动现有同步/异步推理方法，保持向后兼容。

#### 1.4 修改 `main.py`

**`on_analysis_request()`（行 127-168）**：
- 构建粗采帧列表（4帧）和 pose_selector 关键帧列表
- 调用 `analyze_fitness_frames_two_pass()` 替换原调用

**`on_vlm_result()`（行 93-124）**：
- 新增 format validation loop：检查【关键问题】【下一组口令】是否存在
- 不合格 → 触发补充生成（带 constraint prompt），最多重试 2 次

**新增 `SessionMemory` 类（~30 行）**：
- 本地 JSON 文件（`session_history.json`）
- `update(action, errors)` / `get_context()` → 注入 system prompt

---

### Phase 2：Agentic SFT 数据生成

**目标：** 把 Fitness-AQA 的 `(视频, 时间戳, 错误描述)` 转成多步轨迹训练数据

#### 2.1 新增 `training/build_agentic_trajectories.py`

核心转换逻辑，输入 `train.json`，输出 `train_agentic.json`：

```python
# 输入（现有 train.json 格式）
{
  "video_path": "...",
  "error_timestamps": [1.8, 2.1, 2.4],   # 秒
  "errors": ["膝关节内扣", "重心前移"],
  "gt_response": "【关键问题1】..."
}

# 输出（train_agentic.json 格式）
{
  "video_path": "...",
  "trajectory": [
    {
      "role": "assistant",
      "content": "<think>观察整体动作节奏</think>\n<action>identify_phases</action>"
    },
    {
      "role": "tool",
      "content": "<observation>下降 0-2.1s，底部 2.1-2.8s，上升 2.8-4.5s</observation>"
      # 由 pose_selector 离线运行生成，或规则模板填充
    },
    {
      "role": "assistant",
      "content": "<think>底部转换需要精看</think>\n<action>sample_frames([1.8, 2.1, 2.4])</action>"
      # timestamps 直接填入，作为 oracle 中间行动
    },
    {
      "role": "tool",
      "content": "<observation>[关键帧图像占位]</observation>"
    },
    {
      "role": "assistant",
      "content": "【关键问题1】膝关节内扣..."   # 原 gt_response
    }
  ],
  "key_frame_timestamps": [1.8, 2.1, 2.4]   # 用于 Phase 3 评分
}
```

两步生成策略：
1. **快速 bootstrap**：规则模板自动填充 think/action/observation 文本
2. **质量提升**：用 Qwen2.5-72B 把模板内容改写成自然语言（离线，跑一次）

#### 2.2 修改 `training/data_builder.py`

新增 `AgenticFitnessDataset`，平行于现有 `FitnessAQADataset`：

```python
class AgenticFitnessDataset(Dataset):
    # 加载 train_agentic.json
    # __getitem__ 返回完整轨迹的 messages

def build_agentic_messages(trajectory, frames_by_step) -> List[dict]:
    # 在轨迹中正确位置插入图像帧
    # tool observation 步骤插入对应的 PIL Image

def tokenize_trajectory(messages, processor, tokenizer) -> dict:
    # 多 turn label masking：
    # 只有 role=assistant 的 turn 计算 loss
    # role=tool 的 observation 不计算 loss（但保留在输入）
```

关键点：每个 assistant turn 都计算 loss（不只是最后一步），这是 Agentic SFT 和普通 SFT 的核心区别。

#### 2.3 新增 `training/train_sft_agentic.py`

复用 `train_sft.py` 的大部分配置，替换 Dataset 类：

```python
# 主要差异（相比 train_sft.py）
dataset = AgenticFitnessDataset(train_agentic_path, ...)  # 替换
# LoRA 配置、DeepSpeed 配置、TrainingArguments 保持不变
# max_seq_length 需要适当增大（轨迹比单步输出长 ~3x）
```

---

### Phase 3：Process-level GRPO

**目标：** GRPO 的评分对象从单步输出升级为完整轨迹

#### 3.1 修改 `training/train_grpo.py`

**`GRPOFitnessDataset`**：
- prompt 格式改为轨迹起始（包含粗采帧 + 第一步 observation）
- 模型 rollout 生成完整轨迹（think/action/observation/最终输出）

**`AIJudgeReward`（行 162-222）扩展**：

```python
# 新增评分维度
def score_trajectory(trajectory_text, gt_timestamps, gt_errors) -> float:
    # 原有维度（保留，降权）：错误识别、修正建议、格式规范
    # 新增维度：
    #   frame_selection_quality：模型选的时间戳是否靠近 gt_timestamps（±0.5s）
    #   reasoning_coherence：think 内容是否和 action 一致（LLM judge）
    # 加权求和
```

**reward_fn（行 229-244）**：
- 输入改为完整轨迹文本（含 think/action/observation）
- 把 `key_frame_timestamps`（来自 train_agentic.json）传给 judge

---

### Phase 4：Benchmark

**目标：** 可量化地对比单步 baseline vs agentic 各阶段的提升

#### 4.1 新建 `eval/` 目录

```
eval/
├── run_benchmark.py      # 主评估脚本
├── metrics.py            # 各指标计算
└── judge_prompts.py      # 评估用的 judge prompt（与训练 judge 分离）
```

#### 4.2 `eval/metrics.py`

```python
def error_detection_f1(pred_text, gt_errors) -> dict:
    # 用 LLM judge 判断 pred 是否覆盖每个 gt error
    # 返回 precision / recall / f1

def temporal_hit_rate(selected_timestamps, gt_timestamps, window=0.5) -> float:
    # 模型选的帧中，有多少在 gt_timestamps ±window 秒内

def coaching_quality(pred_text, gt_errors) -> float:
    # LLM judge 评分：动作识别(2) + 错误覆盖(3) + 修正可执行(3) + 格式(2)
    # 复用 train_grpo.py 的评分标准，但 judge 模型独立，避免泄漏
```

#### 4.3 `eval/run_benchmark.py`

对比实验矩阵：

| 配置 | 说明 |
|------|------|
| baseline | 原始 Qwen2-VL-2B，单步推理 |
| +pose_select | 加姿态估计关键帧选择，单步推理 |
| +two_pass | 两阶段推理，pose 选帧 |
| sft | SFT 微调后，单步推理 |
| sft_agentic | Agentic SFT，两阶段推理 |
| sft_agentic+grpo | Agentic SFT + Process GRPO |

---

## 文件改动总览

```
Workout-Coach/
├── pose_selector.py              # 新建：MediaPipe 关键帧选择
├── action_profiles.py            # 修改：拆分两阶段 prompt
├── vlm_inference.py              # 修改：新增两阶段推理方法
├── main.py                       # 修改：两阶段调用 + 格式验证 + session 记忆
├── requirements.txt              # 修改：加 mediapipe
│
├── training/
│   ├── build_agentic_trajectories.py   # 新建：轨迹数据生成
│   ├── data_builder.py                 # 修改：新增 AgenticFitnessDataset
│   ├── train_sft_agentic.py            # 新建：Agentic SFT 训练脚本
│   └── train_grpo.py                   # 修改：轨迹级评分
│
└── eval/
    ├── run_benchmark.py          # 新建：评估主脚本
    ├── metrics.py                # 新建：F1、Temporal Hit Rate、口令质量
    └── judge_prompts.py          # 新建：评估用 judge prompt
```

---

## 实施顺序

```
Phase 1（推理侧，1-2天）
    → 可以立即看到效果，验证两阶段推理的价值
    → pose_selector → action_profiles → vlm_inference → main

Phase 2（数据生成，2-3天）
    → 最关键的一步，决定训练质量上限
    → build_agentic_trajectories → data_builder → train_sft_agentic

Phase 4（Benchmark，穿插进行）
    → Phase 1 完成后就开始建 eval 框架，边做边量化

Phase 3（GRPO，最后）
    → 依赖 Phase 2 的模型作为初始化
    → 修改 train_grpo.py 支持轨迹级评分
```
