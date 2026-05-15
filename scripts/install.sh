#!/usr/bin/env bash
# ============================================================================
# Workout-Coach 安装脚本（顺序敏感）
# ============================================================================
# 安装顺序之所以重要：
#   - SGLang / FlashInfer 的 wheel 与具体 (CUDA 版本, PyTorch 主次版本) 强绑定
#   - 如果一次性 pip install 多个会让 pip 各自解版本，结果常互不兼容
#   - 顺序：torch → flashinfer → sglang → 项目 leaf 依赖
# ============================================================================
#
# 用法：
#   conda create -n workout-coach python=3.11 -y
#   conda activate workout-coach
#   bash scripts/install.sh                       # 默认 CUDA 12.1 / Torch 2.5
#   CUDA_VARIANT=cu118 TORCH_VERSION=2.4.1 bash scripts/install.sh
# ============================================================================

set -euo pipefail

CUDA_VARIANT="${CUDA_VARIANT:-cu121}"            # cu118 / cu121 / cu124
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCH_MAJOR_MINOR="${TORCH_VERSION%.*}"          # 例如 2.5
SGLANG_VERSION_SPEC="${SGLANG_VERSION_SPEC:-sglang[all]>=0.4.6}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "============================================================"
echo "[install] CUDA=${CUDA_VARIANT}  Torch=${TORCH_VERSION}  Python=$(${PYTHON_BIN} --version)"
echo "[install] SGLang spec: ${SGLANG_VERSION_SPEC}"
echo "============================================================"

# 防呆：要求在 conda 环境里跑，避免污染 base
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" == "base" ]]; then
    echo "[install] ❌ 必须在已激活的 conda 环境（非 base）中执行"
    echo "          conda create -n workout-coach python=3.11 -y && conda activate workout-coach"
    exit 1
fi

# ----------------------------------------------------------------------
# Step 1: PyTorch（决定后续所有 wheel 的 CUDA / ABI 一致性，必须最先装）
# ----------------------------------------------------------------------
echo ""
echo "[install] [1/4] 安装 PyTorch ${TORCH_VERSION} (${CUDA_VARIANT})..."
pip install --no-cache-dir \
    "torch==${TORCH_VERSION}" "torchvision" "torchaudio" \
    --index-url "https://download.pytorch.org/whl/${CUDA_VARIANT}"

# 验证 CUDA 可用
${PYTHON_BIN} -c "
import torch
assert torch.cuda.is_available(), 'CUDA 不可用'
print(f'[install] ✓ torch={torch.__version__}  cuda={torch.version.cuda}  device_count={torch.cuda.device_count()}')
"

# ----------------------------------------------------------------------
# Step 2: FlashInfer（SGLang 默认 attention backend，独立 wheel index）
# ----------------------------------------------------------------------
echo ""
echo "[install] [2/4] 安装 FlashInfer..."
pip install --no-cache-dir flashinfer-python \
    -i "https://flashinfer.ai/whl/${CUDA_VARIANT}/torch${TORCH_MAJOR_MINOR}/" \
    || {
        echo "[install] ⚠️ FlashInfer 从专用 index 安装失败，尝试从 PyPI 装通用版本..."
        pip install --no-cache-dir flashinfer-python
    }

# ----------------------------------------------------------------------
# Step 3: SGLang（依赖 transformers/triton 等，由 [all] 一起带入）
# ----------------------------------------------------------------------
echo ""
echo "[install] [3/4] 安装 SGLang..."
pip install --no-cache-dir "${SGLANG_VERSION_SPEC}"

# ----------------------------------------------------------------------
# Step 4: 项目 leaf 依赖（必须最后，否则可能反向覆盖 torch/sglang/numpy）
# ----------------------------------------------------------------------
echo ""
echo "[install] [4/4] 安装项目 leaf 依赖（requirements.txt）..."
pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------------------
# 收尾验证
# ----------------------------------------------------------------------
echo ""
echo "[install] 验证关键依赖版本..."
${PYTHON_BIN} - <<'PY'
import importlib
for pkg in ["torch", "sglang", "flashinfer", "transformers", "openai", "mediapipe", "cv2", "PIL"]:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "?")
        print(f"  {pkg:<14}{ver}")
    except Exception as e:
        print(f"  {pkg:<14}❌ 未安装: {e}")
PY

echo ""
echo "============================================================"
echo "[install] ✓ 完成"
echo "[install] 下一步:"
echo "  1) 下载模型: python download_model.py --model Qwen/Qwen3-VL-8B-Instruct --source modelscope"
echo "  2) 启动 SGLang server: bash scripts/launch_sglang.sh"
echo "  3) 另开终端运行客户端: python main.py"
echo "============================================================"
