#!/usr/bin/env bash
# ============================================================================
# 启动 SGLang server（OpenAI 兼容 API）承载 Qwen3-VL 推理
# ============================================================================
# 用法：
#   bash scripts/launch_sglang.sh
#
# 调整参数（环境变量）：
#   VLM_MODEL_NAME   模型 ID 或本地路径
#   SGLANG_HOST      监听地址
#   SGLANG_PORT      监听端口
#   SGLANG_DP_SIZE   数据并行副本数（每张 GPU 一份），≤ 可用 GPU 数
#   SGLANG_TP_SIZE   张量并行（默认 1；想把一个模型横跨多卡时设 ≥2）
#   SGLANG_MEM_FRACTION  静态显存占用比例（0.85 = 留 15% 给 KV cache 弹性）
#   SGLANG_QUANTIZATION  none / fp8 / awq / int4 等
#   SGLANG_EXTRA_ARGS  额外 server 参数，直接拼接
#
# 多卡示例（4× 单卡装下完整模型，做 DP）：
#   SGLANG_DP_SIZE=4 bash scripts/launch_sglang.sh
#
# 单卡装不下时（TP=2 横跨两卡 + DP=2 拷贝两组）：
#   SGLANG_TP_SIZE=2 SGLANG_DP_SIZE=2 bash scripts/launch_sglang.sh
# ============================================================================

set -euo pipefail

MODEL_PATH="${VLM_MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
HOST="${SGLANG_HOST:-127.0.0.1}"
PORT="${SGLANG_PORT:-30000}"
DP_SIZE="${SGLANG_DP_SIZE:-1}"
TP_SIZE="${SGLANG_TP_SIZE:-1}"
MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"
QUANTIZATION="${SGLANG_QUANTIZATION:-}"
EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"

echo "============================================================"
echo "[launch_sglang] model=${MODEL_PATH}"
echo "[launch_sglang] host=${HOST}  port=${PORT}"
echo "[launch_sglang] dp_size=${DP_SIZE}  tp_size=${TP_SIZE}"
echo "[launch_sglang] mem_fraction=${MEM_FRACTION}"
[[ -n "${QUANTIZATION}" ]] && echo "[launch_sglang] quantization=${QUANTIZATION}"
echo "============================================================"

CMD=(
    python -m sglang.launch_server
    --model-path "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --dp-size "${DP_SIZE}"
    --tp-size "${TP_SIZE}"
    --mem-fraction-static "${MEM_FRACTION}"
    --mm-attention-backend fa3
)

if [[ -n "${QUANTIZATION}" ]]; then
    CMD+=(--quantization "${QUANTIZATION}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    CMD+=(${EXTRA_ARGS})
fi

echo "[launch_sglang] cmd: ${CMD[*]}"
echo ""
exec "${CMD[@]}"
