"""
离线模型下载脚本（支持中国大陆镜像）

用途：
1) 预下载 Qwen2-VL-2B-Instruct 到本地缓存
2) 后续可在离线模式启动 main.py
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_MIRROR = "https://hf-mirror.com"


def _is_win_symlink_privilege_error(err: Exception) -> bool:
    err_text = str(err)
    return (
        "WinError 1314" in err_text
        or "所需的特权" in err_text
        or "required privilege" in err_text.lower()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download VLM model for offline startup")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument(
        "--cache-dir",
        default=str(Path("./hf_cache").resolve()),
        help="Model cache directory (HF_HOME)",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("HF_ENDPOINT", DEFAULT_MIRROR),
        help="Hugging Face endpoint/mirror, e.g. https://hf-mirror.com",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable mirror and use official endpoint",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Optional allow patterns, repeatable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 配置缓存目录（transformers/huggingface_hub 通用）
    os.environ["HF_HOME"] = str(cache_dir)

    # 关闭 hf_transfer，网络不稳定环境更稳
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    # 仅关闭告警，不影响实际下载逻辑
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if not args.no_mirror and args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
        os.environ["HUGGINGFACE_HUB_ENDPOINT"] = args.endpoint

    print("=" * 78)
    print("[download_model] 开始下载模型")
    print(f"[download_model] model        : {args.model}")
    print(f"[download_model] cache_dir    : {cache_dir}")
    print(f"[download_model] endpoint     : {os.getenv('HF_ENDPOINT', 'https://huggingface.co')}")
    print(f"[download_model] hf_transfer  : {os.getenv('HF_HUB_ENABLE_HF_TRANSFER')}")
    print("=" * 78)

    endpoint = os.getenv("HF_ENDPOINT")

    try:
        snapshot_path = snapshot_download(
            repo_id=args.model,
            cache_dir=str(cache_dir),
            local_files_only=False,
            allow_patterns=args.allow_pattern if args.allow_pattern else None,
            endpoint=endpoint,
        )
    except Exception as e:
        if _is_win_symlink_privilege_error(e):
            local_model_dir = cache_dir / "local_models" / args.model.replace("/", "--")
            local_model_dir.mkdir(parents=True, exist_ok=True)

            print("\n[download_model] 检测到 Windows 符号链接权限不足（WinError 1314）")
            print("[download_model] 自动切换到无符号链接模式下载...")
            print(f"[download_model] local_dir    : {local_model_dir}")

            try:
                snapshot_path = snapshot_download(
                    repo_id=args.model,
                    local_dir=str(local_model_dir),
                    local_dir_use_symlinks=False,
                    local_files_only=False,
                    allow_patterns=args.allow_pattern if args.allow_pattern else None,
                    endpoint=endpoint,
                )
            except TypeError:
                # 兼容未来版本参数变更
                snapshot_path = snapshot_download(
                    repo_id=args.model,
                    local_dir=str(local_model_dir),
                    local_files_only=False,
                    allow_patterns=args.allow_pattern if args.allow_pattern else None,
                    endpoint=endpoint,
                )
            except Exception as e2:
                print(f"[download_model] ❌ 无符号链接模式仍失败: {e2}")
                return 1

            print("\n[download_model] ✅ 无符号链接模式下载完成")
            print(f"[download_model] local model  : {local_model_dir}")
            print("\n[download_model] 建议离线启动配置（写入 .env）：")
            print(f"VLM_MODEL_NAME={local_model_dir}")
            print(f"HF_HOME={cache_dir}")
            print("VLM_LOCAL_FILES_ONLY=1")
            print("HF_HUB_OFFLINE=1")
            print("HF_DATASETS_OFFLINE=1")
            print("\n[download_model] 然后运行: python main.py")
            return 0

        print(f"[download_model] ❌ 下载失败: {e}")
        return 1

    print("\n[download_model] ✅ 下载完成")
    print(f"[download_model] snapshot path: {snapshot_path}")

    print("\n[download_model] 建议离线启动配置（写入 .env）：")
    print(f"VLM_MODEL_NAME={args.model}")
    print(f"HF_HOME={cache_dir}")
    print("VLM_LOCAL_FILES_ONLY=1")
    print("HF_HUB_OFFLINE=1")
    print("HF_DATASETS_OFFLINE=1")

    print("\n[download_model] 然后运行: python main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
