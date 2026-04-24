"""
VLM 推理核心类 - 基于 Qwen2-VL-2B-Instruct 的多模态推理模块
特别针对 8GB VRAM 的 Windows 环境优化
"""

import gc
import torch
import logging
import os
from typing import List, Optional
from PIL import Image
from threading import Thread, Lock
from queue import Queue

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers.utils.quantization_config import BitsAndBytesConfig

logger = logging.getLogger(__name__)


class FitnessVLM:
    """
    专门用于健身动作分析的多模态视觉语言模型推理类。
    
    特性：
    - 4-bit 量化加载以极限降低显存占用
    - 异步推理接口，不阻塞主线程
    - 自动显存管理和垃圾回收
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cuda",
        verbose: bool = True,
        hf_endpoint: Optional[str] = None,
        local_files_only: bool = False,
        cache_dir: Optional[str] = None,
        use_flash_attention_2: bool = True,
    ):
        """
        初始化 VLM 模型。
        
        Args:
            model_name: Hugging Face 模型名称
            device: 推理设备 ('cuda' 或 'cpu')
            verbose: 是否输出详细日志
        """
        self.model_name = model_name
        self.device = device
        self.verbose = verbose
        self.hf_endpoint = hf_endpoint
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self.use_flash_attention_2 = use_flash_attention_2
        self.model = None
        self.processor = None
        self.inference_lock = Lock()  # 确保显存操作线程安全
        self.inference_queue = Queue()  # 异步推理队列
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        self._load_model()
    
    def _load_model(self):
        """
        使用 4-bit 量化加载模型到 GPU。
        关键点：严格控制显存占用到 8GB 以内。
        """
        logger.info(f"[VLM] 开始加载模型: {self.model_name} (4-bit 量化)")

        # 配置 Hugging Face 端点（支持中国大陆镜像）
        endpoint = self.hf_endpoint or os.getenv("HF_ENDPOINT") or os.getenv("HUGGINGFACE_HUB_ENDPOINT")
        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint
            os.environ["HUGGINGFACE_HUB_ENDPOINT"] = endpoint
            logger.info(f"[VLM] 使用 Hugging Face 端点: {endpoint}")

        # 某些网络环境下关闭 hf_transfer 可提升稳定性
        if not os.getenv("HF_HUB_ENABLE_HF_TRANSFER"):
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        
        # 清理显存
        self._clear_cache()
        
        try:
            # 配置 4-bit 量化
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            
            # 加载处理器
            logger.info("[VLM] 加载 Processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
                cache_dir=self.cache_dir,
            )
            
            # 加载模型
            logger.info("[VLM] 加载模型（4-bit 量化）...")
            model_kwargs = {
                "quantization_config": quantization_config,
                "device_map": "auto",
                "trust_remote_code": True,
                "local_files_only": self.local_files_only,
                "cache_dir": self.cache_dir,
                "low_cpu_mem_usage": True,  # 权重先在 CPU 处理再搬 GPU，消除加载峰值
            }

            if self.use_flash_attention_2:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            try:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    **model_kwargs,
                )
            except Exception as e:
                err = str(e).lower()
                if self.use_flash_attention_2 and ("flashattention" in err or "flash_attn" in err):
                    logger.warning("[VLM] FlashAttention2 不可用，自动回退到默认注意力实现（SDPA/Eager）")
                    model_kwargs.pop("attn_implementation", None)
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        self.model_name,
                        **model_kwargs,
                    )
                else:
                    raise
            
            logger.info("[VLM] ✓ 模型加载完成!")
            self._log_gpu_memory()
            
        except Exception as e:
            logger.error(f"[VLM] 模型加载失败: {e}")
            raise
    
    def _clear_cache(self):
        """清理 GPU 和系统内存缓存。"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    def _log_gpu_memory(self):
        """记录当前 GPU 显存使用情况。"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"[VLM] GPU 显存 - 已分配: {allocated:.2f}GB / 预留: {reserved:.2f}GB")
    
    def analyze_fitness_frames(
        self,
        frames: List[Image.Image],
        system_prompt: str,
        user_query: str = "请分析这些帧中的动作问题。",
        max_new_tokens: int = 256,
    ) -> str:
        """
        同步推理方法：分析一组连续视频帧并返回文本建议。
        
        Args:
            frames: PIL Image 列表，代表连续的视频帧
            system_prompt: 系统指令（角色定义和任务说明）
            user_query: 用户查询文本
            max_new_tokens: 生成的最大 token 数
        
        Returns:
            模型生成的文本建议
        """
        if not frames:
            logger.warning("[VLM] 收到空帧列表，跳过推理")
            return ""
        
        logger.info(f"[VLM] 开始推理 {len(frames)} 帧图像...")
        
        with self.inference_lock:
            try:
                self._clear_cache()
                
                do_sample = os.getenv("VLM_DO_SAMPLE", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
                temperature = float(os.getenv("VLM_TEMPERATURE", "0.6"))
                top_p = float(os.getenv("VLM_TOP_P", "0.9"))
                repetition_penalty = float(os.getenv("VLM_REPETITION_PENALTY", "1.05"))

                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": frame} for frame in frames
                        ] + [{"type": "text", "text": user_query}],
                    }
                ]
                
                # 处理输入
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                logger.debug(f"[VLM] Prompt 长度: {len(text)}")

                # 兼容不同 transformers 版本：
                # - 新版通常通过 qwen-vl-utils 的 process_vision_info 处理
                # - 部分版本没有 processor.process_vision_info
                # 这里做多级回退，优先保证可运行
                if hasattr(self.processor, "process_vision_info"):
                    image_inputs, video_inputs = self.processor.process_vision_info(messages)
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                else:
                    try:
                        # 常见兼容路径：直接传入多帧 PIL 列表
                        inputs = self.processor(
                            text=[text],
                            images=frames,
                            padding=True,
                            return_tensors="pt",
                        )
                    except Exception:
                        # 备用兼容路径：按 batch 维度嵌套一层
                        inputs = self.processor(
                            text=[text],
                            images=[frames],
                            padding=True,
                            return_tensors="pt",
                        )
                
                # 将输入移到 GPU
                inputs = inputs.to(self.device)
                
                # 推理
                logger.info("[VLM] 执行推理...")
                generate_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "repetition_penalty": repetition_penalty,
                }
                if do_sample:
                    generate_kwargs["temperature"] = temperature
                    generate_kwargs["top_p"] = top_p

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        **generate_kwargs,
                    )
                
                # 解码输出
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                logger.info(f"[VLM] ✓ 推理完成，输出长度: {len(output_text)} 字符")
                self._log_gpu_memory()
                
                return output_text.strip()
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error("[VLM] ❌ GPU 显存溢出！")
                    self._clear_cache()
                    return "显存不足，请减少输入帧数或重启程序。"
                else:
                    logger.error(f"[VLM] 推理错误: {e}")
                    raise
            
            except Exception as e:
                logger.error(f"[VLM] 未预期的错误: {e}")
                raise
    
    def analyze_fitness_frames_async(
        self,
        frames: List[Image.Image],
        system_prompt: str,
        callback,
        user_query: str = "请分析这些帧中的动作问题。",
        max_new_tokens: int = 256,
    ):
        """
        异步推理方法：在后台线程中执行推理，完成后调用回调函数。
        
        Args:
            frames: PIL Image 列表
            system_prompt: 系统指令
            callback: 完成后的回调函数 callback(result: str)
            user_query: 用户查询
            max_new_tokens: 最大 token 数
        """
        def _inference_worker():
            try:
                result = self.analyze_fitness_frames(
                    frames=frames,
                    system_prompt=system_prompt,
                    user_query=user_query,
                    max_new_tokens=max_new_tokens,
                )
                callback(result)
            except Exception as e:
                logger.error(f"[VLM] 异步推理异常: {e}")
                callback(f"推理错误: {str(e)}")
        
        thread = Thread(target=_inference_worker, daemon=True)
        thread.start()
    
    def unload_model(self):
        """卸载模型，释放显存。"""
        logger.info("[VLM] 卸载模型...")
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._clear_cache()
        logger.info("[VLM] ✓ 模型已卸载")
