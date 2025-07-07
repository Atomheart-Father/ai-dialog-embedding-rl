"""
模型加载和管理模块
支持MPS加速和量化优化
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
from typing import Optional
import os
import logging
from config import model_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器，负责加载和管理双模型"""

    def __init__(self):
        self.device = model_config.device
        self.tokenizer: Optional[AutoTokenizer] = None
        self.compressor_model: Optional[AutoModelForCausalLM] = None
        self.dialog_model: Optional[AutoModelForCausalLM] = None
        self.generation_config = None

    def load_models(self) -> bool:
        """加载双模型"""
        try:
            logger.info("🔄 开始加载模型...")

            # 创建模型缓存目录
            os.makedirs(model_config.model_cache_dir, exist_ok=True)

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_config.model_name,
                cache_dir=model_config.model_cache_dir,
                trust_remote_code=True
            )

            # 设置pad_token
            if self.tokenizer and hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
                if hasattr(self.tokenizer, 'eos_token'):
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            # 配置量化
            quantization_config = None
            if model_config.use_quantization and model_config.device != "mps" and BitsAndBytesConfig is not None:
                # MPS暂不支持BitsAndBytesConfig，使用torch量化
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

            # 加载压缩模型
            logger.info("📦 加载压缩模型...")
            self.compressor_model = AutoModelForCausalLM.from_pretrained(
                model_config.model_name,
                cache_dir=model_config.model_cache_dir,
                torch_dtype=torch.float16 if model_config.device != "cpu" else torch.float32,
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map="auto" if model_config.device == "cuda" else None,
                low_cpu_mem_usage=True
            )

            # 加载主对话模型
            logger.info("💬 加载主对话模型...")
            self.dialog_model = AutoModelForCausalLM.from_pretrained(
                model_config.model_name,
                cache_dir=model_config.model_cache_dir,
                torch_dtype=torch.float16 if model_config.device != "cpu" else torch.float32,
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map="auto" if model_config.device == "cuda" else None,
                low_cpu_mem_usage=True
            )

            # 移动模型到指定设备（针对MPS）
            if model_config.device == "mps":
                if self.compressor_model and hasattr(self.compressor_model, 'to'):
                    self.compressor_model = self.compressor_model.to(self.device)
                if self.dialog_model and hasattr(self.dialog_model, 'to'):
                    self.dialog_model = self.dialog_model.to(self.device)

            # 设置生成配置 - 优化参数避免重复
            if self.tokenizer:
                self.generation_config = GenerationConfig(
                    max_length=model_config.max_length,
                    temperature=0.8,  # 稍微提高创造性
                    top_p=0.9,
                    top_k=40,
                    do_sample=True,
                    repetition_penalty=1.1,  # 减少重复
                    no_repeat_ngram_size=3,  # 避免3-gram重复
                    pad_token_id=getattr(self.tokenizer, 'pad_token_id', None),
                    eos_token_id=getattr(self.tokenizer, 'eos_token_id', None),
                )

            logger.info("✅ 模型加载完成！")
            return True

        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False

    def count_tokens(self, text: str) -> int:
        """计算文本token数量"""
        if not self.tokenizer:
            return 0
        return len(self.tokenizer.encode(text))

    def generate_text(self, model: AutoModelForCausalLM, prompt: str, max_new_tokens: int = 512) -> str:
        """生成文本"""
        try:
            if not self.tokenizer or not model:
                return ""

            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=model_config.max_length - max_new_tokens
            ).to(self.device)

            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=getattr(self.tokenizer, 'pad_token_id', None),
                    do_sample=True
                )

            # 解码输出（只返回新生成的部分）
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return response.strip()

        except Exception as e:
            logger.error(f"❌ 文本生成失败: {e}")
            return ""

    def cleanup(self):
        """清理模型资源"""
        if self.compressor_model:
            del self.compressor_model
        if self.dialog_model:
            del self.dialog_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()


# 全局模型管理器实例
model_manager = ModelManager()
