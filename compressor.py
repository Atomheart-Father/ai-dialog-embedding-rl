"""
历史对话压缩器模块
智能压缩历史对话，保持语义完整性
"""
import re
from typing import List, Dict, Tuple
import logging
from config import model_config, dialog_config
from models import model_manager

logger = logging.getLogger(__name__)


class DialogHistoryCompressor:
    """历史对话压缩器"""

    def __init__(self):
        self.compression_prompt_template = """请将以下对话历史进行智能摘要，保留重要信息和上下文关联：

对话历史：
{history}

要求：
1. 保留关键信息和主要话题
2. 保持对话逻辑和情感色彩
3. 压缩后内容应简洁明了
4. 总结长度控制在原文的30%左右

摘要："""

        self.system_prompt = """你是一个专业的对话摘要助手，擅长提取对话中的核心信息。"""

    def should_compress(self, history: List[Dict[str, str]]) -> bool:
        """判断是否需要压缩历史对话"""
        if len(history) <= dialog_config.keep_recent_turns * 2:  # *2因为有user和assistant
            return False

        # 计算历史对话的token数
        total_tokens = 0
        for turn in history:
            total_tokens += model_manager.count_tokens(turn['content'])

        return total_tokens > dialog_config.trigger_compression_tokens

    def extract_dialog_text(self, history: List[Dict[str, str]]) -> str:
        """提取对话文本"""
        dialog_text = ""
        for i, turn in enumerate(history):
            role = "用户" if turn['role'] == 'user' else "助手"
            dialog_text += f"{role}: {turn['content']}\n"
        return dialog_text.strip()

    def compress_history(self, history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
        """
        压缩历史对话
        返回: (压缩摘要, 保留的最近对话)
        """
        if not self.should_compress(history):
            return "", history

        # 保留最近几轮对话不压缩
        keep_turns = dialog_config.keep_recent_turns * 2  # user + assistant
        recent_history = history[-keep_turns:] if len(history) > keep_turns else []
        compress_history = history[:-keep_turns] if len(history) > keep_turns else history

        if not compress_history:
            return "", recent_history

        try:
            # 提取需要压缩的对话文本
            dialog_text = self.extract_dialog_text(compress_history)

            # 构建压缩提示
            compression_prompt = self.compression_prompt_template.format(history=dialog_text)

            # 使用压缩模型进行摘要
            logger.info("🔄 开始压缩历史对话...")
            if model_manager.compressor_model is not None:
                compressed_summary = model_manager.generate_text(
                    model=model_manager.compressor_model,
                    prompt=compression_prompt,
                    max_new_tokens=int(len(dialog_text.split()) * model_config.compression_ratio * 2)  # 估算压缩后token数
                )
            else:
                compressed_summary = "压缩模型未加载，使用简化摘要"

            # 清理摘要文本
            compressed_summary = self._clean_summary(compressed_summary)

            logger.info(f"✅ 历史压缩完成，原始长度: {len(dialog_text)} 字符，压缩后: {len(compressed_summary)} 字符")

            return compressed_summary, recent_history

        except Exception as e:
            logger.error(f"❌ 历史压缩失败: {e}")
            return "", recent_history

    def _clean_summary(self, summary: str) -> str:
        """清理摘要文本"""
        # 移除可能的格式标记
        summary = re.sub(r'^(摘要：|总结：|Summary:)', '', summary.strip())
        summary = re.sub(r'\n+', ' ', summary)  # 多个换行替换为空格
        summary = re.sub(r'\s+', ' ', summary)  # 多个空格替换为单个空格

        # 确保摘要不为空且不太短
        if len(summary.strip()) < model_config.min_compression_length:
            return "对话历史较短，暂无重要信息需要摘要。"

        return summary.strip()

    def format_context_for_dialog(self, compressed_summary: str, recent_history: List[Dict[str, str]]) -> str:
        """为主对话模型格式化上下文"""
        context = ""

        # 添加压缩摘要
        if compressed_summary:
            context += f"【对话历史摘要】\n{compressed_summary}\n\n"

        # 添加最近的对话
        if recent_history:
            context += "【最近对话】\n"
            for turn in recent_history:
                role = "用户" if turn['role'] == 'user' else "助手"
                context += f"{role}: {turn['content']}\n"

        return context.strip()


# 全局压缩器实例
history_compressor = DialogHistoryCompressor()
