"""
奖励计算器
实现多维度奖励函数，用于强化学习训练
"""
import re
import math
from typing import Dict, List
import logging
from config import rl_config
from models import model_manager

logger = logging.getLogger(__name__)

class RewardCalculator:
    """奖励计算器"""
    
    def __init__(self):
        self.quality_weight = rl_config.quality_reward_weight
        self.compression_weight = rl_config.compression_reward_weight  
        self.coherence_weight = rl_config.coherence_reward_weight
        
        logger.info("🎯 奖励计算器初始化完成")
    
    def calculate_total_reward(self, 
                             original_state,
                             action: int,
                             compressed_summary: str,
                             dialog_response: str,
                             user_input: str) -> float:
        """计算总奖励"""
        
        # 1. 回答质量奖励
        quality_reward = self._calculate_quality_reward(
            dialog_response, user_input, compressed_summary
        )
        
        # 2. 压缩效率奖励
        compression_reward = self._calculate_compression_reward(
            original_state, compressed_summary
        )
        
        # 3. 连贯性奖励
        coherence_reward = self._calculate_coherence_reward(
            original_state.history, compressed_summary, dialog_response
        )
        
        # 加权总奖励
        total_reward = (
            self.quality_weight * quality_reward +
            self.compression_weight * compression_reward +
            self.coherence_weight * coherence_reward
        )
        
        logger.debug(f"奖励分解: 质量={quality_reward:.3f}, 压缩={compression_reward:.3f}, "
                    f"连贯={coherence_reward:.3f}, 总计={total_reward:.3f}")
        
        return total_reward
    
    def _calculate_quality_reward(self, 
                                dialog_response: str, 
                                user_input: str,
                                compressed_summary: str) -> float:
        """计算回答质量奖励"""
        if not dialog_response.strip():
            return -1.0  # 空回复严重惩罚
        
        quality_score = 0.0
        
        # 1. 长度合理性 (避免过短或过长回复)
        response_length = len(dialog_response)
        if 10 <= response_length <= 500:
            quality_score += 0.3
        elif response_length < 10:
            quality_score -= 0.5  # 过短惩罚
        elif response_length > 1000:
            quality_score -= 0.3  # 过长轻微惩罚
        
        # 2. 相关性检查（简化版本）
        user_keywords = self._extract_keywords(user_input)
        response_keywords = self._extract_keywords(dialog_response)
        
        if user_keywords and response_keywords:
            # 计算关键词重叠度
            overlap = len(set(user_keywords) & set(response_keywords))
            relevance_score = min(1.0, overlap / len(user_keywords))
            quality_score += 0.4 * relevance_score
        
        # 3. 流畅性检查（基于句子结构）
        if self._check_fluency(dialog_response):
            quality_score += 0.3
        
        return max(-1.0, min(1.0, quality_score))  # 限制在[-1, 1]范围
    
    def _calculate_compression_reward(self, original_state, compressed_summary: str) -> float:
        """计算压缩效率奖励"""
        if not compressed_summary.strip():
            return -0.5  # 空摘要惩罚
        
        # 计算压缩比
        original_tokens = original_state.token_count
        summary_tokens = model_manager.count_tokens(compressed_summary)
        
        if original_tokens == 0:
            return 0.0
        
        compression_ratio = summary_tokens / original_tokens
        
        # 理想压缩比在 0.2-0.4 之间
        if 0.2 <= compression_ratio <= 0.4:
            ratio_reward = 1.0
        elif compression_ratio < 0.2:
            # 压缩过度，可能丢失信息
            ratio_reward = 0.5
        elif compression_ratio > 0.6:
            # 压缩不足
            ratio_reward = 0.3
        else:
            ratio_reward = 0.7
        
        # Token节省奖励
        tokens_saved = original_tokens - summary_tokens
        save_reward = min(1.0, tokens_saved / 1000)  # 每节省1000 tokens奖励1.0
        
        return 0.6 * ratio_reward + 0.4 * save_reward
    
    def _calculate_coherence_reward(self, 
                                  history: List[Dict], 
                                  compressed_summary: str,
                                  dialog_response: str) -> float:
        """计算连贯性奖励"""
        coherence_score = 0.0
        
        # 1. 摘要与历史的一致性
        if history and compressed_summary:
            history_text = " ".join([turn['content'] for turn in history[-5:]])
            summary_keywords = set(self._extract_keywords(compressed_summary))
            history_keywords = set(self._extract_keywords(history_text))
            
            if history_keywords:
                consistency = len(summary_keywords & history_keywords) / len(history_keywords)
                coherence_score += 0.5 * consistency
        
        # 2. 回复与上下文的连贯性
        if history and dialog_response:
            recent_context = " ".join([turn['content'] for turn in history[-2:]])
            context_keywords = set(self._extract_keywords(recent_context))
            response_keywords = set(self._extract_keywords(dialog_response))
            
            if context_keywords:
                context_coherence = len(context_keywords & response_keywords) / len(context_keywords)
                coherence_score += 0.3 * context_coherence
        
        # 3. 主题连续性（简化检查）
        if self._check_topic_continuity(history, dialog_response):
            coherence_score += 0.2
        
        return max(0.0, min(1.0, coherence_score))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（简化版本）"""
        if not text:
            return []
        
        # 简单的关键词提取：去除标点符号，提取长度大于2的词
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # 过滤常见停用词
        stop_words = {
            '的', '了', '在', '是', '我', '你', '他', '她', '它', '我们', '你们', '他们',
            '这', '那', '这个', '那个', '有', '没有', '可以', '不能', '会', '不会',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has'
        }
        
        keywords = [word for word in words if word not in stop_words]
        return keywords[:10]  # 返回前10个关键词
    
    def _check_fluency(self, text: str) -> bool:
        """检查文本流畅性（简化版本）"""
        if not text.strip():
            return False
        
        # 简单的流畅性指标
        sentences = re.split(r'[.!?。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return False
        
        # 检查平均句子长度
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # 合理的句子长度范围
        return 3 <= avg_length <= 30
    
    def _check_topic_continuity(self, history: List[Dict], response: str) -> bool:
        """检查主题连续性（简化版本）"""
        if not history or not response:
            return True  # 默认认为连贯
        
        # 提取最近几轮对话的关键词
        recent_keywords = set()
        for turn in history[-3:]:
            recent_keywords.update(self._extract_keywords(turn['content']))
        
        response_keywords = set(self._extract_keywords(response))
        
        # 如果回复包含最近对话的关键词，认为主题连续
        if recent_keywords and response_keywords:
            overlap_ratio = len(recent_keywords & response_keywords) / len(recent_keywords)
            return overlap_ratio > 0.1  # 至少10%的关键词重叠
        
        return True
    
    def calculate_episode_bonus(self, episode_stats: Dict) -> float:
        """计算episode级别的奖励加成"""
        bonus = 0.0
        
        # 1. 对话长度奖励（鼓励长对话）
        steps = episode_stats.get('steps', 0)
        if steps > 10:
            bonus += 0.1 * math.log(steps / 10)
        
        # 2. 压缩效率奖励
        if episode_stats.get('compression_count', 0) > 0:
            avg_compression_ratio = episode_stats.get('avg_compression_ratio', 0.5)
            if 0.2 <= avg_compression_ratio <= 0.4:
                bonus += 0.2
        
        return bonus

# 全局奖励计算器实例
reward_calculator = RewardCalculator() 