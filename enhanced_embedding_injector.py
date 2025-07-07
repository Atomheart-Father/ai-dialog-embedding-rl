"""
增强版Embedding注入器
实现高级的embedding与query结合技术，用于将历史记录压缩向量与新查询一起输入Qwen模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class AdvancedEmbeddingInjector:
    """高级embedding注入器"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.embedding_dim = 896  # Qwen2.5-0.5B的hidden_size
        
        # 初始化可学习的投影层
        self.history_projector = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.query_projector = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim, 
            num_heads=8, 
            batch_first=True
        )
        
        # 可学习的embedding池
        self.special_embeddings = nn.Embedding(10, self.embedding_dim)  # 特殊token的embedding
        
        # 融合策略映射
        self.injection_strategies = {
            'direct_concatenation': self._direct_concatenation,
            'weighted_fusion': self._weighted_fusion,
            'attention_based': self._attention_based_fusion,
            'layered_injection': self._layered_injection,
            'adaptive_routing': self._adaptive_routing,
        }
        
        logger.info("🚀 高级Embedding注入器初始化完成")
    
    def extract_layered_embeddings(self, text: str, layers: List[int] = [-3, -2, -1]) -> Dict[str, torch.Tensor]:
        """提取多层embedding表示"""
        if not self.model_manager.dialog_model or not self.model_manager.tokenizer:
            return {f'layer_{i}': torch.zeros(self.embedding_dim) for i in layers}
        
        try:
            inputs = self.model_manager.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.model_manager.device)
            
            with torch.no_grad():
                outputs = self.model_manager.dialog_model(
                    **inputs,
                    output_hidden_states=True
                )
                
                layered_embeddings = {}
                for layer_idx in layers:
                    hidden_state = outputs.hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]
                    
                    # 多种pooling策略
                    mean_pooled = hidden_state.mean(dim=1)  # 平均pooling
                    max_pooled = hidden_state.max(dim=1)[0]  # 最大pooling
                    cls_token = hidden_state[:, 0, :]  # CLS token
                    
                    # 组合pooling
                    combined = (0.5 * mean_pooled + 0.3 * max_pooled + 0.2 * cls_token)
                    layered_embeddings[f'layer_{layer_idx}'] = combined.squeeze(0).cpu()
                
                return layered_embeddings
                
        except Exception as e:
            logger.error(f"提取多层embedding失败: {e}")
            return {f'layer_{i}': torch.zeros(self.embedding_dim) for i in layers}
    
    def _direct_concatenation(self, history_emb: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        """直接拼接融合"""
        # 投影到相同维度
        hist_proj = self.history_projector(history_emb)
        query_proj = self.query_projector(query_emb)
        
        # 简单拼接并降维
        concatenated = torch.cat([hist_proj, query_proj], dim=0)
        # 使用线性层降维到原始维度
        reduced = concatenated[:self.embedding_dim] + concatenated[self.embedding_dim:]
        
        return reduced
    
    def _weighted_fusion(self, history_emb: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        """加权融合"""
        # 计算权重
        similarity = F.cosine_similarity(history_emb, query_emb, dim=0)
        history_weight = torch.sigmoid(similarity)
        query_weight = 1 - history_weight
        
        # 投影并融合
        hist_proj = self.history_projector(history_emb)
        query_proj = self.query_projector(query_emb)
        
        fused = history_weight * hist_proj + query_weight * query_proj
        return fused
    
    def _attention_based_fusion(self, history_emb: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        """基于注意力的融合"""
        # 准备输入 [batch_size, seq_len, embed_dim]
        hist_expanded = history_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
        query_expanded = query_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
        
        # 注意力融合
        combined_input = torch.cat([hist_expanded, query_expanded], dim=1)  # [1, 2, embed_dim]
        
        try:
            attn_output, attn_weights = self.fusion_attention(
                combined_input, combined_input, combined_input
            )
            
            # 聚合输出
            fused = attn_output.mean(dim=1).squeeze(0)  # [embed_dim]
            return fused
            
        except Exception as e:
            logger.warning(f"注意力融合失败，回退到加权融合: {e}")
            return self._weighted_fusion(history_emb, query_emb)
    
    def _layered_injection(self, history_emb: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        """分层注入融合"""
        # 将embedding分成多个层进行处理
        chunk_size = self.embedding_dim // 4
        fused_chunks = []
        
        for i in range(4):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            
            hist_chunk = history_emb[start_idx:end_idx]
            query_chunk = query_emb[start_idx:end_idx]
            
            # 不同的融合策略
            if i == 0:  # 第一层：保留更多历史信息
                chunk_fused = 0.7 * hist_chunk + 0.3 * query_chunk
            elif i == 1:  # 第二层：平衡融合
                chunk_fused = 0.5 * hist_chunk + 0.5 * query_chunk
            elif i == 2:  # 第三层：侧重查询
                chunk_fused = 0.3 * hist_chunk + 0.7 * query_chunk
            else:  # 第四层：主要是查询信息
                chunk_fused = 0.1 * hist_chunk + 0.9 * query_chunk
            
            fused_chunks.append(chunk_fused)
        
        return torch.cat(fused_chunks, dim=0)
    
    def _adaptive_routing(self, history_emb: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        """自适应路由融合"""
        # 计算历史和查询的特征
        hist_norm = torch.norm(history_emb)
        query_norm = torch.norm(query_emb)
        similarity = F.cosine_similarity(history_emb, query_emb, dim=0)
        
        # 基于特征选择融合策略
        if similarity > 0.8:  # 高相似度，使用简单加权
            return self._weighted_fusion(history_emb, query_emb)
        elif hist_norm > query_norm * 2:  # 历史信息丰富，使用分层注入
            return self._layered_injection(history_emb, query_emb)
        else:  # 使用注意力融合
            return self._attention_based_fusion(history_emb, query_emb)
    
    def create_context_with_embedding(self, 
                                    history_embedding: torch.Tensor,
                                    query_text: str,
                                    injection_strategy: str = "adaptive_routing") -> Tuple[str, Dict]:
        """创建包含embedding信息的上下文"""
        
        # 提取查询的embedding
        query_embeddings = self.extract_layered_embeddings(query_text)
        query_emb = query_embeddings['layer_-1']  # 使用最后一层
        
        # 选择融合策略
        if injection_strategy in self.injection_strategies:
            fusion_func = self.injection_strategies[injection_strategy]
            fused_embedding = fusion_func(history_embedding, query_emb)
        else:
            logger.warning(f"未知的注入策略: {injection_strategy}，使用默认策略")
            fused_embedding = self._adaptive_routing(history_embedding, query_emb)
        
        # 将融合的embedding转换为上下文提示
        context_prompt = self._embedding_to_context_prompt(fused_embedding, query_text)
        
        # 元数据
        metadata = {
            'injection_strategy': injection_strategy,
            'history_embedding_norm': torch.norm(history_embedding).item(),
            'query_embedding_norm': torch.norm(query_emb).item(),
            'fused_embedding_norm': torch.norm(fused_embedding).item(),
            'embedding_similarity': F.cosine_similarity(
                history_embedding, query_emb, dim=0
            ).item()
        }
        
        return context_prompt, metadata
    
    def _embedding_to_context_prompt(self, embedding: torch.Tensor, query_text: str) -> str:
        """将融合的embedding转换为上下文提示"""
        
        # 方法1: 基于embedding的激活模式生成提示
        activation_threshold = embedding.mean() + embedding.std()
        high_activation_dims = (embedding > activation_threshold).sum().item()
        
        if high_activation_dims > embedding.shape[0] * 0.3:
            context_prefix = "基于丰富的历史对话信息"
        elif high_activation_dims > embedding.shape[0] * 0.1:
            context_prefix = "参考相关的历史背景"
        else:
            context_prefix = "结合之前的对话内容"
        
        # 方法2: 将embedding的关键维度映射为特殊token
        top_k = 5
        top_values, top_indices = torch.topk(embedding, top_k)
        context_tokens = []
        
        for val, idx in zip(top_values, top_indices):
            if val > 0:
                token_id = int(idx.item() % 10)  # 映射到0-9
                context_tokens.append(f"<CTX_{token_id}>")
        
        context_token_str = " ".join(context_tokens)
        
        # 构建最终提示
        enhanced_prompt = f"""{context_prefix}，{context_token_str}

用户: {query_text}
助手:"""
        
        return enhanced_prompt
    
    def generate_with_enhanced_context(self, 
                                     history_text: str,
                                     query_text: str,
                                     injection_strategy: str = "adaptive_routing",
                                     max_new_tokens: int = 512) -> Dict:
        """使用增强上下文生成回复"""
        
        # 提取历史embedding
        history_embeddings = self.extract_layered_embeddings(history_text)
        history_emb = history_embeddings['layer_-1']
        
        # 创建增强上下文
        enhanced_prompt, metadata = self.create_context_with_embedding(
            history_emb, query_text, injection_strategy
        )
        
        # 生成回复
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        response = self.model_manager.generate_text(
            model=self.model_manager.dialog_model,
            prompt=enhanced_prompt,
            max_new_tokens=max_new_tokens
        )
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            generation_time = start_time.elapsed_time(end_time) / 1000.0  # 转换为秒
        else:
            generation_time = 0.0
        
        return {
            'response': response,
            'enhanced_prompt': enhanced_prompt,
            'metadata': metadata,
            'generation_time': generation_time,
            'input_tokens': self.model_manager.count_tokens(enhanced_prompt),
            'output_tokens': self.model_manager.count_tokens(response)
        }
    
    def compare_injection_strategies(self, 
                                   history_text: str,
                                   query_text: str) -> Dict:
        """对比不同注入策略的效果"""
        
        results = {}
        
        for strategy in self.injection_strategies.keys():
            try:
                result = self.generate_with_enhanced_context(
                    history_text, query_text, strategy
                )
                results[strategy] = result
                logger.info(f"✅ 策略 {strategy} 完成")
                
            except Exception as e:
                logger.error(f"❌ 策略 {strategy} 失败: {e}")
                results[strategy] = {'error': str(e)}
        
        return results
    
    def save_injector_state(self, filepath: str):
        """保存注入器的可学习参数"""
        state_dict = {
            'history_projector': self.history_projector.state_dict(),
            'query_projector': self.query_projector.state_dict(),
            'fusion_attention': self.fusion_attention.state_dict(),
            'special_embeddings': self.special_embeddings.state_dict(),
        }
        
        torch.save(state_dict, filepath)
        logger.info(f"💾 注入器状态已保存到: {filepath}")
    
    def load_injector_state(self, filepath: str):
        """加载注入器的可学习参数"""
        try:
            state_dict = torch.load(filepath, map_location='cpu')
            
            self.history_projector.load_state_dict(state_dict['history_projector'])
            self.query_projector.load_state_dict(state_dict['query_projector'])
            self.fusion_attention.load_state_dict(state_dict['fusion_attention'])
            self.special_embeddings.load_state_dict(state_dict['special_embeddings'])
            
            logger.info(f"📥 注入器状态已从 {filepath} 加载")
            
        except Exception as e:
            logger.error(f"❌ 加载注入器状态失败: {e}")

# 使用示例和测试函数
def test_enhanced_injector():
    """测试增强版注入器的功能"""
    
    # 这里需要在有model_manager的环境中运行
    # enhanced_injector = AdvancedEmbeddingInjector(model_manager)
    
    # 测试历史和查询
    test_history = """
    用户: 我想学习Python编程
    助手: Python是很好的编程语言，建议从基础语法开始
    用户: 如何学习数据结构？
    助手: 可以从列表、字典等基础数据结构开始学习
    """
    
    test_query = "现在我想学习面向对象编程，应该从哪里开始？"
    
    print("🧪 测试历史文本:")
    print(test_history)
    print(f"\n🔍 测试查询: {test_query}")
    
    # 在实际使用中，需要先初始化model_manager
    # result = enhanced_injector.compare_injection_strategies(test_history, test_query)
    # return result

if __name__ == "__main__":
    test_enhanced_injector() 