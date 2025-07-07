"""
直接使用大模型内部状态的上下文压缩器
通过提取和融合hidden states实现高效的历史信息压缩
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from datetime import datetime
import json

from config import model_config, dialog_config
from models import model_manager

logger = logging.getLogger(__name__)

class HiddenStateBank:
    """历史hidden state存储银行"""
    
    def __init__(self, max_states: int = 50, state_dim: int = 768):
        self.max_states = max_states
        self.state_dim = state_dim
        
        # 存储历史states和对应的metadata
        self.state_bank: List[Dict] = []
        self.attention_weights = None
        
    def add_state(self, hidden_state: torch.Tensor, metadata: Dict):
        """添加新的hidden state"""
        state_entry = {
            'state': hidden_state.detach().clone(),
            'metadata': metadata,
            'timestamp': datetime.now().isoformat(),
            'access_count': 0
        }
        
        self.state_bank.append(state_entry)
        
        # 维持最大容量
        if len(self.state_bank) > self.max_states:
            # 移除最少访问的state
            self.state_bank.sort(key=lambda x: x['access_count'])
            self.state_bank = self.state_bank[1:]
    
    def retrieve_relevant_states(self, query_state: torch.Tensor, top_k: int = 5) -> List[Dict]:
        """检索与query最相关的historical states"""
        if not self.state_bank:
            return []
        
        similarities = []
        for state_entry in self.state_bank:
            sim = F.cosine_similarity(
                query_state.unsqueeze(0), 
                state_entry['state'].unsqueeze(0)
            ).item()
            similarities.append((sim, state_entry))
        
        # 排序并返回top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        relevant_states = [entry for _, entry in similarities[:top_k]]
        
        # 更新访问计数
        for entry in relevant_states:
            entry['access_count'] += 1
        
        return relevant_states
    
    def get_state_summary(self) -> Dict:
        """获取state bank的统计摘要"""
        if not self.state_bank:
            return {}
        
        return {
            'total_states': len(self.state_bank),
            'avg_access_count': np.mean([s['access_count'] for s in self.state_bank]),
            'state_dimension': self.state_dim,
            'memory_usage_mb': len(self.state_bank) * self.state_dim * 4 / (1024*1024)
        }

class DirectEmbeddingCompressor:
    """直接使用大模型内部状态的压缩器"""
    
    def __init__(self):
        self.state_bank = HiddenStateBank()
        
        # 上下文融合策略
        self.fusion_strategies = {
            'attention': self._attention_fusion,
            'weighted_sum': self._weighted_sum_fusion,
            'concatenation': self._concatenation_fusion,
            'interpolation': self._interpolation_fusion
        }
        
        # 当前使用的融合策略
        self.current_strategy = 'attention'
        
        logger.info("🎯 直接Embedding压缩器初始化完成")
    
    def extract_dialog_hidden_states(self, dialog_text: str, 
                                   extract_layers: List[int] = [-1, -2, -3]) -> Dict[str, torch.Tensor]:
        """提取对话文本的多层hidden states"""
        if not model_manager.dialog_model or not model_manager.tokenizer:
            logger.warning("模型未加载，返回零向量")
            return {'combined': torch.zeros(768)}
        
        try:
            # Tokenize输入
            tokenizer = model_manager.tokenizer
            inputs = tokenizer(  # type: ignore
                dialog_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True
            )
            
            # 获取多层hidden states
            with torch.no_grad():
                model = model_manager.dialog_model
                outputs = model(  # type: ignore
                    **inputs,
                    output_hidden_states=True
                )
                
                hidden_states = outputs.hidden_states
                extracted_states = {}
                
                # 提取指定层的states
                for layer_idx in extract_layers:
                    layer_state = hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]
                    
                    # 多种pooling策略
                    mean_pooled = layer_state.mean(dim=1)  # [batch, hidden_dim]
                    max_pooled = layer_state.max(dim=1)[0]  # [batch, hidden_dim]
                    cls_token = layer_state[:, 0, :]  # [batch, hidden_dim] (first token)
                    
                    extracted_states[f'layer_{layer_idx}_mean'] = mean_pooled.squeeze(0)
                    extracted_states[f'layer_{layer_idx}_max'] = max_pooled.squeeze(0)
                    extracted_states[f'layer_{layer_idx}_cls'] = cls_token.squeeze(0)
                
                # 创建组合表示
                all_states = list(extracted_states.values())
                if all_states:
                    # 方案1: 简单平均
                    combined_state = torch.stack(all_states).mean(dim=0)
                    extracted_states['combined'] = combined_state
                    
                    # 方案2: 加权组合 (给最后一层更高权重)
                    weights = torch.softmax(torch.tensor([3.0, 2.0, 1.0] * 3), dim=0)
                    weighted_state = sum(w * s for w, s in zip(weights, all_states))
                    extracted_states['weighted_combined'] = weighted_state
                
                return extracted_states
                
        except Exception as e:
            logger.error(f"提取hidden states失败: {e}")
            return {'combined': torch.zeros(768)}
    
    def compress_history_to_states(self, dialog_history: List[Dict]) -> List[Dict]:
        """将对话历史压缩为hidden states"""
        compressed_states = []
        
        # 按对话轮次处理
        for i in range(0, len(dialog_history), 2):
            if i + 1 < len(dialog_history):
                user_turn = dialog_history[i]
                assistant_turn = dialog_history[i + 1]
                
                if user_turn['role'] == 'user' and assistant_turn['role'] == 'assistant':
                    # 构建对话文本
                    dialog_text = f"用户: {user_turn['content']}\n助手: {assistant_turn['content']}"
                    
                    # 提取hidden states
                    hidden_states = self.extract_dialog_hidden_states(dialog_text)
                    
                    # 创建状态条目
                    state_entry = {
                        'states': hidden_states,
                        'user_input': user_turn['content'],
                        'assistant_response': assistant_turn['content'],
                        'turn_index': i // 2,
                        'token_count': len(dialog_text),
                        'summary': user_turn['content'][:50] + "..."
                    }
                    
                    compressed_states.append(state_entry)
                    
                    # 添加到state bank
                    self.state_bank.add_state(
                        hidden_states['combined'],
                        {
                            'turn_index': i // 2,
                            'summary': state_entry['summary'],
                            'user_input': user_turn['content']
                        }
                    )
        
        return compressed_states
    
    def _attention_fusion(self, query_state: torch.Tensor, 
                         context_states: List[torch.Tensor]) -> torch.Tensor:
        """使用注意力机制融合历史states"""
        if not context_states:
            return query_state
        
        # Stack context states
        context_matrix = torch.stack(context_states)  # [num_contexts, hidden_dim]
        
        # 计算attention weights
        attention_scores = torch.matmul(
            query_state.unsqueeze(0),  # [1, hidden_dim]
            context_matrix.transpose(0, 1)  # [hidden_dim, num_contexts]
        )  # [1, num_contexts]
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 加权融合
        fused_context = torch.matmul(attention_weights, context_matrix).squeeze(0)
        
        # 与query state结合
        enhanced_state = 0.7 * query_state + 0.3 * fused_context
        
        return enhanced_state
    
    def _weighted_sum_fusion(self, query_state: torch.Tensor, 
                           context_states: List[torch.Tensor]) -> torch.Tensor:
        """加权求和融合"""
        if not context_states:
            return query_state
        
        # 根据相似度计算权重
        weights = []
        for ctx_state in context_states:
            sim = F.cosine_similarity(query_state, ctx_state, dim=0)
            weights.append(sim)
        
        weights = F.softmax(torch.tensor(weights), dim=0)
        
        # 加权求和
        weighted_context = sum(w * ctx for w, ctx in zip(weights, context_states))
        
        return 0.6 * query_state + 0.4 * weighted_context
    
    def _concatenation_fusion(self, query_state: torch.Tensor, 
                            context_states: List[torch.Tensor]) -> torch.Tensor:
        """拼接融合（需要降维）"""
        if not context_states:
            return query_state
        
        # 选择最相关的几个context states
        top_states = context_states[:3]  # 最多3个
        
        # 拼接
        all_states = [query_state] + top_states
        concatenated = torch.cat(all_states, dim=0)
        
        # 降维到原始维度（简单的线性变换）
        hidden_dim = query_state.shape[0]
        current_dim = concatenated.shape[0]
        
        if current_dim > hidden_dim:
            # 简单的分块平均
            chunks = concatenated.chunk(current_dim // hidden_dim + 1)
            reduced = torch.stack(chunks[:-1] if len(chunks[-1]) < hidden_dim else chunks).mean(dim=0)
            return reduced[:hidden_dim]
        
        return concatenated
    
    def _interpolation_fusion(self, query_state: torch.Tensor, 
                            context_states: List[torch.Tensor]) -> torch.Tensor:
        """插值融合"""
        if not context_states:
            return query_state
        
        # 计算context states的中心
        context_center = torch.stack(context_states).mean(dim=0)
        
        # 插值参数基于相似度
        similarity = F.cosine_similarity(query_state, context_center, dim=0)
        alpha = torch.sigmoid(similarity)  # 自适应插值权重
        
        return alpha * query_state + (1 - alpha) * context_center
    
    def generate_enhanced_context(self, current_input: str, 
                                max_context_tokens: int = 1500) -> Tuple[str, Dict]:
        """生成增强的上下文（融合历史states）"""
        # 提取当前输入的hidden state
        current_text = f"用户: {current_input}"
        current_states = self.extract_dialog_hidden_states(current_text)
        query_state = current_states['combined']
        
        # 检索相关的历史states
        relevant_entries = self.state_bank.retrieve_relevant_states(query_state, top_k=5)
        
        if not relevant_entries:
            # 无历史信息，直接返回
            return f"用户: {current_input}\n助手:", {'fusion_used': False}
        
        # 提取相关的states
        context_states = [entry['state'] for entry in relevant_entries]
        
        # 使用当前策略融合states
        fusion_func = self.fusion_strategies[self.current_strategy]
        enhanced_state = fusion_func(query_state, context_states)
        
        # 将融合后的state转换回自然语言描述
        context_description = self._state_to_description(enhanced_state, relevant_entries)
        
        # 构建最终的prompt
        enhanced_prompt = f"""基于历史上下文的对话：

{context_description}

当前对话：
用户: {current_input}
助手:"""
        
        # 检查长度限制
        if model_manager.count_tokens(enhanced_prompt) > max_context_tokens:
            # 简化上下文
            simplified_context = self._simplify_context(context_description, max_context_tokens // 2)
            enhanced_prompt = f"""历史摘要: {simplified_context}

用户: {current_input}
助手:"""
        
        metadata = {
            'fusion_used': True,
            'fusion_strategy': self.current_strategy,
            'num_context_states': len(context_states),
            'relevant_turns': [entry['metadata']['summary'] for entry in relevant_entries],
            'enhanced_state_norm': torch.norm(enhanced_state).item()
        }
        
        return enhanced_prompt, metadata
    
    def _state_to_description(self, fused_state: torch.Tensor, 
                            relevant_entries: List[Dict]) -> str:
        """将融合的state转换为自然语言描述"""
        # 方案1: 基于相关条目的元数据
        summaries = []
        for entry in relevant_entries[:3]:  # 最多3个
            summary = entry['metadata']['summary']
            summaries.append(f"- {summary}")
        
        # 方案2: 基于state的"激活模式"
        state_activation = torch.sigmoid(fused_state)
        high_activation_dims = (state_activation > 0.7).sum().item()
        activation_pattern = "高度相关" if high_activation_dims > 100 else "部分相关"
        
        description = f"""相关历史信息 ({activation_pattern}):
{chr(10).join(summaries)}"""
        
        return description
    
    def _simplify_context(self, context: str, max_length: int) -> str:
        """简化上下文描述"""
        if len(context) <= max_length:
            return context
        
        lines = context.split('\n')
        simplified = []
        current_length = 0
        
        for line in lines:
            if current_length + len(line) <= max_length:
                simplified.append(line)
                current_length += len(line)
            else:
                break
        
        return '\n'.join(simplified) + "..."
    
    def switch_fusion_strategy(self, strategy: str):
        """切换融合策略"""
        if strategy in self.fusion_strategies:
            self.current_strategy = strategy
            logger.info(f"切换到融合策略: {strategy}")
        else:
            logger.warning(f"未知的融合策略: {strategy}")
    
    def get_compression_statistics(self) -> Dict:
        """获取压缩统计信息"""
        bank_summary = self.state_bank.get_state_summary()
        
        stats = {
            'state_bank_info': bank_summary,
            'current_fusion_strategy': self.current_strategy,
            'available_strategies': list(self.fusion_strategies.keys()),
            'compression_efficiency': self._calculate_compression_efficiency()
        }
        
        return stats
    
    def _calculate_compression_efficiency(self) -> Dict:
        """计算压缩效率"""
        if not self.state_bank.state_bank:
            return {}
        
        # 估算原始文本大小 vs state大小
        total_original_chars = sum(
            len(entry['metadata'].get('user_input', '')) 
            for entry in self.state_bank.state_bank
        )
        
        total_state_memory = len(self.state_bank.state_bank) * self.state_bank.state_dim * 4  # bytes
        original_memory = total_original_chars * 2  # 假设UTF-8编码
        
        return {
            'compression_ratio': total_state_memory / original_memory if original_memory > 0 else 1.0,
            'memory_savings_mb': (original_memory - total_state_memory) / (1024*1024),
            'states_per_mb': len(self.state_bank.state_bank) / (total_state_memory / (1024*1024))
        }
    
    def save_state_bank(self, filepath: str):
        """保存state bank到文件"""
        serializable_data = []
        for entry in self.state_bank.state_bank:
            serializable_entry = {
                'state': entry['state'].detach().cpu().numpy().tolist(),
                'metadata': entry['metadata'],
                'timestamp': entry['timestamp'],
                'access_count': entry['access_count']
            }
            serializable_data.append(serializable_entry)
        
        save_data = {
            'state_bank': serializable_data,
            'fusion_strategy': self.current_strategy,
            'save_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"State bank已保存到: {filepath}")
    
    def load_state_bank(self, filepath: str):
        """从文件加载state bank"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 重建state bank
            self.state_bank.state_bank = []
            for entry_data in data['state_bank']:
                entry = {
                    'state': torch.tensor(entry_data['state']),
                    'metadata': entry_data['metadata'],
                    'timestamp': entry_data['timestamp'],
                    'access_count': entry_data['access_count']
                }
                self.state_bank.state_bank.append(entry)
            
            # 恢复融合策略
            if 'fusion_strategy' in data:
                self.current_strategy = data['fusion_strategy']
            
            logger.info(f"从 {filepath} 加载了 {len(self.state_bank.state_bank)} 个states")
        
        except Exception as e:
            logger.error(f"加载state bank失败: {e}")

# 创建全局实例
direct_compressor = DirectEmbeddingCompressor() 