"""
基于Embedding的上下文压缩器
使用模型内部向量表示来压缩和表示历史对话
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import logging
from datetime import datetime
try:
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    from sklearn.cluster import KMeans  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    cosine_similarity = None
    KMeans = None
    print("Warning: sklearn not available. Some clustering features will be disabled.")

from config import model_config, dialog_config
from models import model_manager

logger = logging.getLogger(__name__)

class EmbeddingCompressor:
    """基于Embedding的历史压缩器"""
    
    def __init__(self):
        self.embedding_dim = 768  # Qwen模型的隐层维度
        self.max_history_embeddings = 20  # 最大保存的历史embedding数量
        self.similarity_threshold = 0.7  # 相似度阈值
        
        # 历史embedding存储
        self.history_embeddings = []  # List of (embedding, metadata)
        self.current_session_embeddings = []
        
        # 特殊token定义
        self.history_summary_token = "<HIST_EMB>"
        self.context_separator_token = "<CTX_SEP>"
        
        logger.info("🧠 Embedding压缩器初始化完成")
    
    def extract_text_embedding(self, text: str, use_pooling: str = 'mean') -> torch.Tensor:
        """提取文本的embedding表示"""
        if not model_manager.dialog_model or not model_manager.tokenizer:
            logger.warning("模型未加载，返回零向量")
            return torch.zeros(self.embedding_dim)
        
        try:
            # Tokenize输入
            tokenizer = model_manager.tokenizer
            inputs = tokenizer(  # type: ignore
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # 获取模型输出
            with torch.no_grad():
                model = model_manager.dialog_model
                outputs = model(  # type: ignore
                    **inputs,
                    output_hidden_states=True
                )
                
                # 获取最后一层隐状态
                last_hidden_state = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
                
                # 应用不同的pooling策略
                if use_pooling == 'mean':
                    # 平均pooling
                    embedding = last_hidden_state.mean(dim=1)  # [batch, hidden_dim]
                elif use_pooling == 'cls':
                    # 使用[CLS] token (第一个token)
                    embedding = last_hidden_state[:, 0, :]
                elif use_pooling == 'max':
                    # 最大值pooling
                    embedding = last_hidden_state.max(dim=1)[0]
                else:
                    embedding = last_hidden_state.mean(dim=1)
                
                return embedding.squeeze(0)  # 移除batch维度
                
        except Exception as e:
            logger.error(f"提取embedding失败: {e}")
            return torch.zeros(self.embedding_dim)
    
    def compress_dialog_turn(self, user_input: str, assistant_response: str) -> Dict:
        """压缩单轮对话为embedding"""
        # 构建对话文本
        dialog_text = f"用户: {user_input}\n助手: {assistant_response}"
        
        # 提取embedding
        embedding = self.extract_text_embedding(dialog_text)
        
        # 创建元数据
        metadata = {
            'user_input': user_input,
            'assistant_response': assistant_response,
            'timestamp': datetime.now().isoformat(),
            'token_count': len(user_input) + len(assistant_response),
            'turn_summary': user_input[:50] + "..." if len(user_input) > 50 else user_input
        }
        
        # 存储到当前会话
        embedding_data = {
            'embedding': embedding,
            'metadata': metadata
        }
        
        self.current_session_embeddings.append(embedding_data)
        
        return embedding_data
    
    def compress_history_to_embeddings(self, history: List[Dict]) -> List[Dict]:
        """将完整历史压缩为embedding列表"""
        compressed_embeddings = []
        
        # 按对话轮次处理
        for i in range(0, len(history), 2):
            if i + 1 < len(history):
                user_turn = history[i]
                assistant_turn = history[i + 1]
                
                if user_turn['role'] == 'user' and assistant_turn['role'] == 'assistant':
                    embedding_data = self.compress_dialog_turn(
                        user_turn['content'],
                        assistant_turn['content']
                    )
                    compressed_embeddings.append(embedding_data)
        
        return compressed_embeddings
    
    def retrieve_relevant_embeddings(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """检索与查询最相关的历史embedding"""
        if not self.history_embeddings:
            return []
        
        # 获取查询的embedding
        query_embedding = self.extract_text_embedding(query_text)
        query_np = query_embedding.detach().cpu().numpy().reshape(1, -1)
        
        # 计算与所有历史embedding的相似度
        similarities = []
        for hist_data in self.history_embeddings:
            hist_embedding = hist_data['embedding'].detach().cpu().numpy().reshape(1, -1)
            if SKLEARN_AVAILABLE and cosine_similarity is not None:
                similarity = cosine_similarity(query_np, hist_embedding)[0][0]
            else:
                # 使用PyTorch的余弦相似度作为fallback
                query_tensor = torch.from_numpy(query_np[0])
                hist_tensor = torch.from_numpy(hist_embedding[0])
                similarity = torch.nn.functional.cosine_similarity(
                    query_tensor.unsqueeze(0), hist_tensor.unsqueeze(0)
                ).item()
            similarities.append((similarity, hist_data))
        
        # 排序并返回top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # 过滤相似度阈值
        relevant = [
            data for sim, data in similarities[:top_k] 
            if sim > self.similarity_threshold
        ]
        
        logger.info(f"检索到 {len(relevant)} 个相关的历史embedding")
        return relevant
    
    def cluster_embeddings(self, embeddings: List[Dict], n_clusters: int = 3) -> Dict:
        """对embedding进行聚类以发现主题"""
        if not SKLEARN_AVAILABLE or KMeans is None:
            logger.warning("sklearn未安装，跳过聚类分析")
            return {'clusters': [], 'centroids': []}
            
        if len(embeddings) < n_clusters:
            return {'clusters': [], 'centroids': []}
        
        # 提取embedding矩阵
        embedding_matrix = np.stack([
            emb['embedding'].detach().cpu().numpy() 
            for emb in embeddings
        ])
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embedding_matrix)
        
        # 组织聚类结果
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(embeddings[i])
        
        return {
            'clusters': clusters,
            'centroids': kmeans.cluster_centers_,
            'labels': cluster_labels
        }
    
    def embedding_to_natural_language(self, embeddings: List[Dict]) -> str:
        """将embedding转换回自然语言描述"""
        if not embeddings:
            return ""
        
        # 方案1: 直接使用metadata中的摘要
        summaries = []
        for emb_data in embeddings:
            metadata = emb_data['metadata']
            summary = f"- {metadata['turn_summary']}"
            summaries.append(summary)
        
        # 方案2: 基于聚类的主题摘要
        if len(embeddings) > 3:
            cluster_result = self.cluster_embeddings(embeddings, n_clusters=min(3, len(embeddings)))
            topic_summaries = []
            
            for cluster_id, cluster_items in cluster_result['clusters'].items():
                topics = [item['metadata']['turn_summary'] for item in cluster_items]
                topic_summary = f"主题{cluster_id + 1}: " + "; ".join(topics[:2])
                topic_summaries.append(topic_summary)
            
            return "\n".join(topic_summaries)
        
        return "\n".join(summaries)
    
    def generate_context_with_embeddings(self, current_input: str, max_context_length: int = 1500) -> str:
        """生成包含embedding信息的上下文"""
        # 检索相关历史
        relevant_embeddings = self.retrieve_relevant_embeddings(current_input, top_k=5)
        
        if not relevant_embeddings:
            return current_input
        
        # 转换为自然语言
        history_summary = self.embedding_to_natural_language(relevant_embeddings)
        
        # 构建上下文
        context = f"""相关历史摘要:
{history_summary}

当前对话:
用户: {current_input}
助手:"""
        
        # 检查长度限制
        if model_manager.count_tokens(context) > max_context_length:
            # 截断历史摘要
            lines = history_summary.split('\n')
            truncated_summary = '\n'.join(lines[:3])  # 只保留前3行
            context = f"""相关历史摘要:
{truncated_summary}

当前对话:
用户: {current_input}
助手:"""
        
        return context
    
    def update_history_embeddings(self):
        """更新历史embedding存储"""
        # 将当前会话的embedding合并到历史中
        self.history_embeddings.extend(self.current_session_embeddings)
        
        # 保持最大数量限制
        if len(self.history_embeddings) > self.max_history_embeddings:
            # 移除最旧的embedding
            self.history_embeddings = self.history_embeddings[-self.max_history_embeddings:]
        
        # 清空当前会话
        self.current_session_embeddings = []
        
        logger.info(f"历史embedding已更新，当前存储 {len(self.history_embeddings)} 个")
    
    def get_compression_stats(self) -> Dict:
        """获取压缩统计信息"""
        if not self.history_embeddings:
            return {}
        
        # 计算原始token数
        original_tokens = sum(
            emb['metadata']['token_count'] 
            for emb in self.history_embeddings
        )
        
        # embedding占用的"等效token数" (假设1个embedding = 10个token)
        embedding_equivalent_tokens = len(self.history_embeddings) * 10
        
        compression_ratio = embedding_equivalent_tokens / original_tokens if original_tokens > 0 else 0
        
        return {
            'total_embeddings': len(self.history_embeddings),
            'original_tokens': original_tokens,
            'embedding_equivalent_tokens': embedding_equivalent_tokens,
            'compression_ratio': compression_ratio,
            'memory_efficiency': f"{compression_ratio:.1%}",
            'average_similarity': self._calculate_average_similarity()
        }
    
    def _calculate_average_similarity(self) -> float:
        """计算历史embedding之间的平均相似度"""
        if len(self.history_embeddings) < 2:
            return 0.0
        
        embeddings = [emb['embedding'].detach().cpu().numpy() for emb in self.history_embeddings]
        embedding_matrix = np.stack(embeddings)
        
        # 计算所有pair的相似度
        similarities = []
        for i in range(len(embedding_matrix)):
            for j in range(i + 1, len(embedding_matrix)):
                if SKLEARN_AVAILABLE and cosine_similarity is not None:
                    sim = cosine_similarity(
                        embedding_matrix[i:i+1], 
                        embedding_matrix[j:j+1]
                    )[0][0]
                else:
                    # 使用PyTorch的余弦相似度作为fallback
                    vec_i = torch.from_numpy(embedding_matrix[i])
                    vec_j = torch.from_numpy(embedding_matrix[j])
                    sim = torch.nn.functional.cosine_similarity(
                        vec_i.unsqueeze(0), vec_j.unsqueeze(0)
                    ).item()
                similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def save_embeddings(self, filepath: str):
        """保存embedding到文件"""
        # 转换为可序列化格式
        serializable_data = []
        for emb_data in self.history_embeddings:
            serializable_data.append({
                'embedding': emb_data['embedding'].detach().cpu().numpy().tolist(),
                'metadata': emb_data['metadata']
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Embedding已保存到 {filepath}")
    
    def load_embeddings(self, filepath: str):
        """从文件加载embedding"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换回tensor格式
            self.history_embeddings = []
            for item in data:
                embedding_tensor = torch.tensor(item['embedding'])
                self.history_embeddings.append({
                    'embedding': embedding_tensor,
                    'metadata': item['metadata']
                })
            
            logger.info(f"从 {filepath} 加载了 {len(self.history_embeddings)} 个embedding")
        
        except Exception as e:
            logger.error(f"加载embedding失败: {e}")

# 创建全局实例
embedding_compressor = EmbeddingCompressor() 