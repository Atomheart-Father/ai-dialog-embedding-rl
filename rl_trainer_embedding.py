"""
增强版强化学习训练器 - 支持Embedding压缩
结合向量表示和强化学习优化对话系统性能
"""
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Dict, Tuple, Optional
import logging
import json
import os
from datetime import datetime

from config import rl_config, model_config, dialog_config
from models import model_manager
from embedding_compressor import embedding_compressor
from reward_calculator import RewardCalculator

logger = logging.getLogger(__name__)

# 定义经验回放中的转换（增强版）
EmbeddingTransition = namedtuple('EmbeddingTransition', 
    ('state_embedding', 'action', 'next_state_embedding', 'reward', 'metadata'))

class EmbeddingRLTrainer:
    """基于Embedding的强化学习训练器"""
    
    def __init__(self):
        self.reward_calculator = RewardCalculator()
        
        # 经验回放缓冲区（专门存储embedding）
        self.embedding_memory = deque(maxlen=5000)
        
        # 动作空间（简化版，专注于embedding相关决策）
        self.action_space = {
            'use_embedding': 0,      # 使用embedding压缩
            'use_full_context': 1,   # 使用完整上下文
            'hybrid_approach': 2     # 混合方法
        }
        
        # 训练统计
        self.episode_rewards = []
        self.embedding_efficiency = []
        self.context_length_savings = []
        self.current_episode = 0
        
        # 探索参数
        self.epsilon = 0.3
        
        # 创建训练目录
        self.training_dir = "rl_embedding_training"
        os.makedirs(self.training_dir, exist_ok=True)
        
        logger.info("🧠 Embedding RL训练器初始化完成")
    
    def extract_state_embedding(self, dialog_history: List[Dict]) -> torch.Tensor:
        """提取对话状态的embedding表示"""
        if not dialog_history:
            return torch.zeros(embedding_compressor.embedding_dim)
        
        # 合并最近几轮对话
        recent_context = ""
        for turn in dialog_history[-6:]:  # 最近3轮对话
            role = "用户" if turn['role'] == 'user' else "助手"
            recent_context += f"{role}: {turn['content']}\n"
        
        # 提取状态embedding
        state_embedding = embedding_compressor.extract_text_embedding(recent_context)
        return state_embedding
    
    def select_compression_strategy(self, state_embedding: torch.Tensor, training: bool = True) -> int:
        """选择压缩策略"""
        if training and random.random() < self.epsilon:
            # 探索：随机选择策略
            return random.choice(list(self.action_space.values()))
        else:
            # 利用：基于状态embedding选择最优策略
            return self._select_optimal_strategy(state_embedding)
    
    def _select_optimal_strategy(self, state_embedding: torch.Tensor) -> int:
        """基于状态embedding选择最优策略"""
        # 计算embedding的"信息密度"
        embedding_norm = torch.norm(state_embedding).item()
        embedding_variance = torch.var(state_embedding).item()
        
        # 基于信息密度决策
        if embedding_norm > 2.0 and embedding_variance > 0.1:
            # 高信息密度，适合embedding压缩
            return self.action_space['use_embedding']
        elif embedding_norm < 1.0:
            # 低信息密度，使用完整上下文
            return self.action_space['use_full_context']
        else:
            # 中等信息密度，混合方法
            return self.action_space['hybrid_approach']
    
    def execute_compression_strategy(self, 
                                   action: int, 
                                   dialog_history: List[Dict], 
                                   current_input: str) -> Tuple[str, Dict]:
        """执行压缩策略"""
        if action == self.action_space['use_embedding']:
            # 策略1: 纯embedding压缩
            context = embedding_compressor.generate_context_with_embeddings(current_input)
            
            # 压缩历史为embedding
            if len(dialog_history) > 2:
                compressed_data = embedding_compressor.compress_history_to_embeddings(dialog_history)
                embedding_compressor.current_session_embeddings.extend(compressed_data)
                embedding_compressor.update_history_embeddings()
            
            metadata = {
                'strategy': 'embedding_only',
                'compression_used': True,
                'context_length': len(context)
            }
            
        elif action == self.action_space['use_full_context']:
            # 策略2: 完整上下文
            context = ""
            for turn in dialog_history:
                role = "用户" if turn['role'] == 'user' else "助手"
                context += f"{role}: {turn['content']}\n"
            context += f"用户: {current_input}\n助手:"
            
            metadata = {
                'strategy': 'full_context',
                'compression_used': False,
                'context_length': len(context)
            }
            
        else:  # hybrid_approach
            # 策略3: 混合方法
            # 保留最近2轮 + embedding摘要
            recent_context = ""
            for turn in dialog_history[-4:]:  # 最近2轮
                role = "用户" if turn['role'] == 'user' else "助手"
                recent_context += f"{role}: {turn['content']}\n"
            
            # 获取embedding信息
            if len(dialog_history) > 4:
                embedding_context = embedding_compressor.generate_context_with_embeddings(current_input)
                context = embedding_context + "\n" + recent_context + f"用户: {current_input}\n助手:"
            else:
                context = recent_context + f"用户: {current_input}\n助手:"
            
            metadata = {
                'strategy': 'hybrid',
                'compression_used': len(dialog_history) > 4,
                'context_length': len(context)
            }
        
        return context, metadata
    
    def calculate_embedding_reward(self, 
                                 original_state: torch.Tensor,
                                 action: int,
                                 context: str,
                                 response: str,
                                 metadata: Dict,
                                 user_input: str) -> float:
        """计算基于embedding的奖励"""
        reward = 0.0
        
        # 1. 效率奖励 (30%)
        if metadata['compression_used']:
            # 奖励压缩效率
            estimated_full_length = len(user_input) * 10  # 估算完整对话长度
            actual_length = metadata['context_length']
            efficiency = max(0, (estimated_full_length - actual_length) / estimated_full_length)
            reward += efficiency * 0.3
        
        # 2. 质量奖励 (40%)
        # 基于回复长度和相关性的简化评估
        response_quality = min(1.0, len(response) / 200)  # 长度适中性
        reward += response_quality * 0.4
        
        # 3. 策略选择奖励 (30%)
        # 奖励合适的策略选择
        embedding_norm = torch.norm(original_state).item()
        if action == self.action_space['use_embedding'] and embedding_norm > 2.0:
            reward += 0.3  # 奖励在高信息密度时使用embedding
        elif action == self.action_space['use_full_context'] and embedding_norm < 1.0:
            reward += 0.3  # 奖励在低信息密度时使用完整上下文
        elif action == self.action_space['hybrid_approach']:
            reward += 0.15  # 混合方法得到中等奖励
        
        return reward
    
    def store_embedding_transition(self, 
                                 state_embedding: torch.Tensor,
                                 action: int,
                                 next_state_embedding: torch.Tensor,
                                 reward: float,
                                 metadata: Dict):
        """存储embedding转换到经验回放缓冲区"""
        transition = EmbeddingTransition(
            state_embedding, action, next_state_embedding, reward, metadata
        )
        self.embedding_memory.append(transition)
    
    def train_embedding_step(self) -> float:
        """执行一步embedding训练"""
        if len(self.embedding_memory) < 50:
            return 0.0
        
        # 从经验回放中采样
        batch_size = min(16, len(self.embedding_memory))
        transitions = random.sample(self.embedding_memory, batch_size)
        batch = EmbeddingTransition(*zip(*transitions))
        
        # 计算简化的策略损失
        rewards = torch.tensor(batch.reward, dtype=torch.float32)
        actions = torch.tensor(batch.action, dtype=torch.long)
        
        # 策略梯度的简化版本
        policy_loss = -torch.mean(rewards)  # 最大化奖励
        
        # 更新epsilon
        self.epsilon = max(0.1, self.epsilon * 0.995)
        
        return policy_loss.item()
    
    def train_embedding_episode(self, test_inputs: List[str]) -> Dict:
        """训练一个embedding episode"""
        episode_reward = 0.0
        episode_steps = 0
        context_savings = []
        
        dialog_history = []
        
        logger.info(f"🎮 开始Embedding RL Episode {self.current_episode}")
        
        for i, user_input in enumerate(test_inputs):
            # 添加用户输入到历史
            dialog_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            })
            
            # 提取当前状态embedding
            current_state_embedding = self.extract_state_embedding(dialog_history)
            
            # 选择压缩策略
            action = self.select_compression_strategy(current_state_embedding, training=True)
            
            # 执行压缩策略
            context, metadata = self.execute_compression_strategy(
                action, dialog_history, user_input
            )
            
            # 生成回复
            if model_manager.dialog_model:
                response = model_manager.generate_text(
                    model=model_manager.dialog_model,
                    prompt=context,
                    max_new_tokens=256
                )
            else:
                response = f"Embedding回复{i+1}: 关于'{user_input[:20]}...'的智能回答"
            
            # 添加回复到历史
            dialog_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })
            
            # 提取下一状态embedding
            next_state_embedding = self.extract_state_embedding(dialog_history)
            
            # 计算奖励
            reward = self.calculate_embedding_reward(
                current_state_embedding, action, context, response, metadata, user_input
            )
            
            # 存储经验
            self.store_embedding_transition(
                current_state_embedding, action, next_state_embedding, reward, metadata
            )
            
            episode_reward += reward
            episode_steps += 1
            
            # 记录上下文节省
            if metadata['compression_used']:
                estimated_full = len(' '.join([t['content'] for t in dialog_history]))
                actual = metadata['context_length']
                saving = max(0, (estimated_full - actual) / estimated_full)
                context_savings.append(saving)
        
        # 执行训练步骤
        loss = self.train_embedding_step()
        
        # 记录统计信息
        self.episode_rewards.append(episode_reward)
        if context_savings:
            self.context_length_savings.append(np.mean(context_savings))
        self.current_episode += 1
        
        episode_stats = {
            'episode': self.current_episode,
            'total_reward': episode_reward,
            'steps': episode_steps,
            'loss': loss,
            'epsilon': self.epsilon,
            'avg_context_saving': np.mean(context_savings) if context_savings else 0.0,
            'memory_size': len(self.embedding_memory)
        }
        
        logger.info(f"✅ Embedding Episode {self.current_episode} 完成: "
                   f"奖励={episode_reward:.3f}, 上下文节省={episode_stats['avg_context_saving']:.1%}")
        
        return episode_stats
    
    def evaluate_embedding_performance(self, test_inputs: List[str]) -> Dict:
        """评估embedding压缩性能"""
        print("📊 评估Embedding压缩性能...")
        
        total_compression_ratio = 0.0
        total_response_quality = 0.0
        strategy_usage = {strategy: 0 for strategy in ['embedding_only', 'full_context', 'hybrid']}
        
        dialog_history = []
        
        for user_input in test_inputs:
            dialog_history.append({'role': 'user', 'content': user_input})
            
            # 提取状态
            state_embedding = self.extract_state_embedding(dialog_history)
            
            # 选择策略（评估模式，不探索）
            action = self.select_compression_strategy(state_embedding, training=False)
            
            # 执行策略
            context, metadata = self.execute_compression_strategy(
                action, dialog_history, user_input
            )
            
            # 统计策略使用
            strategy_usage[metadata['strategy']] += 1
            
            # 模拟回复
            response = f"评估回复: 关于{user_input[:20]}的回答"
            dialog_history.append({'role': 'assistant', 'content': response})
            
            # 计算压缩比
            if metadata['compression_used']:
                estimated_full = len(' '.join([t['content'] for t in dialog_history]))
                compression_ratio = metadata['context_length'] / estimated_full
                total_compression_ratio += compression_ratio
        
        avg_compression = total_compression_ratio / len(test_inputs)
        
        performance_report = {
            'average_compression_ratio': avg_compression,
            'strategy_distribution': strategy_usage,
            'total_episodes_trained': self.current_episode,
            'current_epsilon': self.epsilon,
            'embedding_memory_size': len(self.embedding_memory)
        }
        
        return performance_report
    
    def save_embedding_checkpoint(self):
        """保存embedding训练检查点"""
        checkpoint = {
            'episode': self.current_episode,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'context_length_savings': self.context_length_savings,
            'memory_size': len(self.embedding_memory),
            'action_space': self.action_space,
            'training_timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(
            self.training_dir,
            f"embedding_checkpoint_episode_{self.current_episode}.json"
        )
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Embedding检查点已保存: {checkpoint_path}")

# 创建全局embedding RL训练器实例
embedding_rl_trainer = EmbeddingRLTrainer() 