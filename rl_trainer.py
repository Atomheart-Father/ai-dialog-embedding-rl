"""
强化学习训练器
实现压缩模型和主对话模型的联合训练架构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from reward_calculator import RewardCalculator

logger = logging.getLogger(__name__)

# 定义经验回放中的转换
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class CompressionActionSpace:
    """压缩动作空间定义"""
    
    def __init__(self):
        # 定义压缩策略的动作空间
        self.compression_ratios = [0.2, 0.3, 0.4, 0.5]  # 压缩比例选择
        self.focus_strategies = [
            'recent_focus',      # 侧重最近对话
            'topic_focus',       # 侧重主题相关
            'entity_focus',      # 侧重实体信息
            'balanced',          # 平衡策略
        ]
        self.action_dim = len(self.compression_ratios) * len(self.focus_strategies)
    
    def decode_action(self, action_idx: int) -> Dict:
        """将动作索引解码为具体策略"""
        ratio_idx = action_idx // len(self.focus_strategies)
        strategy_idx = action_idx % len(self.focus_strategies)
        
        return {
            'compression_ratio': self.compression_ratios[ratio_idx],
            'focus_strategy': self.focus_strategies[strategy_idx],
            'action_idx': action_idx
        }
    
    def sample_random_action(self) -> int:
        """随机采样动作"""
        return random.randint(0, self.action_dim - 1)

class DialogState:
    """对话状态表示"""
    
    def __init__(self, history: List[Dict], compressed_summary: str = ""):
        self.history = history.copy()
        self.compressed_summary = compressed_summary
        self.token_count = self._count_tokens()
        self.turn_count = len([h for h in history if h['role'] == 'user'])
        
    def _count_tokens(self) -> int:
        """计算总token数"""
        total = 0
        for turn in self.history:
            total += model_manager.count_tokens(turn['content'])
        if self.compressed_summary:
            total += model_manager.count_tokens(self.compressed_summary)
        return total
    
    def to_tensor(self, max_length: int = 2048) -> torch.Tensor:
        """将状态转换为tensor表示"""
        # 简化的状态表示：使用token嵌入的平均值
        all_text = ""
        if self.compressed_summary:
            all_text += self.compressed_summary + " "
        
        for turn in self.history[-5:]:  # 只取最近5轮
            all_text += f"{turn['role']}: {turn['content']} "
        
        # 简化的状态表示：使用固定长度向量
        # 基于文本长度和轮次数生成特征向量
        features = [
            len(all_text) / 1000.0,  # 文本长度特征
            self.turn_count / 10.0,   # 轮次数特征
            self.token_count / 2000.0  # token数特征
        ]
        
        # 扩展到指定长度
        feature_vector = torch.zeros(max_length)
        for i, feat in enumerate(features[:max_length]):
            feature_vector[i] = feat
        
        return feature_vector

class RLTrainer:
    """强化学习训练器"""
    
    def __init__(self):
        self.action_space = CompressionActionSpace()
        self.reward_calculator = RewardCalculator()
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=10000)
        
        # 训练统计
        self.episode_rewards = []
        self.training_losses = []
        self.current_episode = 0
        
        # 探索参数
        self.epsilon = rl_config.epsilon_start
        
        # 创建训练目录
        self.training_dir = "rl_training"
        os.makedirs(self.training_dir, exist_ok=True)
        
        logger.info("🤖 RL训练器初始化完成")
    
    def select_compression_action(self, state: DialogState, training: bool = True) -> int:
        """选择压缩动作（使用ε-贪心策略）"""
        if training and random.random() < self.epsilon:
            # 探索：随机选择动作
            return self.action_space.sample_random_action()
        else:
            # 利用：基于当前策略选择最优动作
            return self._select_optimal_action(state)
    
    def _select_optimal_action(self, state: DialogState) -> int:
        """选择最优动作（简化版本）"""
        # 简化的策略：基于历史长度和token数选择
        if state.token_count > 2000:
            # 长对话，选择更高压缩比
            return 0  # 0.2压缩比 + recent_focus
        elif state.token_count > 1500:
            # 中等长度，平衡压缩
            return 7  # 0.3压缩比 + balanced
        else:
            # 短对话，轻度压缩
            return 11  # 0.4压缩比 + recent_focus
    
    def execute_compression_action(self, state: DialogState, action: int) -> Tuple[str, DialogState]:
        """执行压缩动作"""
        action_params = self.action_space.decode_action(action)
        
        # 构建专门的压缩提示
        compression_prompt = self._build_compression_prompt(
            state.history, 
            action_params
        )
        
        # 使用压缩模型生成摘要
        if model_manager.compressor_model is None:
            compressed_summary = "压缩模型未加载，无法生成摘要"
        else:
            compressed_summary = model_manager.generate_text(
                model=model_manager.compressor_model,
                prompt=compression_prompt,
                max_new_tokens=int(len(state.history) * action_params['compression_ratio'] * 10)
            )
        
        # 创建新状态（压缩后保留最近几轮）
        recent_history = state.history[-dialog_config.keep_recent_turns * 2:]
        new_state = DialogState(recent_history, compressed_summary)
        
        return compressed_summary, new_state
    
    def _build_compression_prompt(self, history: List[Dict], action_params: Dict) -> str:
        """构建压缩提示"""
        focus_instructions = {
            'recent_focus': "重点保留最近的对话内容和用户关切",
            'topic_focus': "重点保留主要话题和关键概念",
            'entity_focus': "重点保留人名、地名、专业术语等实体信息",
            'balanced': "平衡保留各类信息，确保摘要完整性"
        }
        
        history_text = ""
        for turn in history:
            role = "用户" if turn['role'] == 'user' else "助手"
            history_text += f"{role}: {turn['content']}\n"
        
        return f"""请对以下对话历史进行智能压缩摘要：

对话历史：
{history_text}

压缩要求：
- 压缩比例：{action_params['compression_ratio']*100:.0f}%
- 策略重点：{focus_instructions[action_params['focus_strategy']]}
- 保持语义连贯性和关键信息

摘要："""
    
    def calculate_reward(self, 
                        original_state: DialogState,
                        action: int,
                        compressed_summary: str,
                        dialog_response: str,
                        user_input: str) -> float:
        """计算奖励"""
        return self.reward_calculator.calculate_total_reward(
            original_state=original_state,
            action=action,
            compressed_summary=compressed_summary,
            dialog_response=dialog_response,
            user_input=user_input
        )
    
    def store_transition(self, state: DialogState, action: int, next_state: DialogState, reward: float):
        """存储转换到经验回放缓冲区"""
        transition = Transition(state, action, next_state, reward)
        self.memory.append(transition)
    
    def train_step(self) -> float:
        """执行一步训练"""
        if len(self.memory) < 100:  # 等待足够的经验
            return 0.0
        
        # 从经验回放中采样
        batch_size = min(32, len(self.memory))
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        
        # 计算损失（简化版本）
        rewards = torch.tensor(batch.reward, dtype=torch.float32)
        loss = F.mse_loss(rewards, torch.zeros_like(rewards))  # 简化的损失计算
        
        # 更新epsilon
        self._update_epsilon()
        
        return loss.item()
    
    def _update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(
            rl_config.epsilon_end,
            rl_config.epsilon_start - (self.current_episode / rl_config.epsilon_decay)
        )
    
    def train_episode(self, simulate_user_inputs: List[str]) -> Dict:
        """训练一个episode"""
        episode_reward = 0.0
        episode_steps = 0
        
        # 初始化对话状态
        current_state = DialogState([])
        
        logger.info(f"🎮 开始训练Episode {self.current_episode}")
        
        for user_input in simulate_user_inputs:
            # 添加用户输入到状态
            current_state.history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            })
            
            # 判断是否需要压缩
            if current_state.token_count > dialog_config.trigger_compression_tokens:
                # 选择压缩动作
                action = self.select_compression_action(current_state, training=True)
                
                # 执行压缩
                compressed_summary, compressed_state = self.execute_compression_action(
                    current_state, action
                )
                
                # 使用压缩后状态生成回复
                dialog_response = self._generate_dialog_response(compressed_state, user_input)
                
                # 计算奖励
                reward = self.calculate_reward(
                    current_state, action, compressed_summary, dialog_response, user_input
                )
                
                # 存储经验
                self.store_transition(current_state, action, compressed_state, reward)
                
                # 更新状态
                current_state = compressed_state
                episode_reward += reward
            else:
                # 无需压缩，直接对话
                dialog_response = self._generate_dialog_response(current_state, user_input)
                # 小的正奖励用于无压缩情况
                reward = 0.1
                episode_reward += reward
            
            # 添加助手回复到历史
            current_state.history.append({
                'role': 'assistant',
                'content': dialog_response,
                'timestamp': datetime.now().isoformat()
            })
            
            episode_steps += 1
        
        # 执行训练步骤
        loss = self.train_step()
        
        # 记录统计信息
        self.episode_rewards.append(episode_reward)
        self.training_losses.append(loss)
        self.current_episode += 1
        
        episode_stats = {
            'episode': self.current_episode,
            'total_reward': episode_reward,
            'steps': episode_steps,
            'loss': loss,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory)
        }
        
        logger.info(f"✅ Episode {self.current_episode} 完成: 奖励={episode_reward:.3f}, 损失={loss:.6f}")
        
        # 定期保存检查点
        if self.current_episode % rl_config.save_checkpoint_interval == 0:
            self.save_checkpoint()
        
        return episode_stats
    
    def _generate_dialog_response(self, state: DialogState, user_input: str) -> str:
        """生成对话回复"""
        # 构建上下文
        context = ""
        if state.compressed_summary:
            context += f"历史摘要：{state.compressed_summary}\n\n"
        
        # 添加最近对话
        for turn in state.history[-6:]:  # 最近3轮对话
            role = "用户" if turn['role'] == 'user' else "助手"
            context += f"{role}: {turn['content']}\n"
        
        context += f"用户: {user_input}\n助手:"
        
        # 生成回复
        if model_manager.dialog_model is None:
            response = "对话模型未加载，无法生成回复"
        else:
            response = model_manager.generate_text(
                model=model_manager.dialog_model,
                prompt=context,
                max_new_tokens=512
            )
        
        return response
    
    def save_checkpoint(self):
        """保存训练检查点"""
        checkpoint = {
            'episode': self.current_episode,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'training_losses': self.training_losses,
            'memory_size': len(self.memory),
            'config': {
                'learning_rate': rl_config.learning_rate,
                'discount_factor': rl_config.discount_factor
            }
        }
        
        checkpoint_path = os.path.join(
            self.training_dir, 
            f"checkpoint_episode_{self.current_episode}.json"
        )
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 检查点已保存: {checkpoint_path}")
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-100:]  # 最近100个episode
        
        return {
            'total_episodes': self.current_episode,
            'current_epsilon': self.epsilon,
            'avg_reward_recent': np.mean(recent_rewards),
            'max_reward': max(self.episode_rewards),
            'memory_utilization': len(self.memory) / 10000,
            'recent_loss': self.training_losses[-1] if self.training_losses else 0.0
        }

# 全局RL训练器实例
rl_trainer = RLTrainer() 