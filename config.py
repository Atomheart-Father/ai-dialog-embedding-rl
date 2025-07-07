"""
项目配置文件
"""
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """模型配置"""
    # 模型相关
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # 目前使用可用模型，后续替换为Qwen3-0.6B
    model_cache_dir: str = "./models"
    use_quantization: bool = True
    quantization_bits: int = 4  # 4bit量化节省内存
    
    # 设备配置
    device: str = "auto"  # 自动检测设备
    use_mps: bool = True  # M4 Pro使用MPS加速
    
    # 推理参数
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # 压缩相关
    max_history_tokens: int = 1500  # 历史最大token数
    compression_ratio: float = 0.3  # 压缩比例
    min_compression_length: int = 200  # 最小压缩长度

@dataclass 
class DialogConfig:
    """对话配置"""
    # 移除最大对话轮数限制 - 实现无轮次数限制的对话
    save_logs: bool = True
    log_dir: str = "./logs"
    
    # 压缩触发条件
    trigger_compression_tokens: int = 1200  # 触发压缩的token数
    keep_recent_turns: int = 3  # 保留最近几轮对话不压缩

@dataclass
class RLConfig:
    """强化学习配置"""
    # 训练相关
    enable_rl_training: bool = True
    training_episodes: int = 1000
    learning_rate: float = 1e-4
    discount_factor: float = 0.95
    
    # 奖励权重
    quality_reward_weight: float = 0.4  # 回答质量奖励权重
    compression_reward_weight: float = 0.3  # 压缩效率奖励权重
    coherence_reward_weight: float = 0.3  # 连贯性奖励权重
    
    # 评估相关
    evaluation_interval: int = 100  # 每N个episode评估一次
    save_checkpoint_interval: int = 500  # 检查点保存间隔
    
    # 探索相关
    epsilon_start: float = 1.0  # 初始探索率
    epsilon_end: float = 0.1   # 最终探索率
    epsilon_decay: int = 500   # 探索率衰减步数

def get_device():
    """获取最佳可用设备"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

# 全局配置实例
model_config = ModelConfig()
dialog_config = DialogConfig()
rl_config = RLConfig()

# 更新设备配置
if model_config.device == "auto":
    model_config.device = get_device()
    
print(f"🚀 使用设备: {model_config.device}")
print(f"📱 模型: {model_config.model_name}")
print(f"🔧 量化: {'启用' if model_config.use_quantization else '禁用'}")
print(f"🤖 RL训练: {'启用' if rl_config.enable_rl_training else '禁用'}") 