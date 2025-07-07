"""
强化学习训练主程序
提供RL训练、评估和可视化功能
"""
import argparse
import logging
import json
import random
from typing import List
from datetime import datetime

from config import rl_config, model_config
from models import model_manager
from rl_trainer import rl_trainer, DialogState
from dialog_manager import DialogManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rl_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RLTrainingEnvironment:
    """RL训练环境"""
    
    def __init__(self):
        self.dialog_manager = DialogManager()
        
        # 模拟用户输入集合（用于训练）
        self.training_scenarios = self._load_training_scenarios()
        
        logger.info("🏋️ RL训练环境初始化完成")
    
    def _load_training_scenarios(self) -> List[List[str]]:
        """加载训练场景"""
        # 预定义的训练对话场景
        scenarios = [
            # 技术讨论场景
            [
                "你好，我想了解一下机器学习的基础知识",
                "什么是监督学习和无监督学习的区别？",
                "能给我举个具体的例子吗？",
                "那强化学习又是什么概念呢？",
                "强化学习在实际中有哪些应用？",
                "我想深入学习强化学习，有什么建议吗？"
            ],
            # 日常交流场景
            [
                "今天天气真不错",
                "你觉得周末去哪里玩比较好？",
                "我比较喜欢户外活动",
                "那你推荐一些适合的地方吧",
                "这些地方需要准备什么装备吗？",
                "谢谢你的建议，很有帮助"
            ],
            # 问题解决场景
            [
                "我的电脑最近运行很慢",
                "是什么原因导致的呢？",
                "我应该怎么检查内存使用情况？",
                "如果是硬盘空间不足怎么办？",
                "有什么软件可以帮助清理系统吗？",
                "这些方法我都试试，谢谢"
            ],
            # 长对话场景（测试压缩能力）
            [
                "我想创业做一个AI产品",
                "市场上有哪些类似的产品？",
                "我的想法是做一个智能助手",
                "需要什么技术栈？",
                "预算大概需要多少？",
                "团队需要哪些角色？",
                "如何进行市场推广？",
                "投资人一般关注什么？",
                "产品迭代周期怎么规划？",
                "如何处理用户反馈？",
                "数据安全和隐私怎么保证？",
                "法律合规方面要注意什么？"
            ]
        ]
        
        # 随机生成更多场景
        topics = [
            "编程", "旅游", "美食", "健康", "投资", "教育", 
            "科技", "艺术", "音乐", "体育", "历史", "文学"
        ]
        
        for topic in topics:
            scenario = [
                f"我对{topic}很感兴趣",
                f"能介绍一下{topic}的基础知识吗？",
                f"{topic}领域有哪些经典作品或案例？",
                f"初学者应该从哪里开始学习{topic}？",
                f"有什么实用的{topic}技巧可以分享吗？"
            ]
            scenarios.append(scenario)
        
        logger.info(f"📚 加载了 {len(scenarios)} 个训练场景")
        return scenarios
    
    def run_training(self, num_episodes = None):
        """运行RL训练"""
        if num_episodes is None:
            num_episodes = rl_config.training_episodes
        else:
            num_episodes = int(num_episodes)
        
        logger.info(f"🚀 开始RL训练，共 {num_episodes} 个episodes")
        
        training_stats = []
        
        for episode in range(num_episodes):
            # 随机选择训练场景
            scenario = random.choice(self.training_scenarios)
            
            # 训练一个episode
            episode_stats = rl_trainer.train_episode(scenario)
            training_stats.append(episode_stats)
            
            # 定期评估和日志
            if episode % rl_config.evaluation_interval == 0:
                self._evaluate_performance(episode)
                self._save_training_progress(training_stats)
        
        logger.info("✅ RL训练完成")
        self._save_final_results(training_stats)
    
    def _evaluate_performance(self, episode: int):
        """评估模型性能"""
        logger.info(f"📊 Episode {episode} 性能评估")
        
        # 获取训练统计
        stats = rl_trainer.get_training_stats()
        
        logger.info(f"  平均奖励: {stats.get('avg_reward_recent', 0):.3f}")
        logger.info(f"  当前探索率: {stats.get('current_epsilon', 0):.3f}")
        logger.info(f"  内存利用率: {stats.get('memory_utilization', 0)*100:.1f}%")
        
        # 运行评估对话
        test_scenario = [
            "你好，我想测试一下系统的对话能力",
            "请介绍一下你的功能特点",
            "在长对话中你是如何保持上下文连贯性的？",
            "压缩历史记录时会丢失重要信息吗？",
            "你认为当前的表现如何？"
        ]
        
        logger.info("🧪 运行评估对话...")
        current_state = DialogState([])
        
        for user_input in test_scenario:
            current_state.history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            })
            
            if current_state.token_count > 500:  # 触发压缩测试
                action = rl_trainer.select_compression_action(current_state, training=False)
                summary, new_state = rl_trainer.execute_compression_action(current_state, action)
                response = rl_trainer._generate_dialog_response(new_state, user_input)
                current_state = new_state
                logger.info(f"  🔄 压缩: {len(summary)} 字符摘要")
            else:
                response = rl_trainer._generate_dialog_response(current_state, user_input)
            
            current_state.history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"  用户: {user_input}")
            logger.info(f"  助手: {response[:100]}...")
    
    def _save_training_progress(self, stats: List):
        """保存训练进度"""
        progress_file = f"rl_training/training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    
    def _save_final_results(self, stats: List):
        """保存最终训练结果"""
        results = {
            'training_config': {
                'episodes': len(stats),
                'learning_rate': rl_config.learning_rate,
                'discount_factor': rl_config.discount_factor,
                'model': model_config.model_name
            },
            'final_stats': rl_trainer.get_training_stats(),
            'episode_history': stats
        }
        
        results_file = f"rl_training/final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 训练结果已保存: {results_file}")

def run_interactive_rl_dialog():
    """运行交互式RL对话（演示模式）"""
    logger.info("🎮 启动交互式RL对话演示")
    
    dialog_manager = DialogManager()
    
    print("=" * 60)
    print("🤖 强化学习双模型对话系统")
    print("=" * 60)
    print("特色功能：")
    print("✨ 智能历史压缩 - 无轮次限制对话")
    print("🎯 RL联合训练 - 压缩与对话协同优化") 
    print("🔄 实时适应 - 动态调整压缩策略")
    print("💬 输入 'quit' 退出，'stats' 查看RL统计")
    print("=" * 60)
    
    try:
        while True:
            user_input = input("\n👤 用户: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'stats':
                stats = rl_trainer.get_training_stats()
                print("\n📊 RL训练统计:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
                continue
            elif user_input.lower() == 'reset':
                dialog_manager.reset_conversation()
                print("🔄 对话已重置")
                continue
            
            if not user_input:
                continue
            
            # 生成回复（包含RL策略）
            response = dialog_manager.chat_with_rl(user_input)
            print(f"🤖 助手: {response}")
            
            # 显示系统状态
            state = dialog_manager.get_rl_state()
            print(f"📈 状态: {state['token_count']} tokens, "
                  f"压缩 {state['compression_count']} 次")
    
    except KeyboardInterrupt:
        print("\n\n👋 对话结束")
    except Exception as e:
        logger.error(f"对话过程中发生错误: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="强化学习双模型对话系统")
    parser.add_argument('--mode', choices=['train', 'chat', 'eval'], 
                       default='chat', help='运行模式')
    parser.add_argument('--episodes', type=int, default=None,
                       help='训练episodes数量')
    parser.add_argument('--scenario', type=str, default=None,
                       help='指定训练场景文件')
    
    args = parser.parse_args()
    
    try:
        # 初始化模型
        logger.info("🔧 初始化模型...")
        model_manager.load_models()
        
        if args.mode == 'train':
            # RL训练模式
            env = RLTrainingEnvironment()
            env.run_training(args.episodes)
            
        elif args.mode == 'chat':
            # 交互对话模式
            run_interactive_rl_dialog()
            
        elif args.mode == 'eval':
            # 评估模式
            env = RLTrainingEnvironment()
            env._evaluate_performance(0)
    
    except Exception as e:
        logger.error(f"程序运行错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 