#!/usr/bin/env python3
"""
快速启动脚本 - 演示强化学习对话系统实验
"""

def main():
    print("🚀 强化学习对话系统实验 - 快速启动")
    print("=" * 50)
    
    print("📚 当前系统的强化学习奖励机制：")
    print("  质量奖励 (40%): 长度合理性 + 相关性 + 流畅性")
    print("  压缩奖励 (30%): 压缩比0.2-0.4最优 + Token节省")
    print("  连贯性奖励 (30%): 摘要一致性 + 上下文连贯性 + 主题连续性")
    
    print("\n🧪 四个对照组设计：")
    print("  1. 基线组: 无压缩 + 无RL训练")
    print("  2. 压缩组: 有压缩 + 无RL训练")
    print("  3. 训练组: 无压缩 + 有RL训练")
    print("  4. 完整组: 有压缩 + 有RL训练")
    
    print("\n📊 测试数据集：")
    print("  - technical_discussion: 技术概念保持能力")
    print("  - problem_solving: 逻辑推理能力")
    print("  - casual_conversation: 日常对话语境保持")
    print("  - knowledge_qa: 知识一致性")
    print("  - mixed_topics: 主题切换适应性")
    
    print("\n💡 使用方法：")
    print("  1. 安装依赖: pip install -r requirements.txt")
    print("  2. 运行Jupyter: jupyter notebook")
    print("  3. 打开: rl_experiment.ipynb")
    print("  4. 或运行: python run_experiment.py")
    
    print("\n📈 预期结果：")
    print("  完整组 > 训练组 > 压缩组 > 基线组")
    print("  预计改进: 压缩15-25%, RL 10-20%, 组合25-40%")
    
    print("\n📋 输出报告将包含：")
    print("  - 性能对比图表")
    print("  - 统计分析结果")
    print("  - 详细实验数据")
    print("  - 可视化Loss曲线")
    
    print("\n🎯 核心创新点：")
    print("  - 双模型RL架构")
    print("  - 智能历史压缩")
    print("  - 16种动作空间")
    print("  - 多维度奖励函数")
    
    print("\n✅ 开始你的实验之旅吧！")

if __name__ == "__main__":
    main() 