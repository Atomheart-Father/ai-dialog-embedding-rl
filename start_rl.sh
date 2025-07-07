#!/bin/bash

# 强化学习双模型对话系统启动脚本
# 支持训练、对话和评估模式

echo "🤖 强化学习双模型对话系统"
echo "================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3"
    echo "请安装Python 3.8+"
    exit 1
fi

# 检查是否在虚拟环境中
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ 虚拟环境: $VIRTUAL_ENV"
else
    echo "⚠️  建议在虚拟环境中运行"
    echo "创建虚拟环境: python3 -m venv rl_env"
    echo "激活: source rl_env/bin/activate"
fi

# 安装依赖
echo "📦 检查依赖..."
pip install -r requirements.txt --quiet

# 创建必要目录
mkdir -p logs
mkdir -p models
mkdir -p rl_training

echo ""
echo "🎯 选择运行模式:"
echo "1) 交互对话 (默认)"
echo "2) RL训练"
echo "3) 性能评估"
echo "4) 查看训练历史"
echo ""

read -p "请选择模式 [1-4]: " mode

case $mode in
    2)
        echo "🏋️ 启动RL训练模式"
        read -p "训练episodes数量 [默认:100]: " episodes
        episodes=${episodes:-100}
        python3 rl_main.py --mode train --episodes $episodes
        ;;
    3)
        echo "📊 启动性能评估"
        python3 rl_main.py --mode eval
        ;;
    4)
        echo "📈 查看训练历史"
        if [ -d "rl_training" ]; then
            echo "训练检查点:"
            ls -la rl_training/checkpoint_*.json 2>/dev/null || echo "无训练检查点"
            echo ""
            echo "训练结果:"
            ls -la rl_training/final_results_*.json 2>/dev/null || echo "无训练结果"
        else
            echo "未找到训练目录"
        fi
        ;;
    *)
        echo "🎮 启动交互对话模式"
        python3 rl_main.py --mode chat
        ;;
esac

echo ""
echo "👋 RL系统已结束" 