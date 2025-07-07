#!/bin/bash

# 双模型对话系统快速启动脚本
# 适用于Apple M4 Pro MacBook

echo "🤖 双模型对话系统快速启动"
echo "========================="

# 检查Python版本
python_version=$(python3 --version 2>&1)
echo "📋 Python版本: $python_version"

# 检查是否在虚拟环境中
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ 虚拟环境: $VIRTUAL_ENV"
else
    echo "⚠️  建议在虚拟环境中运行"
    echo "   创建虚拟环境: python3 -m venv dual_dialog_env"
    echo "   激活环境: source dual_dialog_env/bin/activate"
fi

# 检查依赖文件
if [ ! -f "requirements.txt" ]; then
    echo "❌ 错误: 找不到requirements.txt文件"
    exit 1
fi

# 提供选择菜单
echo ""
echo "请选择操作:"
echo "1) 安装依赖环境 (首次运行)"
echo "2) 启动交互式对话"
echo "3) 运行自动化测试"
echo "4) 查看项目结构"
echo "5) 退出"

read -p "请输入选择 (1-5): " choice

case $choice in
    1)
        echo "🔄 开始安装依赖..."
        python3 setup.py
        ;;
    2)
        echo "🚀 启动对话系统..."
        python3 main.py
        ;;
    3)
        echo "🧪 运行测试..."
        python3 test_dialog.py
        ;;
    4)
        echo "📁 项目结构:"
        tree -I "__pycache__|*.pyc|.git" . 2>/dev/null || ls -la
        ;;
    5)
        echo "👋 再见!"
        exit 0
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac 