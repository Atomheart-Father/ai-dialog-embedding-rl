#!/usr/bin/env python3
"""
双模型对话系统安装脚本
自动检查环境并安装依赖
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 9):
        print("❌ 错误: 需要Python 3.9或更高版本")
        print(f"   当前版本: {sys.version}")
        return False
    print(f"✅ Python版本检查通过: {sys.version.split()[0]}")
    return True

def check_system():
    """检查系统环境"""
    system = platform.system()
    machine = platform.machine()
    
    print(f"🖥️  系统信息: {system} {machine}")
    
    if system == "Darwin" and "arm" in machine.lower():
        print("✅ 检测到Apple Silicon芯片，将启用MPS加速")
        return True
    elif system == "Darwin":
        print("⚠️  检测到Intel Mac，MPS可能不可用")
        return True
    else:
        print("⚠️  非Mac系统，将使用CPU模式")
        return True

def run_command(command, description):
    """运行命令并显示进度"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ {description}完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description}失败:")
        print(f"   错误: {e.stderr}")
        return False

def install_dependencies():
    """安装依赖包"""
    print("\n📦 开始安装依赖包...")
    
    # 检查requirements.txt
    if not Path("requirements.txt").exists():
        print("❌ 错误: 找不到requirements.txt文件")
        return False
    
    # 升级pip
    if not run_command("pip install --upgrade pip", "升级pip"):
        return False
    
    # 安装PyTorch (Apple Silicon优化版本)
    system = platform.system()
    machine = platform.machine()
    
    if system == "Darwin" and "arm" in machine.lower():
        torch_command = "pip install torch torchvision torchaudio"
    else:
        torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    if not run_command(torch_command, "安装PyTorch"):
        return False
    
    # 安装其他依赖
    if not run_command("pip install -r requirements.txt", "安装其他依赖包"):
        return False
    
    return True

def test_imports():
    """测试关键包导入"""
    print("\n🧪 测试包导入...")
    
    test_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("colorama", "Colorama"),
        ("accelerate", "Accelerate")
    ]
    
    for package, name in test_packages:
        try:
            __import__(package)
            print(f"✅ {name}导入成功")
        except ImportError as e:
            print(f"❌ {name}导入失败: {e}")
            return False
    
    return True

def test_mps():
    """测试MPS可用性"""
    print("\n🚀 测试MPS加速...")
    try:
        import torch
        if torch.backends.mps.is_available():
            print("✅ MPS加速可用")
            return True
        else:
            print("⚠️  MPS加速不可用，将使用CPU模式")
            return True
    except Exception as e:
        print(f"❌ MPS测试失败: {e}")
        return False

def create_directories():
    """创建必要的目录"""
    print("\n📁 创建目录结构...")
    
    directories = ["models", "logs"]
    
    for directory in directories:
        try:
            Path(directory).mkdir(exist_ok=True)
            print(f"✅ 创建目录: {directory}")
        except Exception as e:
            print(f"❌ 创建目录失败 {directory}: {e}")
            return False
    
    return True

def show_next_steps():
    """显示后续步骤"""
    print(f"""
🎉 安装完成！

📋 后续步骤:
1. 运行系统:
   python main.py

2. 运行测试:
   python test_dialog.py

3. 查看帮助:
   python main.py
   然后输入 'help'

⚠️  注意事项:
• 首次运行会下载模型文件（约1-2GB）
• 确保网络连接稳定
• 模型下载可能需要几分钟时间

🔧 如有问题:
• 查看README.md获取详细说明
• 检查system.log日志文件
• 确认M4 Pro芯片和24GB+内存

祝您使用愉快！🚀
""")

def main():
    """主安装流程"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    🤖 双模型对话系统安装程序                      ║
║                                                                  ║
║  本程序将自动检查环境并安装所需依赖                               ║
║  适用于Apple M4 Pro MacBook                                     ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # 检查基本环境
    if not check_python_version():
        sys.exit(1)
    
    if not check_system():
        sys.exit(1)
    
    # 安装依赖
    if not install_dependencies():
        print("\n❌ 依赖安装失败，请检查网络连接和权限")
        sys.exit(1)
    
    # 测试导入
    if not test_imports():
        print("\n❌ 包导入测试失败，请检查安装")
        sys.exit(1)
    
    # 测试MPS
    if not test_mps():
        print("\n⚠️  MPS测试失败，但不影响使用")
    
    # 创建目录
    if not create_directories():
        print("\n❌ 目录创建失败")
        sys.exit(1)
    
    # 显示后续步骤
    show_next_steps()

if __name__ == "__main__":
    main() 