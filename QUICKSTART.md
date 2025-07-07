# 🚀 快速启动指南

## 一键启动

```bash
# 1. 赋予脚本执行权限
chmod +x start.sh

# 2. 运行启动脚本
./start.sh
```

## 手动安装

### 步骤1: 创建虚拟环境

```bash
# 使用conda (推荐)
conda create -n dual_dialog python=3.9
conda activate dual_dialog

# 或使用venv
python3 -m venv dual_dialog_env
source dual_dialog_env/bin/activate
```

### 步骤2: 安装依赖

```bash
# 自动安装
python setup.py

# 或手动安装
pip install -r requirements.txt
```

### 步骤3: 运行系统

```bash
# 启动对话系统
python main.py

# 或运行测试
python test_dialog.py
```

## 核心功能演示

### 基本对话
```
👤 您: 你好，我想了解Python编程
🤖 助手: 您好！我很高兴为您介绍Python编程...

👤 您: 继续介绍Python的特点
🤖 助手: Python有以下主要特点...
```

### 历史压缩触发
```
# 当对话历史超过1200 tokens时
📦 触发历史压缩...
✅ 历史压缩完成，原始长度: 2500 字符，压缩后: 750 字符
```

### 查看统计
```
👤 您: stats
📊 对话统计信息：
• 会话ID: 20241201_143022
• 用户发言轮数: 8
• 助手回复轮数: 8
• 总Token数: 1845
• 历史压缩状态: ✅ 已激活
• 压缩摘要长度: 312 字符
```

## 系统命令

| 命令 | 功能 |
|------|------|
| `help` | 显示帮助信息 |
| `stats` | 查看对话统计 |
| `reset` | 重置对话历史 |
| `quit`/`exit` | 退出系统 |

## 配置调优

### 内存优化 (config.py)
```python
# 启用量化减少内存占用
use_quantization = True
quantization_bits = 4

# 调整上下文长度
max_length = 2048
```

### 压缩策略
```python
# 压缩触发阈值
trigger_compression_tokens = 1200

# 保留最近对话轮数
keep_recent_turns = 3

# 压缩比例
compression_ratio = 0.3
```

## 故障排除

### 常见问题

1. **模型下载慢**
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

2. **MPS不可用**
   ```python
   # 在config.py中设置
   device = "cpu"  # 强制使用CPU
   ```

3. **内存不足**
   - 启用量化: `use_quantization = True`
   - 减少批次大小
   - 关闭其他应用释放内存

### 日志检查

```bash
# 查看系统日志
tail -f system.log

# 查看对话日志
ls logs/
cat logs/dialog_20241201_143022.json
```

## 性能基准

### M4 Pro预期性能
- **推理速度**: 20-35 tokens/秒 (INT4量化)
- **内存占用**: ~1.5GB (含系统开销)
- **响应延迟**: ~150ms (首token)

### 压缩效果
- **压缩比**: 30% (可调)
- **压缩耗时**: ~2-3秒
- **语义保持**: 85%+ (主观评估)

## 目录结构

```
dual-dialog-system/
├── config.py          # 📝 系统配置
├── models.py           # 🤖 模型管理
├── compressor.py       # 📦 历史压缩
├── dialog_manager.py   # 💬 对话管理
├── main.py            # 🚀 主程序
├── test_dialog.py     # 🧪 测试脚本
├── setup.py           # ⚙️  安装脚本
├── start.sh           # 🏃 快速启动
├── models/            # 📁 模型缓存
├── logs/              # 📋 对话日志
└── system.log         # 🔍 系统日志
```

## 下一步

1. **体验基本功能**: 运行几轮对话测试
2. **触发压缩机制**: 进行长对话观察压缩效果
3. **调整配置**: 根据需要修改config.py
4. **查看日志**: 了解系统运行状态
5. **性能优化**: 根据硬件情况调优参数

🎉 **现在就开始体验您的双模型对话系统吧！** 