# 🤖 强化学习对话系统

基于强化学习的智能多轮对话系统，实现历史压缩与对话生成的联合优化。

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)

## ✨ 特色功能

### 🎯 新推出：直接Embedding压缩方案
- 🧠 **直接使用模型内部状态**：提取hidden states进行上下文压缩
- 💾 **智能State Bank**：存储和检索历史对话的向量表示
- 🔧 **四种融合策略**：attention、weighted_sum、concatenation、interpolation
- ⚡ **零依赖**：无需额外embedding模型，直接利用主模型
- 📏 **高效压缩**：固定长度向量 vs 可变长度文本

### 📋 原有功能  
- 🧠 **智能历史压缩**：基于强化学习的自适应压缩策略
- 🎯 **多策略压缩**：支持4种压缩策略和动态压缩比
- 🏆 **多维奖励机制**：质量+压缩+连贯性的综合优化
- 📊 **完整实验框架**：四对照组科学评估体系
- 🚀 **MPS/CUDA加速**：支持Apple Silicon和NVIDIA GPU

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>

# 安装依赖
pip install -r requirements.txt
```

### 2. 🎯 推荐：体验直接Embedding压缩

**这是基于您需求设计的核心方案：直接使用大模型internal state！**

```bash
# 直接Embedding压缩演示（推荐！）
python quick_start_direct_embedding.py

# 完整测试所有功能
python test_direct_embedding.py

# Jupyter实验对比
jupyter notebook rl_experiment.ipynb
```

### 3. 📋 传统RL方案体验

```bash
# 快速对话体验
python quick_start.py

# 完整系统演示
python main.py

# 强化学习训练
python rl_main.py
```

## 🏗️ 系统架构

### 🎯 直接Embedding压缩架构
```
📦 核心组件
├── 🧠 Direct Embedding Compressor  # 直接embedding压缩器
│   ├── 🏦 Hidden State Bank        # 状态银行
│   ├── 🔧 Fusion Strategies        # 融合策略
│   └── 🔍 Similarity Retrieval     # 相似度检索
├── 🎯 Dialog Manager              # 对话管理器
└── 🔧 Model Manager               # 模型管理器
```

### 📋 传统RL架构
```
📦 核心组件
├── 🎯 Dialog Manager    # 对话管理器
├── 🗜️ Compressor        # 历史压缩器  
├── 🤖 RL Trainer        # 强化学习训练器
├── 🏆 Reward Calculator # 奖励计算器
└── 🔧 Model Manager     # 模型管理器
```

## 📊 四对照组实验

| 组别 | 历史压缩 | RL训练 | 预期效果 |
|------|:--------:|:------:|----------|
| 基线组 | ❌ | ❌ | 基准性能 |
| 压缩组 | ✅ | ❌ | 压缩效果 (+15-25%) |
| 训练组 | ❌ | ✅ | RL优化 (+10-20%) |
| 完整组 | ✅ | ✅ | 最优性能 (+25-40%) |

## 💡 核心算法

### 压缩策略空间
- **压缩比**：[0.2, 0.3, 0.4, 0.5]
- **策略类型**：最近重点、主题重点、实体重点、平衡策略
- **动作空间**：16种策略组合 (4×4)

### 奖励函数设计
```python
总奖励 = 质量奖励×0.4 + 压缩奖励×0.3 + 连贯性奖励×0.3
```

## 📁 项目结构

```
vest.ai1/
🎯 直接Embedding方案
├── 🐍 direct_embedding_compressor.py    # 直接embedding压缩器
├── 🐍 test_direct_embedding.py          # 直接embedding测试
├── 🐍 quick_start_direct_embedding.py   # 快速启动脚本
├── 🐍 rl_trainer_embedding.py           # embedding增强RL训练器

📋 传统方案
├── 🐍 main.py                   # 主程序入口
├── 🐍 compressor.py             # 历史压缩器
├── 🐍 rl_trainer.py             # RL训练器
├── 🐍 embedding_compressor.py   # 传统embedding压缩器

🔧 共享组件
├── 🐍 config.py                 # 配置文件
├── 🐍 models.py                 # 模型管理器
├── 🐍 dialog_manager.py         # 对话管理器
├── 🐍 reward_calculator.py      # 奖励计算器
├── 📓 rl_experiment.ipynb       # 完整实验框架

📁 数据和日志
├── 📁 models/                   # 模型存储
├── 📁 rl_training/              # 训练记录
└── 📁 logs/                     # 日志文件
```

## 🎛️ 关键配置

```python
# 压缩触发条件
trigger_compression_tokens = 1500  # Token数阈值
keep_recent_turns = 3              # 保留最近轮次

# RL训练参数  
learning_rate = 0.001              # 学习率
epsilon_start = 1.0                # 初始探索率
epsilon_decay = 1000               # 探索衰减

# 模型设置
model_name = 'Qwen/Qwen2.5-0.5B-Instruct'  # 基础模型
```

## 📈 预期结果

### 性能排名
1. 🥇 **完整组** (压缩+RL) - 最优性能
2. 🥈 **训练组** (仅RL) - 中高性能  
3. 🥉 **压缩组** (仅压缩) - 中等性能
4. 📊 **基线组** (无优化) - 基准性能

### 评估指标
- **自动化指标**：BLEU、ROUGE、BERTScore
- **对话指标**：连贯性、信息保持、恰当性
- **压缩指标**：压缩比、信息损失、压缩质量

## 🔧 故障排除

### 常见问题
```bash
# 模型加载失败
python -c "from models import model_manager; model_manager.download_model()"

# 内存不足
# 在config.py中设置: torch_dtype = torch.float16

# 训练不收敛  
# 调整学习率: learning_rate = 0.0001
```

## 📚 详细文档

- 📖 [完整项目文档](PROJECT_OVERVIEW.md) - 详细的系统介绍和配置说明
- 📓 [实验指南](RL_EXPERIMENT_GUIDE.md) - 实验设计和评估方法
- 🚀 [快速入门](QUICKSTART.md) - 新手友好的使用指南

## 🤝 贡献

欢迎提交Issue和Pull Request！请查看[贡献指南](CONTRIBUTING.md)了解详情。

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)

---

**🎉 开始您的强化学习对话系统之旅！**

如有问题，请查看[详细文档](PROJECT_OVERVIEW.md)或提交Issue。 