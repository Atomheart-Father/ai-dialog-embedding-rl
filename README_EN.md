# 🤖 Reinforcement Learning Dialog System

An intelligent multi-turn dialog system based on reinforcement learning that jointly optimizes historical compression and dialog generation.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)

**Language**: [🇨🇳 中文](README.md) | [🇺🇸 English](README_EN.md)

## ✨ Key Features

### 🎯 New: Direct Embedding Compression Solution
- 🧠 **Direct Model Internal States**: Extract hidden states for context compression
- 💾 **Smart State Bank**: Store and retrieve vector representations of historical dialogs
- 🔧 **Four Fusion Strategies**: attention, weighted_sum, concatenation, interpolation
- ⚡ **Zero Dependencies**: No additional embedding models, directly utilize main model
- 📏 **Efficient Compression**: Fixed-length vectors vs variable-length text

### 📋 Legacy Features  
- 🧠 **Smart Historical Compression**: Adaptive compression strategies based on reinforcement learning
- 🎯 **Multi-Strategy Compression**: Supports 4 compression strategies and dynamic compression ratios
- 🏆 **Multi-Dimensional Reward System**: Comprehensive optimization of quality + compression + coherence
- 📊 **Complete Experimental Framework**: Scientific evaluation system with four control groups
- 🚀 **MPS/CUDA Acceleration**: Supports Apple Silicon and NVIDIA GPU

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the project
git clone https://github.com/Atomheart-Father/ai-dialog-embedding-rl.git
cd ai-dialog-embedding-rl

# Install dependencies
pip install -r requirements.txt
```

### 2. 🎯 Recommended: Experience Direct Embedding Compression

**This is the core solution designed based on your requirements: directly using large model internal states!**

```bash
# Direct Embedding compression demo (Recommended!)
python quick_start_direct_embedding.py

# Complete testing of all features
python test_direct_embedding.py

# Jupyter experimental comparison
jupyter notebook rl_experiment.ipynb
```

### 3. 📋 Traditional RL Solution Experience

```bash
# Quick dialog experience
python quick_start.py

# Complete system demonstration
python main.py

# Reinforcement learning training
python rl_main.py
```

## 🏗️ System Architecture

### 🎯 Direct Embedding Compression Architecture
```
📦 Core Components
├── 🧠 Direct Embedding Compressor  # Direct embedding compressor
│   ├── 🏦 Hidden State Bank        # State bank
│   ├── 🔧 Fusion Strategies        # Fusion strategies
│   └── 🔍 Similarity Retrieval     # Similarity retrieval
├── 🎯 Dialog Manager              # Dialog manager
└── 🔧 Model Manager               # Model manager
```

### 📋 Traditional RL Architecture
```
📦 Core Components
├── 🎯 Dialog Manager    # Dialog manager
├── 🗜️ Compressor        # Historical compressor  
├── 🤖 RL Trainer        # Reinforcement learning trainer
├── 🏆 Reward Calculator # Reward calculator
└── 🔧 Model Manager     # Model manager
```

## 📊 Four Control Group Experiments

| Group | Historical Compression | RL Training | Expected Effect |
|-------|:---------------------:|:-----------:|----------------|
| Baseline | ❌ | ❌ | Benchmark performance |
| Compression | ✅ | ❌ | Compression effect (+15-25%) |
| Training | ❌ | ✅ | RL optimization (+10-20%) |
| Complete | ✅ | ✅ | Optimal performance (+25-40%) |

## 💡 Core Algorithms

### Compression Strategy Space
- **Compression Ratio**: [0.2, 0.3, 0.4, 0.5]
- **Strategy Types**: Recent focus, Topic focus, Entity focus, Balanced strategy
- **Action Space**: 16 strategy combinations (4×4)

### Reward Function Design
```python
Total Reward = Quality Reward×0.4 + Compression Reward×0.3 + Coherence Reward×0.3
```

## 📁 Project Structure

```
vest.ai1/
🎯 Direct Embedding Solution
├── 🐍 direct_embedding_compressor.py    # Direct embedding compressor
├── 🐍 test_direct_embedding.py          # Direct embedding tests
├── 🐍 quick_start_direct_embedding.py   # Quick start script
├── 🐍 rl_trainer_embedding.py           # Embedding-enhanced RL trainer

📋 Traditional Solution
├── 🐍 main.py                   # Main program entry
├── 🐍 compressor.py             # Historical compressor
├── 🐍 rl_trainer.py             # RL trainer
├── 🐍 embedding_compressor.py   # Traditional embedding compressor

🔧 Shared Components
├── 🐍 config.py                 # Configuration file
├── 🐍 models.py                 # Model manager
├── 🐍 dialog_manager.py         # Dialog manager
├── 🐍 reward_calculator.py      # Reward calculator
├── 📓 rl_experiment.ipynb       # Complete experimental framework

📁 Data and Logs
├── 📁 models/                   # Model storage
├── 📁 rl_training/              # Training records
└── 📁 logs/                     # Log files
```

## 🎛️ Key Configuration

```python
# Compression trigger conditions
trigger_compression_tokens = 1500  # Token threshold
keep_recent_turns = 3              # Keep recent turns

# RL training parameters  
learning_rate = 0.001              # Learning rate
epsilon_start = 1.0                # Initial exploration rate
epsilon_decay = 1000               # Exploration decay

# Model settings
model_name = 'Qwen/Qwen2.5-0.5B-Instruct'  # Base model
```

## 📈 Expected Results

### Performance Ranking
1. 🥇 **Complete Group** (Compression + RL) - Optimal performance
2. 🥈 **Training Group** (RL only) - Medium-high performance  
3. 🥉 **Compression Group** (Compression only) - Medium performance
4. 📊 **Baseline Group** (No optimization) - Benchmark performance

### Evaluation Metrics
- **Automated Metrics**: BLEU, ROUGE, BERTScore
- **Dialog Metrics**: Coherence, Information retention, Appropriateness
- **Compression Metrics**: Compression ratio, Information loss, Compression quality

## 🔧 Troubleshooting

### Common Issues
```bash
# Model loading failure
python -c "from models import model_manager; model_manager.download_model()"

# Out of memory
# Set in config.py: torch_dtype = torch.float16

# Training not converging  
# Adjust learning rate: learning_rate = 0.0001
```

## 📚 Detailed Documentation

- 📖 [Complete Project Documentation](PROJECT_OVERVIEW.md) - Detailed system introduction and configuration
- 📓 [Experiment Guide](RL_EXPERIMENT_GUIDE.md) - Experimental design and evaluation methods
- 🚀 [Quick Start Guide](QUICKSTART.md) - Beginner-friendly usage guide

## 🤝 Contributing

We welcome Issues and Pull Requests! Please check the [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the [MIT License](LICENSE)

---

**🎉 Start your reinforcement learning dialog system journey!**

For questions, please check the [detailed documentation](PROJECT_OVERVIEW.md) or submit an Issue. 