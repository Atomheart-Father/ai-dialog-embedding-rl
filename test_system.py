#!/usr/bin/env python3
"""
系统综合测试脚本
测试双模型对话系统的所有主要功能
"""
import os
import sys
import time
import logging
from typing import List, Dict

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """测试所有模块导入"""
    print("🔄 测试模块导入...")
    try:
        import config
        import models
        import compressor
        import dialog_manager
        import embedding_compressor
        import direct_embedding_compressor
        import rl_trainer
        import reward_calculator
        print("✅ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_model_loading():
    """测试模型加载"""
    print("\n🔄 测试模型加载...")
    try:
        from models import model_manager
        
        # 测试模型加载
        success = model_manager.load_models()
        if success:
            print("✅ 模型加载成功")
            print(f"   - Tokenizer: {'已加载' if model_manager.tokenizer else '未加载'}")
            print(f"   - 压缩模型: {'已加载' if model_manager.compressor_model else '未加载'}")
            print(f"   - 对话模型: {'已加载' if model_manager.dialog_model else '未加载'}")
            return True
        else:
            print("❌ 模型加载失败")
            return False
    except Exception as e:
        print(f"❌ 模型加载异常: {e}")
        return False

def test_token_counting():
    """测试token计数功能"""
    print("\n🔄 测试token计数...")
    try:
        from models import model_manager
        
        test_text = "这是一个测试文本，用来验证token计数功能。"
        token_count = model_manager.count_tokens(test_text)
        print(f"✅ Token计数测试成功: '{test_text}' = {token_count} tokens")
        return True
    except Exception as e:
        print(f"❌ Token计数失败: {e}")
        return False

def test_text_generation():
    """测试文本生成"""
    print("\n🔄 测试文本生成...")
    try:
        from models import model_manager
        
        if not model_manager.dialog_model:
            print("❌ 对话模型未加载，跳过文本生成测试")
            return False
        
        test_prompt = "你好，请简单介绍一下自己。"
        response = model_manager.generate_text(
            model=model_manager.dialog_model,
            prompt=test_prompt,
            max_new_tokens=100
        )
        
        if response:
            print(f"✅ 文本生成测试成功")
            print(f"   提示: {test_prompt}")
            print(f"   回复: {response[:100]}...")
            return True
        else:
            print("❌ 文本生成失败：空回复")
            return False
    except Exception as e:
        print(f"❌ 文本生成异常: {e}")
        return False

def test_compression():
    """测试历史压缩功能"""
    print("\n🔄 测试历史压缩...")
    try:
        from compressor import history_compressor
        
        # 创建测试历史
        test_history = [
            {'role': 'user', 'content': '你好，我想了解人工智能的发展历史。'},
            {'role': 'assistant', 'content': '人工智能的发展可以追溯到1950年代，当时图灵提出了著名的图灵测试...'},
            {'role': 'user', 'content': '那么机器学习和深度学习是什么时候兴起的？'},
            {'role': 'assistant', 'content': '机器学习在1980年代开始发展，而深度学习在2010年代迎来了突破...'},
            {'role': 'user', 'content': '现在AI在哪些领域应用最广泛？'},
            {'role': 'assistant', 'content': 'AI目前在计算机视觉、自然语言处理、推荐系统等领域应用广泛...'},
        ]
        
        # 测试是否需要压缩
        should_compress = history_compressor.should_compress(test_history)
        print(f"   历史长度: {len(test_history)} 轮")
        print(f"   是否需要压缩: {should_compress}")
        
        # 测试压缩功能
        if should_compress:
            summary, recent = history_compressor.compress_history(test_history)
            print(f"✅ 历史压缩测试完成")
            print(f"   压缩摘要长度: {len(summary)} 字符")
            print(f"   保留历史长度: {len(recent)} 轮")
        else:
            print("✅ 历史压缩测试完成（无需压缩）")
        
        return True
    except Exception as e:
        print(f"❌ 历史压缩测试失败: {e}")
        return False

def test_embedding_compression():
    """测试embedding压缩功能"""
    print("\n🔄 测试Embedding压缩...")
    try:
        from embedding_compressor import embedding_compressor
        
        # 测试文本embedding提取
        test_text = "这是一个测试用的对话内容。"
        embedding = embedding_compressor.extract_text_embedding(test_text)
        print(f"✅ Embedding提取成功，维度: {embedding.shape}")
        
        # 测试对话轮次压缩
        user_input = "你好，今天天气怎么样？"
        assistant_response = "今天天气很好，阳光明媚，适合出门活动。"
        
        embedding_data = embedding_compressor.compress_dialog_turn(user_input, assistant_response)
        print(f"✅ 对话轮次压缩成功")
        print(f"   用户输入: {user_input}")
        print(f"   助手回复: {assistant_response}")
        
        return True
    except Exception as e:
        print(f"❌ Embedding压缩测试失败: {e}")
        return False

def test_direct_embedding():
    """测试直接embedding压缩功能"""
    print("\n🔄 测试直接Embedding压缩...")
    try:
        from direct_embedding_compressor import direct_compressor
        
        # 测试hidden states提取
        test_text = "这是测试直接embedding压缩的文本。"
        hidden_states = direct_compressor.extract_dialog_hidden_states(test_text)
        print(f"✅ Hidden states提取成功，包含 {len(hidden_states)} 个状态")
        
        # 测试上下文生成
        user_input = "请介绍一下机器学习的基本概念。"
        context, metadata = direct_compressor.generate_enhanced_context(user_input)
        print(f"✅ 增强上下文生成成功")
        print(f"   融合策略: {metadata.get('fusion_strategy', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ 直接Embedding压缩测试失败: {e}")
        return False

def test_dialog_manager():
    """测试对话管理器"""
    print("\n🔄 测试对话管理器...")
    try:
        from dialog_manager import dialog_manager
        
        # 重置对话状态
        dialog_manager.reset_dialog()
        
        # 测试对话开始
        welcome = dialog_manager.start_dialog()
        print(f"✅ 对话开始: {welcome[:50]}...")
        
        # 测试简单对话
        test_inputs = [
            "你好",
            "你能做什么？",
            "谢谢"
        ]
        
        for i, user_input in enumerate(test_inputs):
            print(f"\n   第 {i+1} 轮对话:")
            print(f"   用户: {user_input}")
            response = dialog_manager.process_user_input(user_input)
            print(f"   助手: {response[:100]}...")
        
        # 获取对话统计
        stats = dialog_manager.get_dialog_stats()
        print(f"\n✅ 对话管理器测试完成")
        print(f"   对话轮数: {stats['user_turns']}")
        print(f"   总token数: {stats['total_tokens']}")
        
        return True
    except Exception as e:
        print(f"❌ 对话管理器测试失败: {e}")
        return False

def test_rl_components():
    """测试RL组件"""
    print("\n🔄 测试RL组件...")
    try:
        from rl_trainer import rl_trainer, DialogState
        from reward_calculator import RewardCalculator
        
        # 测试动作空间
        action_space = rl_trainer.action_space
        print(f"✅ 动作空间初始化成功，维度: {action_space.action_dim}")
        
        # 测试状态表示
        test_history = [
            {'role': 'user', 'content': '测试用户输入'},
            {'role': 'assistant', 'content': '测试助手回复'}
        ]
        state = DialogState(test_history)
        state_tensor = state.to_tensor()
        print(f"✅ 状态表示成功，维度: {state_tensor.shape}")
        
        # 测试奖励计算
        reward_calc = RewardCalculator()
        test_reward = reward_calc.calculate_total_reward(
            original_state=state,
            action=0,
            compressed_summary="测试压缩摘要",
            dialog_response="测试对话回复",
            user_input="测试用户输入"
        )
        print(f"✅ 奖励计算成功: {test_reward:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ RL组件测试失败: {e}")
        return False

def run_comprehensive_test():
    """运行综合测试"""
    print("🚀 开始系统综合测试\n")
    print("=" * 60)
    
    test_results = []
    
    # 运行所有测试
    tests = [
        ("模块导入", test_imports),
        ("模型加载", test_model_loading),
        ("Token计数", test_token_counting),
        ("文本生成", test_text_generation),
        ("历史压缩", test_compression),
        ("Embedding压缩", test_embedding_compression),
        ("直接Embedding", test_direct_embedding),
        ("对话管理器", test_dialog_manager),
        ("RL组件", test_rl_components),
    ]
    
    for test_name, test_func in tests:
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            duration = end_time - start_time
            
            test_results.append({
                'name': test_name,
                'result': result,
                'duration': duration
            })
            
            if result:
                print(f"   ⏱️  耗时: {duration:.2f}秒")
            
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            test_results.append({
                'name': test_name,
                'result': False,
                'duration': 0
            })
        
        print("-" * 60)
    
    # 输出测试结果摘要
    print("\n📊 测试结果摘要:")
    print("=" * 60)
    
    passed = sum(1 for r in test_results if r['result'])
    total = len(test_results)
    success_rate = passed / total * 100
    
    for result in test_results:
        status = "✅ 通过" if result['result'] else "❌ 失败"
        duration = f"{result['duration']:.2f}s" if result['duration'] > 0 else "N/A"
        print(f"{result['name']:15} {status:8} {duration:>8}")
    
    print("-" * 60)
    print(f"总体结果: {passed}/{total} 通过 ({success_rate:.1f}%)")
    
    if success_rate >= 70:
        print("🎉 系统整体功能正常！")
    else:
        print("⚠️ 系统存在较多问题，需要进一步检查。")
    
    return success_rate >= 70

def quick_functionality_test():
    """快速功能测试（不加载完整模型）"""
    print("🏃‍♂️ 快速功能测试（跳过模型加载）\n")
    print("=" * 60)
    
    # 只测试核心逻辑，不测试模型相关功能
    quick_tests = [
        ("模块导入", test_imports),
        ("历史压缩逻辑", test_compression),
        ("RL组件逻辑", test_rl_components),
    ]
    
    results = []
    for test_name, test_func in quick_tests:
        try:
            result = test_func()
            results.append(result)
            print("-" * 60)
        except Exception as e:
            print(f"❌ {test_name}失败: {e}")
            results.append(False)
            print("-" * 60)
    
    passed = sum(results)
    total = len(results)
    print(f"\n快速测试结果: {passed}/{total} 通过")
    
    return all(results)

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = quick_functionality_test()
    else:
        success = run_comprehensive_test()
    
    sys.exit(0 if success else 1) 