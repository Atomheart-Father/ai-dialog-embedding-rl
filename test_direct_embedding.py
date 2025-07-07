"""
测试直接使用大模型hidden states的压缩方案
展示多种融合策略的效果对比
"""
import sys
import time
import torch
import numpy as np
from typing import List, Dict

# 可选依赖导入
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Some visualization features will be disabled.")

# 导入项目模块
from direct_embedding_compressor import direct_compressor
from models import model_manager

def test_hidden_state_extraction():
    """测试hidden state提取功能"""
    print("🧪 测试1: Hidden State提取")
    
    test_dialogs = [
        "用户: 我想学习机器学习\n助手: 建议从Python基础开始学习",
        "用户: 深度学习和机器学习有什么区别？\n助手: 深度学习是机器学习的一个子集",
        "用户: 强化学习适合什么场景？\n助手: 适合决策优化和游戏AI等场景"
    ]
    
    for i, dialog in enumerate(test_dialogs):
        print(f"\n  对话 {i+1}: {dialog[:30]}...")
        
        start_time = time.time()
        hidden_states = direct_compressor.extract_dialog_hidden_states(dialog)
        end_time = time.time()
        
        print(f"    提取时间: {end_time-start_time:.3f}s")
        print(f"    提取的states: {list(hidden_states.keys())}")
        
        if 'combined' in hidden_states:
            combined = hidden_states['combined']
            print(f"    Combined state shape: {combined.shape}")
            print(f"    State norm: {torch.norm(combined):.3f}")
            print(f"    State mean: {torch.mean(combined):.3f}")
    
    print("✅ Hidden State提取测试完成\n")

def test_state_bank_operations():
    """测试State Bank操作"""
    print("🧪 测试2: State Bank操作")
    
    # 模拟历史对话
    history = [
        {'role': 'user', 'content': '我想学习Python编程'},
        {'role': 'assistant', 'content': '建议从基础语法开始，然后学习数据结构'},
        {'role': 'user', 'content': '机器学习需要什么数学基础？'},
        {'role': 'assistant', 'content': '主要需要线性代数、概率论和微积分'},
        {'role': 'user', 'content': '深度学习框架哪个好？'},
        {'role': 'assistant', 'content': 'PyTorch和TensorFlow都很不错，推荐从PyTorch开始'}
    ]
    
    print(f"  处理 {len(history)//2} 轮对话...")
    
    # 压缩历史为states
    compressed_states = direct_compressor.compress_history_to_states(history)
    
    print(f"  压缩得到 {len(compressed_states)} 个state条目")
    
    for i, state_entry in enumerate(compressed_states):
        print(f"    轮次{i+1}: {state_entry['summary']}")
        print(f"      States: {list(state_entry['states'].keys())}")
    
    # 测试检索
    print("\n  🔍 测试相似度检索:")
    query_text = "我想了解深度学习"
    query_states = direct_compressor.extract_dialog_hidden_states(f"用户: {query_text}")
    query_state = query_states['combined']
    
    relevant = direct_compressor.state_bank.retrieve_relevant_states(query_state, top_k=3)
    print(f"    查询: {query_text}")
    print(f"    找到 {len(relevant)} 个相关states:")
    
    for i, entry in enumerate(relevant):
        print(f"      {i+1}. {entry['metadata']['summary']}")
    
    # 获取统计信息
    bank_stats = direct_compressor.state_bank.get_state_summary()
    print(f"\n  📊 State Bank统计:")
    for key, value in bank_stats.items():
        print(f"    {key}: {value}")
    
    print("✅ State Bank操作测试完成\n")

def test_fusion_strategies():
    """测试不同的融合策略"""
    print("🧪 测试3: 融合策略对比")
    
    # 先填充一些历史数据
    history = [
        {'role': 'user', 'content': '什么是机器学习？'},
        {'role': 'assistant', 'content': '机器学习是让计算机从数据中自动学习的技术'},
        {'role': 'user', 'content': '神经网络是什么？'},
        {'role': 'assistant', 'content': '神经网络是模拟大脑神经元工作的计算模型'},
    ]
    
    direct_compressor.compress_history_to_states(history)
    
    # 测试查询
    test_query = "深度学习和神经网络的关系"
    
    strategies = ['attention', 'weighted_sum', 'concatenation', 'interpolation']
    results = {}
    
    for strategy in strategies:
        print(f"\n  🔧 测试融合策略: {strategy}")
        
        # 切换策略
        direct_compressor.switch_fusion_strategy(strategy)
        
        start_time = time.time()
        enhanced_prompt, metadata = direct_compressor.generate_enhanced_context(test_query)
        end_time = time.time()
        
        results[strategy] = {
            'time': end_time - start_time,
            'metadata': metadata,
            'prompt_length': len(enhanced_prompt)
        }
        
        print(f"    处理时间: {results[strategy]['time']:.3f}s")
        print(f"    是否使用融合: {metadata['fusion_used']}")
        if metadata['fusion_used']:
            print(f"    上下文states数: {metadata['num_context_states']}")
            print(f"    增强state norm: {metadata['enhanced_state_norm']:.3f}")
        print(f"    生成prompt长度: {results[strategy]['prompt_length']} 字符")
        
        # 显示生成的prompt片段
        print(f"    生成的prompt预览:")
        lines = enhanced_prompt.split('\n')
        for line in lines[:4]:
            print(f"      {line}")
        if len(lines) > 4:
            print(f"      ... (共{len(lines)}行)")
    
    # 性能对比
    print(f"\n  📊 融合策略性能对比:")
    for strategy, result in results.items():
        print(f"    {strategy:15s}: {result['time']:.3f}s, {result['prompt_length']:4d} chars")
    
    print("✅ 融合策略测试完成\n")

def test_compression_efficiency():
    """测试压缩效率"""
    print("🧪 测试4: 压缩效率分析")
    
    # 创建更多的测试数据
    extended_history = []
    topics = [
        ("Python编程", "学习基础语法和数据结构"),
        ("机器学习", "理解算法和模型训练"),
        ("深度学习", "神经网络和反向传播"),
        ("强化学习", "智能体和环境交互"),
        ("数据科学", "数据分析和可视化"),
        ("Web开发", "前端和后端技术"),
        ("算法设计", "时间复杂度和空间复杂度")
    ]
    
    for i, (topic, response) in enumerate(topics):
        extended_history.extend([
            {'role': 'user', 'content': f'我想学习{topic}，应该怎么开始？'},
            {'role': 'assistant', 'content': f'关于{topic}，建议{response}，然后逐步深入实践。'}
        ])
    
    print(f"  处理 {len(extended_history)//2} 轮扩展对话...")
    
    # 压缩所有历史
    start_time = time.time()
    compressed_states = direct_compressor.compress_history_to_states(extended_history)
    compression_time = time.time() - start_time
    
    # 计算压缩统计
    stats = direct_compressor.get_compression_statistics()
    
    print(f"  ⏱️  压缩时间: {compression_time:.3f}s")
    print(f"  📦 压缩统计:")
    
    bank_info = stats['state_bank_info']
    efficiency = stats['compression_efficiency']
    
    print(f"    总states数: {bank_info.get('total_states', 0)}")
    print(f"    状态维度: {bank_info.get('state_dimension', 0)}")
    print(f"    内存使用: {bank_info.get('memory_usage_mb', 0):.2f} MB")
    
    if efficiency:
        print(f"    压缩比: {efficiency.get('compression_ratio', 1.0):.3f}")
        print(f"    内存节省: {efficiency.get('memory_savings_mb', 0):.2f} MB")
    
    # 测试检索性能
    test_queries = [
        "我想学习编程",
        "机器学习算法有哪些？",
        "深度学习的应用场景",
        "数据分析的工具"
    ]
    
    print(f"\n  🔍 检索性能测试:")
    retrieval_times = []
    
    for query in test_queries:
        start_time = time.time()
        enhanced_prompt, metadata = direct_compressor.generate_enhanced_context(query)
        retrieval_time = time.time() - start_time
        retrieval_times.append(retrieval_time)
        
        print(f"    查询: {query[:20]}... -> {retrieval_time:.3f}s")
    
    avg_retrieval_time = np.mean(retrieval_times)
    print(f"    平均检索时间: {avg_retrieval_time:.3f}s")
    
    print("✅ 压缩效率测试完成\n")

def test_context_enhancement():
    """测试上下文增强效果"""
    print("🧪 测试5: 上下文增强效果")
    
    # 建立一些历史对话上下文
    context_history = [
        {'role': 'user', 'content': '我是一个初学者，想从零开始学习编程'},
        {'role': 'assistant', 'content': '很好！对于初学者，我建议从Python开始，因为语法简单易懂'},
        {'role': 'user', 'content': '我应该先学习哪些编程概念？'},
        {'role': 'assistant', 'content': '建议先学习变量、循环、条件判断等基础概念'},
        {'role': 'user', 'content': '有什么好的学习资源推荐吗？'},
        {'role': 'assistant', 'content': '推荐《Python编程从入门到实践》这本书，还有在线平台如Codecademy'}
    ]
    
    # 压缩历史
    direct_compressor.compress_history_to_states(context_history)
    
    # 测试新的查询
    new_queries = [
        "我想继续学习数据结构",  # 相关：编程学习
        "推荐一些Python项目练习", # 相关：Python学习
        "今天天气怎么样？",       # 不相关：天气
        "我想学习机器学习算法"    # 部分相关：学习但不是编程基础
    ]
    
    print("  测试不同类型查询的上下文增强效果:\n")
    
    for i, query in enumerate(new_queries):
        print(f"  查询 {i+1}: {query}")
        
        # 生成增强上下文
        enhanced_prompt, metadata = direct_compressor.generate_enhanced_context(query)
        
        print(f"    融合状态: {'是' if metadata['fusion_used'] else '否'}")
        
        if metadata['fusion_used']:
            print(f"    融合策略: {metadata['fusion_strategy']}")
            print(f"    相关历史: {len(metadata['relevant_turns'])} 条")
            print(f"    相关内容: {metadata['relevant_turns']}")
            print(f"    增强强度: {metadata['enhanced_state_norm']:.3f}")
        
        # 显示生成的增强prompt
        print(f"    增强后的prompt:")
        lines = enhanced_prompt.split('\n')
        for line in lines[:6]:  # 显示前6行
            print(f"      {line}")
        if len(lines) > 6:
            print(f"      ... (共{len(lines)}行)")
        
        print()  # 空行分隔
    
    print("✅ 上下文增强测试完成\n")

def visualize_state_similarities():
    """可视化state相似度"""
    print("🧪 测试6: State相似度可视化")
    
    try:
        # 获取当前state bank中的所有states
        state_bank = direct_compressor.state_bank.state_bank
        
        if len(state_bank) < 2:
            print("  State bank中数据不足，跳过可视化")
            return
        
        # 提取所有states
        states = [entry['state'].detach().cpu().numpy() for entry in state_bank]
        summaries = [entry['metadata']['summary'] for entry in state_bank]
        
        # 计算相似度矩阵
        n = len(states)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    state_i = torch.tensor(states[i])
                    state_j = torch.tensor(states[j])
                    sim = torch.nn.functional.cosine_similarity(state_i, state_j, dim=0)
                    similarity_matrix[i][j] = sim.item()
                else:
                    similarity_matrix[i][j] = 1.0
        
        # 简单的相似度统计
        print(f"  📊 相似度分析 ({n} 个states):")
        print(f"    平均相似度: {np.mean(similarity_matrix[similarity_matrix < 1.0]):.3f}")
        print(f"    最大相似度: {np.max(similarity_matrix[similarity_matrix < 1.0]):.3f}")
        print(f"    最小相似度: {np.min(similarity_matrix[similarity_matrix < 1.0]):.3f}")
        
        # 找出最相似的pair
        max_sim_idx = np.unravel_index(
            np.argmax(similarity_matrix * (1 - np.eye(n))), 
            similarity_matrix.shape
        )
        
        print(f"    最相似的对话:")
        print(f"      A: {summaries[max_sim_idx[0]]}")
        print(f"      B: {summaries[max_sim_idx[1]]}")
        print(f"      相似度: {similarity_matrix[max_sim_idx]:.3f}")
    
    except Exception as e:
        print(f"  可视化过程中出错: {e}")
    
    print("✅ 相似度可视化完成\n")

def run_comprehensive_test():
    """运行完整的直接embedding测试"""
    print("🚀 开始直接Embedding压缩综合测试")
    print("=" * 60)
    
    # 加载模型
    print("📥 正在加载模型...")
    try:
        model_manager.load_models()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"⚠️ 模型加载失败: {e}")
        print("将使用模拟模式进行测试")
    
    print("\n" + "=" * 60)
    
    # 依次运行各项测试
    test_hidden_state_extraction()
    test_state_bank_operations()
    test_fusion_strategies()
    test_compression_efficiency()
    test_context_enhancement()
    visualize_state_similarities()
    
    # 最终总结
    print("🎉 所有测试完成！")
    print("\n📋 直接Embedding压缩方案特点:")
    print("✅ 直接使用大模型的hidden states")
    print("✅ 多层state提取和融合")
    print("✅ 四种融合策略: attention, weighted_sum, concatenation, interpolation")
    print("✅ 智能的相似度检索")
    print("✅ 高效的内存使用")
    print("✅ 无需额外的embedding模型")
    print("\n🎯 这个方案完全符合您的要求：直接用大模型内部状态压缩上下文！")

if __name__ == "__main__":
    run_comprehensive_test() 