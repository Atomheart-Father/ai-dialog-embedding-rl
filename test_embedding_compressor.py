"""
Embedding压缩器测试脚本
测试基于向量空间的上下文压缩性能
"""
import sys
import time
import numpy as np
from typing import List, Dict

# 导入项目模块
from embedding_compressor import embedding_compressor
from models import model_manager
from config import model_config

def test_embedding_extraction():
    """测试embedding提取功能"""
    print("🧪 测试1: Embedding提取功能")
    
    test_texts = [
        "你好，我想了解Python编程",
        "强化学习是机器学习的一个重要分支",
        "今天天气很好，适合出门散步",
        "深度学习模型需要大量的计算资源"
    ]
    
    print(f"  测试文本数量: {len(test_texts)}")
    
    for i, text in enumerate(test_texts):
        start_time = time.time()
        embedding = embedding_compressor.extract_text_embedding(text)
        end_time = time.time()
        
        print(f"  文本{i+1}: 维度={embedding.shape}, 时间={end_time-start_time:.3f}s")
        print(f"    内容: {text[:30]}...")
        print(f"    向量范围: [{embedding.min():.3f}, {embedding.max():.3f}]")
    
    print("✅ Embedding提取测试完成\n")

def test_dialog_compression():
    """测试对话压缩功能"""
    print("🧪 测试2: 对话压缩功能")
    
    # 模拟多轮对话
    dialog_history = [
        ("我想学习机器学习，从哪里开始比较好？", "建议从Python基础开始，然后学习numpy、pandas等库"),
        ("我已经会Python了，可以直接学习算法吗？", "可以的，建议先学习线性回归、决策树等基础算法"),
        ("监督学习和无监督学习有什么区别？", "监督学习有标签数据，无监督学习没有标签，用于发现数据模式"),
        ("深度学习需要什么基础？", "需要线性代数、概率论基础，以及对神经网络的理解"),
        ("强化学习适合什么场景？", "适合决策优化场景，如游戏AI、机器人控制、推荐系统等")
    ]
    
    print(f"  对话轮数: {len(dialog_history)}")
    
    # 逐轮压缩
    compressed_data = []
    for i, (user_input, assistant_response) in enumerate(dialog_history):
        embedding_data = embedding_compressor.compress_dialog_turn(user_input, assistant_response)
        compressed_data.append(embedding_data)
        
        metadata = embedding_data['metadata']
        print(f"  轮次{i+1}: {metadata['turn_summary']}")
        print(f"    Token数: {metadata['token_count']}, 向量维度: {embedding_data['embedding'].shape}")
    
    # 计算压缩统计
    stats = embedding_compressor.get_compression_stats()
    if stats:
        print(f"\n📊 压缩统计:")
        print(f"  总embedding数: {stats['total_embeddings']}")
        print(f"  原始token数: {stats['original_tokens']}")
        print(f"  等效token数: {stats['embedding_equivalent_tokens']}")
        print(f"  压缩比: {stats['memory_efficiency']}")
    
    print("✅ 对话压缩测试完成\n")
    return compressed_data

def test_similarity_retrieval(compressed_data: List[Dict]):
    """测试相似度检索功能"""
    print("🧪 测试3: 相似度检索功能")
    
    # 更新历史embedding
    embedding_compressor.current_session_embeddings = compressed_data
    embedding_compressor.update_history_embeddings()
    
    # 测试查询
    test_queries = [
        "什么是机器学习？",
        "深度学习和神经网络的关系",
        "推荐系统是怎么工作的？",
        "我想了解算法基础知识"
    ]
    
    for query in test_queries:
        print(f"\n  查询: {query}")
        relevant = embedding_compressor.retrieve_relevant_embeddings(query, top_k=3)
        
        if relevant:
            for i, emb_data in enumerate(relevant):
                metadata = emb_data['metadata']
                print(f"    相关{i+1}: {metadata['turn_summary']}")
        else:
            print("    未找到相关历史")
    
    print("\n✅ 相似度检索测试完成\n")

def test_context_generation():
    """测试上下文生成功能"""
    print("🧪 测试4: 上下文生成功能")
    
    test_inputs = [
        "我想深入了解神经网络",
        "强化学习有哪些具体应用？",
        "数据预处理有什么技巧？"
    ]
    
    for input_text in test_inputs:
        print(f"\n  输入: {input_text}")
        context = embedding_compressor.generate_context_with_embeddings(input_text)
        
        print(f"  生成的上下文:")
        lines = context.split('\n')
        for line in lines[:8]:  # 只显示前8行
            print(f"    {line}")
        if len(lines) > 8:
            print(f"    ... (共{len(lines)}行)")
    
    print("\n✅ 上下文生成测试完成\n")

def test_clustering_analysis(compressed_data: List[Dict]):
    """测试聚类分析功能"""
    print("🧪 测试5: 聚类分析功能")
    
    try:
        from sklearn.cluster import KMeans
        
        if len(compressed_data) >= 3:
            cluster_result = embedding_compressor.cluster_embeddings(compressed_data, n_clusters=2)
            
            print(f"  聚类结果:")
            for cluster_id, items in cluster_result['clusters'].items():
                print(f"    簇{cluster_id + 1} ({len(items)}个对话):")
                for item in items:
                    print(f"      - {item['metadata']['turn_summary']}")
        else:
            print("  对话数量不足，跳过聚类测试")
    
    except ImportError:
        print("  Sklearn未安装，跳过聚类测试")
    
    print("✅ 聚类分析测试完成\n")

def test_performance_comparison():
    """测试性能对比"""
    print("🧪 测试6: 性能对比")
    
    test_text = "机器学习是人工智能的一个重要分支，它使用算法和统计模型来让计算机系统逐步改善特定任务的性能。通过分析和识别数据中的模式，机器学习算法可以在没有明确编程的情况下做出预测或决策。"
    
    # 测试embedding提取速度
    times = []
    for _ in range(5):
        start_time = time.time()
        embedding = embedding_compressor.extract_text_embedding(test_text)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    print(f"  Embedding提取平均时间: {avg_time:.4f}s")
    print(f"  文本长度: {len(test_text)} 字符")
    print(f"  压缩比: 1个向量 vs ~{len(test_text)//4} tokens")
    
    # 内存使用估算
    embedding_size = embedding.numel() * 4  # float32
    text_size = len(test_text.encode('utf-8'))
    
    print(f"  向量内存: {embedding_size} bytes")
    print(f"  文本内存: {text_size} bytes") 
    print(f"  内存效率: {embedding_size/text_size:.2f}x")
    
    print("✅ 性能对比测试完成\n")

def run_comprehensive_test():
    """运行综合测试"""
    print("🚀 开始Embedding压缩器综合测试\n")
    print("=" * 50)
    
    # 初始化模型
    print("📥 正在加载模型...")
    try:
        model_manager.load_models()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("使用模拟模式进行测试...")
    
    print("\n" + "=" * 50)
    
    # 运行各项测试
    test_embedding_extraction()
    compressed_data = test_dialog_compression()
    test_similarity_retrieval(compressed_data)
    test_context_generation()
    test_clustering_analysis(compressed_data)
    test_performance_comparison()
    
    print("🎉 所有测试完成！")
    print("\n📊 Embedding压缩器特点总结:")
    print("✅ 固定长度向量表示，节省token")
    print("✅ 语义相似度检索，智能匹配")
    print("✅ 聚类分析，发现对话主题")
    print("✅ 快速压缩，无需文本生成")
    print("✅ 可扩展存储，持久化保存")

if __name__ == "__main__":
    run_comprehensive_test() 