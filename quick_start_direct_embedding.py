#!/usr/bin/env python3
"""
直接Embedding压缩方案快速启动脚本
展示如何使用大模型内部hidden states进行上下文压缩
"""
import os
import sys
import time
from datetime import datetime

def main():
    print("🎯 直接Embedding压缩方案 - 快速体验")
    print("=" * 60)
    print("💡 这个方案直接使用大模型的hidden states压缩上下文")
    print("⚡ 无需额外模型，直接利用模型内部表示")
    print()
    
    try:
        # 导入模块
        from direct_embedding_compressor import direct_compressor
        from models import model_manager
        
        print("📥 正在加载模型...")
        start_time = time.time()
        
        try:
            model_manager.load_models()
            print(f"✅ 模型加载完成 ({time.time() - start_time:.2f}s)")
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            print("🔄 将使用模拟模式进行演示")
        
        print("\n" + "=" * 60)
        
        # 模拟一个对话场景
        print("🎬 场景演示: 编程学习咨询")
        
        # 1. 建立历史对话
        print("\n📚 建立历史对话上下文...")
        history_dialogs = [
            {'role': 'user', 'content': '我是编程新手，想学习Python'},
            {'role': 'assistant', 'content': '很好的选择！Python语法简洁，适合初学者入门'},
            {'role': 'user', 'content': '我应该先学哪些基础概念？'},
            {'role': 'assistant', 'content': '建议先掌握变量、数据类型、循环和条件判断'},
            {'role': 'user', 'content': '有什么好的学习资源推荐？'},
            {'role': 'assistant', 'content': '推荐《Python编程快速上手》和官方文档'}
        ]
        
        for i, turn in enumerate(history_dialogs):
            role = "用户" if turn['role'] == 'user' else "助手"
            print(f"  {role}: {turn['content']}")
        
        # 2. 压缩历史为hidden states
        print(f"\n🧠 将 {len(history_dialogs)//2} 轮对话压缩为hidden states...")
        start_time = time.time()
        compressed_states = direct_compressor.compress_history_to_states(history_dialogs)
        compression_time = time.time() - start_time
        
        print(f"✅ 压缩完成 ({compression_time:.3f}s)")
        print(f"📦 生成了 {len(compressed_states)} 个state条目")
        
        # 3. 测试不同融合策略
        print(f"\n🔧 测试4种融合策略...")
        new_query = "我想学习数据结构和算法"
        
        strategies = ['attention', 'weighted_sum', 'concatenation', 'interpolation']
        results = {}
        
        for strategy in strategies:
            print(f"\n  策略: {strategy}")
            
            # 切换策略
            direct_compressor.switch_fusion_strategy(strategy)
            
            # 生成增强上下文
            start_time = time.time()
            enhanced_prompt, metadata = direct_compressor.generate_enhanced_context(new_query)
            end_time = time.time()
            
            results[strategy] = {
                'time': end_time - start_time,
                'metadata': metadata,
                'prompt': enhanced_prompt
            }
            
            print(f"    处理时间: {results[strategy]['time']:.3f}s")
            print(f"    融合使用: {'是' if metadata['fusion_used'] else '否'}")
            
            if metadata['fusion_used']:
                print(f"    相关states: {metadata['num_context_states']}")
                print(f"    增强强度: {metadata['enhanced_state_norm']:.3f}")
            
            print(f"    生成prompt长度: {len(enhanced_prompt)} 字符")
            
            # 显示生成的prompt预览
            print(f"    生成的增强prompt:")
            lines = enhanced_prompt.split('\n')
            for line in lines[:4]:
                print(f"      {line}")
            if len(lines) > 4:
                print(f"      ... (共{len(lines)}行)")
        
        # 4. 性能对比
        print(f"\n📊 策略性能对比:")
        print(f"{'策略':<15} {'时间(s)':<10} {'长度':<8} {'使用融合'}")
        print("-" * 45)
        
        for strategy, result in results.items():
            metadata = result['metadata']
            fusion_used = "是" if metadata['fusion_used'] else "否"
            print(f"{strategy:<15} {result['time']:<10.3f} {len(result['prompt']):<8} {fusion_used}")
        
        # 5. 显示State Bank统计
        print(f"\n💾 State Bank统计信息:")
        stats = direct_compressor.get_compression_statistics()
        if stats and 'state_bank_info' in stats:
            bank_info = stats['state_bank_info']
            print(f"  存储states: {bank_info.get('total_states', 0)} 个")
            print(f"  State维度: {bank_info.get('state_dimension', 0)}")
            print(f"  内存使用: {bank_info.get('memory_usage_mb', 0):.2f} MB")
            print(f"  平均访问: {bank_info.get('avg_access_count', 0):.1f} 次")
        
        # 6. 压缩效率分析
        if stats and 'compression_efficiency' in stats:
            efficiency = stats['compression_efficiency']
            print(f"\n⚡ 压缩效率:")
            print(f"  压缩比: {efficiency.get('compression_ratio', 1.0):.3f}")
            print(f"  内存节省: {efficiency.get('memory_savings_mb', 0):.2f} MB")
        
        # 7. 交互式测试
        print(f"\n🎮 交互式测试 (输入 'quit' 退出):")
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['quit', 'exit', '退出']:
                    break
                
                if not user_input:
                    continue
                
                # 使用最佳策略 (attention)
                direct_compressor.switch_fusion_strategy('attention')
                
                start_time = time.time()
                enhanced_prompt, metadata = direct_compressor.generate_enhanced_context(user_input)
                end_time = time.time()
                
                print(f"\n🤖 系统响应 ({end_time-start_time:.3f}s):")
                print(f"  融合状态: {'启用' if metadata['fusion_used'] else '未启用'}")
                
                if metadata['fusion_used']:
                    print(f"  相关历史: {len(metadata['relevant_turns'])} 条")
                    print(f"  相关内容: {', '.join(metadata['relevant_turns'])}")
                
                print(f"\n📝 增强后的prompt:")
                lines = enhanced_prompt.split('\n')
                for line in lines[:6]:
                    print(f"  {line}")
                if len(lines) > 6:
                    print(f"  ... (共{len(lines)}行)")
                
            except KeyboardInterrupt:
                print(f"\n👋 感谢使用！")
                break
            except Exception as e:
                print(f"\n❌ 处理错误: {e}")
        
        print(f"\n" + "=" * 60)
        print(f"🎉 直接Embedding压缩方案演示完成！")
        print(f"\n✨ 核心特点:")
        print(f"🧠 直接使用大模型hidden states")
        print(f"💾 智能State Bank存储管理")
        print(f"🔧 多种融合策略可选")
        print(f"⚡ 高效压缩，显著节省token")
        print(f"🎯 完全符合您的设计理念！")
        
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
    
    except Exception as e:
        print(f"❌ 运行错误: {e}")

if __name__ == "__main__":
    main() 