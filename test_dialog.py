"""
测试脚本 - 自动化测试双模型对话系统
"""
import time
import logging
from colorama import init, Fore, Style
from models import model_manager
from dialog_manager import dialog_manager

# 初始化colorama
init(autoreset=True)

# 配置简化日志
logging.basicConfig(level=logging.WARNING)

def print_test_header(test_name: str):
    """打印测试头部"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"🧪 测试: {test_name}")
    print(f"{'='*60}{Style.RESET_ALL}\n")

def test_basic_dialog():
    """测试基本对话功能"""
    print_test_header("基本对话功能")
    
    test_inputs = [
        "你好！",
        "请介绍一下Python编程语言",
        "Python有哪些主要特点？",
        "能给我一个简单的Python代码示例吗？"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"{Fore.BLUE}👤 用户 [{i}]: {user_input}{Style.RESET_ALL}")
        
        response = dialog_manager.process_user_input(user_input)
        print(f"{Fore.GREEN}🤖 助手 [{i}]: {response}{Style.RESET_ALL}\n")
        
        time.sleep(1)  # 短暂暂停

def test_compression_trigger():
    """测试历史压缩触发"""
    print_test_header("历史压缩触发测试")
    
    # 重置对话以开始新测试
    dialog_manager.reset_dialog()
    
    # 生成足够长的对话来触发压缩
    long_conversation = [
        "请详细解释什么是机器学习？",
        "机器学习有哪些主要类型？请详细说明每种类型的特点和应用场景。",
        "监督学习和无监督学习有什么区别？请举例说明。",
        "深度学习是什么？它和传统机器学习有什么不同？",
        "神经网络的基本结构是怎样的？请解释神经元、层、权重等概念。",
        "反向传播算法是如何工作的？",
        "过拟合是什么问题？如何防止过拟合？",
        "现在请总结一下我们刚才讨论的所有机器学习相关内容。"
    ]
    
    for i, user_input in enumerate(long_conversation, 1):
        print(f"{Fore.BLUE}👤 用户 [{i}]: {user_input}{Style.RESET_ALL}")
        
        response = dialog_manager.process_user_input(user_input)
        print(f"{Fore.GREEN}🤖 助手 [{i}]: {response[:100]}{'...' if len(response) > 100 else ''}{Style.RESET_ALL}")
        
        # 显示当前统计
        stats = dialog_manager.get_dialog_stats()
        compression_status = "✅ 已激活" if stats['compression_active'] else "❌ 未激活"
        print(f"{Fore.YELLOW}📊 Token数: {stats['total_tokens']}, 压缩状态: {compression_status}{Style.RESET_ALL}\n")
        
        time.sleep(0.5)

def test_context_continuity():
    """测试上下文连续性"""
    print_test_header("上下文连续性测试")
    
    dialog_manager.reset_dialog()
    
    continuity_test = [
        "我的名字是张明，我是一名软件工程师。",
        "我正在学习人工智能和机器学习。",
        "请记住我刚才说的信息。我的职业是什么？",
        "我的名字是什么？",
        "我正在学习什么技术？"
    ]
    
    for i, user_input in enumerate(continuity_test, 1):
        print(f"{Fore.BLUE}👤 用户 [{i}]: {user_input}{Style.RESET_ALL}")
        
        response = dialog_manager.process_user_input(user_input)
        print(f"{Fore.GREEN}🤖 助手 [{i}]: {response}{Style.RESET_ALL}\n")
        
        time.sleep(0.5)

def run_performance_test():
    """运行性能测试"""
    print_test_header("性能测试")
    
    dialog_manager.reset_dialog()
    
    start_time = time.time()
    
    # 测试多轮快速对话
    quick_questions = [
        "1+1等于几？",
        "今天天气怎么样？",
        "推荐一本好书",
        "什么是AI？",
        "谢谢你的回答"
    ]
    
    for i, question in enumerate(quick_questions, 1):
        print(f"{Fore.BLUE}👤 [{i}]: {question}{Style.RESET_ALL}")
        
        question_start = time.time()
        response = dialog_manager.process_user_input(question)
        question_time = time.time() - question_start
        
        print(f"{Fore.GREEN}🤖 [{i}]: {response}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}⏱️  响应时间: {question_time:.2f}秒{Style.RESET_ALL}\n")
    
    total_time = time.time() - start_time
    print(f"{Fore.CYAN}📈 总测试时间: {total_time:.2f}秒{Style.RESET_ALL}")

def main():
    """主测试函数"""
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════╗
║                    🧪 双模型对话系统测试套件                      ║
║                                                                  ║
║  本测试将验证以下功能：                                           ║
║  • 基本对话功能                                                  ║
║  • 历史压缩触发机制                                              ║
║  • 上下文连续性                                                  ║
║  • 系统性能                                                      ║
╚══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")
    
    try:
        # 加载模型
        print(f"{Fore.YELLOW}🔄 正在加载模型进行测试...{Style.RESET_ALL}")
        if not model_manager.load_models():
            print(f"{Fore.RED}❌ 模型加载失败，测试中止{Style.RESET_ALL}")
            return
        
        print(f"{Fore.GREEN}✅ 模型加载完成，开始测试{Style.RESET_ALL}")
        
        # 运行各项测试
        test_basic_dialog()
        test_context_continuity()
        test_compression_trigger()
        run_performance_test()
        
        # 最终统计
        final_stats = dialog_manager.get_dialog_stats()
        print(f"\n{Fore.CYAN}📊 最终测试统计：")
        print(f"• 总对话轮数: {final_stats['user_turns']}")
        print(f"• 总Token数: {final_stats['total_tokens']}")
        print(f"• 压缩状态: {'✅ 已激活' if final_stats['compression_active'] else '❌ 未激活'}")
        print(f"• 会话ID: {final_stats['session_id']}")
        print(f"\n🎉 所有测试完成！{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"\n{Fore.RED}❌ 测试过程中出现错误: {e}{Style.RESET_ALL}")
    
    finally:
        # 清理资源
        model_manager.cleanup()
        print(f"\n{Fore.YELLOW}🧹 测试资源清理完成{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 