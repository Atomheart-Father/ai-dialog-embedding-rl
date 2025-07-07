"""
双模型对话系统 - 主运行脚本
基于Qwen模型的历史压缩对话系统
"""
import sys
import logging
from colorama import init, Fore, Style
from models import model_manager
from dialog_manager import dialog_manager

# 初始化colorama
init(autoreset=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def print_banner():
    """打印系统横幅"""
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════╗
║                    🤖 双模型对话系统 v1.0                        ║
║                                                                  ║
║  ✨ 特性：                                                       ║
║     • 智能历史压缩，避免上下文溢出                                ║
║     • 双模型架构：压缩器 + 主对话模型                             ║
║     • 完全本地化，无外部依赖                                      ║
║     • 适配 Apple M4 Pro 芯片                                     ║
║                                                                  ║
║  🚀 正在启动系统...                                              ║
╚══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)

def print_help():
    """打印帮助信息"""
    help_text = f"""
{Fore.YELLOW}📚 使用说明：
• 直接输入您的问题开始对话
• 输入 'stats' 查看对话统计
• 输入 'reset' 重置对话历史
• 输入 'help' 显示此帮助
• 输入 'quit' 或 'exit' 退出系统{Style.RESET_ALL}
"""
    print(help_text)

def handle_special_commands(user_input: str) -> bool:
    """处理特殊命令，返回是否继续对话"""
    command = user_input.lower().strip()
    
    if command == 'help':
        print_help()
        return True
    
    elif command == 'stats':
        stats = dialog_manager.get_dialog_stats()
        print(f"\n{Fore.GREEN}📊 对话统计信息：")
        print(f"• 会话ID: {stats['session_id']}")
        print(f"• 用户发言轮数: {stats['user_turns']}")
        print(f"• 助手回复轮数: {stats['assistant_turns']}")
        print(f"• 总Token数: {stats['total_tokens']}")
        print(f"• 历史压缩状态: {'✅ 已激活' if stats['compression_active'] else '❌ 未激活'}")
        if stats['compression_active']:
            print(f"• 压缩摘要长度: {stats['compressed_summary_length']} 字符")
        print(f"{Style.RESET_ALL}")
        return True
    
    elif command == 'reset':
        dialog_manager.reset_dialog()
        print(f"\n{Fore.GREEN}✅ 对话历史已重置！{Style.RESET_ALL}\n")
        return True
    
    return False

def main():
    """主函数"""
    print_banner()
    
    try:
        # 加载模型
        print(f"{Fore.YELLOW}🔄 正在加载模型，请稍候...{Style.RESET_ALL}")
        if not model_manager.load_models():
            print(f"{Fore.RED}❌ 模型加载失败，请检查网络连接和配置{Style.RESET_ALL}")
            return
        
        print(f"{Fore.GREEN}✅ 模型加载完成！{Style.RESET_ALL}")
        
        # 开始对话
        welcome_msg = dialog_manager.start_dialog()
        print(f"\n{Fore.GREEN}{welcome_msg}{Style.RESET_ALL}")
        print_help()
        
        # 主对话循环
        while True:
            try:
                # 获取用户输入
                user_input = input(f"\n{Fore.BLUE}👤 您: {Style.RESET_ALL}").strip()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if handle_special_commands(user_input):
                    continue
                
                # 检查退出命令
                if user_input.lower() in ['quit', 'exit', '退出', '再见']:
                    break
                
                # 处理用户输入并获取回复
                print(f"{Fore.YELLOW}🤖 正在思考...{Style.RESET_ALL}", end="", flush=True)
                response = dialog_manager.process_user_input(user_input)
                
                # 清除"思考中"提示并显示回复
                print(f"\r{' ' * 20}\r", end="")  # 清除提示
                print(f"{Fore.GREEN}🤖 助手: {Style.RESET_ALL}{response}")
                
            except KeyboardInterrupt:
                print(f"\n\n{Fore.YELLOW}⚠️  检测到中断信号{Style.RESET_ALL}")
                break
            except EOFError:
                print(f"\n\n{Fore.YELLOW}⚠️  输入结束{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"\n{Fore.RED}❌ 处理错误: {e}{Style.RESET_ALL}")
                logger.error(f"主循环错误: {e}")
                continue
        
        # 结束对话
        end_msg = dialog_manager._end_dialog()
        print(f"\n{Fore.CYAN}{end_msg}{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"\n{Fore.RED}❌ 系统启动失败: {e}{Style.RESET_ALL}")
        logger.error(f"系统启动失败: {e}")
    
    finally:
        # 清理资源
        try:
            model_manager.cleanup()
            print(f"\n{Fore.YELLOW}🧹 资源清理完成{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"资源清理失败: {e}")

if __name__ == "__main__":
    main() 