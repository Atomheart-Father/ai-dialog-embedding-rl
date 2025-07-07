"""
对话管理器
协调压缩器和主对话模型，实现完整的对话流程
"""
import json
import os
from datetime import datetime
from typing import List, Dict
import logging
from config import dialog_config
from models import model_manager
from compressor import history_compressor

logger = logging.getLogger(__name__)


class DialogManager:
    """对话管理器"""

    def __init__(self):
        self.dialog_history: List[Dict[str, str]] = []
        self.compressed_summary = ""
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.turn_count = 0

        # 创建日志目录
        if dialog_config.save_logs:
            os.makedirs(dialog_config.log_dir, exist_ok=True)

        # 系统提示
        self.system_prompt = """你是一个智能助手，请根据提供的对话历史和当前问题，给出有帮助、准确且友好的回答。
如果有对话历史摘要，请参考其中的信息来保持对话的连贯性。"""

    def start_dialog(self) -> str:
        """开始对话会话"""
        welcome_msg = "🤖 双模型对话系统已启动！\n我可以记住我们之前的对话内容，请随时与我交流。\n\n输入 'quit' 或 'exit' 退出对话。"
        logger.info(f"🚀 开始新的对话会话: {self.session_id}")
        return welcome_msg

    def process_user_input(self, user_input: str) -> str:
        """处理用户输入并返回回复"""
        if not user_input.strip():
            return "请输入您的问题。"

        # 检查退出命令
        if user_input.lower().strip() in ['quit', 'exit', '退出', '再见']:
            return self._end_dialog()

        try:
            # 增加轮次计数
            self.turn_count += 1
            logger.info(f"🔄 处理第 {self.turn_count} 轮对话")

            # 添加用户输入到历史
            self.dialog_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            })

            # 检查是否需要压缩历史
            if history_compressor.should_compress(self.dialog_history):
                logger.info("📦 触发历史压缩...")
                self.compressed_summary, recent_history = history_compressor.compress_history(
                    self.dialog_history[:-1]  # 不包含当前用户输入
                )
                # 更新历史记录，只保留最近的对话
                self.dialog_history = recent_history + [self.dialog_history[-1]]

            # 构建对话上下文
            context = self._build_dialog_context(user_input)

            # 生成回复
            if model_manager.dialog_model is not None:
                response = model_manager.generate_text(
                    model=model_manager.dialog_model,
                    prompt=context,
                    max_new_tokens=512
                )
            else:
                response = ""

            if not response:
                response = "抱歉，我现在无法生成回复，请稍后再试。"

            # 添加助手回复到历史
            self.dialog_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })

            # 保存对话日志
            if dialog_config.save_logs:
                self._save_dialog_log()

            return response

        except Exception as e:
            logger.error(f"❌ 处理用户输入失败: {e}")
            return "抱歉，处理您的请求时出现了错误，请重试。"

    def _build_dialog_context(self, user_input: str) -> str:
        """构建对话上下文"""
        context_parts = [self.system_prompt]

        # 添加压缩的历史摘要和最近对话
        if self.compressed_summary or self.dialog_history[:-1]:  # 不包含当前用户输入
            formatted_context = history_compressor.format_context_for_dialog(
                self.compressed_summary,
                self.dialog_history[:-1]
            )
            if formatted_context:
                context_parts.append(formatted_context)

        # 添加当前用户输入
        context_parts.append(f"用户: {user_input}")
        context_parts.append("助手:")

        return "\n\n".join(context_parts)

    def _end_dialog(self) -> str:
        """结束对话"""
        # 保存最终日志
        if dialog_config.save_logs:
            self._save_dialog_log()

        # 获取对话统计
        total_turns = len([turn for turn in self.dialog_history if turn['role'] == 'user'])

        end_msg = f"👋 对话已结束！\n📊 本次对话统计：\n- 对话轮数：{total_turns}\n- 历史压缩：{'是' if self.compressed_summary else '否'}\n- 会话ID：{self.session_id}"

        logger.info(f"✅ 对话会话结束: {self.session_id}")
        return end_msg

    def _save_dialog_log(self):
        """保存对话日志"""
        try:
            log_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'turn_count': self.turn_count,
                'compressed_summary': self.compressed_summary,
                'dialog_history': self.dialog_history,
                'config': {
                    'model_name': getattr(model_manager.tokenizer, 'name_or_path', 'unknown') if model_manager.tokenizer else "unknown",
                    'compression_enabled': bool(self.compressed_summary)
                }
            }

            log_file = os.path.join(dialog_config.log_dir, f"dialog_{self.session_id}.json")
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"❌ 保存对话日志失败: {e}")

    def get_dialog_stats(self) -> Dict:
        """获取对话统计信息"""
        user_turns = len([turn for turn in self.dialog_history if turn['role'] == 'user'])
        assistant_turns = len([turn for turn in self.dialog_history if turn['role'] == 'assistant'])

        total_tokens = sum(model_manager.count_tokens(turn['content']) for turn in self.dialog_history)

        return {
            'session_id': self.session_id,
            'user_turns': user_turns,
            'assistant_turns': assistant_turns,
            'total_tokens': total_tokens,
            'compression_active': bool(self.compressed_summary),
            'compressed_summary_length': len(self.compressed_summary) if self.compressed_summary else 0
        }

    def reset_dialog(self):
        """重置对话状态"""
        self.dialog_history.clear()
        self.compressed_summary = ""
        self.turn_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"🔄 对话状态已重置，新会话ID: {self.session_id}")

    # RL相关方法
    def reset_conversation(self):
        """重置对话历史（RL模式）"""
        self.reset_dialog()

    def chat_with_rl(self, user_input: str) -> str:
        """使用RL策略进行对话"""
        try:
            # 动态导入避免循环依赖
            from rl_trainer import rl_trainer, DialogState

            # 构建当前状态
            current_state = DialogState(self.dialog_history.copy(), self.compressed_summary)

            # 添加用户输入
            user_turn = {
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            }
            current_state.history.append(user_turn)

            # 判断是否需要压缩（使用RL策略）
            if current_state.token_count > dialog_config.trigger_compression_tokens:
                # 使用RL选择压缩动作
                action = rl_trainer.select_compression_action(current_state, training=False)

                # 执行压缩
                compressed_summary, new_state = rl_trainer.execute_compression_action(current_state, action)

                # 更新状态
                self.dialog_history = new_state.history.copy()
                self.compressed_summary = compressed_summary
                self.turn_count += 1

                logger.info(f"🤖 RL压缩执行: 动作={action}, 压缩后={len(compressed_summary)}字符")

                # 使用压缩后状态生成回复
                response = rl_trainer._generate_dialog_response(new_state, user_input)
            else:
                # 无需压缩，直接对话
                response = rl_trainer._generate_dialog_response(current_state, user_input)
                # 更新历史
                self.dialog_history.append(user_turn)
                self.turn_count += 1

            # 添加助手回复到历史
            assistant_turn = {
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            }
            self.dialog_history.append(assistant_turn)

            # 保存对话日志
            if dialog_config.save_logs:
                self._save_dialog_log()

            return response

        except Exception as e:
            logger.error(f"RL对话过程中发生错误: {e}")
            # 回退到普通对话模式
            return self.process_user_input(user_input)

    def get_rl_state(self) -> Dict:
        """获取RL状态信息"""
        try:
            # 动态导入避免循环依赖
            from rl_trainer import rl_trainer

            stats = rl_trainer.get_training_stats()

            # 计算token数
            total_tokens = sum(model_manager.count_tokens(turn['content']) for turn in self.dialog_history)
            if self.compressed_summary:
                total_tokens += model_manager.count_tokens(self.compressed_summary)

            return {
                'token_count': total_tokens,
                'compression_count': 1 if self.compressed_summary else 0,
                'turn_count': len([h for h in self.dialog_history if h['role'] == 'user']),
                'has_summary': bool(self.compressed_summary),
                'summary_length': len(self.compressed_summary) if self.compressed_summary else 0,
                'rl_epsilon': stats.get('current_epsilon', 0),
                'rl_episodes': stats.get('total_episodes', 0)
            }
        except Exception as e:
            logger.error(f"获取RL状态失败: {e}")
            # 计算基础token数
            total_tokens = sum(model_manager.count_tokens(turn['content']) for turn in self.dialog_history)
            if self.compressed_summary:
                total_tokens += model_manager.count_tokens(self.compressed_summary)

            return {
                'token_count': total_tokens,
                'compression_count': 1 if self.compressed_summary else 0,
                'turn_count': len([h for h in self.dialog_history if h['role'] == 'user']),
                'has_summary': bool(self.compressed_summary),
                'summary_length': len(self.compressed_summary) if self.compressed_summary else 0
            }


# 全局对话管理器实例
dialog_manager = DialogManager()
