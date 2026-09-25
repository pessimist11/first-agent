# -*- coding: utf-8 -*-
"""
Agent 组装层：模型 + 工具 + 记忆 + 护栏。

这一层只负责"把零件拼成一个 agent"，不负责交互（交互在 cli.py）。
想看最小版本的话，把 build_agent() 里的 middleware 全删掉照样能跑。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 必须赶在读环境变量之前加载 .env
load_dotenv(Path(__file__).with_name(".env"))

from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import (  # noqa: E402
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from tools import TOOLS  # noqa: E402

MODEL = os.getenv("AGENT_MODEL", "deepseek-flash")
BASE_URL = os.getenv("AGENT_BASE_URL", "https://api.deepseek.com")

SYSTEM_PROMPT = """你是一个中文助手，运行在用户本机的命令行里，手上有一组工具。

工作规则：
1. 需要实时信息（天气、当前时间）或需要算数时，必须调用工具，不要凭记忆编造。
2. 调用工具前先用一句话说明你准备做什么。
3. 回答简洁直接，不要把工具返回的原始文本整段复述一遍。
4. 工具返回失败时，如实告诉用户失败原因，不要假装成功。
"""


def build_model():
    """接模型。

    DeepSeek 用官方集成最省事：它会自动读 DEEPSEEK_API_KEY，也自带工具调用支持。
    如果想换成任意 OpenAI 兼容端点（Kimi / Qwen / 本地 vLLM），换成：
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=MODEL, base_url=BASE_URL,
                          api_key=os.getenv("DEEPSEEK_API_KEY"))
    """
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(
        model=MODEL,
        temperature=0.3,   # 要调工具的场景，低温度更稳
        max_retries=2,
        timeout=120,
    )


def build_agent(checkpointer=None):
    """拼出 agent。

    checkpointer 决定"记忆存在哪"：
      - InMemorySaver  ：进程内存，重启就忘，适合本地玩
      - SqliteSaver    ：存文件，重启还在（见 README 的升级路线）
    """
    return create_agent(
        model=build_model(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer or InMemorySaver(),
        middleware=[
            # 护栏 1：一次提问最多让模型思考 12 轮，防止工具调用死循环烧钱
            ModelCallLimitMiddleware(run_limit=12, exit_behavior="end"),
            # 护栏 2：一次提问里同一个工具最多调 8 次
            ToolCallLimitMiddleware(run_limit=8, exit_behavior="continue"),
            # 护栏 3：上下文到模型窗口 70% 或消息超 60 条时，自动摘要压缩历史
            SummarizationMiddleware(
                model=build_model(),
                trigger=[("fraction", 0.7), ("messages", 60)],
                keep=("messages", 20),
            ),
        ],
        name="first-agent",
    )
