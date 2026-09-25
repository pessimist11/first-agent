#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行交互入口：流式输出 + 多轮记忆。

运行：
    .venv/bin/python cli.py
    .venv/bin/python cli.py --thread demo     # 指定会话 id，同一个 id 就是同一段记忆

交互命令：
    /new    开一个新会话（清空记忆）
    /exit   退出
"""

import argparse
import json
import sys
import uuid

from langchain_core.messages import AIMessageChunk

from agent import MODEL, build_agent
from tools import TOOLS

GRAY, CYAN, YELLOW, DIM, RESET = "\033[90m", "\033[36m", "\033[33m", "\033[2m", "\033[0m"


def text_of(chunk) -> str:
    """兼容 content 是 str 或 content_blocks 列表两种情况。"""
    content = chunk.content
    if isinstance(content, str):
        return content
    out = []
    for block in content or []:
        if isinstance(block, str):
            out.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            out.append(block.get("text", ""))
    return "".join(out)


def short(value, limit=200) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    s = " ".join(s.split())
    return s if len(s) <= limit else s[:limit] + "..."


def run_turn(agent, config, user_text: str, show_thinking: bool = True) -> None:
    """跑一轮对话，把过程实时打在屏幕上。"""
    usage = {"input": 0, "output": 0}
    state = {"mode": None}  # 当前在打"思考"还是"回答"

    def switch_to(mode: str) -> None:
        if state["mode"] == mode:
            return
        if state["mode"] is not None:
            print(RESET)
        if mode == "thinking":
            print(GRAY + "  [思考] ", end="", flush=True)
        state["mode"] = mode

    print()
    events = agent.stream(
        {"messages": [{"role": "user", "content": user_text}]},
        config,
        stream_mode=["messages", "updates"],  # messages=逐token, updates=每个节点跑完的中间状态
    )
    for mode, payload in events:
        if mode == "messages":
            chunk, _meta = payload
            if not isinstance(chunk, AIMessageChunk):
                continue

            reasoning = (chunk.additional_kwargs or {}).get("reasoning_content")
            if reasoning and show_thinking:
                switch_to("thinking")
                print(reasoning, end="", flush=True)

            piece = text_of(chunk)
            if piece:
                switch_to("answer")
                print(piece, end="", flush=True)

            if chunk.usage_metadata:
                usage["input"] += chunk.usage_metadata.get("input_tokens", 0)
                usage["output"] += chunk.usage_metadata.get("output_tokens", 0)

        elif mode == "updates":
            # 这里能看到"节点级"的事件：模型决定调工具、工具返回了什么
            for _node, update in (payload or {}).items():
                if not isinstance(update, dict):
                    continue
                for msg in update.get("messages") or []:
                    kind = getattr(msg, "type", "")
                    if kind == "tool":
                        print("\n%s  ← %s 返回: %s%s" % (CYAN, getattr(msg, "name", "tool"), short(msg.content), RESET))
                    elif kind == "ai" and getattr(msg, "tool_calls", None):
                        for call in msg.tool_calls:
                            print("%s  → 调用 %s(%s)%s" % (YELLOW, call["name"], short(call["args"], 120), RESET))

    if state["mode"] is not None:
        print(RESET)
    if usage["input"] or usage["output"]:
        print("%s  [本轮 token] 输入 %d / 输出 %d%s" % (DIM, usage["input"], usage["output"], RESET))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default=None, help="会话 id；同一个 id 共享记忆")
    parser.add_argument("--no-thinking", action="store_true", help="不显示模型的思考过程")
    args = parser.parse_args()

    agent = build_agent()
    thread_id = args.thread or ("cli-" + uuid.uuid4().hex[:8])
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 62)
    print(" LangChain 轻量 Agent   模型: %s   线程: %s" % (MODEL, thread_id))
    print(" 工具: %s" % "、".join(t.name for t in TOOLS))
    print(" 命令: /new 开新会话   /exit 退出   Ctrl-C 中断本轮")
    print("=" * 62)

    while True:
        try:
            user_text = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return

        if not user_text:
            continue
        if user_text in ("/exit", "/quit", "exit", "quit"):
            print("再见。")
            return
        if user_text == "/new":
            thread_id = "cli-" + uuid.uuid4().hex[:8]
            config = {"configurable": {"thread_id": thread_id}}
            print("已开新会话：%s" % thread_id)
            continue

        try:
            run_turn(agent, config, user_text, show_thinking=not args.no_thinking)
        except KeyboardInterrupt:
            print("\n[已中断本轮]")
        except Exception as exc:  # 网络/接口报错不该让整个 CLI 挂掉
            print("\n[出错] %s: %s" % (type(exc).__name__, exc), file=sys.stderr)


if __name__ == "__main__":
    main()
