#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次最小的 LLM 调用演示（只用 Python 标准库，无需 pip install）。

运行：
    python3 call_llm.py basic    # 最裸的一次请求：看清楚发出去什么、收回来什么
    python3 call_llm.py stream   # 流式输出（SSE）：打字机效果是怎么来的
    python3 call_llm.py tool     # 工具调用：Agent 的本质就是这个循环
    python3 call_llm.py all

核心认知：调用 LLM 就是一次普通的 HTTPS POST。
  请求体 = 模型名 + 一串 messages + 几个参数
  响应体 = JSON，里面 choices[0].message.content 就是模型的话
所谓 "Agent / LangChain / 框架"，都是在这个 HTTP 请求外面包的循环和管理逻辑。
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request

# ---------------------------------------------------------------- 配置区
# 只有这两行是"换供应商"要改的地方：任何 OpenAI 兼容接口都长这样
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"

API_KEY_ENV = "DEEPSEEK_API_KEY"


_KEY_CACHE = None


def load_api_key():
    """优先读环境变量；没有则回退到本机的 DSH 凭据文件。"""
    global _KEY_CACHE
    if _KEY_CACHE:
        return _KEY_CACHE

    key = os.environ.get(API_KEY_ENV)
    if key:
        _KEY_CACHE = key
        return key

    # 仅为了本机演示方便。真实项目请用环境变量 / 密钥管理服务，别把 key 写进代码。
    cred = os.path.expanduser("~/.dsh/.credentials.yaml")
    if os.path.exists(cred):
        m = re.search(r"%s:\s*(\S+)" % API_KEY_ENV, open(cred, encoding="utf-8").read())
        if m:
            print("[i] 未设置环境变量，已从 %s 读取 key（末尾 %s）" % (cred, m.group(1)[-4:]), file=sys.stderr)
            _KEY_CACHE = m.group(1)
            return _KEY_CACHE

    sys.exit("找不到 API key，请先： export %s=sk-xxxx" % API_KEY_ENV)


def chat(payload):
    """发一次 POST /chat/completions，返回 http response 对象（可直接迭代读取流）。"""
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + load_api_key(),
        },
        method="POST",
    )
    try:
        return urllib.request.urlopen(req, timeout=180)
    except urllib.error.HTTPError as e:
        # HTTP 错误时服务端也会返回 JSON，里面写着原因，一定要打出来
        sys.exit("HTTP %s\n%s" % (e.code, e.read().decode("utf-8", "replace")))


def show(title):
    print("\n" + "=" * 64 + "\n" + title + "\n" + "=" * 64)


# ---------------------------------------------------------------- 1. 最裸的一次调用
def demo_basic():
    show("1. 一次最普通的调用（非流式）")

    messages = [
        # system：设定身份/规则，模型最优先听它
        {"role": "system", "content": "你是一个说话极简的助手，回答不超过 40 个字。"},
        # user：这次要问的问题
        {"role": "user", "content": "什么是 REST API？"},
    ]
    payload = {
        "model": MODEL,       # 用哪个模型
        "messages": messages,  # 对话历史，就是一个数组，越往后越是最近说的话
        "max_tokens": 512,     # 最多生成多少 token
        "temperature": 1.0,    # 随机性；调低更稳定，调高更发散
    }

    print("[请求] POST %s/chat/completions" % BASE_URL)
    print(json.dumps(payload, ensure_ascii=False, indent=2))  # 注意：这里没有 key

    with chat(payload) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    print("\n[响应原文]（截断显示）")
    print(json.dumps(raw, ensure_ascii=False, indent=2)[:1200])

    msg = raw["choices"][0]["message"]
    print("\n[模型最终回答] " + (msg.get("content") or ""))
    if msg.get("reasoning_content"):
        # 思维链：模型先"想"再回答，思考过程单独放在这个字段里
        print("[思考过程] " + msg["reasoning_content"][:200] + " ...")
    print("[结束原因] " + raw["choices"][0]["finish_reason"])
    print("[用量] " + json.dumps(raw["usage"], ensure_ascii=False))
    # usage 就是账单依据：prompt_tokens 是输入，completion_tokens 是输出


# ---------------------------------------------------------------- 2. 流式输出
def demo_stream():
    show("2. 流式输出：SSE 打字机效果")

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "用三句话介绍 Python。"}],
        "max_tokens": 512,
        "stream": True,  # 关键就这一行
    }

    print("[请求] 加了一个 stream: true，响应就变成一行行的 data: {...}")
    print("[输出] ", end="")

    with chat(payload) as resp:
        # 流式响应体是 SSE：每行形如 `data: {json}`，最后一行是 `data: [DONE]`
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue  # 空行、注释行，跳过
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("reasoning_content"):
                pass  # 思考阶段，这里选择不打印；想看得更细可以自己输出
            if delta.get("content"):
                print(delta["content"], end="", flush=True)  # 一个个 token 往外吐
    print()


# ---------------------------------------------------------------- 3. 工具调用 = Agent 的内核
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名，如 北京"}},
                "required": ["city"],
            },
        },
    }
]


def get_weather(city):
    """假的本地工具。真实世界里这里可能是查数据库、调内部 API、读文件。"""
    fake = {"北京": "晴，24°C", "上海": "小雨，27°C"}
    return fake.get(city, "暂无该城市数据")


def demo_tool():
    show("3. 工具调用（Function Calling）：Agent 的本质")

    messages = [{"role": "user", "content": "北京现在天气怎么样？顺便说说该穿什么。"}]

    # ---- 第 1 轮：把 tool 的"说明书"一起发给模型
    payload = {"model": MODEL, "messages": messages, "tools": TOOLS, "max_tokens": 1024}
    print("[第 1 轮请求] messages + tools 一起发过去")
    with chat(payload) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    msg = raw["choices"][0]["message"]
    print("[模型返回] finish_reason=%s" % raw["choices"][0]["finish_reason"])
    print(json.dumps(msg.get("tool_calls"), ensure_ascii=False, indent=2))

    # 关键点：模型并不会真的执行工具，它只是"填了个申请表"告诉你它想调什么
    if not msg.get("tool_calls"):
        print("[!] 模型这次没要求调工具，直接答了：" + (msg.get("content") or ""))
        return

    call = msg["tool_calls"][0]
    name = call["function"]["name"]
    args = json.loads(call["function"]["arguments"])
    print("[我们（代码）执行工具] %s(%s)" % (name, json.dumps(args, ensure_ascii=False)))

    result = get_weather(args["city"])  # 真正的执行发生在这里，控制权在你手上
    print("[工具结果] " + result)

    # ---- 第 2 轮：把"模型的话"和"工具结果"都追加进 messages，再问一次
    messages.append(msg)  # 必须原样把 assistant 的 tool_calls 塞回去
    messages.append({
        "role": "tool",           # 工具结果用特有的 role
        "tool_call_id": call["id"],  # 对应是哪次调用
        "content": result,
    })

    print("\n[第 2 轮请求] messages = 历史 + 模型的工具申请 + 工具结果")
    with chat({"model": MODEL, "messages": messages, "max_tokens": 1024}) as resp:
        raw2 = json.loads(resp.read().decode("utf-8"))

    print("[最终回答] " + (raw2["choices"][0]["message"].get("content") or ""))
    print("\n>>> 这就是 Agent：模型决定调什么 → 你执行 → 结果喂回去 → 模型再决定。")
    print(">>> 把这两轮包成一个 while 循环，加上更多工具，就是 Claude Code / 我这种 Agent。")


DEMOS = {"basic": demo_basic, "stream": demo_stream, "tool": demo_tool}


def main():
    load_api_key()  # 提前把 key 准备好，免得提示混在流式输出里
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "all":
        for fn in (demo_basic, demo_stream, demo_tool):
            fn()
    elif what in DEMOS:
        DEMOS[what]()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
