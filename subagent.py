# -*- coding: utf-8 -*-
"""
最小可用的 subagent 系统。

对应设计文档里的关键取舍：
  1. 上下文隔离   —— 每个 subagent 是全新的 messages，看不到主 agent 的对话
  2. 结果压缩     —— 只把"最后一条回答"回流，中间过程留在子 agent 里烧掉
  3. 权限白名单   —— 子 agent 拿不到 save_note（写操作）和 spawn_subagent（防递归爆炸）
  4. 深度限制     —— 结构上只有 1 层：子 agent 的工具集里没有 spawn
  5. 预算配额     —— 并发数 / 总数 / 总 token 三重上限
  6. 真异步       —— spawn 立即返回 id，后台线程跑，主 agent 继续干别的活
  7. 可中断       —— 协作式取消，每个执行步检查一次
  8. 事件通知     —— 完成事件进队列，不做忙轮询
"""

import os
import threading
import time
import uuid
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# 子 agent 只能看到这几个工具：只读 + 计算，没有写操作，也没有 spawn
from tools import calculate, current_time, get_weather, list_files, read_file

CHILD_TOOLS = [get_weather, calculate, list_files, read_file, current_time]

SUBAGENT_MODEL = os.getenv("SUBAGENT_MODEL", os.getenv("AGENT_MODEL", "deepseek-flash"))
RESULT_MAX_CHARS = 4000

CHILD_SYSTEM_PROMPT = """你是被主 agent 派来的子 agent，要独立完成一个具体任务。

规则：
1. 你看不到主 agent 的对话，只能依据收到的任务描述行事。
2. 需要实时信息（天气、时间）或计算时调用工具，不要凭记忆编造。
3. 任务完成、确认无法完成、或发现任务描述有歧义时，立即停止并汇报。
   不要反复用同一种方式重试。
4. 你的输出是给主 agent 看的：只给「结论 + 支撑证据 + 未解决项」，
   不要复述你读过的文件原文，也不要罗列你的操作步骤。
5. 尽量简短，300 字以内。
"""


# ---------------------------------------------------------------- 预算与登记表
@dataclass
class SubAgent:
    id: str
    name: str
    prompt: str
    status: str = "running"  # running | completed | failed | timed_out | killed
    result: str = ""
    error: str = ""
    tokens: int = 0
    created: float = field(default_factory=time.time)
    finished: float = field(default=None)
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread = field(default=None)

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.created


class Budget:
    """预算的树形分配：这里是"全局配额"的简化版。

    真实系统里应该让父 agent 把自己的剩余预算切一块给子 agent，
    否则深度一深就是指数放大。这里只有一层，所以用全局上限即可。
    """

    def __init__(self, max_concurrent: int = 3, max_total: int = 8, max_tokens: int = 80000):
        self.max_concurrent = max_concurrent
        self.max_total = max_total
        self.max_tokens = max_tokens
        self.spawned = 0
        self.tokens = 0
        self._lock = threading.Lock()

    def check_spawn(self):
        """返回 None 表示允许派发；返回字符串表示拒绝的理由。"""
        with _io_lock:  # 遍历 _agents 必须持这把锁，否则并发 spawn 时会炸
            running = sum(1 for a in _agents.values() if a.status == "running")
        with self._lock:
            if running >= self.max_concurrent:
                return "已达并发上限（%d 个正在运行）" % self.max_concurrent
            if self.spawned >= self.max_total:
                return "已达本次会话的 subagent 总数上限（%d 个）" % self.max_total
            if self.tokens >= self.max_tokens:
                return "subagent token 预算已耗尽（%d/%d）" % (self.tokens, self.max_tokens)
            self.spawned += 1
        return None

    def add_tokens(self, n: int) -> None:
        with self._lock:
            self.tokens += n


_agents: dict = {}
_events: list = []          # 待主 agent 取走的通知（不打印，避免打乱流式输出）
_io_lock = threading.Lock()
_budget = Budget()
_model_factory = None


def configure(model_factory) -> None:
    """由 agent.py 注入模型构造函数，避免模块间循环 import。"""
    global _model_factory
    _model_factory = model_factory


def _build_model():
    if _model_factory is not None:
        return _model_factory()
    # 独立使用本模块时的兜底：子 agent 允许用更便宜的模型
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(model=SUBAGENT_MODEL, temperature=0.2, timeout=120)


def _emit(message: str) -> None:
    with _io_lock:
        _events.append("[%.0f] %s" % (time.time(), message))


def drain_events() -> list:
    """取走并清空待通知事件（主循环每轮调用一次）。"""
    with _io_lock:
        pending, _events[:] = list(_events), []
    return pending


# ---------------------------------------------------------------- 结果提取
def _text_of(message) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _final_text(state) -> str:
    """从最终状态里取"最后一条有文本的 AI 消息"——这就是要回流的压缩结论。"""
    for message in reversed((state or {}).get("messages") or []):
        if getattr(message, "type", "") == "ai":
            text = _text_of(message).strip()
            if text:
                return text[:RESULT_MAX_CHARS]
    return "（子 agent 没有返回任何文本）"


def _sum_tokens(state) -> int:
    total = 0
    for message in (state or {}).get("messages") or []:
        usage = getattr(message, "usage_metadata", None) or {}
        total += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    return total


# ---------------------------------------------------------------- 后台执行
def _run(child: SubAgent, timeout_seconds: float) -> None:
    state = None
    child_agent = None
    config = None
    try:
        child_agent = create_agent(
            model=_build_model(),
            tools=CHILD_TOOLS,  # ← 权限白名单 + 深度限制都靠这一行
            system_prompt=CHILD_SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),  # 只为了让中断后还能取回已产生的用量
            middleware=[ModelCallLimitMiddleware(run_limit=8, exit_behavior="end")],
        )
        config = {"configurable": {"thread_id": "sub-" + child.id}, "recursion_limit": 40}

        # 用 values 模式逐步拿状态：好处是每一步都能检查取消/超时。
        # 顺序讲究：先记 state 再判取消，否则中断时会漏掉"刚好到达那一步"的用量
        for step in child_agent.stream(
            {"messages": [{"role": "user", "content": child.prompt}]}, config, stream_mode="values"
        ):
            state = step
            if child.cancel.is_set():
                child.status, child.result = "killed", "被主 agent 中断（协作式取消，在步与步之间生效）"
                break
            if time.time() - child.created > timeout_seconds:
                child.status, child.result = "timed_out", "超过 %.0f 秒被判定超时" % timeout_seconds
                break

        if child.status == "running":
            child.status = "completed"
            child.result = _final_text(state)
    except Exception as exc:
        child.status = "failed"
        child.error = "%s: %s" % (type(exc).__name__, exc)
        child.result = "子 agent 执行失败 —— " + child.error
    finally:
        # 被中断/超时时 state 可能是 None，但钱已经花了：从 checkpoint 把用量捞回来，避免漏账
        if state is None and child_agent is not None and config is not None:
            try:
                state = child_agent.get_state(config).values
            except Exception:
                state = None
        child.tokens = _sum_tokens(state)
        child.finished = time.time()
        _budget.add_tokens(child.tokens)  # 失败也照样计费，不能白烧
        _emit(
            "subagent %s（%s）→ %s，用时 %.1fs，%d tokens"
            % (child.id, child.name, child.status, child.elapsed, child.tokens)
        )


# ---------------------------------------------------------------- 暴露给主 agent 的工具
@tool
def spawn_subagent(prompt: str, name: str = "") -> str:
    """派发一个子 agent 去独立完成一个任务，它有自己的上下文，看不到我们的对话。

    prompt 必须自包含，至少包含：任务目标、必要背景、输入（文件路径等）、
    边界（不要做什么）、验收标准、输出格式、什么情况下该放弃。
    想并行就一次发出多个 spawn，不要串行等。

    本工具立即返回、不等待结果。取结果用 wait_subagent，看进度用 list_subagents。"""
    refusal = _budget.check_spawn()
    if refusal:
        return "派发被拒绝：%s。请自己完成，或先等已有的 subagent 结束。" % refusal

    child = SubAgent(id="sub-" + uuid.uuid4().hex[:6], name=name or "未命名", prompt=prompt)
    with _io_lock:
        _agents[child.id] = child
    child.thread = threading.Thread(target=_run, args=(child, 300.0), daemon=True)
    child.thread.start()
    _emit("已派发 subagent %s（%s）" % (child.id, child.name))
    return (
        "已派发 subagent：%s（%s），正在后台运行。\n"
        "取结果：wait_subagent(agent_id=\"%s\")\n"
        "看进度：list_subagents()" % (child.id, child.name, child.id)
    )


@tool
def wait_subagent(agent_id: str, timeout_seconds: int = 180) -> str:
    """等待某个 subagent 结束并取回它的结论。只在真正需要它的结果时调用，不要用它轮询进度。"""
    child = _agents.get(agent_id)
    if child is None:
        return "没有这个 subagent：%s。用 list_subagents() 看看有哪些。" % agent_id
    if child.thread is not None:
        child.thread.join(timeout=max(1, timeout_seconds))

    if child.status == "running":
        return "subagent %s 仍在运行（已等 %d 秒，已耗时 %.1fs）。可以先做别的事，稍后再来取。" % (
            agent_id,
            timeout_seconds,
            child.elapsed,
        )
    return "subagent %s（%s）状态：%s，用时 %.1fs，%d tokens\n\n【结论】\n%s" % (
        agent_id,
        child.name,
        child.status,
        child.elapsed,
        child.tokens,
        child.result,
    )


@tool
def list_subagents() -> str:
    """列出所有 subagent 的状态、耗时和 token 消耗。非阻塞，用来查看进度。"""
    if not _agents:
        return "还没有派发过任何 subagent。"
    with _io_lock:
        snapshot = list(_agents.values())
        spawned, tokens = _budget.spawned, _budget.tokens
    rows = [
        "%s  %-8s  %-10s  %5.1fs  %6d tokens  %s"
        % (a.id, a.name[:8], a.status, a.elapsed, a.tokens, a.prompt[:40].replace("\n", " "))
        for a in snapshot
    ]
    return "\n".join(rows) + "\n\n预算：已派发 %d/%d，累计 %d/%d tokens" % (
        spawned,
        _budget.max_total,
        tokens,
        _budget.max_tokens,
    )


@tool
def interrupt_subagent(agent_id: str) -> str:
    """请求中断一个正在运行的 subagent。注意这是协作式取消：它会在下一个执行步才停下来，不是立即。"""
    child = _agents.get(agent_id)
    if child is None:
        return "没有这个 subagent：%s" % agent_id
    if child.status != "running":
        return "subagent %s 已经是 %s 状态，无需中断。" % (agent_id, child.status)
    child.cancel.set()
    return "已向 %s 发送中断请求（会在步与步之间生效）。" % agent_id


PARENT_TOOLS = [spawn_subagent, wait_subagent, list_subagents, interrupt_subagent]
