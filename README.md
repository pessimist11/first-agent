# first-agent

一个**轻量但真的能用**的 LangChain Agent：命令行对话、会调工具、有记忆、有护栏。

比 `call_llm.py`（手写 HTTP 调 LLM）上了一个台阶：这里用 LangChain 的 `create_agent`，
把"模型 → 决定调工具 → 执行 → 结果喂回模型"这个循环交给框架管，你只写工具和提示词。

## 环境

- Python **3.13**（已由 uv 装好，系统自带的 3.9 用不了，LangChain 要求 ≥3.10）
- 依赖已在 `.venv/` 里装好：`langchain 1.4.2`、`langgraph 1.2.12`、`langchain-deepseek 1.1.1`

从零重建环境：

```bash
uv venv --python 3.13
uv pip install langchain langchain-deepseek python-dotenv
cp .env.example .env      # 然后填入自己的 DEEPSEEK_API_KEY
```

## 跑起来

```bash
.venv/bin/python cli.py                 # 随机开一个新会话
.venv/bin/python cli.py --thread demo   # 固定会话 id（同一个 id 共享记忆）
.venv/bin/python cli.py --no-thinking   # 不显示模型的思考过程
```

交互命令：`/new` 开新会话（清空记忆），`/exit` 退出，`Ctrl-C` 中断本轮。

试试这几句，能直观看出 agent 在做什么：

```
北京现在天气怎么样？
把结果存成笔记，标题叫「北京天气」
(1200-350)*0.85 等于多少
我刚问了你什么？          ← 验证记忆
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `agent.py` | 组装层：模型 + 工具 + 记忆 + 3 个护栏中间件 |
| `tools.py` | 6 个工具（天气/计算/列目录/读文件/写笔记/时间） |
| `cli.py` | 命令行交互：流式输出、工具调用可视化、多轮记忆 |
| `call_llm.py` | 上一课：纯标准库手写 HTTP 调 LLM，用来看清底层 |
| `.env` | 密钥和模型配置（**已被 .gitignore 排除**） |

## 关键设计点

**1. 工具写得好不好，直接决定 agent 好不好用**

```python
@tool
def get_weather(city: str) -> str:
    """查询某个城市当前的实时天气。city 传城市名，中英文都可以。"""
```

- docstring 就是给模型看的说明书，模型只靠它选工具
- 类型注解自动变成参数 JSON Schema，别用 `*args`
- **出错也要 `return` 一句人话，别 `raise`**——`raise` 会打断整轮对话，模型没机会补救
- 文件类工具都用 `_safe_path()` 限制在工作区内，防路径越界

**2. `calculate` 为什么不用 `eval()`**

`eval("__import__('os').system('rm -rf ~')")` 是真的会执行的。
这里用 AST 白名单只放行数字和四则运算，模型拿不到任何执行能力。
**凡是模型能决定参数的地方，都是安全边界。**

**3. 三个护栏中间件**（都在 `agent.py`）

| 中间件 | 挡什么 |
|---|---|
| `ModelCallLimitMiddleware(run_limit=12)` | 工具调用死循环，防止一次提问烧掉几十次模型调用 |
| `ToolCallLimitMiddleware(run_limit=8)` | 单个工具被反复调用 |
| `SummarizationMiddleware(trigger=[("fraction",0.7),("messages",60)])` | 上下文爆炸，超阈值自动摘要压缩历史 |

## 已验证的行为

- ✅ 工具调用：模型返回 `tool_calls` → 框架执行 → 结果回灌 → 生成回答
- ✅ 多轮记忆：`thread_id` 相同的两轮对话能正确引用前文
- ✅ 失败自愈：工具报错后模型会换个参数重试，重试仍失败就如实告知不编造
- ✅ 安全护栏：`../../../../etc/passwd` 被拒；`eval` 注入被拒
- ✅ 流式输出：思考过程（灰色）+ 回答（打字机）+ 工具调用（黄色）分开显示

## 已知边界 / 下一步升级路线

| 现状 | 升级方向 |
|---|---|
| 记忆在 `InMemorySaver`，进程重启就忘 | 换 `langgraph.checkpoint.sqlite.SqliteSaver` 落盘 |
| 写文件工具直接执行，无人确认 | 加 `HumanInTheLoopMiddleware(interrupt_on={"save_note": True})`，危险操作前暂停等确认 |
| 调试靠终端输出 | 设 `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY`，在 LangSmith 看完整 trace |
| 只有 CLI | FastAPI 包一层 HTTP 接口 |
| 工具是手写的 | 接 MCP server 复用现成工具生态 |

## 注意事项

- `.env` 里是**真实密钥**，已在 `.gitignore` 中；别提交、别外发。如果泄露去 DeepSeek 后台轮换。
- 端口/网络：`get_weather` 走 `wttr.in`，网络不通时该工具会返回"查询失败"，agent 会如实告知。
- DeepSeek 返回的 `reasoning_content`（思考过程）和最终回答是分开的，`cli.py` 里两种颜色分别显示。
