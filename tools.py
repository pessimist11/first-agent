# -*- coding: utf-8 -*-
"""
Agent 的工具集。

写工具的 3 条铁律（决定 agent 好不好用）：
  1. docstring 就是给模型看的说明书 —— 模型只靠它决定调不调、怎么调，写含糊它必选错
  2. 类型注解 = 参数 schema，模型据此生成 JSON；别用 *args/**kwargs
  3. 返回值统一是字符串；出错也要 return 一句人话，别 raise（raise 会直接中断整轮对话）
"""

import ast
import operator
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

# 所有文件操作都被限制在这个目录里，防止模型读到工作区外面的东西
WORKSPACE = Path(__file__).resolve().parent


def _safe_path(relative: str) -> Path:
    """把用户/模型给的相对路径解析到工作区内；越界就报错。"""
    target = (WORKSPACE / relative).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ValueError("拒绝访问：只允许操作工作区(%s)内的文件" % WORKSPACE.name)
    return target


# ---------------------------------------------------------------- 1. 联网工具
@tool
def get_weather(city: str) -> str:
    """查询某个城市当前的实时天气。city 传城市名，中文英文都可以，例如「北京」「Tokyo」。"""
    # URL 必须是纯 ASCII：城市名和 format 里的中文都要百分号编码。
    # quote(..., safe="%") 是为了保留 %l %c %t 这些 wttr.in 的占位符不被编码成 %25；
    # 空格交给 quote 变成 %20（写成 + 会被编码成 %2B，结果原样显示出来）
    fmt = "%l: %c %t 风速%w 湿度%h"
    url = "https://wttr.in/%s?format=%s&lang=zh" % (
        urllib.parse.quote(city),
        urllib.parse.quote(fmt, safe="%"),
    )
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})  # wttr.in 要求非浏览器 UA
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8").strip() or "查询失败：返回内容为空"
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace").strip()
        if "not found" in body.lower() or "unknown location" in body.lower():
            return "查不到「%s」这个城市，请确认城市名是否正确。" % city
        return "天气服务返回 HTTP %s：%s" % (e.code, body[:200] or "无内容")
    except Exception as e:
        return "天气查询失败（%s）。可以稍后重试，或直接告诉用户暂时查不到。" % e


# ---------------------------------------------------------------- 2. 计算工具
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:  # 防止 9**9**9 把机器算死
            raise ValueError("指数过大")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("只支持数字和 + - * / // ** 以及括号")


@tool
def calculate(expression: str) -> str:
    """计算一个数学表达式并返回结果。expression 传纯数学式子，例如 "(1200-350)*0.85" 或 "2**10"。
    不要传中文、单位或变量名。"""
    try:
        # 为什么不用 eval()：它会把 os.system(...) 一起执行了。
        # 这里用 AST 白名单解析，模型拿不到任何执行能力。
        tree = ast.parse(expression, mode="eval")
        return "%s = %s" % (expression, _eval_node(tree))
    except Exception as e:
        return "计算失败：%s" % e


# ---------------------------------------------------------------- 3. 读文件的工具
@tool
def list_files(subdir: str = ".") -> str:
    """列出工作区某个目录下的文件。subdir 传相对路径，默认 "." 表示工作区根目录。"""
    try:
        base = _safe_path(subdir)
        if not base.exists():
            return "目录不存在：%s" % subdir
        rows = []
        for p in sorted(base.iterdir()):
            if p.name in (".venv", ".uv-cache", "__pycache__", ".git"):
                continue  # 这些目录又大又没信息量，直接跳过省 token
            rows.append("%s  %s" % ("DIR " if p.is_dir() else "%5d B" % p.stat().st_size, p.name))
        return "\n".join(rows) if rows else "（空目录）"
    except Exception as e:
        return "列出文件失败：%s" % e


@tool
def read_file(path: str, max_chars: int = 3000) -> str:
    """读取工作区里某个文本文件的内容。path 是相对于工作区的路径，例如 "README.md"。
    max_chars 限制返回字符数，默认 3000，文件很大时不要一次全读。"""
    try:
        f = _safe_path(path)
        if not f.is_file():
            return "文件不存在：%s" % path
        text = f.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            return text[:max_chars] + "\n...（已截断，共 %d 字符）" % len(text)
        return text
    except Exception as e:
        return "读取失败：%s" % e


# ---------------------------------------------------------------- 4. 写文件的工具
@tool
def save_note(title: str, content: str) -> str:
    """把一段内容保存成 Markdown 笔记，存放在工作区的 notes/ 目录下。
    title 只用来当文件名（不要在 content 里重复写标题），content 是正文。这个工具会真正写磁盘。"""
    try:
        notes_dir = WORKSPACE / "notes"
        notes_dir.mkdir(exist_ok=True)
        safe_name = "".join(c for c in title if c.isalnum() or c in "-_ ").strip() or "note"
        target = _safe_path("notes/%s.md" % safe_name)
        target.write_text("# %s\n\n%s\n" % (title, content), encoding="utf-8")
        return "已保存到 %s（正文 %d 字符）" % (target.relative_to(WORKSPACE), len(content))
    except Exception as e:
        return "保存失败：%s" % e


@tool
def current_time() -> str:
    """获取当前本地日期和时间。用户问「现在几点」「今天几号」时用它，不要靠猜。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


TOOLS = [get_weather, calculate, list_files, read_file, save_note, current_time]
