"""读取 WebUI 页面源码，供按 AST 检查页面函数的测试使用。

页面代码由 webui/Main.py 拆分到 webui/common.py 与 webui/components/。直接导入
页面会执行整套渲染，因此这些测试读取源码后只编译目标函数；函数位于哪个模块
不影响测试，它们只需要在页面源码中查找。
"""

from pathlib import Path

WEBUI_DIR = Path(__file__).resolve().parents[2] / "webui"
WEBUI_MAIN = WEBUI_DIR / "Main.py"


def webui_page_files() -> list[Path]:
    return [
        WEBUI_MAIN,
        WEBUI_DIR / "common.py",
        *sorted((WEBUI_DIR / "components").glob("*.py")),
    ]


def webui_source() -> str:
    """返回全部页面模块的源码；各文件以空行分隔，拼接后仍可整体 ast.parse。"""
    return "\n\n".join(path.read_text(encoding="utf-8") for path in webui_page_files())
