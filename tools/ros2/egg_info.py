"""Write a Python package's egg-info (PKG-INFO + entry_points.txt) from its setup.py.

    egg_info.py SETUP_PY OUT_DIR

ros2cli finds its commands and verbs through entry points (importlib.metadata).
Some RCR ros2cli packages ship their code without that metadata, so their
commands don't show up; this recreates it from the `setup(...)` call (read with
ast, not run). Only literal arguments and module-level string constants are
supported, which is how the ros2cli packages write them.
"""

import ast
import pathlib
import sys


def _setup_kwargs(source: str) -> dict:
    tree = ast.parse(source)
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass

    def value(node):
        if isinstance(node, ast.Name) and node.id in constants:
            return constants[node.id]
        return ast.literal_eval(node)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup":
            kwargs = {}
            for kw in node.keywords:
                if kw.arg in ("name", "version", "entry_points"):
                    kwargs[kw.arg] = value(kw.value)
            return kwargs
    raise ValueError("no setup(...) call")


def egg_info(source: str) -> dict[str, str]:
    """{file name: content} of the egg-info directory."""
    kwargs = _setup_kwargs(source)
    lines = []
    for group, entries in kwargs.get("entry_points", {}).items():
        lines.append(f"[{group}]")
        lines.extend(e.replace(" ", "") .replace("=", " = ", 1) for e in entries)
        lines.append("")
    return {
        "PKG-INFO": f"Metadata-Version: 2.1\nName: {kwargs['name']}\nVersion: {kwargs['version']}\n",
        "entry_points.txt": "\n".join(lines),
    }


def main():
    setup_py, out_dir = sys.argv[1:]
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, content in egg_info(pathlib.Path(setup_py).read_text()).items():
        (out / name).write_text(content)


if __name__ == "__main__":
    main()
