#!/usr/bin/env python3
"""99h_check_py_syntax.py — 检查 .py 里「本机 Python 能过、服务器 Python 过不了」的语法。

背景（2026-09-23 实测）
    本机 WorkBuddy 的 Python 是 3.13，而 vggt_human conda env 是 **3.10.20**。
    3.12+ 才合法的写法会在本机一路通过、到服务器直接 SyntaxError 挂在 import 阶段。
    99g 第 434 行就是这么挂的：
        label += f" · 距锚点 {math.dist(原点, ctx["anchor"]):.2f} m"
                                              ^^^^^^^^^^^ 报 f-string: unmatched '['

    规则已在 Python 3.10.20 上逐条实测（不是猜的）：

    会被 3.10 判 SyntaxError 的：
      · f"{d["k"]}"                     单引号外层 + 内层同类引号
      · f'{d['k']}'                     同上（外层单引号）
      · f"{s.replace('\\n', '')}"       表达式段里出现反斜杠（任何位置）
    3.10 合法的（别误报）：
      · f'''{d["k"]}'''                 三引号外层 + 内层单双引号
      · f"{d['k']}"                     内外引号类型不同
      · f"{f'{x}'}"                     嵌套 f-string，内外引号类型不同

    （本文件自身就踩过同类坑：docstring 里写三引号示例会把 docstring 提前闭合。）

为什么不用别的办法
    · `ast.parse(src, feature_version=(3, 10))` **查不出来**：PEG 解析器不再按旧规则限制
      f-string（本机实测 3.10/3.11/3.12 三种 feature_version 全部通过）。
    · 手写字符扫描会误报：注释、普通字符串、三引号模板里的同类模式都会被算进去。
    所以用 tokenize：真实词法分析，注释与普通字符串天然排除。

用法:
    python vggt_human/99h_check_py_syntax.py           # 扫 vggt_human/*.py
    python vggt_human/99h_check_py_syntax.py a.py b/   # 指定文件或目录（递归）

退出码：0 = 干净，1 = 有命中。需 Python ≥ 3.12 运行本检查（低版本不会产出 FSTRING_* token）。
"""
import io
import sys
import tokenize
from pathlib import Path


def delim_of(text: str):
    """从 'f"' / 'rb\\'\\'\\'' 这类 token 文本里取出 (引号字符, 是否三引号)。"""
    i = 0
    while i < len(text) and text[i] not in "\"'":
        i += 1
    if i >= len(text):
        return "", False
    q = text[i]
    return q, text[i:i + 3] == q * 3


def scan_file(path: Path):
    """返回 [(行号, 行内容, 原因)]。"""
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as e:
        return [(-1, f"⚠️ 无法 tokenize：{e}", "")]

    hits = []
    fstack = []      # [(外层引号字符, 是否三引号)]
    depth = 0        # 当前 f-string 里 {...} 的嵌套层数
    for tok in toks:
        t = tok.type
        if t == tokenize.FSTRING_START:
            fstack.append(delim_of(tok.string))
            depth = 0
            continue
        if t == tokenize.FSTRING_END:
            if fstack:
                fstack.pop()
            depth = 0
            continue
        if not fstack:
            continue
        if t == tokenize.OP and tok.string == "{":
            depth += 1
            continue
        if t == tokenize.OP and tok.string == "}":
            depth = max(0, depth - 1)
            continue
        if depth <= 0:
            continue
        # 规则 1：表达式段里出现反斜杠
        if "\\" in tok.string and t != tokenize.FSTRING_MIDDLE:
            hits.append((tok.start[0], tok.line.strip()[:160], "表达式段内含反斜杠"))
            continue
        # 规则 2：内层字符串引号与外层定界符同类（外层单引号时才算，三引号外层合法）
        if t == tokenize.STRING:
            oq, otri = fstack[-1]
            iq, itri = delim_of(tok.string)
            if oq and iq == oq and (not otri or itri):
                hits.append((tok.start[0], tok.line.strip()[:160],
                             f'内层字符串用了与外层相同的定界符 {iq}'))
    seen, out = set(), []
    for ln, frag, why in hits:
        if (ln, why) not in seen:
            seen.add((ln, why))
            out.append((ln, frag, why))
    return out


def iter_targets(args):
    if not args:
        args = [str(Path(__file__).resolve().parent)]
    for a in args:
        p = Path(a)
        if p.is_dir():
            yield from sorted(x for x in p.rglob("*.py") if "__pycache__" not in x.parts)
        elif p.suffix == ".py":
            yield p


def main():
    if not hasattr(tokenize, "FSTRING_START"):
        sys.exit("❌ 需要 Python ≥ 3.12 运行本检查（低版本 tokenize 不产出 FSTRING_* token）")

    targets = list(iter_targets(sys.argv[1:]))
    if not targets:
        sys.exit("❌ 没有找到 .py 文件")

    bad = 0
    for f in targets:
        hits = scan_file(f)
        if hits:
            bad += len(hits)
            print(f"❌ {f}")
            for ln, frag, why in hits:
                print(f"     L{ln}: {frag}")
                print(f"        → {why}：3.12 才合法，3.10 会 SyntaxError。"
                      f"先把值取到局部变量再拼，或内外改用不同引号。")

    print()
    if bad:
        print(f"🎯 扫描 {len(targets)} 个文件，命中 {bad} 处")
        sys.exit(1)
    print(f"✅ 扫描 {len(targets)} 个文件，没有 f-string 兼容性问题")


if __name__ == "__main__":
    main()
