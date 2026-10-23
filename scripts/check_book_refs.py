#!/usr/bin/env python3
r"""Check that the book's includes, links, images and math survive mdbook, without running it.

For every book/src/**/*.md this verifies that
  * `{{#include path}}` and `{{#include path:anchor}}` name an existing file, and that the
    anchor exists in it as `# ANCHOR: anchor`;
  * local `[text](target)`, `![alt](target)` and `<img src="target">` targets exist;
  * math inside `\\( ... \\)` and `\\[ ... \\]` contains nothing Markdown rewrites before
    MathJax sees it. mdbook runs Markdown escapes first, so `\,` `\;` `\{` `\_` and every
    other backslash before ASCII punctuation lose the backslash (`\,` renders as a comma),
    a row break must be written `\\\\`, and smart punctuation turns `'` `"` `--` `...` into
    typographic characters. Letters-only commands (`\cos`, `\cdot`, `\quad`) and `\ ` are
    safe.
Targets inside fenced code blocks and inline code are ignored, as are http(s), mailto and
pure #fragment links. Exits non-zero and lists `file:line -> target` for every miss."""

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "book" / "src"
INCLUDE = re.compile(r"\{\{#include\s+([^}\s:]+)(?::([^}\s]+))?\s*\}\}")
MD_REF = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMG_REF = re.compile(r"<img\b[^>]*?\bsrc=[\"']([^\"']+)[\"']", re.IGNORECASE)
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
INLINE_CODE = re.compile(r"(`+)(?:.+?)\1")
MATH = re.compile(r"\\\\\((.*?)\\\\\)|\\\\\[(.*?)\\\\\]", re.DOTALL)
BACKSLASHES = re.compile(r"(\\+)(.?)", re.DOTALL)
SMART = re.compile(r"'|\"|--|\.\.\.|\*")


def external(target):
    return target.startswith(("http://", "https://", "mailto:", "#", "data:"))


def bad_math(text):
    """(line, snippet) for every sequence in a math span that Markdown would rewrite."""
    bad = []
    for m in MATH.finditer(text):
        body = m.group(1) if m.group(1) is not None else m.group(2)
        lineno = text.count("\n", 0, m.start()) + 1
        hits = [b for b in BACKSLASHES.finditer(body)
                if not (len(b.group(1)) == 4 or (len(b.group(1)) == 1 and (b.group(2).isalpha() or b.group(2) == " ")))]
        hits += list(SMART.finditer(body))
        for h in hits:
            bad.append((lineno, "math: " + body[max(0, h.start() - 12):h.end() + 12].replace("\n", " ")))
    return bad


def check(md):
    misses, in_fence = bad_math(md.read_text()), False
    for lineno, line in enumerate(md.read_text().splitlines(), start=1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for path, anchor in INCLUDE.findall(line):
            target = (md.parent / path).resolve()
            if not target.exists():
                misses.append((lineno, path))
            elif anchor and f"# ANCHOR: {anchor}" not in target.read_text():
                misses.append((lineno, f"{path}:{anchor}"))
        for target in MD_REF.findall(INLINE_CODE.sub("", line)) + IMG_REF.findall(line):
            if external(target) or "{{" in target:
                continue
            clean = target.split("#", 1)[0].split("?", 1)[0]
            if clean and not (md.parent / clean).exists():
                misses.append((lineno, target))
    return misses


def main():
    pages = sorted(SRC.rglob("*.md"))
    misses = [(md, lineno, t) for md in pages for lineno, t in check(md)]
    for md, lineno, target in misses:
        print(f"  {md.relative_to(SRC.parent.parent)}:{lineno}  ->  {target}", file=sys.stderr)
    if misses:
        print(f"FAIL: {len(misses)} unresolved reference(s)", file=sys.stderr)
        return 1
    print(f"OK: all references resolved across {len(pages)} markdown file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
