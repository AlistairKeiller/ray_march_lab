#!/usr/bin/env python3
"""Check that the book's includes, links and images resolve, without needing mdbook.

For every book/src/**/*.md this verifies that
  * `{{#include path}}` and `{{#include path:anchor}}` name an existing file, and that the
    anchor exists in it as `# ANCHOR: anchor`;
  * local `[text](target)`, `![alt](target)` and `<img src="target">` targets exist.
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


def external(target):
    return target.startswith(("http://", "https://", "mailto:", "#", "data:"))


def check(md):
    misses, in_fence = [], False
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
