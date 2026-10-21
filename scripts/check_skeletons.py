#!/usr/bin/env python3
"""Check that problem skeletons stay in sync with their solutions.

The teaching contract: a file under problems/pNN/ is identical to its solutions/pNN/
counterpart except that

1. each `# ANCHOR: NAME` / `# ANCHOR_END: NAME` marker gains a `_solution` suffix on the
   solution side (the book includes both), and
2. the implementation inside each fill-in region is replaced, on the problem side, by one or
   more `# FILL ...` hint comments (plus an optional `pass` / `...` so an empty body still
   parses).

So solving a puzzle should only ever *add* lines. Any other line that is in a problem but not
in its solution means the two have drifted. Read-only; exits non-zero on drift."""

import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def is_fill(line):
    s = line.strip()
    return s in ("...", "pass") or s.lower().startswith("# fill")


def normalize(text):
    """The lines that must match: no blanks, no anchor markers, no trailing whitespace."""
    return [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith(("# ANCHOR:", "# ANCHOR_END:"))
    ]


def drift(problem, solution):
    """Lines present in the problem but absent from the solution that aren't fill hints."""
    p, s = normalize(problem.read_text()), normalize(solution.read_text())
    bad = []
    for tag, i1, i2, _, _ in difflib.SequenceMatcher(a=p, b=s, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            bad += [line for line in p[i1:i2] if not is_fill(line)]
    return bad


def main():
    solutions = sorted((ROOT / "solutions").glob("p*/p*.py")) + sorted((ROOT / "solutions").glob("p*/p*.mojo"))
    pairs = [(ROOT / "problems" / s.relative_to(ROOT / "solutions"), s) for s in solutions]
    pairs = [(p, s) for p, s in pairs if p.exists()]
    if not pairs:
        print(f"error: found no problem/solution pairs under {ROOT}")
        return 1

    diverged = []
    for problem, solution in pairs:
        rel = problem.relative_to(ROOT)
        bad = drift(problem, solution)
        if bad:
            diverged.append((rel, bad))
            print(f"DIVERGED  {rel}  ({len(bad)} unexpected line(s))")
        else:
            print(f"IN-SYNC   {rel}")

    print()
    if diverged:
        for rel, bad in diverged:
            print(f"--- {rel}")
            for line in bad[:25]:
                print(f"    {line}")
        print("\nProblem skeletons must equal their solution outside the '# FILL ...' regions.")
        return 1
    print(f"All {len(pairs)} problem/solution pairs are in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
