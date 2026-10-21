#!/usr/bin/env python3
"""Check that every puzzle where the learner writes a kernel uses the canary.

A puzzle's value assertions can pass even when the kernel wrote outside `output`, so a puzzle
without the canary silently loses that check. A puzzle is in scope when its problem file has a
`# FILL` marker; its problem and solution must then both import `PuzzleMemory`, take their
output from it and call `verify()`. Puzzles in EXCLUDED are driven from a benchmark loop
instead and check their kernel variants against each other."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDED = {"p06"}  # the benchmark: no single guarded output, the layouts cross-check
FILL = re.compile(r"#\s*FILL")
REQUIRED = (
    ("from harness.canary import PuzzleMemory", "the harness import"),
    ("mem.output(", "an output buffer from the harness"),
    ("mem.verify()", "a call to mem.verify()"),
)


def main():
    problems = sorted((ROOT / "problems").glob("p*/p*.py")) + sorted((ROOT / "problems").glob("p*/p*.mojo"))
    failures, checked = [], 0
    for problem in problems:
        if problem.stem in EXCLUDED or not FILL.search(problem.read_text()):
            continue
        checked += 1
        solution = ROOT / "solutions" / problem.relative_to(ROOT / "problems")
        for path in (problem, solution):
            if not path.is_file():
                failures.append(f"{path.relative_to(ROOT)}: missing")
                continue
            text = path.read_text()
            for needle, description in REQUIRED:
                if needle not in text:
                    failures.append(f"{path.relative_to(ROOT)}: missing {description}")

    if failures:
        print(f"Checked {checked} learner-authored puzzles.\n")
        for f in failures:
            print(f"  FAIL  {f}")
        print(f"\n{len(failures)} problem(s) found.")
        return 1
    print(f"All {checked} learner-authored puzzles use the canary.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
