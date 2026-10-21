#!/bin/bash
# Run every unfilled problem skeleton and check that it gets as far as its own check: it
# compiles, runs, prints what it got against what it expected, and fails there. A skeleton
# that no longer compiles (an API moved under it) or that already passes (a solution leaked
# into it) fails this instead.
#   bash scripts/check_problems.sh        # all puzzles
#   bash scripts/check_problems.sh p03    # one puzzle
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

run() {  # the interpreter for a puzzle file, by extension
    case "$1" in
        *.py) uv run python "$1" ;;
        *.mojo) uv run mojo -I . "$1" ;;
    esac
}

pass=0
fail=0
failed=()
for f in problems/p*/p*.py problems/p*/p*.mojo; do
    [ -f "$f" ] || continue
    p="$(basename "${f%.*}")"
    [ $# -gt 0 ] && [ "$p" != "$1" ] && continue
    printf '%s ... ' "$p"
    out="$(run "$f" 2>&1)"
    status=$?
    if [ "$status" -ne 0 ] && grep -q -E "expected:|Layouts disagree" <<<"$out"; then
        echo "✓ (reaches its check and fails there, as an unfilled skeleton should)"
        pass=$((pass + 1))
    else
        if [ "$status" -eq 0 ]; then
            echo "✗ (passes: the skeleton contains a solution)"
        else
            echo "✗ (does not reach its check)"
        fi
        tail -20 <<<"$out" | sed 's/^/    /'
        fail=$((fail + 1))
        failed+=("$p")
    fi
done

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ] || { printf '  %s\n' "${failed[@]}"; exit 1; }
