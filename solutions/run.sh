#!/bin/bash
# Run every reference solution and the harness tests, and check that each reports ✅.
#   bash solutions/run.sh        # everything
#   bash solutions/run.sh p03    # one puzzle
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
for f in solutions/p*/p*.py solutions/p*/p*.mojo harness/test/test_*.py harness/test/test_*.mojo; do
    [ -f "$f" ] || continue
    p="$(basename "${f%.*}")"
    [ $# -gt 0 ] && [ "$p" != "$1" ] && continue
    printf '%s ... ' "$p"
    if out="$(run "$f" 2>&1)" && grep -q "✅" <<<"$out"; then
        echo "✓"
        pass=$((pass + 1))
    else
        echo "✗"
        tail -20 <<<"$out" | sed 's/^/    /'
        fail=$((fail + 1))
        failed+=("$p")
    fi
done

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ] || { printf '  %s\n' "${failed[@]}"; exit 1; }
