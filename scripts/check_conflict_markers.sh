#!/bin/sh
# Fail when a tracked file contains Git conflict markers.
#
# Opening and closing markers at the start of a line are unambiguous. A bare
# "=======" line is also a legitimate reStructuredText or Markdown underline,
# so it is only reported in a file that contains an opening marker as well.
#
# Usage: sh scripts/check_conflict_markers.sh [repository path]
set -eu
cd "${1:-.}"

found=0
if git grep -n -I -E '^(<<<<<<< |>>>>>>> )' -- . ; then
    found=1
fi
for file in $(git grep -l -I -E '^<<<<<<< ' -- . ); do
    if git grep -n -E '^=======$' -- "$file"; then
        found=1
    fi
done

if [ "$found" -ne 0 ]; then
    echo "Conflict markers found in tracked files (see above)." >&2
    exit 1
fi
echo "No conflict markers in tracked files."
