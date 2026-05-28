#!/usr/bin/env bash
# Detect PyMC-stack imports in files Claude reads.
# Runs as a PostToolUse hook for the Read tool. Receives JSON on stdin
# with .tool_input.file_path. Emits additionalContext directing Claude to
# load the pymc-modeling skill when the read file imports pymc, pytensor,
# or arviz. Silent and exit 0 otherwise -- hooks must never fail.

set -euo pipefail

input=$(cat)
file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null || true)

if [ -z "$file_path" ] || [ ! -r "$file_path" ]; then
  exit 0
fi

# Extension filter (case-insensitive). Only Python source and notebooks.
shopt -s nocasematch
case "$file_path" in
  *.py|*.ipynb) ;;
  *) exit 0 ;;
esac
shopt -u nocasematch

# Size guard: skip files larger than ~5 MB.
size=$(wc -c < "$file_path" 2>/dev/null || echo 0)
if [ "$size" -gt 5242880 ]; then
  exit 0
fi

# Pattern: import or from-import of the three stack libraries.
# `arviz_stats` and `pymc_extras` etc. are caught by the prefix match.
py_pattern='^[[:space:]]*(import|from)[[:space:]]+(pymc|pytensor|arviz)([[:space:]]|\.|$)'

matched=""
case "$file_path" in
  *.py|*.PY)
    if grep -qE "$py_pattern" "$file_path" 2>/dev/null; then
      matched=$(grep -oE '(pymc|pytensor|arviz)' "$file_path" | sort -u | tr '\n' ',' | sed 's/,$//')
    fi
    ;;
  *.ipynb|*.IPYNB)
    # Extract code-cell source lines via jq; fall back to loose grep if jq fails.
    if extracted=$(jq -r '.cells[]? | select(.cell_type=="code") | .source[]?' "$file_path" 2>/dev/null); then
      if echo "$extracted" | grep -qE "$py_pattern"; then
        matched=$(echo "$extracted" | grep -oE '(pymc|pytensor|arviz)' | sort -u | tr '\n' ',' | sed 's/,$//')
      fi
    else
      # Malformed notebook -- try a looser match on JSON-escaped source lines.
      if grep -qE '"[[:space:]]*(import|from)[[:space:]]+(pymc|pytensor|arviz)' "$file_path" 2>/dev/null; then
        matched=$(grep -oE '(pymc|pytensor|arviz)' "$file_path" | sort -u | tr '\n' ',' | sed 's/,$//')
      fi
    fi
    ;;
esac

if [ -z "$matched" ]; then
  exit 0
fi

ctx="Load the pymc-modeling skill before responding. The file just read (${file_path}) imports ${matched}; the user is working on PyMC / PyTensor / ArviZ code, so the PyMC 6+, PyTensor 3+, ArviZ 1.0+ API guidance in this skill is needed to answer correctly. If the work involves testing, prior elicitation, model comparison, or pymc-extras features, also load the matching sister skill (pymc-testing, prior-elicitation, model-evaluation, pymc-extras)."

jq -n --arg ctx "$ctx" '{
  "hookSpecificOutput": {
    "hookEventName": "PostToolUse",
    "additionalContext": $ctx
  }
}'

exit 0
