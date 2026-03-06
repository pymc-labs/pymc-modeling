#!/usr/bin/env bash
# PostToolUse hook on Write|Edit that reminds to add diagnostics after pm.sample().
# Reads tool_use input from stdin JSON.

set -euo pipefail

input=$(cat)

# Extract file path from the tool input
file_path=$(echo "$input" | jq -r '.tool_input.file_path // .tool_input.path // empty' 2>/dev/null || true)

if [ -z "$file_path" ]; then
  exit 0
fi

# Only process Python files
case "$file_path" in
  *.py) ;;
  *) exit 0 ;;
esac

# Get file content: use content from input if available, otherwise read from disk
content=$(echo "$input" | jq -r '.tool_input.content // empty' 2>/dev/null || true)
if [ -z "$content" ] && [ -f "$file_path" ]; then
  content=$(cat "$file_path" 2>/dev/null || true)
fi

if [ -z "$content" ]; then
  exit 0
fi

# Check if pm.sample( appears in the content
if echo "$content" | grep -qE 'pm\.sample\('; then
  jq -n '{
    "systemMessage": "Remember to add convergence diagnostics after pm.sample(): check divergences, r_hat, ESS, and run posterior predictive checks. Save results immediately with .to_netcdf()."
  }'
fi

exit 0
