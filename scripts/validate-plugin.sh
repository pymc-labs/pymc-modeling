#!/usr/bin/env bash
# Validate pymc-modeling plugin structure and files
set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PASS=0
FAIL=0

check() {
  local desc="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    echo "  PASS: ${desc}"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: ${desc}"
    FAIL=$((FAIL + 1))
  fi
}

echo "Validating pymc-modeling plugin at ${PLUGIN_DIR}"
echo ""

# Check directories
echo "Directories:"
for dir in skills agents commands hooks hooks/scripts mcp-server .claude-plugin scripts; do
  check "${dir}/" test -d "${PLUGIN_DIR}/${dir}"
done
echo ""

# Check key files
echo "Key files:"
check "plugin.json" test -f "${PLUGIN_DIR}/.claude-plugin/plugin.json"
check "hooks.json" test -f "${PLUGIN_DIR}/hooks/hooks.json"
check "mcp-server/src/pymc_docs_server/server.py" test -f "${PLUGIN_DIR}/mcp-server/src/pymc_docs_server/server.py"
check "install.sh" test -f "${PLUGIN_DIR}/install.sh"
check "README.md" test -f "${PLUGIN_DIR}/README.md"
check "LICENSE" test -f "${PLUGIN_DIR}/LICENSE"
echo ""

# Check SKILL.md files
echo "Skill files:"
for skill_dir in "${PLUGIN_DIR}"/skills/*/; do
  if [ -d "$skill_dir" ]; then
    skill_name=$(basename "$skill_dir")
    check "skills/${skill_name}/SKILL.md" test -f "${skill_dir}/SKILL.md"
  fi
done
echo ""

# Check agent files
echo "Agent files:"
for agent_file in "${PLUGIN_DIR}"/agents/*.md; do
  if [ -f "$agent_file" ]; then
    agent_name=$(basename "$agent_file")
    check "agents/${agent_name}" test -f "$agent_file"
  fi
done
echo ""

# Check command files
echo "Command files:"
for cmd_file in "${PLUGIN_DIR}"/commands/*.md; do
  if [ -f "$cmd_file" ]; then
    cmd_name=$(basename "$cmd_file")
    check "commands/${cmd_name}" test -f "$cmd_file"
  fi
done
echo ""

# Validate JSON files
echo "JSON validation:"
for json_file in .claude-plugin/plugin.json hooks/hooks.json; do
  if [ -f "${PLUGIN_DIR}/${json_file}" ]; then
    check "${json_file} is valid JSON" jq empty "${PLUGIN_DIR}/${json_file}"
  else
    check "${json_file} exists" false
  fi
done
if [ -f "${PLUGIN_DIR}/.mcp.json" ]; then
  check ".mcp.json is valid JSON" jq empty "${PLUGIN_DIR}/.mcp.json"
fi
echo ""

# Check shell scripts are executable
echo "Executable scripts:"
for script in "${PLUGIN_DIR}"/hooks/scripts/*.sh; do
  if [ -f "$script" ]; then
    script_name=$(basename "$script")
    check "hooks/scripts/${script_name} is executable" test -x "$script"
  fi
done
for script in "${PLUGIN_DIR}"/scripts/*.sh; do
  if [ -f "$script" ]; then
    script_name=$(basename "$script")
    check "scripts/${script_name} is executable" test -x "$script"
  fi
done
echo ""

# Check pi extension
echo "Pi extension:"
PI_EXT_DIR="${PLUGIN_DIR}/.pi/extensions/pymc-modeling"
if [ -d "${PI_EXT_DIR}" ]; then
  check ".pi/extensions/pymc-modeling/index.ts" test -f "${PI_EXT_DIR}/index.ts"
  
  if [ -f "${PI_EXT_DIR}/package.json" ]; then
    check ".pi/extensions/pymc-modeling/package.json is valid JSON" jq empty "${PI_EXT_DIR}/package.json"
  else
    check ".pi/extensions/pymc-modeling/package.json exists" false
  fi
  
  echo "  Pi extension data files:"
  for data_file in pymc_api.json arviz_api.json patterns.json; do
    check ".pi/extensions/pymc-modeling/data/${data_file}" test -f "${PI_EXT_DIR}/data/${data_file}"
  done
else
  check ".pi/extensions/pymc-modeling/ directory exists" false
fi
echo ""

# Summary
echo "Results: ${PASS} passed, ${FAIL} failed"
if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
echo "All checks passed."
