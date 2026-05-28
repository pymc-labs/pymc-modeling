#!/usr/bin/env bash
# Install script for pymc-modeling plugin (Claude Code and pi)
# Supports Linux and macOS

set -euo pipefail

PLUGIN_NAME="pymc-modeling"
PLUGIN_DIR="${HOME}/.claude/plugins/${PLUGIN_NAME}"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect OS
case "$(uname -s)" in
  Linux*)  OS="Linux" ;;
  Darwin*) OS="macOS" ;;
  *)       echo "Unsupported OS: $(uname -s)"; exit 1 ;;
esac

echo "Installing ${PLUGIN_NAME} plugin on ${OS}..."

# Create plugin directory
mkdir -p "${PLUGIN_DIR}"

# Shared directories (symlinked for Claude Code, also used by pi via project root)
SHARED_DIRS=("skills" "agents" "commands" "hooks" "mcp-server" "scripts")

# Claude Code-specific directories
CLAUDE_DIRS=(".claude-plugin")

for dir in "${SHARED_DIRS[@]}" "${CLAUDE_DIRS[@]}"; do
  if [ -d "${SOURCE_DIR}/${dir}" ]; then
    # Remove existing symlink or directory
    rm -rf "${PLUGIN_DIR}/${dir}"
    ln -sf "${SOURCE_DIR}/${dir}" "${PLUGIN_DIR}/${dir}"
    echo "  Linked ${dir}/"
  fi
done

# Install pi extension (pi auto-discovers extensions in ~/.pi/agent/extensions/)
PI_EXT_DIR="${HOME}/.pi/agent/extensions"
if [ -d "${SOURCE_DIR}/.pi/extensions/pymc-modeling" ]; then
  mkdir -p "${PI_EXT_DIR}"
  rm -rf "${PI_EXT_DIR}/pymc-modeling"
  ln -sf "${SOURCE_DIR}/.pi/extensions/pymc-modeling" "${PI_EXT_DIR}/pymc-modeling"
  echo "  Linked pi extension to ${PI_EXT_DIR}/pymc-modeling"
fi

# Copy top-level files
for file in README.md AGENTS.md LICENSE CHANGELOG.md; do
  if [ -f "${SOURCE_DIR}/${file}" ]; then
    cp "${SOURCE_DIR}/${file}" "${PLUGIN_DIR}/${file}"
  fi
done

# Make hook scripts executable
if [ -d "${SOURCE_DIR}/hooks/scripts" ]; then
  chmod +x "${SOURCE_DIR}/hooks/scripts/"*.sh 2>/dev/null || true
fi

# Make validation script executable
if [ -f "${SOURCE_DIR}/scripts/validate-plugin.sh" ]; then
  chmod +x "${SOURCE_DIR}/scripts/validate-plugin.sh"
fi

echo ""
echo "Successfully installed ${PLUGIN_NAME} plugin to ${PLUGIN_DIR}"
echo "Plugin source: ${SOURCE_DIR}"
echo ""
echo "To validate the installation, run:"
echo "  bash ${SOURCE_DIR}/scripts/validate-plugin.sh"
