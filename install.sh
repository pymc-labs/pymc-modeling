#!/usr/bin/env bash
# Install script for pymc-modeling across supported agent harnesses.
# Supports Linux and macOS.

set -euo pipefail

PLUGIN_NAME="pymc-modeling"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


PLUGIN_SKILLS=(
  model-evaluation
  prior-elicitation
  pymc-extras
  pymc-modeling
  pymc-testing
)
PLUGIN_AGENTS=(
  diagnostics-agent.md
  model-review-agent.md
  prior-elicitation-agent.md
)
PLUGIN_COMMANDS=(
  model-compare.md
  prior-check.md
  pymc-diagnose.md
  shape-check.md
)

# Detect OS
case "$(uname -s)" in
  Linux*)  OS="Linux" ;;
  Darwin*) OS="macOS" ;;
  *)       echo "Unsupported OS: $(uname -s)"; exit 1 ;;
esac

usage() {
  cat <<EOF
Usage: bash install.sh <target>

Targets:
  claude-code   Claude Code plugin at ~/.claude/plugins/pymc-modeling
  omp           Oh My Pi / pi-compatible harness at \${PI_CODING_AGENT_DIR:-~/.omp/agent}
  pi            Legacy pi harness at ~/.pi/agent
  codex         Codex user resources at ~/.codex
  gemini        Gemini user resources at ~/.gemini
  opencode      OpenCode skills and commands at ~/.config/opencode
  agents        Generic Agent Skills at ~/.agents/skills

EOF
}

remove_path() {
  local target_path="$1"
  local label="$2"

  if [ -e "${target_path}" ] || [ -L "${target_path}" ]; then
    rm -rf "${target_path}"
    echo "  Removed existing ${label} at ${target_path}"
  fi
}

link_path() {
  local source_path="$1"
  local target_path="$2"
  local label="$3"

  if [ ! -e "${source_path}" ] && [ ! -L "${source_path}" ]; then
    return
  fi

  mkdir -p "$(dirname "${target_path}")"
  remove_path "${target_path}" "${label}"
  ln -s "${source_path}" "${target_path}"
  echo "  Linked ${label} -> ${target_path}"
}

uninstall_named_children() {
  local target_dir="$1"
  local label="$2"
  shift 2
  local name

  for name in "$@"; do
    remove_path "${target_dir}/${name}" "${label}/${name}"
  done
}

uninstall_claude_code_plugin() {
  remove_path "${HOME}/.claude/plugins/${PLUGIN_NAME}" "Claude Code plugin"
}

uninstall_extension() {
  local agent_dir="$1"
  local label="$2"

  remove_path "${agent_dir}/extensions/${PLUGIN_NAME}" "${label} extension"
}

uninstall_native_resources() {
  local agent_dir="$1"
  local label="$2"

  uninstall_named_children "${agent_dir}/skills" "${label} skills" "${PLUGIN_SKILLS[@]}"
  uninstall_named_children "${agent_dir}/agents" "${label} agents" "${PLUGIN_AGENTS[@]}"
  uninstall_named_children "${agent_dir}/commands" "${label} commands" "${PLUGIN_COMMANDS[@]}"
}

uninstall_skill_resources() {
  local skills_dir="$1"
  local label="$2"

  uninstall_named_children "${skills_dir}" "${label} skills" "${PLUGIN_SKILLS[@]}"
}

uninstall_command_resources() {
  local commands_dir="$1"
  local label="$2"

  uninstall_named_children "${commands_dir}" "${label} commands" "${PLUGIN_COMMANDS[@]}"
}

link_children() {
  local source_dir="$1"
  local target_dir="$2"
  local label="$3"
  local entry
  local name

  if [ ! -d "${source_dir}" ]; then
    return
  fi

  mkdir -p "${target_dir}"
  for entry in "${source_dir}"/*; do
    if [ ! -e "${entry}" ] && [ ! -L "${entry}" ]; then
      continue
    fi
    name="$(basename "${entry}")"
    link_path "${entry}" "${target_dir}/${name}" "${label}/${name}"
  done
}

copy_if_exists() {
  local source_path="$1"
  local target_path="$2"

  if [ -f "${source_path}" ]; then
    mkdir -p "$(dirname "${target_path}")"
    cp "${source_path}" "${target_path}"
  fi
}

install_claude_code_plugin() {
  local plugin_dir="${HOME}/.claude/plugins/${PLUGIN_NAME}"
  local dir
  local file

  mkdir -p "${plugin_dir}"

  # Claude Code plugin layout.
  for dir in skills agents commands hooks mcp-server scripts .claude-plugin; do
    link_path "${SOURCE_DIR}/${dir}" "${plugin_dir}/${dir}" "Claude Code ${dir}"
  done

  for file in README.md AGENTS.md LICENSE CHANGELOG.md; do
    copy_if_exists "${SOURCE_DIR}/${file}" "${plugin_dir}/${file}"
  done

  echo "  Installed Claude Code plugin at ${plugin_dir}"
}

install_extension() {
  local agent_dir="$1"
  local label="$2"
  local extension_source="${SOURCE_DIR}/.pi/extensions/${PLUGIN_NAME}"

  link_path "${extension_source}" "${agent_dir}/extensions/${PLUGIN_NAME}" "${label} extension"
}

install_native_resources() {
  local agent_dir="$1"
  local label="$2"

  link_children "${SOURCE_DIR}/skills" "${agent_dir}/skills" "${label} skills"
  link_children "${SOURCE_DIR}/agents" "${agent_dir}/agents" "${label} agents"
  link_children "${SOURCE_DIR}/commands" "${agent_dir}/commands" "${label} commands"
}

install_skill_resources() {
  local skills_dir="$1"
  local label="$2"

  link_children "${SOURCE_DIR}/skills" "${skills_dir}" "${label} skills"
}

install_command_resources() {
  local commands_dir="$1"
  local label="$2"

  link_children "${SOURCE_DIR}/commands" "${commands_dir}" "${label} commands"
}

TARGET="${1:-}"
OMP_AGENT_DIR="${PI_CODING_AGENT_DIR:-${HOME}/.omp/agent}"
LEGACY_PI_AGENT_DIR="${HOME}/.pi/agent"

if [ -z "${TARGET}" ] || [ "${TARGET}" = "-h" ] || [ "${TARGET}" = "--help" ]; then
  usage
  if [ -z "${TARGET}" ]; then
    exit 1
  fi
  exit 0
fi

echo "Installing ${PLUGIN_NAME} resources for ${TARGET} on ${OS}..."

case "${TARGET}" in
  claude-code | claude_code | claude)
    uninstall_claude_code_plugin
    install_claude_code_plugin
    INSTALLED_TARGET="Claude Code plugin: ${HOME}/.claude/plugins/${PLUGIN_NAME}"
    ;;
  omp | oh-my-pi | oh_my_pi)
    uninstall_extension "${OMP_AGENT_DIR}" "Oh My Pi"
    uninstall_native_resources "${OMP_AGENT_DIR}" "Oh My Pi"
    install_extension "${OMP_AGENT_DIR}" "Oh My Pi"
    install_native_resources "${OMP_AGENT_DIR}" "Oh My Pi"
    INSTALLED_TARGET="Oh My Pi extension/resources: ${OMP_AGENT_DIR}"
    ;;
  pi | legacy-pi | legacy_pi)
    uninstall_extension "${LEGACY_PI_AGENT_DIR}" "legacy pi"
    uninstall_skill_resources "${LEGACY_PI_AGENT_DIR}/skills" "legacy pi"
    install_extension "${LEGACY_PI_AGENT_DIR}" "legacy pi"
    install_skill_resources "${LEGACY_PI_AGENT_DIR}/skills" "legacy pi"
    INSTALLED_TARGET="legacy pi extension/resources: ${LEGACY_PI_AGENT_DIR}"
    ;;
  codex)
    uninstall_native_resources "${HOME}/.codex" "Codex"
    install_native_resources "${HOME}/.codex" "Codex"
    INSTALLED_TARGET="Codex user resources: ${HOME}/.codex"
    ;;
  gemini)
    uninstall_native_resources "${HOME}/.gemini" "Gemini"
    install_native_resources "${HOME}/.gemini" "Gemini"
    INSTALLED_TARGET="Gemini user resources: ${HOME}/.gemini"
    ;;
  opencode)
    uninstall_skill_resources "${HOME}/.config/opencode/skills" "OpenCode"
    uninstall_command_resources "${HOME}/.config/opencode/commands" "OpenCode"
    install_skill_resources "${HOME}/.config/opencode/skills" "OpenCode"
    install_command_resources "${HOME}/.config/opencode/commands" "OpenCode"
    INSTALLED_TARGET="OpenCode resources: ${HOME}/.config/opencode"
    ;;
  agents | agent-skills | agent_skills)
    uninstall_skill_resources "${HOME}/.agents/skills" "generic agents"
    install_skill_resources "${HOME}/.agents/skills" "generic agents"
    INSTALLED_TARGET="generic Agent Skills: ${HOME}/.agents/skills"
    ;;
  *)
    echo "Unknown target: ${TARGET}" >&2
    echo "" >&2
    usage >&2
    exit 1
    ;;
esac

# Make hook scripts executable
if [ -d "${SOURCE_DIR}/hooks/scripts" ]; then
  chmod +x "${SOURCE_DIR}/hooks/scripts/"*.sh 2>/dev/null || true
fi

# Make validation script executable
if [ -f "${SOURCE_DIR}/scripts/validate-plugin.sh" ]; then
  chmod +x "${SOURCE_DIR}/scripts/validate-plugin.sh"
fi

echo ""
echo "Successfully installed ${PLUGIN_NAME} resources."
echo "Plugin source: ${SOURCE_DIR}"
echo "Installed target: ${INSTALLED_TARGET}"
echo ""
echo "To validate the plugin structure, run:"
echo "  bash ${SOURCE_DIR}/scripts/validate-plugin.sh"
