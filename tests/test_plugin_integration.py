"""End-to-end plugin structure integration test.

Verifies that all plugin components work together: skills load and parse,
hooks.json references resolve, MCP server data loads, and cross-references
between skills and references are valid.
"""

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

PLUGIN_ROOT = Path(__file__).parent.parent


class TestPluginIntegration:
    def test_all_skills_loadable(self):
        """Every SKILL.md can be read and has valid frontmatter."""
        skills_dir = PLUGIN_ROOT / "skills"
        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            assert skill_md.exists(), f"Missing SKILL.md in {skill_dir.name}"
            text = skill_md.read_text()
            assert text.startswith("---"), f"{skill_dir.name}/SKILL.md missing frontmatter"
            parts = text.split("---", 2)
            fm = yaml.safe_load(parts[1])
            assert "name" in fm and "description" in fm

    def test_hooks_json_scripts_exist(self):
        """All scripts referenced in hooks.json exist and are executable."""
        hooks_path = PLUGIN_ROOT / "hooks" / "hooks.json"
        data = json.loads(hooks_path.read_text())
        for event_entries in data["hooks"].values():
            for entry in event_entries:
                for hook in entry["hooks"]:
                    if hook["type"] != "command":
                        continue
                    cmd = hook["command"]
                    if "${CLAUDE_PLUGIN_ROOT}/" in cmd:
                        rel = cmd.split("${CLAUDE_PLUGIN_ROOT}/", 1)[1].split()[0]
                        script = PLUGIN_ROOT / rel
                        assert script.is_file(), f"Missing script: {rel}"
                        assert os.access(script, os.X_OK), f"Not executable: {rel}"

    def test_mcp_data_files_load(self):
        """MCP server JSON data files load without errors."""
        data_dir = PLUGIN_ROOT / "mcp-server" / "data"
        for name in ["pymc_api.json", "arviz_api.json", "patterns.json"]:
            path = data_dir / name
            assert path.exists(), f"Missing {name}"
            data = json.loads(path.read_text())
            assert isinstance(data, list) and len(data) > 0, f"{name} is empty or not a list"

    def test_skill_reference_dirs_exist(self):
        """Skills that reference a references/ dir actually have one."""
        skills_dir = PLUGIN_ROOT / "skills"
        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            text = skill_md.read_text()
            if "references/" in text:
                refs = skill_dir / "references"
                assert refs.is_dir(), (
                    f"{skill_dir.name}/SKILL.md mentions references/ but dir doesn't exist"
                )
                assert len(list(refs.glob("*.md"))) > 0, (
                    f"{skill_dir.name}/references/ exists but has no .md files"
                )

    def test_agent_files_not_empty(self):
        """All agent definition files have content."""
        agents_dir = PLUGIN_ROOT / "agents"
        if not agents_dir.is_dir():
            pytest.skip("No agents directory")
        for agent_file in agents_dir.glob("*.md"):
            text = agent_file.read_text().strip()
            assert len(text) > 50, f"Agent {agent_file.name} is too short"

    def test_command_files_not_empty(self):
        """All command definition files have content."""
        commands_dir = PLUGIN_ROOT / "commands"
        if not commands_dir.is_dir():
            pytest.skip("No commands directory")
        for cmd_file in commands_dir.glob("*.md"):
            text = cmd_file.read_text().strip()
            assert len(text) > 50, f"Command {cmd_file.name} is too short"
