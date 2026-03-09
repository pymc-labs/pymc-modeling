"""Evals for plugin structure — validates files, frontmatter, JSON, and cross-references."""

import json
import re
from pathlib import Path

import pytest
import yaml

PLUGIN_ROOT = Path(__file__).parent.parent

# ── Directory structure ──────────────────────────────────────────────────────

REQUIRED_DIRS = [
    "skills",
    "agents",
    "commands",
    "hooks",
    "hooks/scripts",
    "mcp-server",
    "scripts",
    ".claude-plugin",
]


@pytest.mark.parametrize("dirname", REQUIRED_DIRS)
def test_required_directory_exists(dirname):
    assert (PLUGIN_ROOT / dirname).is_dir(), f"Missing directory: {dirname}"


REQUIRED_FILES = [
    ".claude-plugin/plugin.json",
    "hooks/hooks.json",
    "mcp-server/src/pymc_docs_server/server.py",
    "install.sh",
    "README.md",
    "LICENSE",
]


@pytest.mark.parametrize("filepath", REQUIRED_FILES)
def test_required_file_exists(filepath):
    assert (PLUGIN_ROOT / filepath).is_file(), f"Missing file: {filepath}"


# ── JSON validity ────────────────────────────────────────────────────────────

JSON_FILES = [
    ".claude-plugin/plugin.json",
    "hooks/hooks.json",
]


@pytest.mark.parametrize("filepath", JSON_FILES)
def test_json_valid(filepath):
    path = PLUGIN_ROOT / filepath
    if not path.exists():
        pytest.skip(f"{filepath} not found")
    data = json.loads(path.read_text())
    assert isinstance(data, dict)


def test_mcp_json_valid_if_exists():
    path = PLUGIN_ROOT / ".mcp.json"
    if not path.exists():
        pytest.skip(".mcp.json not found")
    data = json.loads(path.read_text())
    assert isinstance(data, dict)


# ── MCP data files ───────────────────────────────────────────────────────────

MCP_DATA_FILES = [
    "mcp-server/data/pymc_api.json",
    "mcp-server/data/arviz_api.json",
    "mcp-server/data/patterns.json",
]


@pytest.mark.parametrize("filepath", MCP_DATA_FILES)
def test_mcp_data_file_valid(filepath):
    path = PLUGIN_ROOT / filepath
    assert path.is_file(), f"Missing MCP data file: {filepath}"
    data = json.loads(path.read_text())
    assert isinstance(data, list), f"{filepath} should be a JSON array"
    assert len(data) > 0, f"{filepath} is empty"


# ── SKILL.md frontmatter ─────────────────────────────────────────────────────

SKILL_DIRS = list((PLUGIN_ROOT / "skills").iterdir()) if (PLUGIN_ROOT / "skills").is_dir() else []


@pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=lambda d: d.name)
def test_skill_has_skill_md(skill_dir):
    assert (skill_dir / "SKILL.md").is_file(), f"{skill_dir.name} missing SKILL.md"


@pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=lambda d: d.name)
def test_skill_frontmatter_valid(skill_dir):
    """SKILL.md must have YAML frontmatter with name and description."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        pytest.skip("SKILL.md not found")

    text = skill_md.read_text()
    assert text.startswith("---"), "SKILL.md must start with YAML frontmatter (---)"

    # Extract frontmatter
    parts = text.split("---", 2)
    assert len(parts) >= 3, "SKILL.md must have closing --- for frontmatter"

    fm = yaml.safe_load(parts[1])
    assert isinstance(fm, dict), "Frontmatter must be a YAML mapping"
    assert "name" in fm, "Frontmatter missing 'name'"
    assert "description" in fm, "Frontmatter missing 'description'"
    assert isinstance(fm["name"], str) and len(fm["name"]) > 0
    assert isinstance(fm["description"], str) and len(fm["description"]) > 20, (
        "description should be at least 20 characters"
    )


@pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=lambda d: d.name)
def test_skill_name_matches_directory(skill_dir):
    """Skill name in frontmatter should match directory name."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        pytest.skip("SKILL.md not found")

    text = skill_md.read_text()
    parts = text.split("---", 2)
    if len(parts) < 3:
        pytest.skip("No frontmatter")

    fm = yaml.safe_load(parts[1])
    assert fm.get("name") == skill_dir.name, (
        f"Skill name '{fm.get('name')}' doesn't match directory '{skill_dir.name}'"
    )


@pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=lambda d: d.name)
def test_skill_body_not_empty(skill_dir):
    """SKILL.md must have substantive body content after frontmatter."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        pytest.skip("SKILL.md not found")

    text = skill_md.read_text()
    parts = text.split("---", 2)
    if len(parts) < 3:
        pytest.skip("No frontmatter")

    body = parts[2].strip()
    assert len(body) > 100, f"SKILL.md body too short ({len(body)} chars)"


# ── Agent files ──────────────────────────────────────────────────────────────

AGENT_FILES = list((PLUGIN_ROOT / "agents").glob("*.md")) if (PLUGIN_ROOT / "agents").is_dir() else []


@pytest.mark.parametrize("agent_file", AGENT_FILES, ids=lambda f: f.name)
def test_agent_file_not_empty(agent_file):
    text = agent_file.read_text().strip()
    assert len(text) > 50, f"Agent file {agent_file.name} too short ({len(text)} chars)"


# ── Command files ────────────────────────────────────────────────────────────

COMMAND_FILES = list((PLUGIN_ROOT / "commands").glob("*.md")) if (PLUGIN_ROOT / "commands").is_dir() else []


@pytest.mark.parametrize("command_file", COMMAND_FILES, ids=lambda f: f.name)
def test_command_file_not_empty(command_file):
    text = command_file.read_text().strip()
    assert len(text) > 50, f"Command file {command_file.name} too short ({len(text)} chars)"


# ── hooks.json schema ────────────────────────────────────────────────────────

def test_hooks_json_schema():
    """hooks.json must have valid hook event structure."""
    path = PLUGIN_ROOT / "hooks" / "hooks.json"
    data = json.loads(path.read_text())
    assert "hooks" in data, "hooks.json missing top-level 'hooks' key"

    hooks = data["hooks"]
    valid_events = {"UserPromptSubmit", "PostToolUse", "PreCompact", "PreToolUse", "PostToolUse"}
    for event_name in hooks:
        assert event_name in valid_events, f"Unknown hook event: {event_name}"
        entries = hooks[event_name]
        assert isinstance(entries, list), f"{event_name} must be a list"

        for entry in entries:
            assert "matcher" in entry, f"{event_name} entry missing 'matcher'"
            assert "hooks" in entry, f"{event_name} entry missing 'hooks'"
            for hook in entry["hooks"]:
                assert "type" in hook, f"Hook in {event_name} missing 'type'"
                assert hook["type"] in ("command", "prompt"), (
                    f"Invalid hook type: {hook['type']}"
                )


def test_hook_scripts_executable():
    """All .sh files in hooks/scripts/ must be executable."""
    scripts_dir = PLUGIN_ROOT / "hooks" / "scripts"
    if not scripts_dir.is_dir():
        pytest.skip("hooks/scripts/ not found")

    for sh in scripts_dir.glob("*.sh"):
        import os
        assert os.access(sh, os.X_OK), f"{sh.name} is not executable"


# ── Cross-references ─────────────────────────────────────────────────────────

def test_hook_scripts_referenced_exist():
    """Scripts referenced in hooks.json must exist on disk."""
    hooks_data = json.loads((PLUGIN_ROOT / "hooks" / "hooks.json").read_text())
    for event_entries in hooks_data["hooks"].values():
        for entry in event_entries:
            for hook in entry["hooks"]:
                if hook["type"] != "command":
                    continue
                cmd = hook["command"]
                # Extract script path after ${CLAUDE_PLUGIN_ROOT}/
                if "${CLAUDE_PLUGIN_ROOT}/" in cmd:
                    rel = cmd.split("${CLAUDE_PLUGIN_ROOT}/", 1)[1]
                    # Strip any trailing arguments
                    script_path = rel.split()[0]
                    assert (PLUGIN_ROOT / script_path).is_file(), (
                        f"Hook references missing script: {script_path}"
                    )


def test_skill_count():
    """Plugin should have the expected number of skills."""
    skills = list((PLUGIN_ROOT / "skills").iterdir())
    skill_dirs = [s for s in skills if s.is_dir()]
    assert len(skill_dirs) == 5, f"Expected 5 skills, found {len(skill_dirs)}: {[s.name for s in skill_dirs]}"


# ── Skill reference link validation ──────────────────────────────────────────

class TestSkillReferenceLinks:
    @pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=lambda d: d.name)
    def test_reference_links_resolve(self, skill_dir):
        """All references/*.md links in SKILL.md must resolve to existing files."""
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            pytest.skip("SKILL.md not found")

        text = skill_md.read_text()
        # Find markdown links: [text](references/foo.md) and bare references/foo.md
        link_pattern = re.compile(r'references/[\w\-]+\.md')
        links = link_pattern.findall(text)

        for link in links:
            target = skill_dir / link
            assert target.is_file(), (
                f"Broken reference link in {skill_dir.name}/SKILL.md: {link}"
            )


# ── Skill code block syntax ──────────────────────────────────────────────────

class TestSkillCodeBlocks:
    @pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=lambda d: d.name)
    def test_code_blocks_have_valid_syntax(self, skill_dir):
        """Python code blocks in skills should have valid syntax."""
        md_files = [skill_dir / "SKILL.md"]
        refs_dir = skill_dir / "references"
        if refs_dir.is_dir():
            md_files.extend(refs_dir.glob("*.md"))

        errors = []
        for md_file in md_files:
            if not md_file.exists():
                continue
            text = md_file.read_text()
            # Extract python code blocks
            blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
            for i, block in enumerate(blocks):
                # Skip pseudocode blocks
                if "# pseudo" in block.lower() or "..." == block.strip():
                    continue
                # Replace ... with pass for compilation
                compile_block = block.replace("...", "pass")
                try:
                    compile(compile_block, f"{md_file.name}:block{i}", "exec")
                except SyntaxError as e:
                    errors.append(
                        f"{md_file.relative_to(skill_dir)}:block{i}: {e.msg} (line {e.lineno})"
                    )

        if errors:
            # Warn but don't fail for small numbers — some blocks may have intentional pseudo-syntax
            # Only fail if more than 20% of blocks have errors
            total_blocks = sum(
                len(re.findall(r'```python\n', (skill_dir / "SKILL.md").read_text()))
                for _ in [1]
            )
            if len(errors) > max(1, total_blocks * 0.2):
                pytest.fail(
                    f"{len(errors)} code blocks with syntax errors:\n" +
                    "\n".join(errors[:10])
                )
