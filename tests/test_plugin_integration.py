"""Cross-component integration tests.

Tests that verify different plugin components work together correctly.
File-existence and schema-validation tests live in test_plugin_structure.py;
this file focuses on cross-talk between skills, hooks, MCP data, and agents.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

PLUGIN_ROOT = Path(__file__).parent.parent


class TestSkillHookIntegration:
    """Verify that skills referenced in hooks actually exist."""

    def test_suggested_skills_exist(self):
        """Every skill name in suggest-skill.sh maps to an actual skill directory."""
        script = PLUGIN_ROOT / "hooks" / "scripts" / "suggest-skill.sh"
        text = script.read_text()
        # Extract skill names from the script (look for skill directory references)
        skill_refs = set(re.findall(r'skills/([\w-]+)', text))
        skills_dir = PLUGIN_ROOT / "skills"
        existing = {d.name for d in skills_dir.iterdir() if d.is_dir()}

        for ref in skill_refs:
            assert ref in existing, (
                f"suggest-skill.sh references skill '{ref}' but no skills/{ref}/ exists"
            )


class TestMCPSkillConsistency:
    """Verify MCP server data aligns with skill content."""

    def test_api_lookup_covers_skill_mentioned_functions(self):
        """Core functions mentioned in pymc-modeling SKILL.md should be in the MCP API data."""
        skill_md = PLUGIN_ROOT / "skills" / "pymc-modeling" / "SKILL.md"
        if not skill_md.exists():
            pytest.skip("pymc-modeling SKILL.md not found")

        text = skill_md.read_text()
        # Extract pm.Function references from the skill
        pm_refs = set(re.findall(r'(pm\.\w+)', text))
        # These are the most important ones that should have MCP entries
        core_refs = pm_refs & {"pm.sample", "pm.Model", "pm.Normal", "pm.HalfNormal", "pm.Data"}

        api_path = PLUGIN_ROOT / "mcp-server" / "data" / "pymc_api.json"
        api_data = json.loads(api_path.read_text())
        api_names = {e["name"] for e in api_data}

        missing = core_refs - api_names
        assert not missing, (
            f"SKILL.md references these core functions not in MCP API data: {missing}"
        )

    def test_error_patterns_cover_skill_antipatterns(self):
        """Error patterns in MCP data should cover antipatterns mentioned in skills."""
        error_path = PLUGIN_ROOT / "mcp-server" / "data"
        # Load all error pattern keywords
        from pymc_docs_server.server import ERROR_PATTERNS
        all_keywords = set()
        for ep in ERROR_PATTERNS:
            all_keywords.update(k.lower() for k in ep["keywords"])

        # Key error conditions that skills warn about should have error coverage
        expected_coverage = ["divergen", "theano"]
        for term in expected_coverage:
            assert any(term in kw for kw in all_keywords), (
                f"No error pattern covers '{term}', but skills warn about it"
            )


class TestSkillReferenceCompleteness:
    """Verify skill references/ dirs have content matching SKILL.md claims."""

    def test_skill_reference_dirs_have_content(self):
        """Skills that reference a references/ dir have matching .md files."""
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


class TestCommandAgentCoverage:
    """Verify commands and agents reference valid skills or tools."""

    def test_commands_reference_existing_skills(self):
        """Slash commands that mention skill names should reference real skills."""
        commands_dir = PLUGIN_ROOT / "commands"
        if not commands_dir.is_dir():
            pytest.skip("No commands directory")

        skills_dir = PLUGIN_ROOT / "skills"
        existing_skills = {d.name for d in skills_dir.iterdir() if d.is_dir()}

        for cmd_file in commands_dir.glob("*.md"):
            text = cmd_file.read_text()
            # Look for skill references in command files
            skill_refs = set(re.findall(r'skills/([\w-]+)', text))
            for ref in skill_refs:
                assert ref in existing_skills, (
                    f"Command {cmd_file.name} references skill '{ref}' which doesn't exist"
                )
