# Changelog

## 1.0.0 - 2025-07-01

Initial release of the pymc-modeling Claude Code plugin.

### Added

- **Skills**: pymc-modeling, pymc-testing, prior-elicitation, model-evaluation, pymc-extras
- **Agents**: diagnostics-agent, model-review-agent, prior-elicitation-agent
- **Commands**: /pymc-diagnose, /prior-check, /shape-check, /model-compare
- **Hooks**: UserPromptSubmit skill suggestion, PostToolUse PyMC code checks, PostToolUse diagnostics reminder, PreCompact modeling context preservation
- **MCP Server**: PyMC documentation search tool
- **Benchmark**: Model quality evaluation suite
- Plugin manifest, install script, and validation tooling
