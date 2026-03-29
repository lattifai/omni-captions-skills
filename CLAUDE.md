# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

**omni-captions-skills** - Claude Code plugin for media transcription and caption translation using AI APIs.

## Structure

```
omni-captions-skills/
├── .claude-plugin/
│   ├── plugin.json          # Plugin manifest (required)
│   └── marketplace.json     # Marketplace distribution
├── skills/
│   ├── omnicaptions-pipeline/SKILL.md    # Primary: download+transcribe+translate
│   ├── omnicaptions-transcribe/SKILL.md  # Specialized: transcription only
│   ├── omnicaptions-convert/SKILL.md     # Specialized: format conversion
│   └── omnicaptions-LaiCut/SKILL.md      # Specialized: forced alignment
├── src/omnicaptions/
│   ├── __init__.py          # Exports
│   ├── caption.py           # GeminiCaption class (legacy, wraps services)
│   ├── cli.py               # CLI entry point (all commands support --json)
│   ├── mcp_server.py        # MCP Server (Claude Desktop / Cursor integration)
│   ├── config.py            # API key management (show/set/get via CLI)
│   ├── ui.py                # Rich console helpers (spinner, table, progress)
│   ├── services/
│   │   ├── __init__.py      # Exports ToolResult, ErrorCode
│   │   ├── base.py          # ToolResult dataclass, ErrorCode enum
│   │   ├── pipeline.py      # Pipeline orchestration (download→transcribe→align→translate→convert)
│   │   ├── download.py      # YouTube download via yt-dlp
│   │   ├── transcribe.py    # Gemini-based transcription
│   │   ├── translate.py     # Caption translation
│   │   ├── convert.py       # Format conversion (30+ formats)
│   │   ├── align.py         # LattifAI forced alignment
│   │   └── doctor.py        # Preflight environment check
│   └── prompts/
│       └── transcription_dotey.md
└── pyproject.toml
```

## Key Classes

- `ToolResult`: Structured return type from all services (`status`, `message`, `data`, `error_code`). Supports `.to_json()`.
- `ErrorCode`: Enum of machine-readable error categories (MISSING_API_KEY, NETWORK_TIMEOUT, etc.)
- `GeminiCaption`: Legacy class with `transcribe()` and `translate()` methods. Still works but services layer is preferred.
- From `lattifai-captions`: `Caption`, `GeminiReader`, `GeminiWriter`
- From `lattifai`: `LattifAI` client for forced alignment (via `omnicaptions[alignment]`)

## CLI

```bash
# All commands support -o for output and --json for structured JSON output
omnicaptions pipeline <url> [-l lang] [-f fmt] [--no-align] [--no-translate] [--json]
omnicaptions transcribe <input> [-o output] [-m model] [-l lang] [-t lang --bilingual] [--json]
omnicaptions convert <input> [-o output] [-f fmt] [-t fmt] [--json]
omnicaptions translate <input> [-o output] -l <lang> [--bilingual] [--json]
omnicaptions download <url> [-o output] [-q quality] [--json]
omnicaptions LaiCut <audio> <caption> [-o output] [-f format] [-k api-key] [-v] [--json]
omnicaptions doctor [--json]    # Preflight check: ffmpeg, yt-dlp, API keys
omnicaptions config show|set|get [--json]  # Manage API keys
```

The `--json` flag outputs `ToolResult` as JSON, designed for agent consumption.

## Skills

| Skill | Type | Description |
|-------|------|-------------|
| `pipeline` | Primary | Full workflow: download + transcribe + translate |
| `transcribe` | Specialized | YouTube/video transcription only |
| `LaiCut` | Specialized | Forced alignment with word-level timing |
| `convert` | Specialized | Format conversion between 30+ caption formats |

> `download` and `translate` skills were removed — absorbed into the `pipeline` skill.

## MCP Server

Exposes all services as MCP tools for Claude Desktop, Cursor, and other MCP clients.

```bash
# Install
pip install -e ".[mcp]"

# Run (stdio, default)
omnicaptions-mcp

# Run (SSE)
omnicaptions-mcp --transport sse --port 8080
```

### Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "omnicaptions": {
      "command": "omnicaptions-mcp"
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `doctor` | Check environment dependencies and API keys |
| `download` | Download media from YouTube/video platforms |
| `transcribe` | Transcribe audio/video to text with timestamps |
| `check_captions` | Check if a URL has existing captions |
| `translate` | Translate caption files |
| `convert` | Convert between 30+ caption formats |
| `forced_align` | LattifAI forced alignment for precise timing |
| `pipeline` | Full pipeline: download, transcribe, align, translate, convert |

## Development

```bash
# Format
ruff check --fix . && ruff format .

# Test
pytest
```

## Dependencies

- `google-genai`: Gemini API
- `lattifai-captions`: Caption formats
- `omnicaptions[alignment]`: Optional - LattifAI for forced alignment

---

## Skills Development Reference

Reference: https://code.claude.com/docs/en/skills

### SKILL.md Format

```yaml
---
name: skill-name                    # Lowercase, alphanumeric, hyphens (max 64 chars)
description: When to use this skill # Claude uses this for auto-invocation
argument-hint: [filename] [format]  # Optional: hint for autocomplete
disable-model-invocation: true      # Optional: prevent auto-loading, manual /name only
user-invocable: false               # Optional: hide from / menu (background knowledge)
allowed-tools: Read, Grep, Bash(python:*)  # Optional: tools without permission
model: sonnet                       # Optional: model override
context: fork                       # Optional: run in isolated subagent
agent: Explore                      # Optional: subagent type (Explore, Plan, general-purpose)
---

Your skill instructions here...
```

### Frontmatter Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | No | Display name, becomes `/plugin:name` |
| `description` | Recommended | When to use - Claude uses for auto-invocation |
| `argument-hint` | No | Autocomplete hint: `[issue-number]` |
| `disable-model-invocation` | No | `true` = manual only, no auto-load |
| `user-invocable` | No | `false` = background knowledge only |
| `allowed-tools` | No | Tools without permission prompts |
| `model` | No | Model override when skill active |
| `context` | No | `fork` = isolated subagent |
| `agent` | No | Subagent type: `Explore`, `Plan`, `general-purpose` |

### Invocation Control

| Config | User invoke | Claude invoke | Context |
|--------|-------------|---------------|---------|
| (default) | Y | Y | Description always; full on invoke |
| `disable-model-invocation: true` | Y | N | Manual only |
| `user-invocable: false` | N | Y | Background knowledge |

### String Substitutions

| Variable | Description |
|----------|-------------|
| `$ARGUMENTS` | Arguments passed when invoking |
| `${CLAUDE_SESSION_ID}` | Current session ID |

### Dynamic Context Injection

Use `` `!command` `` to run shell and inject output:

```markdown
## Current status
- Branch: !`git branch --show-current`
- Changes: !`git status --short`
```

### Best Practices

1. Keep SKILL.md focused (<500 lines)
2. Reference supporting files for details
3. Write descriptive descriptions for auto-invocation
4. Use `disable-model-invocation: true` for side-effect operations
5. Use `user-invocable: false` for background context

### Plugin Structure

```
plugin/
├── .claude-plugin/
│   └── plugin.json          # Required manifest
├── skills/
│   └── skill-name/SKILL.md  # Creates /plugin:skill-name
└── ...
```

### plugin.json Schema

```json
{
  "name": "plugin-name",           // Namespace for skills
  "description": "What it does",
  "version": "1.0.0",
  "author": { "name": "Author" },
  "repository": "https://github.com/...",
  "license": "MIT"
}
```
