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
│   ├── omnicaptions-transcribe/SKILL.md
│   ├── omnicaptions-convert/SKILL.md
│   ├── omnicaptions-translate/SKILL.md
│   ├── omnicaptions-download/SKILL.md
│   └── omnicaptions-LaiCut/SKILL.md
├── src/omnicaptions/
│   ├── __init__.py          # Exports
│   ├── caption.py           # GeminiCaption class
│   ├── cli.py               # CLI entry point
│   ├── config.py            # API key management
│   └── prompts/
│       └── transcription_dotey.md
└── pyproject.toml
```

## Key Classes

- `GeminiCaption`: Main class with `transcribe()` and `translate()` methods
- `GeminiCaptionConfig`: Configuration dataclass
- From `lattifai-captions`: `Caption`, `GeminiReader`, `GeminiWriter`
- From `lattifai`: `LattifAI` client for forced alignment (via `omnicaptions[alignment]`)

## CLI

```bash
# All commands support -o for output file/directory (optional, defaults to input dir)
omnicaptions transcribe <input> [-o output] [-m model] [-l lang] [-t lang --bilingual]
omnicaptions convert <input> [-o output] [-f fmt] [-t fmt]
omnicaptions translate <input> [-o output] -l <lang> [--bilingual]
omnicaptions download <url> [-o output] [-q quality]
omnicaptions LaiCut <audio> <caption> [-o output] [-f format] [-k api-key] [-v]
```

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
| (default) | ✓ | ✓ | Description always; full on invoke |
| `disable-model-invocation: true` | ✓ | ✗ | Manual only |
| `user-invocable: false` | ✗ | ✓ | Background knowledge |

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
