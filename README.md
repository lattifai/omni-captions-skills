# omni-captions-skills

**Captions Made Easy** — Claude Code Caption Skills

> "I need bilingual captions for this Fireship vibe coding video https://youtube.com/watch?v=Tw18-4U7mts"
>
> One sentence. Claude handles the download, transcription, and translation.

[中文文档](README_zh.md)

## Install

```bash
npx skills add https://github.com/lattifai/omni-captions-skills
```

<details>
<summary>Other installation methods</summary>

**Claude Code Plugin System:**
```bash
/plugin marketplace add lattifai/omni-captions-skills
/plugin install omnicaptions@lattifai-omni-captions-skills
```

**Local Development:**
```bash
git clone https://github.com/lattifai/omni-captions-skills.git
claude --plugin-dir ./omni-captions-skills
```
</details>

## Try It

```
> Make bilingual captions for this Fireship vibe coding video https://youtube.com/watch?v=Tw18-4U7mts
```

```srt
1
00:00:00,000 --> 00:00:03,200
Mass hysteria satisfies a deep human need.

2
00:00:03,200 --> 00:00:07,440
Vibe coding is programming without actually writing any code yourself.
Vibe coding
```

## Skills

| Skill | Description |
|-------|-------------|
| `pipeline` | **Full workflow**: download + transcribe + translate in one shot |
| `transcribe` | YouTube/video -> Markdown with timestamps |
| `convert` | Convert between 30+ caption formats |
| `LaiCut` | **Forced alignment, word-level timing accuracy** |

> Invoke via `/omnicaptions:pipeline` or `/omnicaptions-transcribe`

## LaiCut: Precise Timing

Standard transcription gives "approximate" timestamps. LaiCut uses [LattifAI](https://lattifai.com/) Lattice-1 model to match text precisely to audio waveforms, achieving **word-level accuracy**.

**Install LaiCut:**
```bash
# Using uv (recommended, auto-configures package index)
uv pip install "omni-captions-skills[laicut]" --extra-index-url https://lattifai.github.io/pypi/simple/

# Using pip
pip install "omni-captions-skills[laicut]" --extra-index-url https://lattifai.github.io/pypi/simple/
```

**Supported languages:** English, Chinese, German, and mixed

**Recommended workflow:** Align before translate (translated text doesn't match original audio)

## Setup

| Feature | API Key | Note |
|---------|---------|------|
| Translation | None required | **Uses Claude by default**, works out of the box |
| Transcription | [Gemini API](https://aistudio.google.com/apikey) | Optional, only needed for transcription |
| LaiCut alignment | [LattifAI API](https://lattifai.com/dashboard/api-keys) | Optional, only needed for precise alignment |

> Gemini is only used for video transcription. When a video has no captions, you'll be prompted whether to transcribe -- configure then. Translation uses Claude by default, works out of the box.

API keys are prompted automatically and saved to `~/.config/omnicaptions/config.json`

Run `omnicaptions doctor` to verify your environment (ffmpeg, yt-dlp, API keys).

## OmniCaptions CLI Usage

```bash
# With captions: download -> align -> translate
omnicaptions download "https://youtube.com/watch?v=xxx"
omnicaptions LaiCut video.mp4 video.en.vtt -o video_LaiCut.srt
omnicaptions translate video_LaiCut.srt -l zh --bilingual

# Without captions: transcribe -> align -> translate
omnicaptions transcribe video.mp4
omnicaptions LaiCut video.mp4 video_GeminiUnd.md -o video_LaiCut.srt
omnicaptions translate video_LaiCut.srt -l zh --bilingual

# Environment check
omnicaptions doctor
```

### JSON Output for Agents

All commands support `--json` for structured output, returning a `ToolResult` object:

```bash
omnicaptions transcribe video.mp4 --json
```

```json
{
  "status": "success",
  "error_code": null,
  "message": "Transcribed to video_GeminiUnd.md",
  "data": {"output_path": "video_GeminiUnd.md"}
}
```

On failure, `error_code` provides a machine-readable category (e.g., `MISSING_API_KEY`, `NETWORK_TIMEOUT`).

## MCP Server

Use omnicaptions as an [MCP](https://modelcontextprotocol.io/) server with Claude Desktop, Cursor, or any MCP client.

**Install:**
```bash
pip install "omni-captions-skills[mcp]"
```

**Claude Desktop** — add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "omnicaptions": {
      "command": "omnicaptions-mcp"
    }
  }
}
```

**Available tools:** `doctor`, `download`, `transcribe`, `check_captions`, `translate`, `convert`, `forced_align`

SSE transport is also supported: `omnicaptions-mcp --transport sse --port 8080`

## Architecture

The codebase uses a services layer (`src/omnicaptions/services/`) where each operation (download, transcribe, translate, convert, align, doctor) is an independent module returning structured `ToolResult` objects. The CLI and skills are thin wrappers around these services.

---
Credits: [@dotey](https://x.com/dotey) for the [transcription prompt](https://x.com/dotey/status/1971810075867046131) | Built on [lattifai-captions](https://github.com/lattifai/captions)

[MIT License](LICENSE)
