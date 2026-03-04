---
name: omnicaptions-pipeline
description: End-to-end caption workflow. Use when user wants captions for a video, asks to "subtitle this", "transcribe and translate", or any multi-step caption task. Also use for downloading videos.
allowed-tools: Bash(omnicaptions:*)
---

# OmniCaptions Pipeline

One-shot caption workflow. Choose a template based on user intent. All CLI commands support `--json` for structured output.

## Step 0: Preflight Check

Always run first:
```bash
omnicaptions doctor --json
```

If any check has `status: "missing"`, show the `help` field and ask user to fix before proceeding.

## Templates

### Template A: Full Pipeline (URL -> translated captions)

**When:** User provides a URL and wants captions (optionally translated).

```bash
# 1. Download media + existing captions
omnicaptions download "URL" -q audio --json
# -> Check data.caption_path: if not null, skip step 2

# 2. Transcribe (only if no captions from step 1)
omnicaptions transcribe "URL" --json
# -> data.output_path is the transcript

# 3. Align (optional - if user wants precise timing or karaoke)
omnicaptions LaiCut <audio_path> <caption_path> --json
# -> data.output_path is the aligned caption

# 4. Translate
# Option A - Claude native (no API key needed, default):
#   Read the caption file, translate in conversation, write output
# Option B - Gemini CLI (batch processing, better for large files):
#   omnicaptions translate <caption> -l <lang> --bilingual --json

# 5. Convert to final format (if needed)
omnicaptions convert <caption> -o <output.ass> --style bilingual --json
```

### Template B: Transcribe Only

**When:** User has a local file or URL, just wants a transcript.

```bash
omnicaptions transcribe <file_or_url> --json
```

### Template C: Align + Translate

**When:** User already has captions but wants better timing and/or translation.

```bash
omnicaptions LaiCut <audio> <caption> --json
omnicaptions translate <aligned_caption> -l <lang> --json
omnicaptions convert <translated> -o <output.ass> --style bilingual --json
```

### Template D: Download Only

**When:** User wants to download media/captions from a URL.

```bash
omnicaptions download "URL" -q audio --json
# Quality options: audio (default), best, 1080p, 720p, 480p, 360p
```

## Translation Decision Tree

1. User explicitly asks for Claude -> use Claude native translation
2. User explicitly asks for Gemini -> use `omnicaptions translate` CLI
3. No GEMINI_API_KEY available -> use Claude native (read file, translate, write)
4. Large file (>100 segments) -> recommend Gemini CLI (batch + context-aware)
5. Default -> use Claude native (zero extra API key needed)

## Output Naming Convention

Chain operations in filename: `base.lang_op1_op2.ext`
```
video.en.vtt                         # downloaded caption
video.en_LaiCut.json                 # aligned (JSON for word-level)
video.en_LaiCut.srt                  # aligned (SRT)
video.en_LaiCut_Claude_zh.srt        # aligned + Claude translated
video.en_LaiCut_Gemini_zh.srt        # aligned + Gemini translated
video_GeminiUnd.md                   # transcribed
video_GeminiUnd_LaiCut.json          # transcribed + aligned
```

## Error Handling

All `--json` output follows this schema:
```json
{
  "status": "success|error",
  "error_code": "MISSING_API_KEY|INVALID_INPUT|...|null",
  "message": "human-readable description",
  "data": { ... }
}
```

Common error_codes and actions:
- `MISSING_API_KEY` -> show the help message, ask user to set key
- `MISSING_DEPENDENCY` -> show install command from doctor output
- `INVALID_INPUT` -> check file path
- `DOWNLOAD_FAILED` -> retry or try different URL
