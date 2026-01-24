---
name: omnicaptions-convert
description: Use when converting between caption formats (SRT, VTT, ASS, TTML, Gemini MD, etc.). Supports 30+ caption formats.
allowed-tools: Bash(omnicaptions:*)
---

# Caption Format Conversion

Convert between 30+ caption/caption formats using `lattifai-captions`.

## ⚡ YouTube Workflow

```bash
# 1. Transcribe YouTube video directly
omnicaptions transcribe "https://youtube.com/watch?v=VIDEO_ID" -o transcript.md

# 2. Convert to any format
omnicaptions convert transcript.md -o output.srt
omnicaptions convert transcript.md -o output.ass
omnicaptions convert transcript.md -o output.vtt
```

## When to Use

- Converting SRT to VTT, ASS, TTML, etc.
- Converting Gemini markdown transcript to standard caption formats
- Converting YouTube VTT (with word-level timestamps) to other formats
- Batch format conversion

## When NOT to Use

- Need transcription (use `/omnicaptions:transcribe`)
- Need translation (use `/omnicaptions:translate`)

## Setup

```bash
pip install $PLUGIN_DIR/packages/lattifai_core-*.tar.gz \
  "$(ls $PLUGIN_DIR/packages/lattifai_captions-*.tar.gz)[splitting]" \
  $PLUGIN_DIR/packages/omnicaptions-*.tar.gz
```

## Quick Reference

| Format | Extension | Read | Write |
|--------|-----------|------|-------|
| SRT | `.srt` | ✓ | ✓ |
| VTT | `.vtt` | ✓ | ✓ |
| ASS/SSA | `.ass` | ✓ | ✓ |
| TTML | `.ttml` | ✓ | ✓ |
| Gemini MD | `.md` | ✓ | ✓ |
| JSON | `.json` | ✓ | ✓ |
| TXT | `.txt` | ✓ | ✓ |

Full list: SRT, VTT, ASS, SSA, TTML, DFXP, SBV, SUB, LRC, JSON, TXT, TSV, Audacity, Audition, FCPXML, EDL, and more.

## CLI Usage

```bash
# Convert (auto-output to same directory, only changes extension)
omnicaptions convert input.srt -t vtt           # → ./input.vtt
omnicaptions convert transcript.md              # → ./transcript.srt

# Specify output file or directory
omnicaptions convert input.srt -o output/       # → output/input.srt
omnicaptions convert input.srt -o output.vtt    # → output.vtt

# Specify format explicitly
omnicaptions convert input.txt -o out.srt -f txt -t srt
```

## Python Usage

```python
from omnicaptions import Caption

# Load any format
cap = Caption.read("input.srt")

# Write to any format
cap.write("output.vtt")
cap.write("output.ass")
cap.write("output.ttml")
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Format not detected | Use `--from` / `--to` flags |
| Missing timestamps | Source format must have timing info |
| Encoding error | Specify `encoding="utf-8"` |

## Related Skills

| Skill | Use When |
|-------|----------|
| `/omnicaptions:transcribe` | Need transcript from audio/video |
| `/omnicaptions:translate` | Translate with Gemini API |
| `/omnicaptions:translate` | Translate with Claude (no API key) |
| `/omnicaptions:download` | Download video/captions first |

### Workflow Examples

```
# Transcribe → Convert → Translate (with Claude)
/omnicaptions:transcribe video.mp4
/omnicaptions:convert video_GeminiUnd.md -o video.srt
/omnicaptions:translate video.srt -l zh --bilingual
```
