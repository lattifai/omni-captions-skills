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

## ASS Style Presets

When converting to ASS format, use `--style` to apply preset styles:

```bash
omnicaptions convert input.srt -o output.ass --style default    # White text, bottom
omnicaptions convert input.srt -o output.ass --style top        # White text, top
omnicaptions convert input.srt -o output.ass --style bilingual  # White + Yellow (for bilingual)
omnicaptions convert input.srt -o output.ass --style yellow     # Yellow text, bottom
```

| Preset | Position | Line 1 | Line 2 | Use Case |
|--------|----------|--------|--------|----------|
| `default` | Bottom | White | White | Standard captions |
| `top` | Top | White | White | When bottom is occupied |
| `bilingual` | Bottom | White | Yellow | Bilingual captions (原文 + 译文) |
| `yellow` | Bottom | Yellow | Yellow | High visibility |

### Bilingual Example

If your SRT has two-line captions like:
```
1
00:00:01,000 --> 00:00:03,000
Hello World
你好世界
```

Use `--style bilingual` to make the second line yellow:
```bash
omnicaptions convert bilingual.srt -o output.ass --style bilingual
```

## Karaoke Mode

Generate karaoke subtitles with word-level highlighting. **Requires word-level timing** (use LaiCut alignment first).

```bash
# Basic karaoke (sweep effect - gradual fill)
omnicaptions convert aligned.json -o output.ass --karaoke

# Different effects
omnicaptions convert aligned.json -o output.ass --karaoke sweep    # Gradual fill (default)
omnicaptions convert aligned.json -o output.ass --karaoke instant  # Instant highlight
omnicaptions convert aligned.json -o output.ass --karaoke outline  # Outline then fill

# LRC karaoke (enhanced word timestamps)
omnicaptions convert aligned.json -o output.lrc --karaoke
```

| Effect | ASS Tag | Description |
|--------|---------|-------------|
| `sweep` | `\kf` | Gradual fill from left to right (default) |
| `instant` | `\k` | Instant word highlight |
| `outline` | `\ko` | Outline fills, then text fills |

### Karaoke Workflow

```bash
# 1. Align with LaiCut (get word-level timing in JSON)
omnicaptions LaiCut audio.mp3 lyrics.txt

# 2. Convert to karaoke ASS
omnicaptions convert lyrics_LaiCut.json -o karaoke.ass --karaoke

# Or combine with style
omnicaptions convert lyrics_LaiCut.json -o karaoke.ass --karaoke --style yellow
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
