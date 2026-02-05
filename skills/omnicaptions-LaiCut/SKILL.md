---
name: omnicaptions-LaiCut
description: Use when user needs accurate/precise caption timing, or aligning captions with audio/video using forced alignment. Corrects caption timing to match actual speech. Uses LattifAI Lattice-1 model.
allowed-tools: Read, Bash(omnicaptions:*), Bash(lai:*)
---

# LaiCut

LattifAI's audio-text processing toolkit. Currently supports forced alignment, with translate and speaker diarization coming soon.

**Requires LattifAI API Key** - Get from https://lattifai.com/dashboard/api-keys

## When to Use

- **Accurate/precise timing needed** - When user requests accurate timestamps or precise alignment
- **Sync misaligned captions** - Fix timing drift in downloaded captions
- **Align manual transcripts** - Match text to speech precisely
- **Post-transcription alignment** - Improve timing from auto-generated captions
- **Multi-format support** - SRT, VTT, ASS, LRC, TXT, MD

## When NOT to Use

- Need full transcription (use `/omnicaptions:transcribe`)
- No existing caption/transcript (nothing to align)
- Very short clips (<5 seconds)

## Setup

```bash
pip install "omni-captions-skills[laicut]" --extra-index-url https://lattifai.github.io/pypi/simple/
```

## API Key

Priority: `LATTIFAI_API_KEY` env → `.env` file → `~/.config/omnicaptions/config.json`

If not set, ask user: `Please enter your LattifAI API key (get from https://lattifai.com/dashboard/api-keys):`

Then run with `-k <key>`. Key will be saved to config file automatically.

## CLI Usage

```bash
# Basic alignment (default: JSON with word-level timing, RECOMMENDED)
omnicaptions LaiCut audio.mp3 caption.srt
# → caption_LaiCut.json

# Then convert to desired format (preserves word timing in JSON for future use)
omnicaptions convert caption_LaiCut.json -o caption_LaiCut.srt

# Smart sentence segmentation (for word-level captions like YouTube VTT)
omnicaptions LaiCut video.mp4 caption.vtt --split-sentence
```

| Option | Description |
|--------|-------------|
| `-o, --output` | Output file (default: `<caption>_LaiCut.json`) |
| `-f, --format` | Output format (default: json) |
| `-k, --api-key` | LattifAI API key |
| `--split-sentence` | AI-powered semantic sentence segmentation |
| `-v, --verbose` | Show progress |

## JSON Output (Recommended)

JSON output preserves **word-level timing** for downstream tasks:

```json
[
  {
    "text": "Hello world",
    "start": 0.0,
    "end": 2.5,
    "words": [
      {"word": "Hello", "start": 0.0, "end": 0.5},
      {"word": "world", "start": 0.6, "end": 2.5}
    ]
  }
]
```

Convert JSON to other formats:
```bash
# For playback/editing
omnicaptions convert video.en_LaiCut.json -o video.en_LaiCut.srt

# For bilingual ASS
omnicaptions convert video.en_LaiCut.json -o video.en_LaiCut.ass --style bilingual

# For karaoke
omnicaptions convert lyrics_LaiCut.json -o lyrics_LaiCut_karaoke.ass --karaoke
```

**For translation**: Convert to SRT first (JSON is too large for Claude to read):
```bash
# 1. JSON → SRT
omnicaptions convert video.en_LaiCut.json -o video.en_LaiCut.srt

# 2. Claude 翻译 → video.en_LaiCut_Claude_zh.srt

# 3. 转换为带颜色 ASS
omnicaptions convert video.en_LaiCut_Claude_zh.srt -o video.en_LaiCut_Claude_zh_Color.ass \
  --line1-color "#00FF00" --line2-color "#FFFF00"
```

## Ask User: Enable Smart Sentence Segmentation?

When input captions are word-level or poorly segmented (e.g., YouTube VTT), ask user whether to enable `--split-sentence`:

```
Is the caption word-level or poorly segmented (e.g., YouTube VTT)?
- Yes → Add --split-sentence (AI re-segments into natural sentences)
- No → Keep original segmentation
```

**Use cases**: YouTube VTT, word-aligned captions, messy auto-generated captions

## LattifAI API Key Error Handling

**LaiCut requires LattifAI API key** (NOT Gemini API key)

Error example:
```
API KEY verification error: API KEY is invalid or expired.
```

**Correct handling**:
1. Tell user that LaiCut requires a valid **LattifAI API key**
2. Ask user if they want to provide a new key
3. Direct user to https://lattifai.com/dashboard/api-keys
4. **NEVER** skip alignment step claiming "timing is already accurate enough"

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| API key invalid/expired | Ask user for new key, do NOT skip |
| No API key | Set `LATTIFAI_API_KEY` or use `-k` |
| Audio format error | Convert to WAV/MP3/M4A first |
| Empty output | Check caption has text content |

## Related Skills

| Skill | Use When |
|-------|----------|
| `/omnicaptions:transcribe` | Generate transcript first |
| `/omnicaptions:convert` | Convert caption formats |
| `/omnicaptions:translate` | Translate after alignment |

### Workflow Examples

**Important**:
- LaiCut outputs JSON by default (preserves word-level timing)
- Convert to SRT/ASS when needed for playback or translation
- Generate bilingual captions AFTER alignment

```bash
# No caption: transcribe → align (JSON) → convert → translate
omnicaptions transcribe video.mp4
# → video_GeminiUnd.md
omnicaptions LaiCut video.mp4 video_GeminiUnd.md
# → video_GeminiUnd_LaiCut.json
omnicaptions convert video_GeminiUnd_LaiCut.json -o video_GeminiUnd_LaiCut.srt
# → video_GeminiUnd_LaiCut_Claude_zh.srt (after translate)

# Has caption: download → align (JSON) → convert → translate
omnicaptions download "https://youtu.be/xxx"
# → xxx.en.vtt
omnicaptions LaiCut xxx.mp4 xxx.en.vtt
# → xxx.en_LaiCut.json
omnicaptions convert xxx.en_LaiCut.json -o xxx.en_LaiCut.srt
# → xxx.en_LaiCut_Claude_zh.srt (after translate)
# → xxx.en_LaiCut_Claude_zh_Color.ass (after convert with --style)
```
