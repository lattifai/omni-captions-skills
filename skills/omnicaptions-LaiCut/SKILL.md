---
name: omnicaptions-LaiCut
description: Use when user needs accurate/precise subtitle timing, or aligning subtitles with audio/video using forced alignment. Corrects subtitle timing to match actual speech. Uses LattifAI Lattice-1 model.
allowed-tools: Bash(omnicaptions:*)
---

# LaiCut

LattifAI's audio-text processing toolkit. Currently supports forced alignment, with translate and speaker diarization coming soon.

**Requires LattifAI API Key** - Get from https://lattifai.com/dashboard/api-keys

## When to Use

- **Accurate/precise timing needed** - When user requests accurate timestamps or precise alignment
- **Sync misaligned subtitles** - Fix timing drift in downloaded captions
- **Align manual transcripts** - Match text to speech precisely
- **Post-transcription alignment** - Improve timing from auto-generated subtitles
- **Multi-format support** - SRT, VTT, ASS, LRC, TXT, MD

## When NOT to Use

- Need full transcription (use `/omnicaptions:transcribe`)
- No existing subtitle/transcript (nothing to align)
- Very short clips (<5 seconds)

## Setup

```bash
pip install $PLUGIN_DIR/packages/lattifai_core-*.tar.gz \
  "$(ls $PLUGIN_DIR/packages/lattifai_captions-*.tar.gz)[splitting]" \
  $PLUGIN_DIR/packages/omnicaptions-*.tar.gz
pip install "$(ls $PLUGIN_DIR/packages/lattifai-*.tar.gz)[alignment]"
```

## API Key

Priority: `LATTIFAI_API_KEY` env → `.env` file → `~/.config/omnicaptions/config.json`

If not set, ask user: `Please enter your LattifAI API key (get from https://lattifai.com/dashboard/api-keys):`

Then run with `-k <key>`. Key will be saved to config file automatically.

## CLI Usage

```bash
# Basic alignment
omnicaptions LaiCut audio.mp3 subtitle.srt

# Specify output file
omnicaptions LaiCut video.mp4 transcript.md -o aligned.srt

# Smart sentence segmentation (for word-level captions like YouTube VTT)
omnicaptions LaiCut video.mp4 caption.vtt --split-sentence
```

| Option | Description |
|--------|-------------|
| `-o, --output` | Output file (default: `<caption>_LaiCut.<ext>`) |
| `-f, --format` | Output format (default: same as input) |
| `-k, --api-key` | LattifAI API key |
| `--split-sentence` | AI-powered semantic sentence segmentation |
| `-v, --verbose` | Show progress |

## Ask User: Enable Smart Sentence Segmentation?

When input captions are word-level or poorly segmented (e.g., YouTube VTT), ask user whether to enable `--split-sentence`:

```
Is the caption word-level or poorly segmented (e.g., YouTube VTT)?
- Yes → Add --split-sentence (AI re-segments into natural sentences)
- No → Keep original segmentation
```

**Use cases**: YouTube VTT, word-aligned captions, messy auto-generated subtitles

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

**Important**: Generate bilingual subtitles AFTER alignment, since LaiCut aligns text with original audio. No pre-processing needed for audio/video input.

```bash
# No caption: transcribe → align → translate
omnicaptions transcribe video.mp4
omnicaptions LaiCut video.mp4 video_GeminiUnd.md -o video_LaiCut.srt
omnicaptions translate video_LaiCut.srt -l zh --bilingual

# Has caption: download → align → translate
omnicaptions download "https://youtu.be/xxx"
omnicaptions LaiCut xxx.mp4 xxx.en.vtt -o xxx_LaiCut.srt
omnicaptions translate xxx_LaiCut.srt -l zh --bilingual
```
