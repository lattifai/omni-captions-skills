# omni-captions-skills

[中文文档](README_zh.md)

Claude Code skills for YouTube or Local media file transcription and Caption translation using AI APIs.

> **Credits**: Gemini transcription prompt inspired by [@dotey](https://x.com/dotey)'s [tweet](https://x.com/dotey/status/1971810075867046131), with [minor modifications](https://github.com/lattifai/omni-captions-skills/commit/3f85975058aaad3c83254d213dbd4136364a2073).

## Install

### Option 1: npx (Recommended)

```bash
npx skills add https://github.com/lattifai/omni-captions-skills
```

### Option 2: Claude Code Plugin System

Add the marketplace and install via Claude Code's built-in plugin system:

```bash
# In Claude Code, run:
/plugin marketplace add lattifai/omni-captions-skills
/plugin install omnicaptions@lattifai-omni-captions-skills
```

### Option 3: Local Development

For testing or development, load the plugin directly:

```bash
git clone https://github.com/lattifai/omni-captions-skills.git
claude --plugin-dir ./omni-captions-skills
```

## Skills

| Skill | Description |
|-------|-------------|
| `/omnicaptions:transcribe` | Transcribe YouTube/video to markdown with timestamps |
| `/omnicaptions:translate` | Translate captions (Gemini API / Claude) |
| `/omnicaptions:convert` | Convert between 30+ caption formats |
| `/omnicaptions:download` | Download videos/captions from YouTube |
| `/omnicaptions:LaiCut` | Forced alignment using LattifAI Lattice-1 model |

> **Note**: `/omnicaptions:transcribe` and `/omnicaptions-transcribe` are equivalent.

Powered by [lattifai-captions](https://github.com/lattifai/captions) - supports 30+ caption formats including messy YouTube VTT with word-level timestamps.

## Usage Examples

### Bilingual Caption from YouTube

```
❯ 制作这个视频的中英双语字幕 https://youtube.com/shorts/H8LwA-daqqA
```

Output:
```srt
1
00:00:00,000 --> 00:00:03,500
Does fast charging hurt the battery?
快充会伤害电池吗？
```

### Chinese Song → English Caption

```
❯ 把王菲的《如愿》翻译成英文字幕 https://youtube.com/watch?v=6fV2dRqJHvw
```

Output:
```srt
1
00:00:37,700 --> 00:00:45,000
你是，遥遥的路，山野大雾里的灯。
You are the distant road, a lamp within the mountain mist.
```

### Download Video for Editing

```
❯ 下载这个视频 https://youtube.com/watch?v=VIDEO_ID
```

### Convert Caption Formats

```
❯ 把这个 SRT 转成 VTT 格式

Claude: [Uses /omnicaptions:convert]
        omnicaptions convert input.srt -t vtt
```

## Accurate Timing with `LaiCut`

Caption timing from different sources often has issues:

| Source | Problem |
|--------|---------|
| YouTube auto-captions | Timing drift, word-level chunking |
| Gemini transcription | Approximate timestamps |
| Manual transcripts | No timing at all |

**LaiCut** uses forced alignment ([LattifAI](https://lattifai.com/) Lattice-1 model) to match text precisely to audio waveforms, achieving **word-level accuracy**.

**Supported languages**: English, Chinese, German, and any mix of these. More languages coming soon.

> **Coming soon**: More LaiCut features including translation, speaker diarization, and clip editing with precise timestamps.

### Why align before translate?

LaiCut aligns text to speech. Bilingual captions contain translations that don't match the audio, so:

1. **Align first** with original language text
2. **Translate after** to preserve accurate timing

## Recommended Workflow

For accurate bilingual captions, align timing before translation:

```bash
# With existing captions: download → align → translate
omnicaptions download "https://youtube.com/watch?v=xxx"
omnicaptions LaiCut xxx.mp4 xxx.en.vtt -o xxx_LaiCut.srt
omnicaptions translate xxx_LaiCut.srt -l zh --bilingual

# Without captions: transcribe → align → translate
omnicaptions transcribe video.mp4
omnicaptions LaiCut video.mp4 video_GeminiUnd.md -o video_LaiCut.srt
omnicaptions translate video_LaiCut.srt -l zh --bilingual
```

## Setup

**Gemini API** (transcribe/translate): https://aistudio.google.com/apikey

**LattifAI API** (LaiCut alignment): https://lattifai.com/dashboard/api-keys

API keys are prompted automatically and saved to `~/.config/omnicaptions/config.json`

## License

MIT
