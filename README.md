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
❯ Make bilingual captions for this Fireship vibe coding video https://youtube.com/watch?v=Tw18-4U7mts
```

```srt
1
00:00:00,000 --> 00:00:03,200
Mass hysteria satisfies a deep human need.
群体性癔症满足了人类某种深层需求。

2
00:00:03,200 --> 00:00:07,440
Vibe coding is programming without actually writing any code yourself.
Vibe coding 就是不用自己写代码的编程方式。
```

## Skills

| Skill | Description |
|-------|-------------|
| `transcribe` | YouTube/video → Markdown with timestamps |
| `translate` | Translate captions, bilingual output supported |
| `convert` | Convert between 30+ caption formats |
| `download` | Download YouTube video/audio/captions |
| `LaiCut` | **Forced alignment, word-level timing accuracy** |

> Invoke via `/omnicaptions:transcribe` or `/omnicaptions-transcribe`

## LaiCut: Precise Timing

Standard transcription gives "approximate" timestamps. LaiCut uses [LattifAI](https://lattifai.com/) Lattice-1 model to match text precisely to audio waveforms, achieving **word-level accuracy**.

**Supported languages:** English, Chinese, German, and mixed

**Recommended workflow:** Align before translate (translated text doesn't match original audio)

## Setup

| Feature | API Key | Note |
|---------|---------|------|
| Translation | None required | **Uses Claude by default**, works out of the box |
| Transcription | [Gemini API](https://aistudio.google.com/apikey) | Optional, only needed for transcription |
| LaiCut alignment | [LattifAI API](https://lattifai.com/dashboard/api-keys) | Optional, only needed for precise alignment |

> Gemini is only used for video transcription. When a video has no captions, you'll be prompted whether to transcribe — configure then. Translation uses Claude by default, works out of the box.

API keys are prompted automatically and saved to `~/.config/omnicaptions/config.json`

## OmniCaptions CLI Usage

```bash
# With captions: download → align → translate
omnicaptions download "https://youtube.com/watch?v=xxx"
omnicaptions LaiCut video.mp4 video.en.vtt -o video_LaiCut.srt
omnicaptions translate video_LaiCut.srt -l zh --bilingual

# Without captions: transcribe → align → translate
omnicaptions transcribe video.mp4
omnicaptions LaiCut video.mp4 video_GeminiUnd.md -o video_LaiCut.srt
omnicaptions translate video_LaiCut.srt -l zh --bilingual
```

---
Credits: [@dotey](https://x.com/dotey) for the [transcription prompt](https://x.com/dotey/status/1971810075867046131) | Built on [lattifai-captions](https://github.com/lattifai/captions)

[MIT License](LICENSE)
