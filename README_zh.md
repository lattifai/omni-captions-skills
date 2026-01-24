# omni-captions-skills

**视频字幕 So Easy** — Claude Code Caption Skills

[English](README.md)

> 我需要 Fireship 这个 vibe coding 视频的中英双语字幕 https://youtube.com/watch?v=Tw18-4U7mts 把视频 1080p 下载下来
>
> 就这一句，Claude 帮你搞定下载、转录、翻译。

## 安装

```bash
npx skills add https://github.com/lattifai/omni-captions-skills
```

<details>
<summary>其他安装方式</summary>

**Claude Code 插件系统：**
```bash
/plugin marketplace add lattifai/omni-captions-skills
/plugin install omnicaptions@lattifai-omni-captions-skills
```

**本地开发：**
```bash
git clone https://github.com/lattifai/omni-captions-skills.git
claude --plugin-dir ./omni-captions-skills
```
</details>

## 试试看

```
❯ 帮我把 Fireship 这个 vibe coding 视频做成中英双语字幕 https://youtube.com/watch?v=Tw18-4U7mts
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

## 技能一览

| 技能 | 功能 |
|------|------|
| `transcribe` | YouTube/视频 → 带时间戳的 Markdown |
| `translate` | 翻译字幕，支持双语输出 |
| `convert` | 30+ 字幕格式互转 |
| `download` | 下载 YouTube 视频/音频/字幕 |
| `LaiCut` | **强制对齐，词级精度时间轴** |

> 调用方式：`/omnicaptions:transcribe` 或 `/omnicaptions-transcribe`

## LaiCut：精准时间轴

普通转录的时间戳只是"大概"，LaiCut 使用 [LattifAI](https://lattifai.com/) Lattice-1 模型将文本与音频波形精确匹配，实现**词级精度**。

**支持语言：** 英语、中文、德语及混合

**推荐工作流：** 先对齐再翻译（翻译文本与原始音频不匹配，无法对齐）

## 配置

| 功能 | API Key | 说明 |
|------|---------|------|
| 翻译 | 无需配置 | **默认使用 Claude**，开箱即用 |
| 转录 | [Gemini API](https://aistudio.google.com/apikey) | 可选，仅转录时需要 |
| LaiCut 对齐 | [LattifAI API](https://lattifai.com/dashboard/api-keys) | 可选，仅精准对齐时需要 |

> Gemini 仅用于视频转录。当视频没有字幕时会提示是否需要转录，届时再配置即可。翻译默认走 Claude，开箱即用。

首次使用时自动提示输入，保存至 `~/.config/omnicaptions/config.json`

## OmniCaptions 命令行使用示例

```bash
# 有字幕：下载 → 对齐 → 翻译
omnicaptions download "https://youtube.com/watch?v=xxx"
omnicaptions LaiCut video.mp4 video.en.vtt -o video_LaiCut.srt
omnicaptions translate video_LaiCut.srt -l zh --bilingual

# 无字幕：转录 → 对齐 → 翻译
omnicaptions transcribe video.mp4
omnicaptions LaiCut video.mp4 video_GeminiUnd.md -o video_LaiCut.srt
omnicaptions translate video_LaiCut.srt -l zh --bilingual
```

## Credits
* Gemini 转录提示词源自 [@dotey](https://x.com/dotey) 的[推文](https://x.com/dotey/status/1971810075867046131)，[略有修改](https://github.com/lattifai/omni-captions-skills/commit/3f85975)。
* 基于 [lattifai-captions](https://github.com/lattifai/captions) 构建
* MIT License
