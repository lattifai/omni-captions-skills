# omni-captions-skills

Claude Code 技能插件，用于 YouTube 和本地音视频的转录与字幕翻译。

> **声明致谢**: Gemini 转录提示词来自 [@dotey](https://x.com/dotey) 的 [推文](https://x.com/dotey/status/1971810075867046131) 并做了[少许修改](https://github.com/lattifai/omni-captions-skills/commit/3f85975058aaad3c83254d213dbd4136364a2073)。

## 安装

### 方式 1: npx（推荐）

```bash
npx skills add https://github.com/lattifai/omni-captions-skills
```

### 方式 2: Claude Code 插件系统

通过 Claude Code 内置插件系统添加 marketplace 并安装：

```bash
# 在 Claude Code 中运行：
/plugin marketplace add lattifai/omni-captions-skills
/plugin install omnicaptions@lattifai-omni-captions-skills
```

### 方式 3: 本地开发

用于测试或开发，直接加载本地插件：

```bash
git clone https://github.com/lattifai/omni-captions-skills.git
claude --plugin-dir ./omni-captions-skills
```

## 技能列表

| 技能 | 功能 |
|------|------|
| `/omnicaptions:transcribe` | 将 YouTube/视频转录为带时间戳的 Markdown |
| `/omnicaptions:translate` | 翻译字幕（Gemini API / Claude） |
| `/omnicaptions:convert` | 支持 30+ 字幕格式互转 |
| `/omnicaptions:download` | 从 YouTube 下载视频/字幕 |
| `/omnicaptions:LaiCut` | 使用 LattifAI Lattice-1 模型进行强制对齐 |

> **提示**: `/omnicaptions:transcribe` 和 `/omnicaptions-transcribe` 是等效的。

基于 [lattifai-captions](https://github.com/lattifai/captions) 构建，支持 30+ 字幕格式，包括混乱的 YouTube VTT（词级时间戳）。

## 使用示例

### YouTube 视频生成中英双语字幕

```
❯ 制作这个视频的中英双语字幕 https://youtube.com/shorts/H8LwA-daqqA
```

输出:
```srt
1
00:00:00,000 --> 00:00:03,500
Does fast charging hurt the battery?
快充会伤害电池吗？
```

### 中文歌曲翻译成英文字幕

```
❯ 把王菲的《如愿》翻译成英文字幕 https://youtube.com/watch?v=6fV2dRqJHvw
```

输出:
```srt
1
00:00:37,700 --> 00:00:45,000
你是，遥遥的路，山野大雾里的灯。
You are the distant road, a lamp within the mountain mist.
```

### 下载视频用于编辑

```
❯ 下载这个视频 https://youtube.com/watch?v=VIDEO_ID
```

### 字幕格式转换

```
❯ 把这个 SRT 转成 VTT 格式

Claude: [使用 /omnicaptions:convert]
        omnicaptions convert input.srt -t vtt
```

## 使用 `LaiCut` 获取精准时间轴

不同来源的字幕时间轴往往存在问题：

| 来源 | 问题 |
|------|------|
| YouTube 自动字幕 | 时间偏移、按词分段 |
| Gemini 转录 | 时间戳不够精确 |
| 手动转录 | 完全没有时间信息 |

**LaiCut** 使用强制对齐技术（[LattifAI](https://lattifai.com/) Lattice-1 模型），将文本精确匹配到音频波形，实现**词级精度**。

**支持语言**: 英语、中文、德语，以及这些语言的混合。更多语言即将支持。

> **即将推出**: 更多 LaiCut 功能，包括翻译、说话人分离、基于精准时间轴的片段剪辑。

### 为什么要先对齐再翻译？

LaiCut 将文本与语音对齐。双语字幕包含与音频不匹配的翻译，因此：

1. **先对齐** - 使用原始语言文本
2. **后翻译** - 保留精准的时间轴

## 推荐工作流

制作精准双语字幕，建议先对齐时间轴再翻译：

```bash
# 有字幕：下载 → 对齐 → 翻译
omnicaptions download "https://youtube.com/watch?v=xxx"
omnicaptions LaiCut xxx.mp4 xxx.en.vtt -o xxx_LaiCut.srt
omnicaptions translate xxx_LaiCut.srt -l zh --bilingual

# 无字幕：转录 → 对齐 → 翻译
omnicaptions transcribe video.mp4
omnicaptions LaiCut video.mp4 video_GeminiUnd.md -o video_LaiCut.srt
omnicaptions translate video_LaiCut.srt -l zh --bilingual
```

## 配置

**Gemini API**（转录/翻译）: https://aistudio.google.com/apikey

**LattifAI API**（LaiCut 对齐）: https://lattifai.com/dashboard/api-keys

API Key 会自动提示输入，并保存到 `~/.config/omnicaptions/config.json`

## 许可证

MIT
