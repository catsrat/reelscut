---
title: Reelscut
emoji: 🎬
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Clip Reels

Paste a long video link → get short, vertical, captioned reels ready to post.

## Pipeline
1. **Download** the video (`yt-dlp`)
2. **Transcribe** speech with word timing (`whisper.cpp`, local — no API needed)
3. **Pick the best moments** — Claude reads the transcript and chooses the most
   engaging clips and writes a title for each. *(Falls back to an automatic
   selector if no AI key is set, so the app works without a key.)*
4. **Render** each clip: cut → crop to vertical 9:16 → burn captions (`ffmpeg`)

## Run
```bash
cd ~/clip-reels
export ANTHROPIC_API_KEY=sk-ant-...   # optional but recommended
./run.sh
```
Then open http://localhost:5050

## Get an AI key (for the "magic" mode)
1. Go to https://console.anthropic.com
2. Create an account, add a little credit (a few dollars covers many videos)
3. Create an API key, then `export ANTHROPIC_API_KEY=sk-ant-...` before `./run.sh`

Cost is small — a few cents of text per video (only the transcript goes to the AI;
the video never leaves your computer).

## Requirements (already installed on this machine)
- `ffmpeg`, `yt-dlp`, `whisper-cli` (via Homebrew)
- Whisper model at `models/ggml-base.en.bin`
- Python deps in `venv/` (`flask`, `anthropic`, `pillow`)

## ⚠️ Important note
Downloading other people's YouTube videos is against YouTube's Terms of Service.
This is fine for testing on your own videos, but a public product that downloads
arbitrary videos carries legal/ToS risk. The usual fix for a real product is to
have users **upload** their own files instead.

## Tuning
- Clip length window: `MIN_CLIP` / `MAX_CLIP` in `pipeline.py`
- Caption look: `FONT_SIZE`, `CAPTION_BOTTOM_MARGIN`, `_render_caption_png` in `pipeline.py`
- Better transcription: download a larger model (e.g. `ggml-small.en.bin`) into
  `models/` and update `MODEL_PATH`
