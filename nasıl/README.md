# Whisper + FFmpeg — extract interesting moments

This repository provides a small script to transcribe a video with OpenAI Whisper and split out "interesting" moments (fast speech or keyword hits) into short clips using FFmpeg.

Prerequisites
- `ffmpeg` (and `ffprobe`) installed and on your PATH
- Python 3.8+
- `OPENAI_API_KEY` set in your environment

Install Python deps:

```bash
python -m pip install -r requirements.txt
```

Basic usage:

```bash
export OPENAI_API_KEY="sk-..."
python extract_interesting.py -i input.mp4 -k "keyword1,keyword2" -o out_clips
```

Options:
- `--wps-threshold`: words-per-second threshold for marking fast speech (default 3.0)
- `--padding`: seconds padding added around each detected moment (default 0.5)
- `--reencode`: re-encode clips for frame-accurate cutting (slower)

Notes
- The script uses the OpenAI Python client. The exact API surface may vary between client versions; if you hit an API error, ensure you have a recent `openai` package installed and check the client method signatures.
- Cutting with `-c copy` is fast but may be slightly imprecise; use `--reencode` for accurate frame cuts.
