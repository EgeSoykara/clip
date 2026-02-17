#!/usr/bin/env python3
"""
extract_interesting.py

Usage: python extract_interesting.py -i input.mp4 --keywords keyword1,keyword2

What it does:
- Uses ffmpeg to extract audio from the input video
- Calls OpenAI Whisper (via OpenAI Python client) to transcribe with timestamps
- Detects "interesting" moments from: fast speech rate and keyword hits
- Uses ffmpeg to create short clip files for each interesting moment

Requirements: ffmpeg installed and available on PATH, Python packages from requirements.txt
Set `OPENAI_API_KEY` in your environment before running.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import time
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import List, Tuple

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# If a .env file exists in the working directory, load simple KEY=VALUE lines
def _load_dotenv_simple(path: str = ".env"):
    try:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                if "=" not in ln:
                    continue
                k, v = ln.split("=", 1)
                k = k.strip()
                v = v.strip().strip("\"\'")
                if not k:
                    continue
                # Always prefer .env for OPENAI_API_KEY to avoid stale shell values.
                if k == "OPENAI_API_KEY":
                    os.environ[k] = v
                elif k not in os.environ:
                    os.environ[k] = v
    except Exception:
        # Best-effort loader; silently ignore errors so it doesn't break execution
        return


# Attempt to load environment variables from .env so users don't have to export them manually
_load_dotenv_simple()


def check_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("ffmpeg/ffprobe not found on PATH. Install ffmpeg and re-run.")
        sys.exit(1)


def run(cmd: List[str], **kwargs):
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def extract_audio(input_path: str, out_audio: str):
    # Convert to mono 16k WAV for reliable transcription
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        out_audio,
    ]
    run(cmd)


def transcribe_with_openai(audio_path: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    if OpenAI is None:
        raise RuntimeError("openai package not available. See requirements.txt")

    client = OpenAI(api_key=api_key)

    # Attempt a transcription call that returns timestamps/segments
    max_retries = int(os.getenv("TRANSCRIBE_RETRIES", "3"))
    base_backoff = float(os.getenv("TRANSCRIBE_BACKOFF", "1.0"))
    last_exc = None
    for attempt in range(1, max_retries + 1):
        with open(audio_path, "rb") as fh:
            try:
                resp = client.audio.transcriptions.create(model="whisper-1", file=fh)
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    backoff = base_backoff * (2 ** (attempt - 1))
                    print(f"Transcription attempt {attempt} failed: {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    continue
                else:
                    # no more retries
                    raise RuntimeError(f"transcription request failed after {max_retries} attempts: {e}")

    # Resp may be an object; convert to dict if necessary
    # Attempt to convert response object to JSON-serializable dict
    try:
        if hasattr(resp, "__dict__"):
            resp_dict = resp.__dict__
        else:
            resp_dict = resp
        # Optionally print a truncated debug view when explicitly enabled
        try:
            dbg = json.dumps(resp_dict)
        except Exception:
            dbg = str(resp_dict)
        if os.getenv("DEBUG_TRANSCRIPTION", "0") == "1":
            print("DEBUG_TRANSCRIPTION_RESP:", dbg[:2000])
        # If json-serializable, use parsed dict
        try:
            resp = json.loads(json.dumps(resp_dict))
        except Exception:
            resp = resp_dict
    except Exception:
        # leave resp as-is
        pass
    return resp


def parse_segments(resp: dict, total_duration: float) -> List[dict]:
    # Try common keys for segments with timestamps
    if isinstance(resp, dict) and "segments" in resp and isinstance(resp["segments"], list):
        segments = []
        for s in resp["segments"]:
            text = s.get("text") or s.get("caption") or s.get("sentence") or ""
            start = float(s.get("start", 0.0))
            end = float(s.get("end", start + max(0.5, len(text.split()) / 2.0)))
            segments.append({"text": text.strip(), "start": start, "end": end})
        return segments

    # Fallback: if we have a full `text` only, return one segment for whole audio
    if isinstance(resp, dict) and "text" in resp:
        text = resp.get("text", "").strip()
        if not text:
            return []

        # Split into sentences heuristically; distribute timestamps proportionally by word count
        sentences = [s.strip() for s in re.split(r'(?<=[\.!?])\s+', text) if s.strip()]
        words_per_sent = [len([w for w in s.split() if w.strip()]) for s in sentences]
        total_words = sum(words_per_sent) or 1
        segments = []
        cur = 0.0
        for count, sent in zip(words_per_sent, sentences):
            frac = count / total_words
            dur = max(0.05, total_duration * frac)
            start = cur
            end = min(total_duration, cur + dur)
            segments.append({"text": sent, "start": float(start), "end": float(end)})
            cur = end
        # If rounding left a gap at the end, extend last segment to total_duration
        if segments:
            segments[-1]["end"] = float(total_duration)
        return segments

    raise RuntimeError("Could not parse transcription segments from API response")


def get_audio_duration(path: str) -> float:
    # ffprobe to get duration
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd, encoding="utf-8").strip()
    try:
        return float(out)
    except Exception:
        return 0.0


def transcribe_in_chunks(audio_path: str, chunk_length: float) -> List[dict]:
    """Split `audio_path` into chunks of `chunk_length` seconds and transcribe each chunk.
    Returns a merged list of segments with timestamps adjusted to the original audio timeline.
    """
    with TemporaryDirectory() as td2:
        pattern = os.path.join(td2, "chunk_%04d.wav")
        # Use ffmpeg segmenter to split audio into consecutive chunks
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            audio_path,
            "-f",
            "segment",
            "-segment_time",
            f"{int(chunk_length)}",
            "-c",
            "copy",
            pattern,
        ]
        try:
            run(cmd)
        except Exception as e:
            raise RuntimeError(f"Failed to split audio into chunks: {e}")

        all_segments: List[dict] = []
        offset = 0.0
        # Process chunk files in order
        for fn in sorted(os.listdir(td2)):
            if not fn.startswith("chunk_"):
                continue
            chunk_path = os.path.join(td2, fn)
            dur = get_audio_duration(chunk_path)
            if dur <= 0.0:
                continue
            try:
                resp = transcribe_with_openai(chunk_path)
            except Exception as e:
                print(f"Warning: transcription failed for chunk {fn}: {e} — skipping")
                offset += dur
                continue
            try:
                segs = parse_segments(resp, dur)
            except Exception as e:
                print(f"Warning: failed to parse segments for chunk {fn}: {e}")
                offset += dur
                continue
            for s in segs:
                all_segments.append({
                    "text": s.get("text", ""),
                    "start": float(s.get("start", 0.0)) + offset,
                    "end": float(s.get("end", 0.0)) + offset,
                })
            offset += dur
        return all_segments


def download_youtube(url: str, outdir: str) -> str:
    """Download a YouTube URL using yt-dlp (command-line). Returns path to downloaded file.
    Requires `yt-dlp` to be installed and available on PATH."""
    if shutil.which("yt-dlp") is None and shutil.which("youtube-dl") is None:
        raise RuntimeError("yt-dlp or youtube-dl not found on PATH. Install yt-dlp to enable YouTube downloads.")

    tmpdir = outdir
    # construct output template
    out_template = os.path.join(tmpdir, "downloaded_video.%(ext)s")
    cmd = [shutil.which("yt-dlp") or shutil.which("youtube-dl"), "-f", "bestvideo+bestaudio/best", "-o", out_template, url]
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)

    # find the downloaded file in tmpdir
    for fn in os.listdir(tmpdir):
        if fn.startswith("downloaded_video."):
            return os.path.join(tmpdir, fn)
    raise RuntimeError("Failed to locate downloaded video file after yt-dlp run")


def extract_youtube_id(url: str) -> str:
    # Try to extract common YouTube id patterns
    try:
        m = re.search(r"(?:v=|/v/|youtu\.be/|/shorts/)([A-Za-z0-9_-]{6,})", url)
        if m:
            return m.group(1)
    except Exception:
        pass
    return ""


def detect_interesting(segments: List[dict], keywords: List[str], wps_threshold: float = 3.0) -> List[Tuple[float, float, str]]:
    matches = []
    kw_regex = None
    if keywords:
        kw_regex = re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b", flags=re.I)

    for seg in segments:
        text = seg.get("text", "")
        start = seg.get("start", 0.0)
        end = seg.get("end", start + 0.01)
        words = len([w for w in text.split() if w.strip()])
        duration = max(0.001, end - start)
        wps = words / duration

        reasons = []
        if kw_regex and kw_regex.search(text):
            reasons.append("keyword")
        if wps >= wps_threshold and words >= 3:
            reasons.append("fast_speech")

        if reasons:
            matches.append((start, end, ",".join(reasons)))

    # Merge overlapping/nearby matches
    if not matches:
        return []

    matches.sort(key=lambda x: x[0])
    merged = []
    cur_s, cur_e, cur_r = matches[0]
    for s, e, r in matches[1:]:
        if s <= cur_e + 0.5:  # merge if within 0.5s
            cur_e = max(cur_e, e)
            cur_r = cur_r + ";" + r
        else:
            merged.append((cur_s, cur_e, cur_r))
            cur_s, cur_e, cur_r = s, e, r
    merged.append((cur_s, cur_e, cur_r))

    # If merged matches are extremely long, split into chunks to avoid full-video clips.
    max_chunk = 60.0  # default max clip length in seconds
    final = []
    for s, e, r in merged:
        length = e - s
        if length <= max_chunk:
            final.append((s, e, r))
        else:
            # split into consecutive chunks of max_chunk length
            cur = s
            while cur < e - 0.01:
                nxt = min(e, cur + max_chunk)
                final.append((cur, nxt, r))
                cur = nxt
    return final


def ffmpeg_extract_clip(
    input_path: str,
    start: float,
    end: float,
    out_path: str,
    reencode: bool = False,
    shorts: bool = False,
    shorts_width: int = 1080,
    shorts_height: int = 1920,
):
    # Add padding to be safe
    start = max(0.0, start)
    if shorts:
        # 9:16 vertical output for YouTube Shorts.
        vf = (
            f"scale={shorts_width}:{shorts_height}:force_original_aspect_ratio=increase,"
            f"crop={shorts_width}:{shorts_height}"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            input_path,
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            out_path,
        ]
    elif reencode:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            input_path,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            out_path,
        ]
    else:
        # Try copy (fast) — may be slightly imprecise depending on container
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            input_path,
            "-c",
            "copy",
            out_path,
        ]
    run(cmd)


def sanitize_filename(name: str, max_len: int = 200) -> str:
    # Remove path separators and replace unsafe characters with underscores
    # Allow letters, numbers, dot, dash and underscore
    name = name.replace(os.path.sep, "_")
    # Replace any character not alnum or .-_ with _
    name = re.sub(r"[^A-Za-z0-9.\-_]", "_", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    # Trim to max length
    if len(name) > max_len:
        name = name[:max_len]
    # Strip leading/trailing dots or underscores
    name = name.strip("._ ")
    if not name:
        return "clip"
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input video file")
    parser.add_argument("-k", "--keywords", default="", help="Comma-separated keywords to prioritize")
    parser.add_argument("-o", "--outdir", default="clips", help="Output directory for clips")
    parser.add_argument("--wps-threshold", type=float, default=2.0, help="Words-per-second threshold for fast speech")
    parser.add_argument("--padding", type=float, default=0.5, help="Seconds of padding to add around detected segments")
    parser.add_argument("--reencode", action="store_true", help="Re-encode clips for frame-accurate cutting (slower)")
    parser.add_argument("--max-clip-length", type=float, default=60.0, help="Maximum clip length in seconds; long segments will be split")
    parser.add_argument("--skip-download", action="store_true", help="If set and input is a YouTube URL, reuse an existing downloaded file instead of calling yt-dlp")
    parser.add_argument("--keep-download", action="store_true", help="If set when downloading a YouTube URL, save a persistent copy under ./downloads/")
    parser.add_argument("--chunk-length", type=float, default=240.0, help="When fallback is used, split audio into this many seconds per chunk for transcription")
    parser.add_argument("--shorts", action="store_true", help="Render clips as vertical 9:16 YouTube Shorts")
    parser.add_argument("--shorts-width", type=int, default=1080, help="Shorts output width in pixels")
    parser.add_argument("--shorts-height", type=int, default=1920, help="Shorts output height in pixels")
    args = parser.parse_args()

    check_ffmpeg()
    os.makedirs(args.outdir, exist_ok=True)

    with TemporaryDirectory() as td:
        input_path = args.input
        downloaded = False
        # If input is a URL (YouTube), attempt to download into the temp dir
        if isinstance(input_path, str) and input_path.startswith("http"):
            # If user asked to skip download, try to find an existing downloaded file
            if args.skip_download:
                vid = extract_youtube_id(input_path)
                candidates = []
                search_dirs = [args.outdir, os.getcwd()]
                for d in search_dirs:
                    try:
                        for fn in os.listdir(d):
                            if vid and vid in fn:
                                candidates.append(os.path.join(d, fn))
                            if fn.startswith("downloaded_video."):
                                candidates.append(os.path.join(d, fn))
                    except Exception:
                        continue
                if candidates:
                    input_path = candidates[0]
                    print("Reusing existing downloaded file:", input_path)
                    downloaded = True
                else:
                    # Fall back to downloading if no existing file found
                    print("--skip-download set but no existing download found; falling back to downloading...")
                    try:
                        dl = download_youtube(input_path, td)
                        input_path = dl
                        downloaded = True
                        # Optionally persist the downloaded file for reuse
                        if args.keep_download:
                            downloads_dir = os.path.join(os.getcwd(), "downloads")
                            os.makedirs(downloads_dir, exist_ok=True)
                            vid = extract_youtube_id(args.input)
                            if vid:
                                dest_name = f"downloaded_{vid}.mp4"
                            else:
                                dest_name = f"downloaded_{int(time.time())}.mp4"
                            dest_path = os.path.join(downloads_dir, dest_name)
                            if not os.path.exists(dest_path):
                                try:
                                    shutil.copy2(input_path, dest_path)
                                    print("Saved downloaded source to:", dest_path)
                                except Exception as e:
                                    print("Warning: failed to save persistent download:", e)
                    except Exception as e:
                        print("Failed to download URL:", e)
                        sys.exit(1)
            else:
                try:
                    print("Detected URL input; downloading video to temporary directory...")
                    dl = download_youtube(input_path, td)
                    input_path = dl
                    downloaded = True
                    # Persist the download if requested
                    if args.keep_download:
                        downloads_dir = os.path.join(os.getcwd(), "downloads")
                        os.makedirs(downloads_dir, exist_ok=True)
                        vid = extract_youtube_id(args.input)
                        if vid:
                            dest_name = f"downloaded_{vid}.mp4"
                        else:
                            dest_name = f"downloaded_{int(time.time())}.mp4"
                        dest_path = os.path.join(downloads_dir, dest_name)
                        if not os.path.exists(dest_path):
                            try:
                                shutil.copy2(input_path, dest_path)
                                print("Saved downloaded source to:", dest_path)
                            except Exception as e:
                                print("Warning: failed to save persistent download:", e)
                except Exception as e:
                    print("Failed to download URL:", e)
                    sys.exit(1)

        audio_path = os.path.join(td, "audio.wav")
        print("Extracting audio to", audio_path)
        extract_audio(input_path, audio_path)

        duration = get_audio_duration(audio_path)
        print("Audio duration:", duration)

        print("Sending audio to OpenAI Whisper for transcription...")
        try:
            resp = transcribe_with_openai(audio_path)
            segments = parse_segments(resp, duration)
        except Exception as e:
            print("Direct transcription failed:", e)
            print("Falling back to chunked transcription (splitting audio)...")
            segments = transcribe_in_chunks(audio_path, chunk_length=float(args.chunk_length))
            print(f"Got {len(segments)} segments from chunked transcription")
        if 'segments' not in locals():
            # already have segments from fallback
            pass
        else:
            print(f"Got {len(segments)} transcription segments")

        keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
        matches = detect_interesting(segments, keywords, wps_threshold=args.wps_threshold)
        # If user provided --max-clip-length, further split long matches
        if args.max_clip_length and args.max_clip_length > 0:
            max_len = float(args.max_clip_length)
            split_matches = []
            for s, e, reason in matches:
                ln = e - s
                if ln <= max_len:
                    split_matches.append((s, e, reason))
                else:
                    cur = s
                    while cur < e - 0.01:
                        nxt = min(e, cur + max_len)
                        split_matches.append((cur, nxt, reason))
                        cur = nxt
            matches = split_matches
        print(f"Detected {len(matches)} interesting segments")

        clips = []
        for idx, (s, e, reason) in enumerate(matches, start=1):
            s2 = max(0.0, s - args.padding)
            e2 = e + args.padding
            base_name = f"clip_{idx:02d}_{int(s)}-{int(e)}_{reason.replace(',', '_')}"
            safe_name = sanitize_filename(base_name, max_len=180) + ".mp4"
            out_name = os.path.join(args.outdir, safe_name)
            print(f"Creating clip {idx}: {s2:.2f}s -> {e2:.2f}s ({reason}) -> {out_name}")
            ffmpeg_extract_clip(
                input_path,
                s2,
                e2,
                out_name,
                reencode=args.reencode,
                shorts=args.shorts,
                shorts_width=args.shorts_width,
                shorts_height=args.shorts_height,
            )
            clips.append(out_name)

        meta = {"input": args.input, "clips": clips}
        with open(os.path.join(args.outdir, "clips.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        print("Done. Clips written to:", args.outdir)


if __name__ == "__main__":
    main()
