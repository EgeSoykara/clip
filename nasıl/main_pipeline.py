#!/usr/bin/env python3
"""
main_pipeline.py

Runs the workflow in order:
1) clip extraction (extract_interesting.py)
2) delete short clips (delsrt.py)
3) upload clips to Google Drive (uploaddrive.py)
"""

import argparse
import subprocess
import sys


def run_step(cmd: list[str], label: str) -> None:
    print(f"\n=== {label} ===")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run clip -> delete -> upload pipeline")
    parser.add_argument("-i", "--input", required=True, help="Input video path or URL")
    parser.add_argument("-k", "--keywords", default="", help="Comma-separated keywords for clipping")
    parser.add_argument("-o", "--outdir", default="out_clips", help="Output folder for extracted clips")

    parser.add_argument("--wps-threshold", type=float, default=2.0)
    parser.add_argument("--padding", type=float, default=0.5)
    parser.add_argument("--reencode", action="store_true")
    parser.add_argument("--max-clip-length", type=float, default=60.0)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--keep-download", action="store_true")
    parser.add_argument("--chunk-length", type=float, default=240.0)
    parser.add_argument("--shorts", dest="shorts", action="store_true", default=True)
    parser.add_argument("--no-shorts", dest="shorts", action="store_false")
    parser.add_argument("--shorts-width", type=int, default=1080)
    parser.add_argument("--shorts-height", type=int, default=1920)

    args = parser.parse_args()

    if args.outdir != "out_clips":
        print("Error: outdir must be 'out_clips' because delsrt.py and uploaddrive.py are currently hardcoded to that folder.")
        return 2

    py = sys.executable

    clip_cmd = [
        py,
        "extract_interesting.py",
        "--input",
        args.input,
        "--keywords",
        args.keywords,
        "--outdir",
        args.outdir,
        "--wps-threshold",
        str(args.wps_threshold),
        "--padding",
        str(args.padding),
        "--max-clip-length",
        str(args.max_clip_length),
        "--chunk-length",
        str(args.chunk_length),
    ]

    if args.reencode:
        clip_cmd.append("--reencode")
    if args.skip_download:
        clip_cmd.append("--skip-download")
    if args.keep_download:
        clip_cmd.append("--keep-download")
    if args.shorts:
        clip_cmd.extend(
            [
                "--shorts",
                "--shorts-width",
                str(args.shorts_width),
                "--shorts-height",
                str(args.shorts_height),
            ]
        )

    try:
        run_step(clip_cmd, "Step 1/3: Extract clips")
        run_step([py, "delsrt.py"], "Step 2/3: Delete short clips")
        run_step([py, "uploaddrive.py"], "Step 3/3: Upload to Google Drive")
    except subprocess.CalledProcessError as e:
        print(f"\nPipeline failed at step with exit code: {e.returncode}")
        return e.returncode

    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
