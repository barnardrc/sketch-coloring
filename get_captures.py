"""Extract frames from authorized videos at a configurable interval."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def process_video(job):
    video_path, output_dir, interval_seconds, trim_start, trim_end = job
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return video_path.name, 0, "could not open video"

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            return video_path.name, 0, "invalid frame rate"

        start_frame = int(fps * trim_start)
        end_frame = total_frames - int(fps * trim_end)
        interval_frames = max(1, int(fps * interval_seconds))
        if end_frame <= start_frame:
            return video_path.name, 0, "video is shorter than configured trims"

        written = 0
        for frame_number in range(start_frame, end_frame, interval_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok:
                break
            output_path = output_dir / f"{video_path.stem}_{frame_number}.jpg"
            if cv2.imwrite(str(output_path), frame):
                written += 1
        return video_path.name, written, None
    finally:
        capture.release()


def extract_frames(input_dir, output_dir, interval_seconds=50, trim_start=30, trim_end=10, workers=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    jobs = [(path, output_dir, interval_seconds, trim_start, trim_end) for path in videos]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(process_video, jobs))


def build_parser():
    parser = argparse.ArgumentParser(description="Extract periodic frames from authorized videos.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--interval-seconds", type=float, default=50)
    parser.add_argument("--trim-start", type=float, default=30)
    parser.add_argument("--trim-end", type=float, default=10)
    parser.add_argument("--workers", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    if args.interval_seconds <= 0 or args.trim_start < 0 or args.trim_end < 0:
        raise SystemExit("Intervals must be positive and trims must be non-negative.")

    results = extract_frames(
        args.input_dir,
        args.output_dir,
        args.interval_seconds,
        args.trim_start,
        args.trim_end,
        args.workers,
    )
    for video, count, error in results:
        print(f"{video}: {error or f'{count} frame(s) written'}")


if __name__ == "__main__":
    main()
