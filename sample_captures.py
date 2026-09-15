"""Create a deterministic sample of extracted images."""

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def sample_images(input_dir, output_dir, count, seed=42, replace=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if output_dir.exists():
        if not replace:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    files = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    selected = random.Random(seed).sample(files, min(count, len(files)))
    for source in selected:
        shutil.copy2(source, output_dir / source.name)
    return selected


def build_parser():
    parser = argparse.ArgumentParser(description="Sample extracted images for dataset review.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replace", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.count < 0:
        raise SystemExit("--count must be non-negative")
    selected = sample_images(args.input_dir, args.output_dir, args.count, args.seed, args.replace)
    print(f"Copied {len(selected)} image(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
