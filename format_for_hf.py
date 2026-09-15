"""Convert class-folder image/caption pairs into a flat Diffusers dataset."""

import argparse
import json
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def convert_to_diffusers_format(source_root, output_dir):
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []

    for image_path in sorted(source_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        caption_path = image_path.with_suffix(".txt")
        if not caption_path.exists():
            print(f"Warning: no caption for {image_path}; skipping")
            continue

        class_name = image_path.parent.name
        destination_name = f"{class_name}_{image_path.name}"
        shutil.copy2(image_path, output_dir / destination_name)
        metadata.append(
            {
                "file_name": destination_name,
                "text": caption_path.read_text(encoding="utf-8").strip(),
            }
        )

    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as destination:
        for entry in metadata:
            destination.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Create a flat Diffusers image-caption dataset.")
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    metadata = convert_to_diffusers_format(args.source_root, args.output_dir)
    print(f"Converted {len(metadata)} image-caption pair(s) into {args.output_dir}")


if __name__ == "__main__":
    main()
