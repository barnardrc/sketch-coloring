"""Classify authorized images, generate captions, and write training pairs."""

import argparse
import json
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_BANNED_TERMS = ("watermark", "logo", "subtitles", "credits", "blurry")


def load_concepts(path):
    concepts = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(concepts, dict) or len(concepts) < 2:
        raise ValueError("Concept configuration must contain at least two named concepts")
    for name, config in concepts.items():
        if not isinstance(config, dict) or not config.get("description") or not config.get("trigger"):
            raise ValueError(f"Concept {name!r} requires description and trigger values")
    return concepts


def clean_caption(caption):
    prefixes = ("an image of ", "a photo of ", "a picture of ", "a close up of ", "screenshot of ")
    cleaned = caption.strip()
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    return cleaned


def generate_best_caption(
    image,
    blip_processor,
    blip_model,
    clip_processor,
    clip_model,
    device,
):
    import torch

    blip_inputs = blip_processor(image, text="a photo of ", return_tensors="pt").to(device)
    with torch.no_grad():
        sequences = blip_model.generate(
            **blip_inputs,
            max_new_tokens=75,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=5,
        )

    candidates = [
        clean_caption(blip_processor.decode(sequence, skip_special_tokens=True))
        for sequence in sequences
    ]
    candidates = [candidate for candidate in candidates if candidate]
    if not candidates:
        return None

    text_inputs = clip_processor(text=candidates, return_tensors="pt", padding=True).to(device)
    image_inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeddings = clip_model.get_text_features(**text_inputs)
        image_embedding = clip_model.get_image_features(**image_inputs)
    text_embeddings /= text_embeddings.norm(p=2, dim=-1, keepdim=True)
    image_embedding /= image_embedding.norm(p=2, dim=-1, keepdim=True)
    scores = (image_embedding @ text_embeddings.t()).squeeze(0)
    return candidates[scores.argmax().item()]


def sort_and_caption(
    input_dir,
    output_dir,
    concepts,
    min_score=0.22,
    min_gap=0.05,
    batch_size=8,
    clip_model_name="openai/clip-vit-large-patch14",
    caption_model_name="Salesforce/blip-image-captioning-large",
):
    import torch
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor, CLIPModel, CLIPProcessor

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    blip_model = BlipForConditionalGeneration.from_pretrained(caption_model_name).to(device)
    blip_processor = BlipProcessor.from_pretrained(caption_model_name)

    for concept_name in concepts:
        (output_dir / concept_name).mkdir(parents=True, exist_ok=True)

    files = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    concept_names = list(concepts)
    prompts = [concepts[name]["description"] for name in concept_names]

    for start in range(0, len(files), batch_size):
        batch_paths = files[start:start + batch_size]
        images = []
        valid_paths = []
        for path in batch_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
                valid_paths.append(path)
            except OSError as exc:
                print(f"Skipping {path.name}: {exc}")
        if not images:
            continue

        text_inputs = clip_processor(text=prompts, return_tensors="pt", padding=True).to(device)
        image_inputs = clip_processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            text_embeddings = clip_model.get_text_features(**text_inputs)
            image_embeddings = clip_model.get_image_features(**image_inputs)
        text_embeddings /= text_embeddings.norm(p=2, dim=-1, keepdim=True)
        image_embeddings /= image_embeddings.norm(p=2, dim=-1, keepdim=True)
        similarity = image_embeddings @ text_embeddings.t()

        for index, source_path in enumerate(valid_paths):
            scores = similarity[index]
            top_scores, top_indices = scores.topk(2)
            score = top_scores[0].item()
            gap = score - top_scores[1].item()
            if score < min_score or gap < min_gap:
                print(f"Skipped {source_path.name}: score={score:.3f}, gap={gap:.3f}")
                continue

            concept_name = concept_names[top_indices[0].item()]
            caption = generate_best_caption(
                images[index],
                blip_processor,
                blip_model,
                clip_processor,
                clip_model,
                device,
            )
            if not caption or any(term in caption.lower() for term in DEFAULT_BANNED_TERMS):
                print(f"Skipped {source_path.name}: caption did not pass filters")
                continue

            destination_dir = output_dir / concept_name
            shutil.copy2(source_path, destination_dir / source_path.name)
            trigger = concepts[concept_name]["trigger"]
            (destination_dir / f"{source_path.stem}.txt").write_text(
                f"{trigger}, {caption}",
                encoding="utf-8",
            )
            print(f"Accepted {source_path.name} -> {concept_name}")


def main():
    parser = argparse.ArgumentParser(description="Classify and caption authorized training images.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("concepts", type=Path, help="JSON concept configuration")
    parser.add_argument("--min-score", type=float, default=0.22)
    parser.add_argument("--min-gap", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--caption-model", default="Salesforce/blip-image-captioning-large")
    args = parser.parse_args()

    concepts = load_concepts(args.concepts)
    sort_and_caption(
        args.input_dir,
        args.output_dir,
        concepts,
        args.min_score,
        args.min_gap,
        args.batch_size,
        args.clip_model,
        args.caption_model,
    )


if __name__ == "__main__":
    main()
