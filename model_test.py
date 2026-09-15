"""Generate one qualitative evaluation image from a LoRA checkpoint."""

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def main():
    parser = argparse.ArgumentParser(description="Evaluate a LoRA checkpoint with one prompt.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="blurry, low quality, text, watermark")
    parser.add_argument("--output", type=Path, default=Path("evaluation.png"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipeline = StableDiffusionPipeline.from_pretrained(args.base_model, torch_dtype=dtype)
    pipeline.to(device)
    pipeline.load_lora_weights(str(args.lora))
    image = pipeline(
        args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
    ).images[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
