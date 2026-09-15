# Multi-Concept LoRA Dataset Pipeline

An experimental toolkit for turning a collection of authorized videos or images into a filtered, captioned dataset suitable for multi-concept LoRA training with common Diffusers-style workflows.

## What it demonstrates

- Parallel video-frame extraction at configurable intervals
- Deterministic sampling for manageable dataset review
- CLIP-based concept classification and ambiguity filtering
- BLIP caption generation followed by CLIP reranking
- Conversion from class-folder image/caption pairs to a flat `metadata.jsonl` dataset
- Loading a LoRA checkpoint into a Diffusers pipeline for qualitative evaluation

## Pipeline

```text
authorized videos
    -> frame extraction
    -> deterministic sampling
    -> CLIP concept filtering
    -> BLIP caption candidates
    -> CLIP caption reranking
    -> manual quality review
    -> Diffusers dataset formatting
    -> LoRA training and evaluation
```

No source videos, training images, captions, model checkpoints, or generated results are included. Use only material you own, material licensed for this purpose, or material for which you have explicit permission. Model and dataset licenses remain separate from this repository's source code.

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The captioning and generation steps are resource-intensive. A CUDA-capable GPU is strongly recommended, although PyTorch can fall back to CPU for small experiments.

## 1. Extract frames

```bash
python get_captures.py videos captures --interval-seconds 50 --trim-start 30 --trim-end 10
```

## 2. Sample frames

The output directory must be new unless `--replace` is explicitly supplied:

```bash
python sample_captures.py captures samples --count 500 --seed 42
```

## 3. Classify and caption

Create a JSON configuration whose keys are output class names and whose values contain a visual description and training trigger:

```json
{
  "concept_a": {
    "description": "a visual description of the first authorized concept",
    "trigger": "sks_concept_a"
  },
  "concept_b": {
    "description": "a visual description of the second authorized concept",
    "trigger": "sks_concept_b"
  }
}
```

Then run:

```bash
python to_train/blip_clip_pipe.py samples dataset_sorted concepts.example.json
```

Automated captions and classifications require manual review. They may be incorrect, biased, or unsuitable for training.

## 4. Convert to Diffusers format

```bash
python format_for_hf.py dataset_sorted hf_dataset
```

This copies images into one directory and creates `metadata.jsonl` with `file_name` and `text` fields.

## 5. Evaluate a LoRA checkpoint

```bash
python model_test.py \
  --base-model MODEL_ID_OR_PATH \
  --lora PATH_TO_LORA_WEIGHTS \
  --prompt "your authorized evaluation prompt" \
  --output evaluation.png
```

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
```

## Limitations

- CLIP confidence is not a calibrated statement of correctness.
- BLIP can hallucinate details and may repeat artifacts from its training data.
- Simple confidence gaps do not guarantee clean concept separation.
- Small or imbalanced datasets can overfit quickly.
- Generated output requires both qualitative and quantitative evaluation.
- Training and inference dependencies may impose additional model-specific terms.

## License

No open-source license has been selected. Until one is added, standard copyright applies.
