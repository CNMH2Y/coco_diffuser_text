# Stable Diffusion UNet LoRA Extension

Added in November 2026 to extend the repository's 2024 diffusion experiments.

**Status:** source implementation with syntax, CLI, and input-validation checks only. No dependencies were installed and no model was downloaded, trained, or used to generate images during preparation of this extension. The selected package versions are a reference API baseline, not a tested GPU environment lockfile.

## Scope

- Standard Diffusers-format Stable Diffusion **1.x or 2.x**, using `CLIPTextModel`, `AutoencoderKL`, and `UNet2DConditionModel`.
- One process on **one CUDA GPU** for training. Comparison selects CUDA when available and otherwise uses CPU.
- LoRA on UNet attention projections; VAE, text encoder, and base UNet weights remain frozen.
- SDXL, SD3, FLUX, inpainting models, text-encoder LoRA, distributed training, optimizer-state resume, and automatic quality metrics are outside this implementation.
- This is a new portfolio extension, not a claim that LoRA was present in the original 2024 commit.

## 1. Prepare a server environment

Use a new Python 3.10–3.12 environment, separate from the legacy scripts. Install the CUDA-enabled PyTorch build appropriate for your server using [PyTorch's installation selector](https://pytorch.org/get-started/locally/), then install the remaining packages:

```bash
python -m pip install -r requirements-lora.txt
```

The script uses Diffusers 0.35.1, Transformers 4.56.2, and PEFT 0.17.1 APIs. These versions do not represent the repository's original 2024 environment. Verify the environment on the actual server before launching a longer run. No particular VRAM budget or training time has been measured.

The base model may require prior authentication or acceptance of its model terms. Credentials are handled by Hugging Face tooling, not stored in this script. A downloaded Diffusers-format model directory can be passed instead of a model ID.

## 2. Provide image-caption records

Example layout:

```text
data/train/
  metadata.jsonl
  images/
    example_001.jpg
    example_002.jpg
```

Each nonempty JSONL line contains a relative image path and a caption:

```json
{"file_name":"images/example_001.jpg","text":"a watercolor painting of a small cottage beside a lake"}
{"file_name":"images/example_002.jpg","text":"a watercolor painting of a mountain valley at sunset"}
```

`metadata.example.jsonl` illustrates the format; its image files are **not supplied**. Replace it with actual records. Captions may repeat an image with different descriptions. Paths must remain under the metadata file's directory, including after resolving symbolic links. PNG, JPEG, WebP, and BMP inputs are accepted.

Choose a coherent adaptation task, such as one visual style, and provide descriptive captions. The watercolor examples are placeholders, not a supplied dataset. This workflow does not automatically download or train on COCO. If using COCO, first export your chosen image/caption subset into this format.

Check paths and captions without loading a model:

```bash
python train_sd_lora.py check-data --metadata data/train/metadata.jsonl
```

To additionally check that image files can be decoded (requires Pillow):

```bash
python train_sd_lora.py check-data --metadata data/train/metadata.jsonl --verify-images
```

Keep evaluation images/prompts separate from the training set. Do not use private or unlicensed records in a public release. Input images and generated model artifacts are excluded by the supplied `.gitignore`; merge its rules if the repository already has one.

## 3. Train an adapter

Run from the repository root on the server. This is an example configuration, not a measured best setting:

```bash
python train_sd_lora.py train --model stable-diffusion-v1-5/stable-diffusion-v1-5 --metadata data/train/metadata.jsonl --output-dir outputs/lora-run-01 --rank 8 --resolution 512 --steps 800 --batch-size 1 --gradient-accumulation-steps 4 --learning-rate 0.0001 --mixed-precision fp16 --gradient-checkpointing --seed 42
```

Use `--model /path/to/downloaded/sd-model --local-files-only` for an offline model directory. The original project's compatible SD 2.x model can also be selected; its availability/access and environment must be checked on the server. Hub model references are resolved to a commit hash before component loading, and the same hash is used for comparison. Pass `--revision MODEL_COMMIT_HASH` to select a particular snapshot. Offline Hub loading requires a cached model index or an explicit commit hash. Local model directories must remain unchanged between training and comparison.

Important behavior:

- `--steps` counts **successful optimizer updates**, not microbatches. Accumulation groups carry across dataset epochs instead of flushing a partially filled group.
- With batch size 1 and accumulation 4, four examples contribute to each optimizer update. For larger batch sizes, the final batch in an epoch can be smaller, so effective sample count may vary.
- VAE image encoding stays FP32. Frozen text/UNet weights use the selected inference dtype; trainable LoRA parameters remain FP32. Loss is computed in FP32.
- Epsilon-prediction models target sampled noise; velocity-prediction models use the scheduler's velocity target.
- `lora_alpha` equals rank, so the adapter scaling factor is 1 before the inference-time scale is applied.
- Default preprocessing is RGB conversion, EXIF orientation correction, resize, center crop, and normalization to `[-1, 1]`. `--random-crop` and `--random-flip` are optional. Only enable flips when captions remain correct after mirroring.
- Output directories must be new or empty. This script does not resume interrupted training and writes the final adapter only after all requested updates finish.

On successful completion, the output contains:

```text
outputs/lora-run-01/
  pytorch_lora_weights.safetensors
  run_config.json
  training_loss.csv
```

The manifest records package versions, GPU, seed, model/revision, trainable parameters, dataset metadata hash, and completed updates. The metadata hash covers the JSONL, not image bytes; preserve the dataset separately if exact reproduction is needed. CSV values are **training noise/velocity-prediction MSE**, not accuracy, CLIPScore, or an image-quality metric. No performance numbers are prefilled.

## 4. Generate matched base/LoRA comparisons

Use `prompts.example.json` as a format example and replace it with evaluation prompts for your adaptation task. It is a JSON array of strings. Use several prompts and seeds rather than selecting one favorable example.

```bash
python train_sd_lora.py compare --adapter-dir outputs/lora-run-01 --prompts prompts.example.json --output-dir outputs/comparison-01 --resolution 512 --steps 30 --guidance-scale 7.5 --lora-scale 1.0 --seeds 42 123 2024
```

The script reads the base-model reference from the completed training manifest, checks the adapter hash, generates baseline images **before** loading the adapter, and resets a fresh seed for every corresponding image. Both modes use DDIM with the same settings. It saves:

- Individual `*_base.png` and `*_lora.png` images.
- Side-by-side `*_comparison.png` images.
- `comparison.json` with prompts, seeds, settings, versions, and safety-check flags.

The base model's default safety-checker behavior is retained. Flagged outputs may be filtered; inspect recorded flags before interpreting pairs. Results may differ across GPU hardware or software versions even with the same seed. The tool does not measure CLIPScore, FID, latency, or VRAM savings and does not declare that LoRA improved quality.

Generated outputs stay ignored by default. After reviewing a real run, manually copy a small, representative selection to a public `examples/lora/` directory if you want them in the README. Keep its full prompt/seed/settings context. Do not label placeholder images as model outputs.

## 5. Describe the work accurately

Before a GPU run, the defensible claim is that you added the implementation and comparison workflow. After training and comparison succeed, describe the actual dataset, completed updates, adapter parameter count, and observed behavior. Quantitative improvement requires an appropriate held-out evaluation.

Do not attach unmeasured percentages or retroactively attribute this new extension to the 2024 project. A project date range may include both the original experiments and this later extension, with each phase explained.

## Lightweight checks

From the repository root:

```bash
python -m unittest discover -s tests -p "test_lora_inputs.py" -v
python train_sd_lora.py --help
```

These tests cover malformed metadata, missing files, directory escapes, prompt formatting, and output collision handling using temporary dummy files. They do not verify image decoding, ML dependencies, GPU execution, adapter training/reload, or generation quality.

## Implementation references

- [Diffusers LoRA guide, v0.35.1](https://huggingface.co/docs/diffusers/v0.35.1/en/training/lora)
- [Diffusers v0.35.1 text-to-image LoRA reference](https://github.com/huggingface/diffusers/blob/v0.35.1/examples/text_to_image/train_text_to_image_lora.py)
- [PEFT LoRA configuration](https://huggingface.co/docs/peft/en/package_reference/lora)
- [Accelerate gradient accumulation](https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation)

This is a compact implementation against those public APIs, not a reproduction of benchmark results. Model and dataset licenses continue to apply; no base weights, training images, or license grant for the existing repository are bundled here.
