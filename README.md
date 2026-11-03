# Diffusion Models: Training, Sampling, and LoRA Adaptation

Experiments with PyTorch and Hugging Face Diffusers: denoising networks, unconditional and class-conditioned DDPM training, Stable Diffusion text-to-image sampling, and a new UNet LoRA training/comparison workflow.

## Project history and status

- **November 2024:** original diffusion experiments committed to this repository.
- **November 2026:** added `train_sd_lora.py`, dataset validation, adapter export, and matched-seed base/LoRA comparison code.
- **LoRA execution status:** implementation added; GPU training, adapter reload, and image generation have not yet been exercised end to end. No trained LoRA weights, generated comparison images, or quality improvements are claimed in this release.

## Components

| File | Scope |
| --- | --- |
| `train_sd_lora.py` | New SD 1.x/2.x UNet attention LoRA training and paired image-generation workflow |
| `stable_diffusion_example.py` | SD 2.1 sampling loop with prompt encoding, classifier-free guidance, scheduler updates, and VAE decoding |
| `train_classed_mnist_diffuser.py` | Class embeddings, conditional UNet, DDPM noise-prediction training, checkpoint save/load, and sampling |
| `train_mnist_diffuser_ddpm.py` | Unconditional MNIST DDPM training and sampling |
| `diffuser_example.py` | Pretrained face-generation model with DDIM sampling |
| `train_mnist_diffuser.py`, `train_animeface_diffuser.py` | Earlier custom-UNet denoising experiments |
| `stable_diff.py` | Incomplete COCO text-conditioning training draft, retained as experimental history |
| `gru_exm.py` | Unrelated early pulse-signal experiment; see the separate [pulse-classification project](https://github.com/CNMH2Y/cnn-gru-pulse-classification) |

The legacy scripts use historical APIs and local paths. Their original environment is not reconstructed here. The dependency list below applies to the new LoRA extension.

## LoRA workflow

The new script freezes the VAE, CLIP text encoder, and original UNet weights. It trains low-rank adapters on the UNet attention projections `to_q`, `to_k`, `to_v`, and `to_out.0`. It includes:

- Local image-caption JSONL input with path/caption validation.
- Configurable rank, learning rate, batch size, and optimizer-update count.
- FP16/BF16 options, FP32 trainable adapter weights, gradient accumulation, and optional gradient checkpointing.
- Scheduler-aware noise or velocity prediction targets.
- Safetensors adapter export plus training configuration, parameter counts, and loss CSV.
- Base/LoRA image pairs with identical prompts, seeds, scheduler, guidance, and resolution.

See [README_LORA.md](README_LORA.md) for input format, commands, outputs, and limitations. The implementation uses the public [Diffusers LoRA APIs](https://huggingface.co/docs/diffusers/v0.35.1/en/training/lora) and [PEFT LoRA configuration](https://huggingface.co/docs/peft/en/package_reference/lora).

## Evidence to add after running

Keep a separate prompt set for evaluation, preserve every selected prompt/seed result, and report the base model revision, training steps, trainable parameter count, and dataset scope. Training MSE and a few attractive examples do not establish a quality improvement. Actual checkpoints and measured results should be added only after a real run.

Model, dataset, and upstream code licenses still apply. This extension does not change the licensing of the original repository or redistribute model/data assets.
