#!/usr/bin/env python3
"""Stable Diffusion 1.x/2.x UNet LoRA training and matched-seed comparison.

New repository extension, November 2026. No model results are bundled.
Uses the public Diffusers/PEFT adapter APIs; see README_LORA.md for references.
Heavy dependencies are imported only by train/compare, not by help/check-data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
WEIGHT_NAME = "pytorch_lora_weights.safetensors"


def read_manifest(path: Path) -> list[dict]:
    """Read local image-caption JSONL without importing ML packages."""
    path = path.resolve(strict=True)
    root = path.parent
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_number}: expected an object")
        caption, filename = row.get("text"), row.get("file_name")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Line {line_number}: text must be a nonempty caption")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"Line {line_number}: file_name is required")
        relative = Path(filename)
        if relative.is_absolute() or relative.drive:
            raise ValueError(f"Line {line_number}: file_name must be relative")
        image_path = (root / relative).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError(f"Line {line_number}: image must stay inside the dataset directory")
        if not image_path.is_file():
            raise ValueError(f"Line {line_number}: image not found: {filename}")
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            raise ValueError(f"Line {line_number}: unsupported image extension")
        records.append({"image": image_path, "text": caption.strip()})
    if not records:
        raise ValueError("The manifest contains no image-caption records")
    return records


def read_prompts(path: Path) -> list[str]:
    prompts = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompts must be a nonempty JSON array of strings")
    if any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Every prompt must be a nonempty string")
    return [p.strip() for p in prompts]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def versions() -> dict:
    names = ["torch", "diffusers", "peft", "transformers", "accelerate", "Pillow", "numpy"]
    return {name: importlib.metadata.version(name) for name in names}


def new_output_directory(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output already exists and is not empty; choose a new directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_revision(model: str, requested: str | None, offline: bool) -> str | None:
    """Pin Hub models once so all components and later comparisons share a snapshot."""
    if Path(model).is_dir():
        return None
    from huggingface_hub import model_info, try_to_load_from_cache
    if requested and re.fullmatch(r"[0-9a-fA-F]{40}", requested):
        return requested
    if offline:
        cached = try_to_load_from_cache(model, "model_index.json", revision=requested)
        if isinstance(cached, str) and re.fullmatch(r"[0-9a-fA-F]{40}", Path(cached).parent.name):
            return Path(cached).parent.name
        raise ValueError("Offline Hub loading needs a cached model_index.json or an explicit model commit hash")
    return model_info(model, revision=requested).sha


def check_data(args) -> None:
    records = read_manifest(args.metadata)
    summary = {"caption_records": len(records), "unique_images": len({r["image"] for r in records})}
    if args.verify_images:
        from PIL import Image
        for record in records:
            with Image.open(record["image"]) as image:
                image.verify()
        summary["image_files_verified"] = True
    else:
        summary["image_files_verified"] = False
    print(json.dumps(summary, indent=2))
    print("This checks input files only; it does not download or run a model.")


def make_dataset(records, tokenizer, resolution, random_crop, random_flip):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    class CaptionedImages(torch.utils.data.Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            record = records[index]
            with Image.open(record["image"]) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = image.size
            scale = resolution / min(width, height)
            image = image.resize(
                (max(resolution, math.ceil(width * scale)), max(resolution, math.ceil(height * scale))),
                Image.Resampling.BICUBIC,
            )
            width, height = image.size
            if random_crop:
                left = torch.randint(width - resolution + 1, (1,)).item()
                top = torch.randint(height - resolution + 1, (1,)).item()
            else:
                left, top = (width - resolution) // 2, (height - resolution) // 2
            image = image.crop((left, top, left + resolution, top + resolution))
            if random_flip and torch.rand(()).item() < 0.5:
                image = ImageOps.mirror(image)
            pixels = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
            tokens = tokenizer(
                record["text"], padding="max_length", truncation=True,
                max_length=tokenizer.model_max_length, return_tensors="pt",
            )
            return {
                "pixels": torch.from_numpy(pixels).permute(2, 0, 1).contiguous(),
                "input_ids": tokens.input_ids[0],
                "attention_mask": tokens.attention_mask[0],
            }

    return CaptionedImages()


def train(args) -> None:
    # Check paths/captions before loading a GPU model or downloading weights.
    records = read_manifest(args.metadata)
    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from accelerate.utils import GradientAccumulationPlugin, set_seed
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from tqdm.auto import tqdm
    from transformers import CLIPTextModel, CLIPTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires one CUDA GPU. No model has been downloaded.")
    if args.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; choose fp16 or no")
    # Carry microbatches across epochs so even tiny datasets get full accumulation groups.
    accumulation = GradientAccumulationPlugin(
        num_steps=args.gradient_accumulation_steps, sync_with_dataloader=False,
    )
    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_plugin=accumulation)
    if accelerator.num_processes != 1:
        raise ValueError("This compact implementation supports one process / one GPU only")
    set_seed(args.seed)
    output = new_output_directory(args.output_dir)
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    revision = resolve_revision(args.model, args.revision, args.local_files_only)
    loading = {"revision": revision, "local_files_only": args.local_files_only}
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer", **loading)
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder", **loading)
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", **loading)
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet", **loading)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler", **loading)
    if noise_scheduler.config.prediction_type not in {"epsilon", "v_prediction"}:
        raise ValueError("Only epsilon and v_prediction training targets are supported")
    if unet.config.in_channels != 4 or unet.config.cross_attention_dim != text_encoder.config.hidden_size:
        raise ValueError("Expected a standard SD 1.x/2.x model, not SDXL/SD3/FLUX/inpainting")
    if args.resolution % (2 ** (len(vae.config.block_out_channels) - 1)):
        raise ValueError("Resolution must be divisible by the VAE downsampling factor")

    for module in (unet, vae, text_encoder):
        module.requires_grad_(False)
    vae.eval().to(accelerator.device, dtype=torch.float32)
    text_encoder.eval().to(accelerator.device, dtype=dtype)
    unet.to(accelerator.device, dtype=dtype)
    # alpha=rank keeps export/load scaling unambiguous (alpha / rank = 1).
    unet.add_adapter(LoraConfig(
        r=args.rank, lora_alpha=args.rank, lora_dropout=0.0,
        target_modules=TARGET_MODULES, init_lora_weights="gaussian", bias="none",
    ))
    trainable = []
    for name, parameter in unet.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise RuntimeError(f"Unexpected trainable base parameter: {name}")
            parameter.data = parameter.data.to(torch.float32)
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("No LoRA parameters were attached")
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    unet.train()
    dataset = make_dataset(records, tokenizer, args.resolution, args.random_crop, args.random_flip)
    # num_workers=0 also makes this nested dataset portable on Windows.
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    trainable_count = sum(p.numel() for p in trainable)
    total_count = sum(p.numel() for p in unet.parameters())
    config = {
        "status": "training_in_progress", "extension_added": "2026-11",
        "started_at_utc": utc_now(), "model": args.model, "revision": revision,
        "requested_revision": args.revision,
        "prediction_type": noise_scheduler.config.prediction_type,
        "rank": args.rank, "lora_alpha": args.rank, "target_modules": TARGET_MODULES,
        "trainable_parameters": trainable_count, "unet_parameters_with_adapter": total_count,
        "trainable_fraction": trainable_count / total_count,
        "resolution": args.resolution, "seed": args.seed, "requested_optimizer_steps": args.steps,
        "batch_size": args.batch_size, "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "nominal_effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "batch_size_note": "Last data batch may be smaller than batch_size.",
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm, "mixed_precision": args.mixed_precision,
        "gradient_checkpointing": args.gradient_checkpointing,
        "random_crop": args.random_crop, "random_flip": args.random_flip,
        "caption_records": len(records), "unique_images": len({r["image"] for r in records}),
        "metadata_sha256": sha256(args.metadata), "script_sha256": sha256(Path(__file__)),
        "package_versions": versions(), "gpu": torch.cuda.get_device_name(accelerator.device),
        "cuda_version": torch.version.cuda, "evaluation": "No held-out quality metric computed",
    }
    write_json(output / "run_config.json", config)
    print(f"Trainable adapter parameters: {trainable_count:,} / {total_count:,} ({100 * trainable_count / total_count:.3f}%)")
    unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)
    optimizer.zero_grad(set_to_none=True)
    step, loss_sum, microbatches, consecutive_skips = 0, 0.0, 0, 0
    progress = tqdm(total=args.steps, desc="Optimizer updates")
    with (output / "training_loss.csv").open("w", newline="", encoding="utf-8") as log:
        writer = csv.writer(log)
        writer.writerow(["optimizer_step", "mean_microbatch_mse", "learning_rate"])
        while step < args.steps:
            for batch in loader:
                with accelerator.accumulate(unet):
                    with torch.no_grad():
                        # FP32 VAE encoding avoids fp16 VAE overflow; only LoRA gets gradients.
                        pixels = batch["pixels"].to(accelerator.device, dtype=torch.float32)
                        latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                        latents = latents.to(dtype)
                        mask = batch["attention_mask"] if getattr(text_encoder.config, "use_attention_mask", False) else None
                        conditioning = text_encoder(batch["input_ids"], attention_mask=mask)[0]
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (latents.shape[0],), device=latents.device, dtype=torch.long,
                    )
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    else:
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    with accelerator.autocast():
                        prediction = unet(noisy_latents, timesteps, encoder_hidden_states=conditioning).sample
                        loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("Non-finite training loss; no completed adapter was written")
                    loss_sum += loss.detach().item()
                    microbatches += 1
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    if accelerator.optimizer_step_was_skipped:
                        consecutive_skips += 1
                        if consecutive_skips >= 10:
                            raise FloatingPointError("Ten consecutive AMP updates were skipped; try bf16 or no precision reduction")
                    else:
                        consecutive_skips = 0
                        step += 1
                        mean_loss = loss_sum / microbatches
                        writer.writerow([step, mean_loss, optimizer.param_groups[0]["lr"]])
                        log.flush()
                        progress.update(1)
                        progress.set_postfix(mse=f"{mean_loss:.4f}")
                    loss_sum, microbatches = 0.0, 0
                    if step >= args.steps:
                        break
    progress.close()
    accelerator.wait_for_everyone()
    adapter = get_peft_model_state_dict(accelerator.unwrap_model(unet))
    adapter = {name: value.detach().to(device="cpu", dtype=torch.float32) for name, value in adapter.items()}
    StableDiffusionPipeline.save_lora_weights(
        str(output), unet_lora_layers=convert_state_dict_to_diffusers(adapter),
        weight_name=WEIGHT_NAME, safe_serialization=True,
    )
    config.update({
        "status": "training_completed", "completed_at_utc": utc_now(),
        "completed_optimizer_steps": step, "adapter_sha256": sha256(output / WEIGHT_NAME),
    })
    write_json(output / "run_config.json", config)
    accelerator.end_training()
    print(f"Saved LoRA adapter to {output}. Training loss is not a generation-quality score.")


def compare(args) -> None:
    prompts = read_prompts(args.prompts)
    adapter_dir = args.adapter_dir.resolve(strict=True)
    config = json.loads((adapter_dir / "run_config.json").read_text(encoding="utf-8"))
    if config.get("status") != "training_completed":
        raise ValueError("The adapter directory does not describe a completed training run")
    weights = adapter_dir / WEIGHT_NAME
    if not weights.is_file() or sha256(weights) != config.get("adapter_sha256"):
        raise ValueError("Adapter weights are missing or do not match the training manifest")
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from PIL import Image, ImageDraw

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    output = new_output_directory(args.output_dir)
    pipe = StableDiffusionPipeline.from_pretrained(
        config["model"], revision=config.get("revision"), torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    settings = {
        "created_at_utc": utc_now(), "model": config["model"], "revision": config.get("revision"),
        "adapter_sha256": config["adapter_sha256"], "lora_scale": args.lora_scale,
        "prompts": prompts, "negative_prompt": args.negative_prompt, "seeds": args.seeds,
        "scheduler": "DDIMScheduler", "scheduler_config": dict(pipe.scheduler.config),
        "inference_steps": args.steps, "guidance_scale": args.guidance_scale,
        "resolution": args.resolution, "eta": 0.0, "device": device,
        "dtype": str(dtype), "package_versions": versions(),
        "script_sha256": sha256(Path(__file__)), "status": "comparison_in_progress",
        "interpretation": "Qualitative paired samples; no measured improvement or held-out score.",
    }
    write_json(output / "comparison.json", settings)
    rows = []
    # Generate every baseline image before loading the adapter.
    for mode in ("base", "lora"):
        if mode == "lora":
            pipe.load_lora_weights(str(adapter_dir), weight_name=WEIGHT_NAME, adapter_name="portfolio")
            pipe.set_adapters(["portfolio"], adapter_weights=[args.lora_scale])
        for prompt_index, prompt in enumerate(prompts):
            for seed in args.seeds:
                # A fresh CPU generator for EACH call yields matched starting noise.
                generator = torch.Generator(device="cpu").manual_seed(seed)
                with torch.inference_mode():
                    result = pipe(
                        prompt=prompt, negative_prompt=args.negative_prompt,
                        generator=generator, num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale, height=args.resolution,
                        width=args.resolution, eta=0.0,
                    )
                filename = f"prompt_{prompt_index:02d}_seed_{seed}_{mode}.png"
                result.images[0].save(output / filename)
                flags = result.nsfw_content_detected
                rows.append({
                    "prompt_index": prompt_index, "prompt": prompt, "seed": seed,
                    "mode": mode, "file": filename,
                    "safety_flag": bool(flags[0]) if flags is not None else None,
                })
    for prompt_index in range(len(prompts)):
        for seed in args.seeds:
            prefix = f"prompt_{prompt_index:02d}_seed_{seed}"
            with Image.open(output / f"{prefix}_base.png") as left, Image.open(output / f"{prefix}_lora.png") as right:
                canvas = Image.new("RGB", (args.resolution * 2, args.resolution + 32), "white")
                canvas.paste(left, (0, 32))
                canvas.paste(right, (args.resolution, 32))
                draw = ImageDraw.Draw(canvas)
                draw.text((8, 8), "Base model", fill="black")
                draw.text((args.resolution + 8, 8), f"LoRA (scale={args.lora_scale:g})", fill="black")
                canvas.save(output / f"{prefix}_comparison.png")
    settings.update({"status": "comparison_completed", "completed_at_utc": utc_now(), "samples": rows})
    write_json(output / "comparison.json", settings)
    print(f"Saved paired images and settings to {output}. Review all prompts/seeds before interpreting results.")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def seed_int(value):
    number = int(value)
    if not 0 <= number < 2**32:
        raise argparse.ArgumentTypeError("seed must be in [0, 2**32)")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-data", help="Check local JSONL paths/captions; no model load")
    check.add_argument("--metadata", type=Path, required=True)
    check.add_argument("--verify-images", action="store_true", help="Also decode-check images with Pillow")
    check.set_defaults(func=check_data)
    fit = sub.add_parser("train", help="Train UNet attention LoRA on one CUDA GPU")
    fit.add_argument("--model", required=True, help="SD 1.x/2.x Diffusers-format model ID or local directory")
    fit.add_argument("--revision", default=None, help="Use a model commit hash for stronger reproducibility")
    fit.add_argument("--metadata", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--resolution", type=positive_int, default=512)
    fit.add_argument("--rank", type=positive_int, default=8)
    fit.add_argument("--steps", type=positive_int, default=800, help="Successful optimizer updates, not microbatches")
    fit.add_argument("--batch-size", type=positive_int, default=1)
    fit.add_argument("--gradient-accumulation-steps", type=positive_int, default=4)
    fit.add_argument("--learning-rate", type=positive_float, default=1e-4)
    fit.add_argument("--weight-decay", type=nonnegative_float, default=0.01)
    fit.add_argument("--max-grad-norm", type=positive_float, default=1.0)
    fit.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="no")
    fit.add_argument("--gradient-checkpointing", action="store_true")
    fit.add_argument("--random-crop", action="store_true")
    fit.add_argument("--random-flip", action="store_true", help="Use only when flips preserve caption meaning")
    fit.add_argument("--seed", type=seed_int, default=42)
    fit.add_argument("--local-files-only", action="store_true")
    fit.set_defaults(func=train)
    render = sub.add_parser("compare", help="Generate base/LoRA pairs with identical prompts and starting noise")
    render.add_argument("--adapter-dir", type=Path, required=True)
    render.add_argument("--prompts", type=Path, required=True, help="JSON array of held-out evaluation prompts")
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--resolution", type=positive_int, default=512)
    render.add_argument("--steps", type=positive_int, default=30)
    render.add_argument("--guidance-scale", type=positive_float, default=7.5)
    render.add_argument("--lora-scale", type=nonnegative_float, default=1.0)
    render.add_argument("--negative-prompt", default="")
    render.add_argument("--seeds", type=seed_int, nargs="+", default=[42, 123, 2024])
    render.add_argument("--local-files-only", action="store_true")
    render.set_defaults(func=compare)
    args = parser.parse_args(argv)
    if hasattr(args, "resolution") and args.resolution % 64:
        parser.error("--resolution must be a multiple of 64")
    if args.command == "compare" and len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    return args


def main():
    args = parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, ImportError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
