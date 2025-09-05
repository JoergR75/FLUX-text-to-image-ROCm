#!/usr/bin/env python3
"""
FLUX text-to-image on AMD ROCm (Optimized for Dual 32GB GPUs)

Benchmark added: TTFT, TaFT, ToD, t/s, VRAM usage

Example:
python3 flux_test7.py \
  --prompt "a hyperrealistic exploring spaceship between other smaller spaceships and a huge planet in space, cinematic" \
  --model black-forest-labs/FLUX.1-dev \
  --steps 50 \
  --width 1280 \
  --height 960 \
  --out spaceship_50.png \
  --hf-token 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

"""

import argparse
import os
import math
import torch
import time
from diffusers import FluxPipeline
from huggingface_hub import login
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*expandable_segments not supported on this platform.*"
)

def parse_args():
    p = argparse.ArgumentParser(description="Generate images with FLUX on AMD ROCm (Multi-GPU optimized)")
    p.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-schnell",
                   help="HF model id")
    p.add_argument("--prompt", type=str, required=True, help="Text prompt")
    p.add_argument("--negative", type=str, default="", help="Negative prompt")
    p.add_argument("--steps", type=int, default=24, help="Num inference steps")
    p.add_argument("--guidance", type=float, default=3.5, help="Classifier-free guidance scale")
    p.add_argument("--width", type=int, default=1024, help="Output width")
    p.add_argument("--height", type=int, default=1024, help="Output height")
    p.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    p.add_argument("--out", type=str, default="flux_out.png", help="Output file path")
    p.add_argument("--dtype", type=str, choices=["bf16", "fp16"], default="bf16",
                   help="Compute dtype (default: bf16)")
    p.add_argument("--hf-token", type=str, default=None,
                   help="Hugging Face token or set HF_TOKEN env var")
    p.add_argument("--low-vram", action="store_true",
                   help="Enable low VRAM optimizations")
    return p.parse_args()

def pick_dtype(opt_dtype: str):
    if opt_dtype == "bf16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            print("[warn] bf16 not supported; falling back to fp16.")
            return torch.float16
    return torch.float16

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("No ROCm-enabled GPU detected!")

    # Optimize memory allocation for multi-GPU ROCm
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["ACCELERATE_USE_BALANCED_MEMORY"] = "1"
    os.environ["ACCELERATE_USE_DEVICE_MAP"] = "balanced"

    device = torch.device("cuda")
    dtype = pick_dtype(args.dtype)

    # Login to HF if needed
    token = args.hf_token or os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("[warn] No Hugging Face token provided. Private models may fail.")

    print(f"[info] Loading model on multiple GPUs: {args.model}")
    pipe = FluxPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        use_safetensors=True,
        device_map="balanced",
    )

    if args.low_vram:
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
        print("[info] Low VRAM mode enabled.")

    # Even dimensions required by Flux
    def even(x): return int(math.floor(x / 2) * 2)
    height, width = even(args.height), even(args.width)

    # Seeding
    generator = torch.Generator(device="cuda").manual_seed(args.seed) if args.seed else None

    # ---------------- Benchmark start ----------------

    print(f"[info] Generating image on multiple GPUs: steps={args.steps}, size={width}x{height}, dtype={dtype}")

    # Reset peak memory counters
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)

    # TTFT: pipeline forward (warm-up + inference)
    ttft0 = time.time()  # start forward timer

    # Warm-up step (optional, can stabilize benchmarking)
    _ = pipe(
        prompt=args.prompt,
        num_inference_steps=1,
        height=height,
        width=width,
        generator=generator,
        output_type="pil",
    )

    t0 = time.time()  # start total timer (after warm-up)
    td0 = time.time()  # start diffusion timer
    image = pipe(
        prompt=args.prompt,
        negative_prompt=(args.negative or None),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=height,
        width=width,
        generator=generator,
        output_type="pil",
    ).images[0]
    td1 = time.time()  # diffusion end
    t1 = time.time()  # total end
    ttft1 = time.time()  # forward end

    # Save image
    image.save(args.out)
    print(f"[done] Saved -> {args.out}")

    # ---------------- Benchmark results ----------------

    taft = t1 - t0
    tod = td1 - td0
    ttft = ttft1 - ttft0
    t_per_s = args.steps / tod if tod > 0 else 0

    print("\n[benchmark]")
    print(f"  TTFT (forward):  {ttft:.2f} sec")
    print(f"  TaFT (total):    {taft:.2f} sec")
    print(f"  ToD  (diffuse):  {tod:.2f} sec")
    print(f"  t/s  (steps/s):  {t_per_s:.2f}")

    # VRAM usage per GPU
    for i in range(torch.cuda.device_count()):
        mem_alloc = torch.cuda.max_memory_allocated(i) / (1024 ** 2)  # MB
        mem_reserved = torch.cuda.max_memory_reserved(i) / (1024 ** 2)  # MB
        print(f"  GPU{i} peak allocated: {mem_alloc:.1f} MB, reserved: {mem_reserved:.1f} MB")

if __name__ == "__main__":
    main()
