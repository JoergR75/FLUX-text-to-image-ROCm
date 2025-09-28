#!/usr/bin/env python3
"""
FLUX text-to-image bot with simple web prompt UI (Gradio) - AMD ROCm optimized

Features:
 - Same multi-GPU/ROCm optimizations as your original script
 - Gradio web UI to enter prompt, negative prompt, size, steps, guidance, seed
 - Model loading / HF token entry from the UI
 - Optional low-vram mode
 - Shows generated image, lets user download it, and prints benchmark stats

Run:
  python3 flux_t2i_bot.py

Notes:
 - Requires `gradio`, `diffusers`, `accelerate`, `huggingface_hub`, and ROCm/PyTorch.
 - If running headless on a remote server, set `GRADIO_SERVER_NAME='0.0.0.0'` and choose a port.
 - This script aims to be a drop-in replacement for your CLI workflow but with a small UI.
"""

import os
import math
import time
import warnings
from functools import lru_cache

import torch
import gradio as gr
from diffusers import FluxPipeline
from huggingface_hub import login

warnings.filterwarnings(
    "ignore",
    message=".*expandable_segments not supported on this platform.*"
)

# ------------------------- Helpers -------------------------

def even(x):
    return int(math.floor(x / 2) * 2)


def pick_dtype(opt_dtype: str):
    if opt_dtype == "bf16":
        # On ROCm/MPS/CUDA we attempt to use bfloat16 when supported
        if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        else:
            print("[warn] bf16 not supported; falling back to fp16.")
            return torch.float16
    return torch.float16


# Global pipeline cache (one pipeline instance per model+dtype+low_vram combination)
@lru_cache(maxsize=4)
def get_pipeline(model_id: str, dtype_str: str, low_vram: bool, hf_token: str | None):
    """Load and return a configured FluxPipeline. This is cached so repeated UI requests reuse it."""
    if not torch.cuda.is_available():
        raise RuntimeError("No ROCm-enabled GPU detected!")

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["ACCELERATE_USE_BALANCED_MEMORY"] = "1"
    os.environ["ACCELERATE_USE_DEVICE_MAP"] = "balanced"

    dtype = pick_dtype(dtype_str)

    if hf_token:
        try:
            login(token=hf_token)
        except Exception as e:
            print(f"[warn] HF login failed: {e}")

    print(f"[info] Loading model on multiple GPUs: {model_id} (dtype={dtype})")

    # Keep device_map balanced for multi-GPU ROCm setups
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        device_map="balanced",
    )

    # Optional compilations (best-effort)
    try:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        pipe.text_encoder = torch.compile(pipe.text_encoder, mode="reduce-overhead", fullgraph=True)
        print("[info] TorchInductor compilation enabled for UNet and text encoder.")
    except Exception as e:
        print(f"[warn] torch.compile not available or failed: {e}")

    # Example LoRA: keep it optional and non-fatal if missing
    try:
        pipe.load_lora_weights("enhanceaiteam/Flux-uncensored", weight_name="lora.safetensors")
    except Exception:
        # Not fatal: continue without the LoRA
        pass

    if low_vram:
        try:
            pipe.enable_attention_slicing()
            pipe.enable_sequential_cpu_offload()
            print("[info] Low VRAM mode enabled.")
        except Exception as e:
            print(f"[warn] Failed to enable low VRAM mode: {e}")

    return pipe


# ------------------------- Generation function -------------------------

def generate_image(
    model_id: str,
    hf_token: str | None,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    seed: int | None,
    dtype_str: str,
    low_vram: bool,
    save_to_disk: bool,
    out_path: str,
):
    """Generate an image and return (pil_image, info_text).

    This function will be called by Gradio. It intentionally tries to be robust for interactive use.
    """
    # sanitize sizes
    width, height = even(width), even(height)

    # load pipeline (cached)
    pipe = get_pipeline(model_id, dtype_str, low_vram, hf_token)

    # prepare generator
    generator = torch.Generator(device="cuda").manual_seed(seed) if seed is not None and seed >= 0 else None

    # reset peak memory counters for accurate benchmarking
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)

    # Warm-up 1 step to stabilize
    try:
        _ = pipe(
            prompt=prompt,
            num_inference_steps=1,
            height=height,
            width=width,
            generator=generator,
            output_type="pil",
        )
    except Exception as e:
        return None, f"[error] Warm-up failed: {e}"

    t0 = time.time()
    td0 = time.time()
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=(negative_prompt or None),
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            generator=generator,
            output_type="pil",
        )
    except Exception as e:
        return None, f"[error] Generation failed: {e}"

    td1 = time.time()
    t1 = time.time()

    image = result.images[0]

    # Save to disk if requested
    if save_to_disk:
        try:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            image.save(out_path)
        except Exception as e:
            return image, f"[warn] Failed to save image: {e}"

    # Benchmark stats
    tod = td1 - td0
    taft = t1 - t0
    t_per_s = steps / tod if tod > 0 else 0.0

    # VRAM usage per GPU
    mem_lines = []
    for i in range(torch.cuda.device_count()):
        mem_alloc = torch.cuda.max_memory_allocated(i) / (1024 ** 2)
        mem_reserved = torch.cuda.max_memory_reserved(i) / (1024 ** 2)
        mem_lines.append(f"GPU{i} peak allocated: {mem_alloc:.1f} MB, reserved: {mem_reserved:.1f} MB")

    info = (
        f"Steps: {steps} | size: {width}x{height} | dtype: {dtype_str}\n"
        f"TaFT (total): {taft:.2f}s | ToD (diffusion): {tod:.2f}s | steps/s: {t_per_s:.2f}\n"
        + "\n".join(mem_lines)
    )

    return image, info


# ------------------------- Gradio UI -------------------------

def build_ui():
    with gr.Blocks(title="FLUX Text-to-Image Bot (ROCm)") as demo:
        gr.Markdown("# FLUX Text-to-Image Bot (ROCm)\nEnter a prompt and press Generate.")

        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Prompt", value="a hyperrealistic exploring spaceship between other smaller spaceships and a huge planet in space, cinematic", lines=4)
                negative = gr.Textbox(label="Negative prompt", value="", lines=2)
                model_id = gr.Textbox(label="Model (HF id)", value="black-forest-labs/FLUX.1-schnell")
                hf_token = gr.Textbox(label="Hugging Face token (optional)", value="", type="password")

                with gr.Row():
                    steps = gr.Slider(minimum=1, maximum=200, value=24, step=1, label="Steps")
                    guidance = gr.Slider(minimum=1.0, maximum=20.0, value=3.5, step=0.1, label="Guidance Scale")

                with gr.Row():
                    width = gr.Number(value=1024, label="Width")
                    height = gr.Number(value=1024, label="Height")
                    seed = gr.Number(value=-1, label="Seed (-1 = random)")

                with gr.Row():
                    dtype_str = gr.Dropdown(choices=["bf16", "fp16"], value="bf16", label="Compute dtype")
                    low_vram = gr.Checkbox(label="Low VRAM mode", value=False)

                save_to_disk = gr.Checkbox(label="Save to disk", value=False)
                out_path = gr.Textbox(label="Output path (if saving)", value="flux_out.png")

                generate_btn = gr.Button("Generate")

            with gr.Column(scale=1):
                image_out = gr.Image(label="Generated Image")
                info_out = gr.Textbox(label="Info / Benchmark", lines=12)

        # Wire up the button
        generate_btn.click(
            fn=generate_image,
            inputs=[model_id, hf_token, prompt, negative, steps, guidance, width, height, seed, dtype_str, low_vram, save_to_disk, out_path],
            outputs=[image_out, info_out],
        )

        gr.Markdown("\n---\nTip: If you use this on a remote server, set `GRADIO_SERVER_NAME='0.0.0.0'` and choose a port.")

    return demo


if __name__ == "__main__":
    ui = build_ui()
    # default: serve on localhost:7860
    ui.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"), server_port=int(os.environ.get("GRADIO_PORT", 7860)))
