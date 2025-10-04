# 🖌️ FLUX Text-to-Image Web UI Agent — ROCm-Ready

A lightweight **Gradio-based web interface** for running the **FLUX text-to-image model** on AMD ROCm systems.  
This project provides an easy-to-use web UI to generate AI images from text prompts, monitor VRAM usage, and optionally share the interface online.
**Optimized for Multi-GPU (Dual Radeon AI PRO R9700 32GB GPUs) · Supports bf16/fp16 · Hugging Face Models**

[![ROCm](https://img.shields.io/badge/AMD-ROCm_6.x-red)](https://rocmdocs.amd.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0.dev%2Brocm6.4-ee4c2c?logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow)](https://huggingface.co/docs/diffusers)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Gradio Quickstart](https://img.shields.io/badge/Gradio-Quickstart-blue)](https://www.gradio.app/guides/quickstart)

<img width="1709" height="952" alt="image" src="https://github.com/user-attachments/assets/2ac83f03-1aec-4c82-b4f0-47a5815b9d8a" />

---

## 🚀 Features

- 🌌 Text-to-image generation with the **FLUX.1** model  
- 🎛 Simple **Gradio web interface** for interactive use  
- 📊 Automatic logging of:
  - Inference speed (tokens/sec, steps/sec)
  - VRAM usage (per GPU)
- ⚡ ROCm compatible (tested on AMD Instinct MI210, Radeon AI PRO R9700 @ ROCm 6.4.3 & 7.0.1)  
- 🌍 Optional public link sharing with `share=True`  

---

## 📦 Requirements

- Python **3.10+**  
- PyTorch (with ROCm support) -> use the [ROCm 6.4.3 automated deployment script](https://github.com/JoergR75/rocm-6.4.3-rdna4-docker-automated-deployment) 
- Hugging Face `diffusers` library  
- Additional packages: `transformers`, `accelerate`, `safetensors`, `gradio`, `psutil` includet in the [ROCm 6.4.3 automated deployment script](https://github.com/JoergR75/rocm-6.4.3-rdna4-docker-automated-deployment) 

# 💻 Code Explanation: FLUX Gradio Web Agent

This script provides a **Gradio-based web UI** for running the **FLUX text-to-image model** with AMD ROCm optimizations.  
It is designed as a drop-in replacement for CLI workflows but with an interactive interface.

---

## 🔹 Imports and Setup

```python
import os, math, time, warnings
from functools import lru_cache

import torch
import gradio as gr
from diffusers import FluxPipeline
from huggingface_hub import login
     ...
```
- torch → deep learning backend with ROCm/CUDA support
- gradio → builds the web interface
- diffusers.FluxPipeline → loads and runs the FLUX model
- huggingface_hub.login → optional Hugging Face authentication for private models
- Warnings about expandable segments are filtered to avoid clutter.

## 🔹 Helper Functions
```python
def even(x): ...
def pick_dtype(opt_dtype: str): ...
```
- even() → ensures image dimensions are even (some models require this).
- pick_dtype() → chooses the compute precision (bf16 if available, otherwise fp16).

## 🔹 Pipeline Loader (with Caching)
```python
@lru_cache(maxsize=4)
def get_pipeline(model_id, dtype_str, low_vram, hf_token):
    ...
```
Uses lru_cache to avoid reloading the model on every generation.
Configures memory management for multi-GPU ROCm setups.
Supports:
- bf16 / fp16 compute
- LoRA weight loading (optional)
- Low VRAM mode (attention slicing + CPU offload)
- Attempts torch.compile to speed up UNet and text encoder.

## 🔹 Image Generation Function
```python
def generate_image(...):
    ...
```
Called by the Gradio UI when Generate is clicked.
Steps:
- Ensures width/height are valid.
- Loads pipeline (cached).
- Prepares random seed (or deterministic if set).
- Runs a 1-step warm-up for stability.
- Runs full inference and measures:
- ToD → time spent in diffusion
- TaFT → total generation time
- Steps/s → throughput
- Collects VRAM usage per GPU.
- Optionally saves the image to disk.
- Returns the image and a benchmark report.

## 🔹 Gradio UI
```python
def build_ui():
    with gr.Blocks(title="FLUX Text-to-Image Bot (ROCm)") as demo:
        ...
```
UI layout:
- Prompt inputs: prompt, negative prompt, model ID, HF token
- Generation settings: steps, guidance scale, width/height, seed, dtype, low-VRAM toggle
- Output options: save-to-disk toggle + path
- Results: generated image + benchmark info
A Generate button triggers the generate_image() function.
Provides a tip for remote usage: GRADIO_SERVER_NAME=0.0.0.0.

## 🔹 Main Launcher
```python
if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
              server_port=int(os.environ.get("GRADIO_PORT", 7860)))
```
Starts the Gradio server on http://127.0.0.1:7860 by default.
Supports environment variables:
- GRADIO_SERVER_NAME=0.0.0.0 → listen on all interfaces (for remote servers)
- GRADIO_PORT=XXXX → choose a custom port

## 🔹 Key Features Recap
- Multi-GPU ROCm optimization
- bf16/fp16 precision with fallback
- Optional low VRAM mode
- Gradio UI for easy prompt entry
- Benchmark logging (steps/sec, VRAM usage, timings)
- Supports Hugging Face authentication for gated/private models

# 🌀 Installation

### Download the the server script

```echo
wget https://raw.githubusercontent.com/JoergR75/FLUX-text-to-image-ROCm/refs/heads/main/Gradio-Web-Agent/FLUX-gradio-web-agent.py
```
<img width="1266" height="199" alt="{282018B2-4594-456B-A4C2-7B58E0AF1EE1}" src="https://github.com/user-attachments/assets/15e50547-eb49-4e11-be74-51569f440432" />

### Launch the Gradio web Agent server

```echo
python3 FLUX-gradio-web-agent.py
```
The Web server will be launched with following parameters:
- http://127.0.0.1:7860 → The Gradio web interface is running locally on your machine. Open this link in your browser to access the app.
- share=True → If you set this option inside the script’s launch() method, Gradio will create a temporary public URL. This allows you to share access with others outside your local network.
> ⚠️ **Attention**  
> The first time you launch the script, it will download the model weights.  
> This process can take **15–20 minutes**, depending on your hardware and internet connection.
<img width="516" height="69" alt="{297F19AF-995D-4A17-BBA1-B49F0CB68F36}" src="https://github.com/user-attachments/assets/c8b90dfb-954e-4306-9352-2afdbcf094f9" />

### Launch the Gradio web Agent from another device connected to same network

First, SSH into the web server and forward port **7860**:
```echo
ssh -L 7860:127.0.0.1:7860 ai1@pc1
```
or use the the server IP address
```echo
ssh -L 7860:127.0.0.1:7860 ai1@192.168.178.xxx
```
Now you can open **http://127.0.0.1:7860** in your local browser to access the Gradio Web Agent.

<img width="1375" height="986" alt="{231D8436-E76C-4EFA-B9FE-A77BCE0896B0}" src="https://github.com/user-attachments/assets/d1fc46b6-8184-4890-baef-6f8281642b4b" />
Prompt example (1280x768 ressolution, 24 iterations, bf16, black-forest-labs/FLUX.1-schnell):

```echo
A sprawling steampunk city with towering brass and iron skyscrapers, glowing steam vents, and intricate gears. Huge airships, zeppelins, and retro-futuristic spaceships float between buildings. Streets filled with mechanical details, glowing lanterns, and rail systems. Cinematic lighting, foggy atmosphere, ultra-realistic, epic perspective, industrial textures, golden brass tones.
```

At the right side, you can review the recorded picture. Below is the benchmark information, including TaFT, TOS, T/s, and the GPU memory usage for the 2× RADEON AI PRO R9700 GPUs.
<img width="1274" height="1045" alt="{E9D010FB-4B93-46B5-AEFA-A88F34355AFC}" src="https://github.com/user-attachments/assets/01f27fcd-8ade-4a3a-b496-413bc8fb09ec" />
