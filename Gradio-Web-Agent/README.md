# FLUX - Gradio Web Agent

A lightweight **Gradio-based web interface** for running the **FLUX text-to-image model** on AMD ROCm and CUDA systems.  
This project provides an easy-to-use web UI to generate AI images from text prompts, monitor VRAM usage, and optionally share the interface online.

---

## 🚀 Features

- 🌌 Text-to-image generation with the **FLUX.1** model  
- 🎛 Simple **Gradio web interface** for interactive use  
- 📊 Automatic logging of:
  - Inference speed (tokens/sec, steps/sec)
  - VRAM usage (per GPU)
- ⚡ ROCm and CUDA compatible (tested on AMD Instinct MI210, ROCm 6.4)  
- 🌍 Optional public link sharing with `share=True`  

---

## 📦 Requirements

- Python **3.9+**  
- PyTorch (with ROCm or CUDA support)  
- Hugging Face `diffusers` library  
- Additional packages: `transformers`, `accelerate`, `safetensors`, `gradio`, `psutil`

Install dependencies:

```bash
# Example for ROCm 6.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# Common dependencies
pip install diffusers transformers accelerate safetensors gradio psutil

