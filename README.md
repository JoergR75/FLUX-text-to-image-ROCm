# 🚀 FLUX Text-to-Image Benchmark on AMD ROCm  
**Optimized for Multi-GPU (Dual 32GB GPUs) · Supports bf16/fp16 · Hugging Face Models**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)  
[![ROCm](https://img.shields.io/badge/AMD-ROCm_6.x-red)](https://rocmdocs.amd.com/)  
[![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow)](https://huggingface.co/docs/diffusers)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 📌 Overview
This repository provides a **highly optimized FLUX text-to-image pipeline** for **AMD ROCm** with support for **multi-GPU acceleration**, **bf16/fp16 compute**, and **detailed benchmarking**.  

The script leverages **[Hugging Face Diffusers](https://huggingface.co/docs/diffusers)** and **FluxPipeline** for efficient, high-quality image generation on **RDNA3/4** and **CDNA2/3** GPUs.

---

## ✨ Features
- ⚡ **Optimized for AMD ROCm 6.x** (multi-GPU, expandable segments)
- 🖼️ **Text-to-Image Generation** using **FluxPipeline**
- 🧠 **bf16 / fp16 mixed precision** for better performance
- 🔀 **Balanced multi-GPU memory allocation**
- 🛠️ **Low VRAM mode** for GPUs with ≤16GB
- 📊 **Built-in benchmarks**:
  - **TTFT** → Time to First Token
  - **TaFT** → Total time
  - **ToD** → Diffusion time
  - **t/s** → Steps per second
  - **VRAM usage per GPU**

---

## 🛠️ Installation

### 1. Download the Python script
```bash
wget https://raw.githubusercontent.com/JoergR75/FLUX-text-to-image-ROCm/refs/heads/main/flux_rocm.py
```

### 2. Install dependencies
```bash
pip3 install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
pip3 install diffusers transformers huggingface_hub accelerate safetensors
```

## ⚡ Usage
Basic Example
```bash
python3 flux_rocm.py \
  --prompt "a hyperrealistic exploring spaceship between other smaller spaceships and a huge planet in space, cinematic" \
  --model black-forest-labs/FLUX.1-dev \
  --steps 10 \
  --width 1280 \
  --height 960 \
  --out spaceship_50.png \
  --hf-token hf_your_token_here
```

### **Key Parameters**

| Parameter     | Type   | Default                            | Description                         |
|--------------|--------|------------------------------------|-------------------------------------|
| `--prompt`   | str    | **required**                       | Text prompt for image generation   |
| `--model`    | str    | `black-forest-labs/FLUX.1-dev`     | Hugging Face model ID             |
| `--steps`    | int    | 24                                 | Number of inference steps         |
| `--guidance` | float  | 3.5                                | Classifier-free guidance scale    |
| `--width`    | int    | 1024                               | Output image width               |
| `--height`   | int    | 1024                               | Output image height              |
| `--dtype`    | str    | `bf16`                             | Compute precision (`bf16` or `fp16`) |
| `--low-vram` | flag   | disabled                           | Enable memory optimizations      |
| `--hf-token` | str    | `None`                             | Hugging Face access token        |
| `--out`      | str    | `flux_out.png`                     | Output image filename            |

## 🧠 Multi-GPU Optimization (ROCm)

This script automatically enables multi-GPU balanced memory allocation via PyTorch & Accelerate:
```bash
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ACCELERATE_USE_BALANCED_MEMORY"] = "1"
os.environ["ACCELERATE_USE_DEVICE_MAP"] = "balanced"
```

If you have 2× 32GB GPUs, the workload is split evenly.
For low-VRAM GPUs, you can enable sequential CPU offloading:
```bash
python3 flux_rocm.py --prompt "..." --low-vram
```

## 📊 Benchmark Metrics

| Metric | Description |
|--------|------------|
| **TTFT** | Time to First Token (pipeline warm-up + inference start) |
| **TaFT** | Total time from start to finish |
| **ToD**  | Diffusion-only time |
| **t/s**  | Inference steps per second |
| **VRAM** | Peak memory allocated & reserved per GPU |

---

### **Example Output**
```bash
[benchmark]
  TTFT (forward):   3.12 sec
  TaFT (total):     18.45 sec
  ToD  (diffuse):   15.88 sec
  t/s  (steps/s):   3.15
  GPU0 peak allocated: 15522.1 MB, reserved: 16200.4 MB
  GPU1 peak allocated: 15489.3 MB, reserved: 16187.6 MB
```
## 📌 Requirements

- **Python** ≥ 3.10  
- **ROCm** ≥ 6.0  
- **PyTorch ROCm** ≥ 2.3  
- **AMD GPU** (RDNA3/4 or CDNA2/3 recommended)  
- **Dual 32GB GPUs** recommended for 1280×960 high-quality generations  

---

## 📄 License

This project is licensed under the **MIT License**.  
See [LICENSE](LICENSE) for more details.

---

## 🧩 Roadmap

- [ ] Add **Flux Turbo** benchmarking  
- [ ] Support **LoRA fine-tuning**  
- [ ] WebUI integration for prompt testing  
- [ ] ONNX + AMD ROCm Graph Mode for speedup  

---

## 🤝 Contributing

Pull requests are welcome!  
If you'd like to add improvements, please open an **issue** first to discuss your ideas.

---

## 🌟 Acknowledgments

- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers)  
- [PyTorch ROCm](https://pytorch.org/)  
- [Black Forest Labs – FLUX Models](https://huggingface.co/black-forest-labs)
