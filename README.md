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
wget 
```
