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

▶️ Usage
Run the web agent:
python3 flux_gradio_web_agent.py
Typical output:
Running on local URL:  http://127.0.0.1:7860
To create a public link, set `share=True` in `launch()`.
Then open your browser and go to:
👉 http://127.0.0.1:7860
💡 Example Prompt
Inside the web UI, enter a prompt such as:
a hyperrealistic exploring spaceship between other smaller spaceships and a huge planet in space, cinematic
The model will generate and display the result directly in the Gradio interface.
⚙️ Configuration
The script can be customized to suit your needs:
Model: defaults to black-forest-labs/FLUX.1-dev
Steps: number of denoising steps (default: 24)
Width / Height: image resolution (default: 1280×960)
Prompt / Negative prompt: adjustable in the UI
share=True: enable to get a public link
📊 Benchmarking
The web agent logs useful performance stats:
TaFT (Time a Frame Took)
ToD (Time on Device)
Throughput (tokens/sec)
VRAM usage per GPU
This is helpful for comparing performance across GPUs and environments.
🖼️ Screenshots & Examples
(Add your own generated images here)
Example:
![Example output](examples/spaceship.png)
