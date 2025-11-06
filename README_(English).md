# PKC MARK Benchmark Tool

🚀 A public benchmarking tool to **measure**, **compare**, and **analyze** the performance of local AI models (LLMs).
Built for real-world use, then refined for public release.

---

## ✍️ Author & Contact

**Author**: PKC
**Blog**: [https://pkc0412.tistory.com/]
**Email**: [pkc0412@gmail.com]

---

## 📜 License

This project uses a **dual license**.

* **Non‑commercial & Open Source** → **GPLv3** (personal / academic / non‑profit).
* **Commercial use** → **Commercial license required** (for commercial or closed‑source products).
  Contact via email for details.

### What You Can Do

* ✅ Use freely for personal, educational, and research purposes.
* ✅ Read and improve the code.
* ✅ Fork and build a different project.
* ✅ Redistribute as open source *(keep original author credit and tool name)*.

> If you’re brave enough to dissect this **monolith**… 😄

### Notes & Expectations

* This is **not** a perfect commercial product.
* Edge cases in very specific environments may cause issues.
* No 24/7 customer support *(personal project! 😊)*.
* Published with an **open‑source spirit**, imperfections included.

---

## PKC Benchmark Tool — MARK

### 1) Overview

A **web‑based** benchmarking tool for local AI models.
Test multiple backends with the **same prompt**, and visualize performance in real time.

**Supported model families**:

* **GGUF (Llama)**
* **Diffusers**
* **Transformers**

**Metrics tracked**:

* **VRAM usage**
* **Inference speed** — **TPS** (Tokens Per Second), **TTFT** (Time To First Token)
* **GPU power (W)**, **GPU temperature (°C)**

### 2) Key Features

* **Intuitive Web UI** → `benchmark_canvas.html` to control settings and watch results live.
* **Auto Model Detection** → scans folders from `config.json` *(GGUF, Diffusers, Transformers)*.
* **Detailed Metrics** → load time, VRAM, TTFT, TPS, GPU power/temperature.
* **Pipeline Testing** → feed analysis‑model output (e.g., emotion analysis) into LLM prompts automatically.
* **Result History & Comparison** → results saved to browser LocalStorage for side‑by‑side reviews.
* **Flexible Options** → caching, sequential/parallel load, auto‑retry on low VRAM, GPU layer count, and more.

### 3) Requirements

* **Python**: 3.11.9 *(recommended)*
* **GPU**: NVIDIA, **CUDA 12.1** *(recommended)*
* **Libraries**: see **Installation** below.

### 4) Installation

**1. Download** all files from this repository.
**2. Create & activate** a virtual environment.

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate
```

**3. Install PyTorch first** (match your CUDA version).
Example — **CUDA 12.1**:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**4. Install remaining dependencies**:

```bash
pip install -r requirements.txt
```

### 5) Essential Configuration (Most Important)

Edit `config.json` **before** running.
Set the **local folder path** that contains your models.

Open `config.json` and set `models_scan_path` to the correct directory.

**Windows example**

```json
{
  "results_dir": "results",
  "models_scan_path": "C:/MyModels"
}
```

**Linux/macOS example**

```json
{
  "results_dir": "results",
  "models_scan_path": "/home/user/models"
}
```

`results_dir` stores benchmark outputs (`.json`, `.html`).
In the `config.json` file, you **must enter your own local path** for the line:

````json
"models_scan_path": "Input user-specific folder path"
``` (`.json`, `.html`).  
Default: `results`.

### 6) Usage
**Run the server**  
- Double‑click `start_server_windows.bat`, **or**  
- Run manually:  
```bash
python benchmark_server.py
````

**Open the UI**

* Your default browser will open `benchmark_canvas.html` shortly after the server starts.
  **Configure settings**
* Choose model, prompt, GPU layers, caching, etc.
  **Start**
* Click **Benchmark Start**.
  **Monitor results**
* Use **Summary**, **Log**, **Chart** tabs for real‑time status.
  **Compare runs**
* Load past results in **Result History** or select multiple runs in **Comparison** for side‑by‑side charts.

### 7) Model Detection Rules

The tool scans all subfolders under `models_scan_path` and classifies them as follows:

* **Llama (GGUF)**
  If a folder contains `.gguf` → register as `llama`.
  If `mmproj` also exists → recognized as **VLM (CLIP)**.

* **Diffusers**
  If no `.gguf`, but `model_index.json` exists → register as `diffusers` (image generation).

* **Transformers**
  If neither applies and `config.json` exists → register as `transformers`.
  If the folder name contains `kluebert` or `emotion` → auto‑classify as `analysis_model`.

---

## Contact (Quick Reference)

**Email**: [pkc0412@gmail.com]
**Blog**: [https://pkc0412.tistory.com/]
