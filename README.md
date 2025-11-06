# pkc-mark-benchmark
A local AI benchmark tool for testing LLM, Diffusers, and Transformers models.

# 🧩 PKC MARK Benchmark Tool

🚀 A **public benchmarking tool** to easily **measure, compare, and analyze** the performance of local AI models (LLM, Diffusers, Transformers).
Originally built as part of a personal multimodal chatbot project, later renewed and released as open source.

---

## ✨ Key Features

* **Web-based UI** — Control all settings and view results in real time via `benchmark_canvas.html`.
* **Auto Model Detection** — Automatically scans and classifies models in the folder specified in `config.json` (supports GGUF, Diffusers, Transformers).
* **Detailed Metrics** — Measure VRAM usage, Time-To-First-Token (TTFT), Tokens Per Second (TPS), GPU power (W), and GPU temperature (°C).
* **Pipeline Testing** — Automatically injects outputs from analysis models (e.g., emotion analysis) into LLM prompts for integrated testing.
* **Result History & Comparison** — All benchmark results are saved to the browser (LocalStorage) and can be compared side-by-side.
* **Flexible Options** — Model caching, parallel loading, auto retry on low VRAM, adjustable GPU layers, and more.

![Mark_Benchmark](./images/PKC_Mark_Benchmark_01.jpg)


## ⚙️ Installation & Execution

1. **Clone or download** this repository.
2. **Create and activate** a virtual environment:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Install PyTorch** according to your CUDA version, then install other dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. **Edit `config.json`** — Set your local model path in the line:

   ```json
   {
       "results_dir": "results",
       "models_scan_path": "Enter your model path"
   }
   ```
5. **Run the server:**

   * Windows: `start_server_windows.bat`
   * Or run manually:

     ```bash
     python benchmark_server.py
     ```
6. Once the server starts, your browser will automatically open `benchmark_canvas.html`. Start your benchmark from there.

---

## 🧠 Model Detection Rules

* `.gguf` → **Llama (GGUF)**
* `model_index.json` → **Diffusers (image generation)**
* `config.json` → **Transformers (text-based)**
* Folder names containing `kluebert` or `emotion` → auto-classified as **analysis_model**

---

## 📜 License

* **Non-commercial / Open Source**: GPLv3
* **Commercial use**: Separate commercial license required (contact: [pkc0412@gmail.com](mailto:pkc0412@gmail.com))

---

## ✍️ Author

**PKC**

* Blog: [https://pkc0412.tistory.com/](https://pkc0412.tistory.com/)
* Email: [pkc0412@gmail.com](mailto:pkc0412@gmail.com)

---
