PKC MARK Benchmark Tool Beta V0.9

⚠️ Important Notice
The user interface (UI), internal logs, and tags for this benchmark tool are all standardized in English.
This document describes the tool's features based on that English UI.

🚀 A public benchmark tool for easily measuring, comparing, and analyzing the performance of various AI models and LLMs in your local environment.

(This tool was originally created for my personal needs during the development of a local AI multi-modal chatbot system and has since been renewed for public release.)


✍️ Creator & Contact

If you have any questions, bug reports, or suggestions, please feel free to contact me.

Creator: PKC

Blog: https://pkc0412.tistory.com/

Email: pkc0412@gmail.com

Github: https://github.com/pkc0412-alt/pkc-mark-benchmark


📜 License

This project follows a dual-license policy:

Non-Profit & Open Source: For personal, academic, and non-profit projects, this tool is licensed under the GPLv3.

Commercial Use: If you wish to use this software for commercial purposes or in a closed-source product, a separate commercial license is required. Please inquire via the email above.


What You Can Do:

✅ Use it freely (for personal, educational, or research purposes)

(If you are the 'hero' who will refactor this monolith...)

✅ Analyze and improve the code

✅ Fork it to create a completely different project

✅ Redistribute it as open source (must retain original attribution and tool name)


⚠️ Reality Check

This project is NOT "perfect commercial software."

It might have issues in highly specific environments.

There is no 24/7 customer support (it's a personal project! 😊).

It's released in the spirit of open source, even if it's not perfect.

PKC MARK Benchmark Tool - Beta V0.9

1. Overview

The PKC MARK Benchmark Tool is a web-based utility designed to measure the performance of local AI models.

It allows you to test various local models like GGUF (Llama), Diffusers, and Transformers using identical prompts. You can measure, compare, and visualize various performance metrics in real-time, including VRAM usage, inference speed (TPS, TTFT), and GPU power/temperature.


2. Key Features

Intuitive Web UI: Control all settings and view real-time results via benchmark_canvas.html.

Automatic Model Detection: Automatically scans and classifies models within the folder specified in config.json (supports GGUF, Diffusers, Transformers).

Detailed Performance Metrics: Measures model load time, VRAM usage, Time to First Token (TTFT), Tokens Per Second (TPS), GPU power (W), and GPU temperature (°C).

Pipeline Testing: Allows for chain testing by automatically injecting the results from an analysis model (e.g., one tagged 'Emotion' or 'Analysis') into the prompt for a language model (LLM).

Result History & Comparison: All test results are saved in the browser (LocalStorage), allowing you to load past results and compare performance side-by-side.

Flexible Configuration: Provides various test options, including model caching, sequential/parallel loading, auto-retry on VRAM error, and setting the number of GPU layers.


3. Requirements

Python 3.11.9 (Recommended)

NVIDIA GPU (CUDA 12.1 environment recommended)

Required Python libraries (see installation)


4. Installation

Download all files from this repository.

(Recommended) Open a terminal, navigate to the project folder, and create a virtual environment:

python -m venv venv


Activate the new virtual environment (on Windows):

.\venv\Scripts\activate


(Important) Install PyTorch first. Check the official PyTorch website for the correct version matching your CUDA version.
(Example for CUDA 12.1):

pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)


Install the remaining libraries from requirements.txt:

pip install -r requirements.txt


5. ⚠ Mandatory Setup (Crucial)

Before running, you must edit the config.json file to set the path to your local models folder.

Open config.json in a text editor.

Change the value of models_scan_path to the actual folder path where your models are stored.

Example (Windows):

{
    "results_dir": "results",
    "models_scan_path": "C:/MyModels"
}


Example (Linux/Mac):

{
    "results_dir": "results",
    "models_scan_path": "/home/user/models"
}


(results_dir is the folder name where benchmark results will be saved as .json and .html files. Default is "results".)


6. How to Use

Start Server: Double-click start_server_windows.bat (or run python benchmark_server.py in your terminal).

Access UI: The server will automatically open the benchmark_canvas.html page in your default web browser. (Note: All UI text is in English.)

Configure: Use the web UI to select models, enter prompts, set GPU layers, caching, etc.

Run: Click the '🚀 Start Benchmark' button to begin the test.

Check Results: View real-time results in the 'Summary', 'Log', and 'Chart' tabs.

Compare: After the test, you can load past results from 'Result History' or select multiple sessions to compare them in the 'Comparison' tab.


7. Model Detection Rules

The tool scans subfolders within the models_scan_path and identifies models based on these rules:

Llama (GGUF): If a .gguf file is present, it's registered as llama type. (If an mmproj file is also present, it's recognized as a VLM (CLIP) model).

Diffusers: If no .gguf file is present but a model_index.json file exists, it's registered as diffusers (image generation) type.

Transformers: If neither of the above conditions is met but a config.json file exists, it's registered as transformers type. (If the folder name includes "kluebert" or "emotion", it's auto-categorized as 'analysis_model').


Email: pkc0412@gmail.com

Blog: https://pkc0412.tistory.com/

Github: https://github.com/pkc0412-alt/pkc-mark-benchmark