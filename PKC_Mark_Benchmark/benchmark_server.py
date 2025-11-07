#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---- NVML Manager (robust, safe to define early) ----
class NvmlManager:
    _initialized = False
    _refcount = 0

    @classmethod
    def _nvml_available(cls):
        # Defer checks to runtime to avoid import-order issues
        return bool(globals().get("NVIDIA_SMI_AVAILABLE", False))

    @classmethod
    def acquire(cls):
        if not cls._nvml_available():
            return False
        try:
            # functions looked up at runtime
            globals()["nvmlInit"]()
            cls._initialized = True
            cls._refcount += 1
            return True
        except Exception:
            # if already initialized or any transient error, keep graceful behavior
            try:
                cls._refcount += 1
            except Exception:
                pass
            return cls._initialized

    @classmethod
    def release(cls):
        if not cls._nvml_available():
            return
        try:
            if cls._refcount > 0:
                cls._refcount -= 1
            if cls._refcount == 0 and cls._initialized:
                globals()["nvmlShutdown"]()
                cls._initialized = False
        except Exception:
            # no-op on shutdown errors
            pass

import sys
import time
import gc
import json
import csv
import psutil
import platform
import asyncio
import traceback
import numpy as np
import io
import re
import os
from contextlib import redirect_stderr
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Library Imports ---
try:
    import torch
    import torch.nn.functional as F
    from llama_cpp import Llama
    from diffusers import DiffusionPipeline
    # [MODIFIED] Import classes for Q&A and CausalLM (generation) models
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, \
        AutoModelForQuestionAnswering, AutoModelForCausalLM 
    try:
        from pynvml import *
        NVIDIA_SMI_AVAILABLE = True
    except ImportError:
        NVIDIA_SMI_AVAILABLE = False
        print("⚠️ [Warning] pynvml library not found. GPU power and temperature measurement will be disabled.")

        print("   - Install: pip install pynvml (or nvidia-ml-py)")

except ImportError as e:
    print(f"❌ Failed to import required libraries: {e}")
    print("   - Example install: pip install uvicorn fastapi pydantic torch llama-cpp-python diffusers transformers psutil numpy pynvml")
    sys.exit(1)

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# --- Config File Loading Helper ---
def load_json_config(filename: str) -> Dict[str, Any]:
    """Safely loads a config file and provides detailed error info if it fails."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found: Please ensure '{filename}' is in the same directory as benchmark_server.py.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Config file format error: The JSON in '{filename}' is invalid. (e.g., trailing comma, missing brackets)")
        print(f"   - Error: {e.msg}")
        print(f"   - Location: Line {e.lineno}, Column {e.colno}")
        sys.exit(1)

# --- Automatic Model Scanning ---
def scan_models_directory(scan_path: str) -> Dict[str, Any]:
    """
    Recursively scans the specified directory (scan_path) to automatically configure models.
    
    Rules:
    1. (Llama/GGUF) If a folder contains a .gguf file, register it as 'llama' type.
       - .gguf or .mmproj files containing 'mmproj' are considered CLIP models.
    2. (Diffusers) If no .gguf and 'model_index.json' exists, register as 'diffusers' type.
    3. (Transformers) If neither of the above and 'config.json' exists, register as 'transformers' type.
    """
    print(f"🤖 Scanning for models: {scan_path}")
    models_found = {}
    
    if not os.path.isdir(scan_path):
        print(f"❌ Model scan path not found: {scan_path}")
        return {}

    for root, dirs, files in os.walk(scan_path, topdown=True, followlinks=True):
        if Path(root) == Path(scan_path):
            continue

        model_id = Path(root).name
        model_path = str(Path(root))
        
        parent_dir = str(Path(root).parent)
        if parent_dir != scan_path and Path(parent_dir).name in models_found:
            dirs.clear() # Stop scanning subdirectories
            continue

        # 1. GGUF (Llama)
        gguf_files = [f for f in files if f.endswith(".gguf")]
        if gguf_files:
            main_model_path = None
            clip_model_path = None
            
            for f in gguf_files:
                if "mmproj" in f.lower():
                    clip_model_path = os.path.join(root, f)
                else:
                    main_model_path = os.path.join(root, f)
            
            if not clip_model_path:
                mmproj_files = [f for f in files if f.endswith(".mmproj")]
                if mmproj_files:
                    clip_model_path = os.path.join(root, mmproj_files[0])

            if main_model_path:
                model_name = model_id.replace(".gguf", "")
                models_found[model_id] = {
                    "name": model_name,
                    "category": "language_model", # GGUF defaults to language model
                    "type": "llama",
                    "paths": {"model_path": main_model_path, "clip_model_path": clip_model_path}
                }
                print(f"  ✅ [Llama] Found: {model_name} (CLIP: {'Yes' if clip_model_path else 'No'})")
                dirs.clear() 
                continue

        # 2. Diffusers (model_index.json)
        if "model_index.json" in files:
            models_found[model_id] = {
                "name": model_id,
                "category": "image_model", # Diffusers is image model
                "type": "diffusers",
                "paths": {"model_path": model_path}
            }
            print(f"  ✅ [Diffusers] Found: {model_id}")
            dirs.clear() 
            continue

        # 3. Transformers (config.json)
        if "config.json" in files:
            category = "language_model"
            if "kluebert" in model_id.lower() or "emotion" in model_id.lower():
                category = "analysis_model"
            
            models_found[model_id] = {
                "name": model_id,
                "category": category,
                "type": "transformers",
                "paths": {"model_path": model_path}
            }
            print(f"  ✅ [Transformers] Found: {model_id} (Category: {category})")
            dirs.clear() 
            continue
            
    if not models_found:
        print(f"⚠️ Scan complete. No models found in {scan_path}.")
    else:
        print(f"🤖 Scan complete: Found {len(models_found)} models total.")
        
    return models_found

# --- Load Config Files ---
CONFIG = load_json_config("config.json")
MODELS = scan_models_directory(CONFIG.get("models_scan_path", "./models"))

# --- FastAPI App Setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}

# --- Global State Management ---
class BenchmarkParams(BaseModel):
    llm_max_tokens: int
    n_gpu_layers: int
    cache_models: bool
    sequential_loading: bool
    auto_retry: bool
    prompts: List[str]
    test_runs: int
    # [REMOVED] llm_language: str = 'ko' 
    repeat_count: int
    repeat_min_len: int
    random_seed: Optional[int] = None
    connect_pipeline: bool = True

class BenchmarkConfig(BaseModel):
    target_gpu: int
    benchmark_params: BenchmarkParams
    selected_models: List[str]
    run_simultaneous: bool
    
benchmark_status = {"running": False, "task": None, "cancelled": False}
model_cache: Dict[str, Any] = {}
log_queue: "asyncio.Queue[str]" = asyncio.Queue()
captured_llama_info: Dict[str, Any] = {}

# ================= Utility & GPU Info Functions =================
def get_gpu_info(handle):
    """Gets GPU info from a given NVML handle."""
    try:
        info = nvmlDeviceGetMemoryInfo(handle)
        power = nvmlDeviceGetPowerUsage(handle) / 1000.0
        temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        return {
            "used_gb": info.used / 1024**3,
            "total_gb": info.total / 1024**3,
            "power_w": power,
            "temp_c": temp,
        }
    except NVMLError as e:
        print(f"⚠️ NVML handle error: {e}")
        return {
            "used_gb": 0, "total_gb": 0, "power_w": 0, "temp_c": 0,
        }


def get_system_info(override_llama_info: Optional[Dict[str, Any]] = None):
    """Collects system hardware and software information."""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": f"{platform.processor()} ({psutil.cpu_count(logical=True)} Threads)",
        "ram_total_gb": psutil.virtual_memory().total / 1024**3,
        "gpus": [],
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cublas_enabled": torch.cuda.is_available(),
        "llama_cpp_info": (override_llama_info if override_llama_info else (captured_llama_info if captured_llama_info else {"status": "Available after model load"}))
    }
    
    if NVIDIA_SMI_AVAILABLE:
        ## NvmlManager used for safe, single init
        try:
            NvmlManager.acquire()
            
            device_count = nvmlDeviceGetCount()
            for i in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                info["gpus"].append({
                    "index": i,
                    "name": nvmlDeviceGetName(handle),
                    "vram_total_gb": nvmlDeviceGetMemoryInfo(handle).total / 1024**3,
                })
            
            NvmlManager.release()
                
        except NVMLError as e:
            print(f"NVML Error (get_system_info): {e}")
            info["gpus"] = []
    
    info["models"] = { mid: {"id": mid, "name": minfo["name"], "category": minfo.get("category", "unknown")} for mid, minfo in MODELS.items() if not mid.startswith('_') }
    return info

def clear_gpu_cache(device_index: int):
    """Clears the cache for the specified GPU device."""
    if torch.cuda.is_available():
        with torch.cuda.device(device_index):
            torch.cuda.empty_cache()
    gc.collect()

def parse_llama_cpp_info(output: str) -> Dict[str, Any]:
    """Parses llama_cpp stderr output to extract acceleration info."""
    info = {}
    patterns = {
        "AVX": r"avx\s*=\s*1", "AVX2": r"avx2\s*=\s*1", "AVX512": r"avx512\s*=\s*1",
        "FMA": r"fma\s*=\s*1", "F16C": r"f16c\s*=\s*1",
        "BLAS": r"blas\s*=\s*1", "CUBLAS": r"cublas\s*=\s*1", "CLBLAST": r"clblast\s*=\s*1",
        "METAL": r"metal\s*=\s*1",
        "ggml_backend": r"ggml_backend\s*=\s*(\w+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            if key == "ggml_backend":
                info[key] = match.group(1).upper()
            else:
                info[key] = True
    return info

# ================= Logger =================
async def send_log(log_type: str, data: Any):
    """Sends a log message to the async log queue."""
    await log_queue.put(json.dumps({"type": log_type, "data": data}, ensure_ascii=False, cls=NumpyJSONEncoder))

# ================= Benchmark Core Logic =================
class BenchmarkRunner:

    # ---- Classification label mapping helper ----
    def _map_classification_label(self, pred_id: int, cfg) -> str:
        """Maps a classification ID to a text label. Prefers config.id2label, falls back based on num_labels."""
        label = None
        id2label = getattr(cfg, "id2label", None)
        if isinstance(id2label, dict):
            label = id2label.get(str(pred_id))
            if label is None:
                try:
                    label = id2label.get(pred_id)
                except Exception:
                    label = None
        if isinstance(label, bytes):
            try:
                label = label.decode("utf-8", errors="ignore")
            except Exception:
                pass
        if isinstance(label, str):
            return label

        # Fallbacks by num_labels
        num_labels = getattr(cfg, "num_labels", None)
        if num_labels == 7:
            fallback7 = ["Fear","Surprise","Anger","Sadness","Neutral","Happiness","Disgust"]
            if 0 <= pred_id < 7:
                return fallback7[pred_id]
        if num_labels == 3:
            fallback3 = ["Negative","Neutral","Positive"]
            if 0 <= pred_id < 3:
                return fallback3[pred_id]

        return f"ID: {pred_id}"
        
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.params = config.benchmark_params
        self.device = f"cuda:{config.target_gpu}"
        self.results: Dict[str, Any] = {}
        self.system_info = {} 
        self.nvml_handle = None
        self.nvml_initialized = False
        self.captured_llama_info = {}

        if NVIDIA_SMI_AVAILABLE:
            try:
                nvmlInit()
                self.nvml_handle = nvmlDeviceGetHandleByIndex(self.config.target_gpu)
                self.nvml_initialized = True
                print(f"✅ NVML Initialized (GPU {self.config.target_gpu})")
            except NVMLError as e:
                print(f"⚠️ NVML Initialization Failed: {e}")
                self.nvml_handle = None
                self.nvml_initialized = False

    def cleanup_nvml(self):
        """Explicitly release NVML via manager."""
        if self.nvml_initialized and NVIDIA_SMI_AVAILABLE:
            NvmlManager.release()
            self.nvml_initialized = False
            print("✅ NVML released by NvmlManager")

    def __del__(self):
        """Destructor: attempts NVML cleanup."""
        self.cleanup_nvml()

    def _get_run_params(self) -> Dict[str, Any]:
        """Returns a dict of current run settings."""
        return {
            "run_mode": "Sequential" if self.params.sequential_loading else "Parallel",
            "cache_enabled": "✓" if self.params.cache_models else "✗",
            "connect_pipeline": "✓" if self.params.connect_pipeline else "✗",
            "llm_max_tokens": self.params.llm_max_tokens,
            "n_gpu_layers": self.params.n_gpu_layers,
            "repeat_count": self.params.repeat_count,
            "repeat_min_len": self.params.repeat_min_len,
            # [REMOVED] llm_language
        }

    def _print_config_transparency(self):
        """Prints benchmark settings to the terminal for transparency."""
        print("\n" + "="*60)
        print("📋 Benchmark Settings Transparency")
        print("="*60)
        
        print(f"🎯 Target GPU: {self.config.target_gpu}")
        print(f"📝 Number of Test Prompts: {len(self.params.prompts)}")
        print(f"🔄 Test Runs per Prompt: {self.params.test_runs}")
        
        print(f"🎛️ LLM Max Tokens: {self.params.llm_max_tokens}")
        print(f"🗄️ GPU Layers: {self.params.n_gpu_layers}")
        
        print(f"🔄 Repetition Count: {self.params.repeat_count}")
        print(f"📏 Repetition Length: {self.params.repeat_min_len}")
        
        print(f"💾 Model Caching: {'Enabled' if self.params.cache_models else 'Disabled'}")
        print(f"⚡ Sequential Loading: {'Enabled' if self.params.sequential_loading else 'Disabled'}")
        print(f"🔄 Auto Retry (VRAM): {'Enabled' if self.params.auto_retry else 'Disabled'}")
        print(f"🔗 Connection Pipeline: {'Enabled' if self.params.connect_pipeline else 'Disabled'}")
        print(f"🎲 Random Seed: {self.params.random_seed if self.params.random_seed else 'Random'}")
        
        print(f"🤖 Selected Models ({len(self.config.selected_models)}):")
        for i, model_id in enumerate(self.config.selected_models, 1):
            model_name = MODELS.get(model_id, {}).get('name', model_id)
            category = MODELS.get(model_id, {}).get('category', 'unknown')
            print(f"   {i}. {model_name} ({category})")
        
        print(f"💬 Test Prompt Preview:")
        for i, prompt in enumerate(self.params.prompts[:3], 1):  # Show max 3
            preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
            print(f"   {i}. {preview}")
        if len(self.params.prompts) > 3:
            print(f"   ... and {len(self.params.prompts) - 3} more.")
        
        print("="*60)
        print("✅ Settings confirmed. Starting benchmark.")
        print("="*60 + "\n")

    async def run(self):
        try:
            self._print_config_transparency()
            
            psutil.cpu_percent(interval=None)
            
            start_time = time.time()

            if not self.params.prompts:
                await send_log("error_report", {"model": "system", "summary": "No prompts provided.", "traceback": "At least one prompt must be provided from the UI."})
                return

            loading_mode = "Sequential Loading" if self.params.sequential_loading else "Parallel Loading"
            await send_log("log", f"✅ Benchmark Started (Target GPU: {self.config.target_gpu}, Prompts: {len(self.params.prompts)}, Mode: {loading_mode})")
            benchmark_status["cancelled"] = False

            for i, prompt in enumerate(self.params.prompts):
                if benchmark_status["cancelled"]: break
                await send_log("log", f"--- Prompt {i+1}/{len(self.params.prompts)} Test Start: \"{prompt[:80]}...\"")
                
                self.results[prompt] = {}
                
                if self.params.sequential_loading:
                    await self.run_sequential_tests(prompt)
                else:
                    await self.run_parallel_tests(prompt)
                
                if not self.params.cache_models:
                    for name in list(model_cache.keys()):
                        await self.unload_model(name)
            
            self.system_info = get_system_info(override_llama_info=self.captured_llama_info)

            end_time = time.time()
            total_duration = end_time - start_time
            self.results["_metadata"] = {"total_duration_s": total_duration}

            if benchmark_status["cancelled"]:
                print("\n🟡 Benchmark was stopped by the user.")
                await send_log("log", "🟡 Benchmark was stopped by the user.")
            else:
                self._save_results()
                print("\n✅ All benchmark tests complete!")
                await send_log("log", "✅ All benchmark tests complete!")
            
            await send_log("final_results", self.results)

        except asyncio.CancelledError:
            print("\n🟡 Benchmark task was cancelled externally.")
            await send_log("log", "🟡 Benchmark task was cancelled externally.")
            await send_log("final_results", self.results)
        except Exception as e:
            tb = traceback.format_exc()
            summary = f"Fatal error during benchmark run: {type(e).__name__}"
            print(f"\n❌ {summary}\n{tb}")
            await send_log("error_report", {"model": "System", "summary": summary, "traceback": tb})
            await send_log("log", f"❌ {summary}")
            await send_log("final_results", self.results)
        finally:
            for name in list(model_cache.keys()):
                await self.unload_model(name)
            self.cleanup_nvml()

    # --- Common: SYS block post-processing function ---
    def _clean_sys_tags(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        # Robust remove of <<SYS>>...<</SYS>> (case/space tolerant)
        pattern = re.compile(r"<<\s*SYS\s*>>.*?<<\s*/\s*SYS\s*>>", re.DOTALL | re.IGNORECASE)
        text = pattern.sub("", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    
    async def _run_tests(self, original_prompt: str, strategy: str = 'parallel'):
        """Unified orchestration for sequential/parallel modes."""
        analysis_models = [m for m in self.config.selected_models if MODELS.get(m, {}).get('category') == 'analysis_model']
        language_models = [m for m in self.config.selected_models if MODELS.get(m, {}).get('category') == 'language_model']
        other_models = [m for m in self.config.selected_models if MODELS.get(m, {}).get('category') not in ['language_model', 'analysis_model', 'image_model']]
        image_models = [m for m in self.config.selected_models if MODELS.get(m, {}).get('category') == 'image_model']

        system_instruction = "You are a helpful AI assistant."
        effective_prompt_for_llms = f"<<SYS>>\n{system_instruction}\n<</SYS>>\n\n[Request] {original_prompt}\n"
        connect_pipeline = getattr(self.params, 'connect_pipeline', True)

        # Stage 1: Analysis
        if analysis_models:
            await send_log("log", f"  -> 🧠 Stage 1: {'Sequential' if strategy=='sequential' else 'Parallel'} analysis with {len(analysis_models)} models")
            for model_id in analysis_models:
                if benchmark_status["cancelled"]: return
                success, result_data = await self._test_wrapper_parallel(model_id, original_prompt, original_prompt)
                if success and result_data:
                    analysis_result = result_data.get('output_text', 'No result').replace('Classification:', '').strip()
                    analysis_type = "Emotion" if "kluebert" in model_id or "emotion" in model_id else "Analysis"
                    if connect_pipeline:
                        effective_prompt_for_llms += f"[{analysis_type}] {analysis_result}\n"
                if strategy == 'sequential':
                    try:
                        await self.unload_model(model_id)
                        await send_log("log", f"    📤 Unloaded {MODELS.get(model_id, {}).get('name', model_id)}")
                    except Exception:
                        pass
    async def run_sequential_tests(self, original_prompt: str):
        return await self._run_tests(original_prompt, strategy='sequential')


    async def run_parallel_tests(self, original_prompt: str):
        return await self._run_tests(original_prompt, strategy='parallel')



    async def _test_wrapper_parallel(self, model_id: str, prompt: str, original_prompt: str) -> Tuple[bool, Optional[Dict]]:
        """Test wrapper for parallel/cached mode."""
        result = {"status": "❌ Failed", "prompt": original_prompt}
        model = None
        
        try:
            model_info = MODELS.get(model_id)
            if not model_info: raise ValueError(f"No definition for {model_id}")
            
            stage_started_at = datetime.now().isoformat()
            stage_type = model_info.get('category', 'other').replace('_model', '')
            await send_log('pipeline_stage_start', {'prompt': original_prompt, 'model': model_id, 'stage': stage_type, 't': stage_started_at})
            
            paths = model_info.get("paths", {})
            if not paths.get("model_path"):
                raise ValueError(f"Model {model_id} has no valid 'model_path'.")

            if model_id not in model_cache and self.nvml_handle:
                gpu_info_data = get_gpu_info(self.nvml_handle)
                available_vram_gb = gpu_info_data['total_gb'] - gpu_info_data['used_gb']
                estimated_vram_gb = self._estimate_vram(paths["model_path"], self.params.n_gpu_layers)
                if available_vram_gb < estimated_vram_gb:
                    raise MemoryError(f"Insufficient VRAM: Need ~{estimated_vram_gb:.2f}GB, Available {available_vram_gb:.2f}GB")

            if model_id in model_cache:
                model, load_time = model_cache[model_id], 0.0
            else:
                log_msg = f"  - 🔥 Loading {model_id}..."
                print(log_msg)
                await send_log("log", log_msg)
                model, load_time, new_llama_info = await self._load_model_with_retry(model_info, paths, self.params.n_gpu_layers)
                if new_llama_info:
                    await send_log("llama_info_update", new_llama_info)
                if self.params.cache_models:
                    model_cache[model_id] = model
            
            inference_results, system_metrics, status = await self._run_inference_suite(model_id, model, model_info, prompt)
            
            result.update({
                "status": status, 
                "load_time_s": load_time, 
                **inference_results, 
                **system_metrics,
                **self._get_run_params()
            })
            
            result['started_at'] = stage_started_at
            result['ended_at'] = datetime.now().isoformat()
            result['analysis_injected'] = (prompt != original_prompt and self.params.connect_pipeline and (model_info.get('category') == 'language_model'))
            if result.get('analysis_injected'):
                result['llm_input_preview'] = (prompt[:200] if isinstance(prompt, str) else str(prompt))
                await send_log('llm_prompt_enriched', {'prompt': original_prompt, 'model': model_id, 'preview': result['llm_input_preview']})
            
            self.results.setdefault(original_prompt, {}).setdefault('pipeline_trace', [])
            self.results[original_prompt]['pipeline_trace'].append({'stage':stage_type,'model':model_id,'t_start':result['started_at'],'t_end':result['ended_at'],'status': 'ok' if result.get('status','').startswith('✅') or result.get('status','').startswith('⚠️') else 'error'})
            await send_log('pipeline_stage_complete', {'prompt': original_prompt, 'model': model_id, 'stage': stage_type, 't': result['ended_at']})
            
            self.results[original_prompt][model_id] = result
            await send_log("result_update", {"model": model_id, "data": result})
            return True, result

        except Exception as e:
            tb = traceback.format_exc()
            summary = f"{type(e).__name__}: {e}"
            await send_log("error_report", {"model": model_id, "summary": summary, "traceback": tb})
            result["error"] = summary
            result.update(self._get_run_params())
            self.results[original_prompt][model_id] = result
            await send_log("result_update", {"model": model_id, "data": result})
            return False, None
        finally:
            if not self.params.cache_models:
                await self.unload_model(model_id, model)
    
    async def _load_model_with_retry(self, model_info: Dict[str, Any], paths: Dict[str, str], n_gpu_layers: int) -> Tuple[Any, float, Optional[Dict]]:
        """Tries to load a model, retrying with fewer GPU layers on VRAM error."""
        try:
            start_time = time.time()
            loader_func = getattr(self, f"_load_{model_info['type']}_model")
            model, new_llama_info = await asyncio.to_thread(loader_func, paths, n_gpu_layers)
            return model, time.time() - start_time, new_llama_info
        except (RuntimeError, MemoryError, ValueError) as e:
            if "out of memory" in str(e).lower() and self.params.auto_retry and n_gpu_layers != 0:
                await send_log("log", f"  - 📉 Out of VRAM. Retrying with n_gpu_layers({n_gpu_layers}) reduced...")
                clear_gpu_cache(self.config.target_gpu)
                new_layers = max(0, n_gpu_layers - 10) if n_gpu_layers > 0 else 30
                return await self._load_model_with_retry(model_info, paths, new_layers)
            else:
                raise e

    async def _run_inference_suite(self, model_id: str, model: Any, model_info: Dict[str, Any], prompt: str) -> Tuple[Dict, Dict, str]:
        """Runs inference test_runs times and returns average/system metrics."""
        all_results = []
        for i in range(self.params.test_runs):
            await send_log("log", f"    - run {i+1}/{self.params.test_runs}")
            runner_func = getattr(self, f"_run_{model_info['type']}_inference")
            res = await asyncio.to_thread(runner_func, model, prompt, model_info)
            all_results.append(res)
        
        final_output = all_results[-1].get("output_text", "")
        if model_info.get('category') == 'language_model' or (model_info.get('category') == 'analysis_model' and model_info.get('type') == 'transformers' and 'Generation:' in final_output):
            await send_log("conversation", {"prompt": prompt, "response": str(final_output), "model_name": model_info.get("name", model_id)})

        status = "✅ Success"
        if "[Repeat detected]" in final_output: status = "⚠️ Quality Warning"
        
        avg_results = {}
        metrics_to_avg = ['ttft_ms', 'tokens_per_second', 'inference_time_s']
        for metric in metrics_to_avg:
            values = [r[metric] for r in all_results if r.get(metric) is not None]
            if values:
                avg_results[metric] = float(np.mean(values))
            else:
                avg_results[metric] = None

        avg_results["output_text"] = final_output
        
        gpu_stats = get_gpu_info(self.nvml_handle) if self.nvml_handle else {}
        system_metrics = {
            "vram_usage_gb": gpu_stats.get('used_gb'),
            "gpu_power_w": gpu_stats.get('power_w'),
            "gpu_temp_c": gpu_stats.get('temp_c'),
            "cpu_util_percent": psutil.cpu_percent(interval=None)
        }
        
        return avg_results, system_metrics, status

    def _is_repeating(self, text: str, min_len: int, repeat_count: int) -> bool:
        """Simple repetition detection logic."""
        if len(text) < min_len * repeat_count: return False
        last_chunk = text[-min_len:]
        return bool(last_chunk.strip()) and text.count(last_chunk) >= repeat_count

    def _run_llama_inference(self, model, prompt, model_info):
        """Run Llama (GGUF) inference, with streaming and repetition detection."""
        is_analysis_model = model_info.get('category') == 'analysis_model'

        if is_analysis_model:
            # Analysis model: no streaming
            start = time.time()
            completion = model(
                prompt,
                max_tokens=self.params.llm_max_tokens,
                stream=False,
                repeat_penalty=1.2
            )
            elapsed = time.time() - start
            full_response = ""
            if 'choices' in completion and len(completion['choices']) > 0:
                choice = completion['choices'][0]
                if 'text' in choice and choice['text']:
                    full_response = choice['text']
            
            clean = self._clean_sys_tags(full_response)
            return {"inference_time_s": elapsed, "output_text": clean, "ttft_ms": None, "tokens_per_second": None}
        else:
            # Language model: streaming
            
            chat_template_prompt = prompt
            
            start = time.time()
            stop_sequences = ["<|im_end|>", "</s>", "<|endoftext|>", "User:", "Assistant:", "[Request]", "<<SYS>>", "<</SYS>>", "[/INST]", "SYSTEM:"] 
            stream = model(
                chat_template_prompt,
                max_tokens=self.params.llm_max_tokens, 
                stream=True, 
                stop=stop_sequences, 
                repeat_penalty=1.2,
                seed=self.params.random_seed if self.params.random_seed is not None else -1
            )
            
            first_chunk_time, full_response = None, ""
            completion_tokens = 0
            for chunk in stream:
                if first_chunk_time is None: first_chunk_time = time.time()
                token = chunk['choices'][0].get('text', '')
                if token is None: continue

                completion_tokens += 1
                full_response += token
                
                if self.params.repeat_count > 1 and completion_tokens % 10 == 0:
                    if self._is_repeating(
                        full_response, 
                        min_len=self.params.repeat_min_len, 
                        repeat_count=self.params.repeat_count
                    ):
                        repeating_chunk = full_response[-self.params.repeat_min_len:]
                        full_response += f" [Repeat detected - chunk: '{repeating_chunk.strip()[:20]}...']"
                        break
            
            elapsed = time.time() - start
            ttft = (first_chunk_time - start) * 1000 if first_chunk_time else 0
            
            time_to_generate = (elapsed - (first_chunk_time - start)) if first_chunk_time else 0
            tps = ((completion_tokens - 1) / time_to_generate) if completion_tokens > 1 and time_to_generate > 0 else 0
            
            clean = self._clean_sys_tags(full_response)
            return {"ttft_ms": ttft, "tokens_per_second": tps, "output_text": clean, "inference_time_s": elapsed}

    def _run_diffusers_inference(self, model, prompt, model_info):
        start = time.time()
        _ = model(prompt, num_inference_steps=8).images[0]
        return {"inference_time_s": time.time() - start, "output_text": f"Image generated: '{prompt}'", "ttft_ms": None, "tokens_per_second": None}

    def _run_transformers_inference(self, model_tuple, prompt, model_info):
        """Run inference for Transformers models (classification, Q&A, generation)."""
        tokenizer, model_instance = model_tuple
        task_type = getattr(model_instance, 'task_type', 'classification') 
        
        start = time.time()
        
        if task_type == "classification":
            # --- 1. Classification ---
            
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model_instance.device)
            with torch.no_grad(): 
                outputs = model_instance(**inputs)
            
            probabilities = F.softmax(outputs.logits, dim=1)[0]
            pred_id = torch.argmax(probabilities).item()
            
            label = self._map_classification_label(pred_id, model_instance.config)

            confidence = probabilities[pred_id].item() * 100
            output_text = f"Classification: {label} ({confidence:.2f}%)"
            prompt_template_used = None
            
        elif task_type == "question_answering":
            parts = prompt.split("Context:", 1)
            question = parts[0].replace("Question:", "").strip()
            context = parts[1].strip() if len(parts) > 1 else ""

            if not question or not context:
                output_text = f"⚠️ Q&A Format Error: Expected 'Question: [Q] Context: [C]'"
            else:
                inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True).to(model_instance.device)
                with torch.no_grad():
                    outputs = model_instance(**inputs)
                
                answer_start_index = outputs.start_logits.argmax()
                answer_end_index = outputs.end_logits.argmax() + 1
                
                input_ids = inputs["input_ids"][0]
                answer_tokens = input_ids[answer_start_index:answer_end_index]
                answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                output_text = f"Answer: {answer}"
            prompt_template_used = None

        elif task_type == "generation":
            # --- 3. Generation (Causal LM) ---
            
            # [REMOVED] Language forcing logic
            concat_text = prompt 
            
            try:
                input_ids = tokenizer(concat_text, return_tensors="pt").to(model_instance.device)
                prompt_template_used = "concat_fallback"
            except Exception:
                input_ids = tokenizer(concat_text, return_tensors="pt").to(model_instance.device)
                prompt_template_used = "concat_fallback"

            generated_ids = model_instance.generate(
                input_ids["input_ids"],
                max_new_tokens=self.params.llm_max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=getattr(tokenizer, "eos_token_id", None) or getattr(tokenizer, "pad_token_id", None),
            )

            input_len = input_ids["input_ids"].shape[1]
            decoded = tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=True)
            
            output_text = f"Generation: {self._clean_sys_tags(decoded.strip())}"
        else:
             output_text = f"❌ Unsupported transformers task type: '{task_type}'"
             prompt_template_used = None

        inference_time = time.time() - start
        
        result = {"inference_time_s": inference_time, "output_text": output_text, "ttft_ms": None, "tokens_per_second": None}
        if prompt_template_used:
            result["_prompt_template_used"] = prompt_template_used
        return result

    async def unload_model(self, model_id: str, model_instance: Any = None):
        """Unloads a model from cache and clears GPU cache."""
        if model_id in model_cache: del model_cache[model_id]
        if model_instance: del model_instance
        clear_gpu_cache(self.config.target_gpu)

    def _save_results(self):
        """Saves benchmark results to JSON and HTML files."""
        results_dir = CONFIG.get("results_dir", "results")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path(results_dir).mkdir(exist_ok=True)
        json_path = Path(results_dir) / f"benchmark_{ts}.json"
        html_path = Path(results_dir) / f"benchmark_{ts}.html"

        full_results = { "results": self.results, "config": self.config.model_dump(), "system_info": self.system_info }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)

        def _esc(x):
            if x is None:
                return ""
            return (str(x)
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))

        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<!doctype html><html><head><meta charset='utf-8'>")
            f.write("<title>PKC Benchmark Tool - MARK (Pipeline Trace Enabled)</title>")
            f.write("<style>")
            f.write("body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Noto Sans,sans-serif;padding:16px;background:#0b1220;color:#e5e7eb;}")
            f.write("h2{margin:0 0 14px;font-size:22px;color:#a5b4fc;} h3{margin:26px 0 10px;font-size:16px;color:#86efac;}")
            f.write(".meta{font-size:12px;color:#9ca3af;margin-bottom:12px;}")
            f.write("table{border-collapse:collapse;width:100%;margin:10px 0 26px;background:#0f172a;font-size:11px;} th,td{border:1px solid #334155;padding:6px 8px;vertical-align:top;text-align:left;} th{background:#111827;color:#e5e7eb;position:sticky;top:0;}")
            f.write(".ok{color:#34d399;} .warn{color:#fbbf24;} .fail{color:#f87171;}")
            f.write(".out{white-space:pre-wrap;line-height:1.6;padding:10px 12px;background:#111827;border:1px solid #374151;border-radius:8px;max-height:100px;overflow:auto;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace;font-size:13px;word-break:break-word;}")
            f.write("</style></head><body>")
            f.write("<h2>PKC Benchmark Tool - MARK</h2>")

            try:
                cfg = self.config.model_dump()
                f.write("<div class='meta'>")
                total_duration_s = self.results.get("_metadata", {}).get("total_duration_s")
                if total_duration_s is not None:
                    m = int(total_duration_s // 60)
                    s = int(total_duration_s % 60)
                    total_duration_str = f"{m}m {s}s" if m > 0 else f"{s}s"
                    f.write(f"Total Test Time: {total_duration_str} | ")
                f.write(f"Target GPU: {_esc(cfg.get('target_gpu'))} | ")
                bp = cfg.get('benchmark_params', {})
                f.write(f"Max Tokens: {_esc(bp.get('llm_max_tokens'))}, n_gpu_layers: {_esc(bp.get('n_gpu_layers'))}, ")
                f.write(f"Runs: {_esc(bp.get('test_runs'))}, ")
                f.write(f"Repeat: {_esc(bp.get('repeat_count'))}@{_esc(bp.get('repeat_min_len'))}")
                f.write("</div>")
            except Exception:
                pass

            llama_info_dict = self.system_info.get("llama_cpp_info", {}) if isinstance(self.system_info, dict) else {}
            backend = llama_info_dict.get('ggml_backend', 'N/A')
            enabled_flags = sorted([key for key, val in llama_info_dict.items() if val is True and key != 'ggml_backend'])
            accel_str = f"Backend: {backend}, Flags: {', '.join(enabled_flags)}" if (llama_info_dict and backend != 'N/A') else "N/A"

            for prompt, models_data in self.results.items():
                if str(prompt).startswith('_'):
                    continue
                f.write(f"<h3>Prompt: {_esc(prompt)}</h3>")
                f.write("<table><thead><tr>")
                headers = [
                    "Model", "Status", "Output", "Load(s)", "VRAM(GB)", "Power(W)", "Temp(C)", "CPU(%)",
                    "TTFT(ms)", "TPS", "Inference(s)",
                    "Mode", "Cache", "Pipeline", "MaxTokens", "GPULayers", "RepeatCnt", "RepeatLen",
                    "Error", "Llama.cpp Accel"
                ]
                for h in headers:
                    f.write(f"<th>{_esc(h)}</th>")
                f.write("</tr></thead><tbody>")

                for model_id, data in models_data.items():
                    if not isinstance(data, dict): continue
                    model_name = MODELS.get(model_id, {}).get("name", model_id)
                    status = data.get("status", "") or ""
                    status_class = "ok" if str(status).startswith("✅") else ("warn" if str(status).startswith("⚠️") else "fail")

                    def fmt(v, fmt_str):
                        try:
                            if v is None or v == "":
                                return ""
                            return fmt_str.format(float(v))
                        except (ValueError, TypeError):
                            return _esc(v)

                    output_block = f"<pre class='out'>{_esc(data.get('output_text',''))}</pre>"

                    row = [
                        _esc(model_name),
                        f"<span class='{status_class}'>{_esc(status)}</span>",
                        output_block,
                        fmt(data.get("load_time_s"), "{:.2f}"),
                        fmt(data.get("vram_usage_gb"), "{:.2f}"),
                        fmt(data.get("gpu_power_w"), "{:.1f}"),
                        fmt(data.get("gpu_temp_c"), "{:.1f}"),
                        fmt(data.get("cpu_util_percent"), "{:.1f}"),
                        fmt(data.get("ttft_ms"), "{:.1f}"),
                        fmt(data.get("tokens_per_second"), "{:.2f}"),
                        fmt(data.get("inference_time_s"), "{:.4f}"),
                        fmt(data.get("run_mode"), "{}"),
                        fmt(data.get("cache_enabled"), "{}"),
                        fmt(data.get("connect_pipeline"), "{}"),
                        fmt(data.get("llm_max_tokens"), "{}"),
                        fmt(data.get("n_gpu_layers"), "{}"),
                        fmt(data.get("repeat_count"), "{}"),
                        fmt(data.get("repeat_min_len"), "{}"),
                        _esc(data.get("error", "")),
                        _esc(accel_str if "llama" in MODELS.get(model_id, {}).get("type", "") else "N/A")
                    ]
                    f.write("<tr>" + "".join([f"<td>{cell}</td>" for cell in row]) + "</tr>")

                f.write("</tbody></table>")

            f.write("</body></html>")


    def _estimate_vram(self, model_path: str, n_gpu_layers: int) -> float:
        """Estimate VRAM usage based on model path and GPU layers."""
        try:
            file_size_gb = Path(model_path).stat().st_size / 1024**3
        except FileNotFoundError:
            file_size_gb = 5.0
            
        if n_gpu_layers == -1: return file_size_gb * 1.2
        if n_gpu_layers == 0: return 0.5
        return file_size_gb * (n_gpu_layers / 35.0) * 1.1

    def _load_llama_model(self, paths: Dict[str, str], n_gpu_layers: int) -> Tuple[Any, Optional[Dict]]:
        """Load Llama (GGUF) model."""
        global captured_llama_info  # kept for backward-compat; instance field is source of truth
        model_path = paths.get("model_path")
        clip_path = paths.get("clip_model_path")
        
        if not model_path:
            raise ValueError("Llama model_path is missing.")

        params = {
            "model_path": model_path, 
            "n_ctx": 4096, 
            "n_gpu_layers": n_gpu_layers, 
            "verbose": True
        }
        if clip_path:
            params["clip_model_path"] = clip_path
            params["n_ctx"] = 2048
        
        f = io.StringIO()
        with redirect_stderr(f): 
            model = Llama(**params)
        output = f.getvalue()
        
        newly_captured_info = parse_llama_cpp_info(output)
        if newly_captured_info:
            self.captured_llama_info = newly_captured_info
            captured_llama_info = newly_captured_info
            
        return model, newly_captured_info

    def _load_diffusers_model(self, paths: Dict[str, str], n_gpu_layers: int):
        """Load Diffusers (image generation) model."""
        model_path = paths.get("model_path")
        if not model_path:
            raise ValueError("Diffusers model_path is missing.")
            
        pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        return pipeline.to(self.device), None

    def _load_transformers_model(self, paths: Dict[str, str], n_gpu_layers: int):
        """Load Transformers (classification, Q&A, generation) model."""
        model_path = paths.get("model_path")
        if not model_path:
            raise ValueError("Transformers model_path is missing.")
            
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        
        if config.architectures and "ForSequenceClassification" in config.architectures[0]:
            model_class = AutoModelForSequenceClassification
            task_type = "classification"
        elif config.architectures and ("ForQuestionAnswering" in config.architectures[0] or "QA" in config.architectures[0]):
            model_class = AutoModelForQuestionAnswering
            task_type = "question_answering"
        elif config.is_decoder or (config.architectures and ("ForCausalLM" in config.architectures[0] or "GPT" in config.architectures[0] or "Llama" in config.architectures[0])):
            model_class = AutoModelForCausalLM
            task_type = "generation"
        else:
            model_class = AutoModelForSequenceClassification
            task_type = "classification"
            print(f"⚠️ [Warning] Could not infer task_type for model {Path(model_path).name}. Defaulting to Classification.")
            
        model = model_class.from_pretrained(model_path).to(self.device)
        model.task_type = task_type 
        return (tokenizer, model), None

# ================= FastAPI Endpoints =================
@app.get("/")
async def read_root(): return {"message": "PKC Benchmark Tool - MARK"}

@app.get("/api/system-info")
async def api_system_info(): 
    """Return system info (with safe NVML logic)."""
    return JSONResponse(content=get_system_info())

@app.post("/api/run-benchmark")
async def run_benchmark_endpoint(config: BenchmarkConfig):
    if benchmark_status["running"]: raise HTTPException(status_code=409, detail="A benchmark is already running.")
    
    async def task_wrapper():
        benchmark_status["running"] = True
        await BenchmarkRunner(config).run()
        benchmark_status["running"] = False; benchmark_status["task"] = None
        
    benchmark_status["task"] = asyncio.create_task(task_wrapper())
    return JSONResponse(content={"message": "Benchmark started."})

@app.post("/api/cancel")
async def cancel_benchmark():
    if not benchmark_status["running"]: return JSONResponse(status_code=400, content={"message": "No benchmark is currently running."})
    benchmark_status["cancelled"] = True
    if benchmark_status["task"]: benchmark_status["task"].cancel()
    return JSONResponse(content={"message": "Benchmark cancellation requested."})

@app.post("/api/clear-cache")
async def clear_cache_endpoint():
    global captured_llama_info  # kept for backward-compat; instance field is source of truth; captured_llama_info = {}; model_cache.clear(); gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    await send_log("log", "🧠 Model cache cleared.")
    return JSONResponse(content={"message": "Model cache cleared."})

@app.get("/api/stream")
async def stream_logs(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected(): break
            try:
                item = await asyncio.wait_for(log_queue.get(), timeout=1.0)
                yield f"data: {item}\n\n"
            except asyncio.TimeoutError: continue
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    print("="*53)
    print("      PKC Benchmark Tool - MARK Server Start     ")
    print("="*53)
    print(f"- UI Access: http://127.0.0.1:8000 (or open benchmark_canvas.html)")
    print(f"- Model Scan Path: {CONFIG.get('models_scan_path', 'N/A')}")
    print(f"- Found {len(MODELS)} models.")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)