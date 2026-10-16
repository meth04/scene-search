"""Optional, direct local text/video embeddings. No metadata or other encoders."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
DIMENSION = 768
FORMAT = "driveretriever-embeddinggemma2-v1"
QUERY_PREFIX = "task: search result | query: "
CACHE = ROOT / "data/model-cache/embeddinggemma2"


class EmbeddingGemma2Error(RuntimeError):
    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(code + (": " + detail if detail else ""))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalized(value, dimension=DIMENSION):
    vectors = np.asarray(value, dtype=np.float32)
    if vectors.ndim not in (1, 2) or vectors.shape[-1] != dimension:
        raise EmbeddingGemma2Error("EG2_DIMENSION_MISMATCH", str(vectors.shape))
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or (norms <= 0).any():
        raise EmbeddingGemma2Error("EG2_NONFINITE_OR_ZERO_EMBEDDING")
    result = vectors / norms
    if not np.isfinite(result).all():
        raise EmbeddingGemma2Error("EG2_NONFINITE_EMBEDDING")
    return result


def safe_dtype(torch):
    # Recent PyTorch reports emulated BF16 as supported on Turing. Use native support.
    try:
        supported = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        supported = torch.cuda.is_bf16_supported()
    return torch.bfloat16 if supported else torch.float32


def validate_config(config):
    required = {"repo": REPO, "revision": REVISION, "dimension": DIMENSION,
                "normalize": True, "query_prefix": QUERY_PREFIX, "audio_config": None}
    for key, expected in required.items():
        if key not in config or config[key] != expected:
            raise EmbeddingGemma2Error("EG2_CONFIG_MISMATCH", key)
    if config.get("dtype") not in ("float32", "bfloat16"):
        raise EmbeddingGemma2Error("EG2_UNSAFE_DTYPE")
    representation = config.get("representation")
    if representation not in ("EG2-A", "EG2-B", "EG2-C"):
        raise EmbeddingGemma2Error("EG2_CONFIG_MISMATCH", "representation")
    if config.get("vision_tokens") not in (70, 140):
        raise EmbeddingGemma2Error("EG2_CONFIG_MISMATCH", "vision_tokens")
    expected = {"EG2-A": ("native", 5), "EG2-B": ("frames", 5), "EG2-C": ("frames", 10)}[representation]
    if (config.get("input_kind"), config.get("frame_count")) != expected or config.get("fps") != 1:
        raise EmbeddingGemma2Error("EG2_CONFIG_MISMATCH", "sampling")
    if config.get("sampling") != "uniform_endpoints" or config.get("per_clip_fallback") is not False:
        raise EmbeddingGemma2Error("EG2_CONFIG_MISMATCH", "sampling policy")
    return config


def frame_indices(total, count):
    if total < count or count < 1:
        raise EmbeddingGemma2Error("EG2_INSUFFICIENT_FRAMES")
    return np.linspace(0, total - 1, count, dtype=np.int64).tolist()


def decode_frames(path, count):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or fps <= 0 or total <= 0 or abs(total / fps - 5) > .15:
            raise EmbeddingGemma2Error("EG2_WINDOW_DURATION_MISMATCH", str(path))
        indices = frame_indices(total, count)
        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise EmbeddingGemma2Error("EG2_VIDEO_DECODE_FAILED", f"{path}: frame {index}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.stack(frames), [index / fps for index in indices]
    finally:
        cap.release()


def cached_model_path():
    path = CACHE / "models--google--embeddinggemma-2/snapshots" / REVISION
    if not all((path / name).is_file() and (path / name).stat().st_size > 0
               for name in ("model.safetensors", "config.json", "tokenizer.json", "modules.json")):
        raise EmbeddingGemma2Error("EG2_MODEL_NOT_CACHED", "Run scripts/prepare_embeddinggemma2.py and cache the pinned model first")
    return path


def worker_python_path():
    python = Path(os.environ.get("EMBEDDINGGEMMA2_PYTHON", ROOT / ".venv_embeddinggemma2/Scripts/python.exe"))
    if not python.is_file():
        raise EmbeddingGemma2Error("EG2_DEPENDENCY_MISSING", "Isolated EmbeddingGemma2 Python is not installed")
    return python


def runtime_readiness():
    cached_model_path()
    python = worker_python_path()
    try:
        result = subprocess.run([str(python), "-B", "-c",
            "import importlib.util; assert all(importlib.util.find_spec(m) is not None for m in ['sentence_transformers','transformers','torch'])"],
            capture_output=True, text=True, timeout=20, creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        if result.returncode:
            raise EmbeddingGemma2Error("EG2_DEPENDENCY_MISSING", result.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EmbeddingGemma2Error("EG2_DEPENDENCY_MISSING", repr(exc)) from exc
    return {"ready": True, "revision": REVISION, "dimension": DIMENSION, "model_cached": True, "offline": True}


class EmbeddingGemma2Encoder:
    """Lives in the isolated worker; imports optional frameworks only when loaded."""
    def __init__(self, config, *, model_factory=None):
        self.config = validate_config(config)
        self.model_factory = model_factory
        self.model = None
        self.load_count = 0
        self.metrics = {}
        self.lock = threading.RLock()

    def load(self):
        with self.lock:
            if self.model is not None:
                return self
            try:
                import psutil
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingGemma2Error("EG2_DEPENDENCY_MISSING", str(exc)) from exc
            if not torch.cuda.is_available():
                raise EmbeddingGemma2Error("EG2_CUDA_UNAVAILABLE")
            dtype = safe_dtype(torch)
            if str(dtype).removeprefix("torch.") != self.config["dtype"]:
                raise EmbeddingGemma2Error("EG2_RUNTIME_DTYPE_MISMATCH")
            torch.set_num_threads(4)
            torch.cuda.reset_peak_memory_stats()
            process = psutil.Process()
            ram_before = process.memory_info().rss
            started = time.perf_counter()
            factory = self.model_factory or SentenceTransformer
            try:
                self.model = factory(str(cached_model_path()), device="cuda", local_files_only=True,
                    model_kwargs={"torch_dtype": dtype}, config_kwargs={"audio_config": None})
                self.model.eval()
                torch.cuda.synchronize()
            except Exception as exc:
                self.model = None
                raise EmbeddingGemma2Error("EG2_CUDA_LOAD_FAILED", repr(exc)) from exc
            if self.model.prompts.get("SearchQuery") != QUERY_PREFIX:
                self.close()
                raise EmbeddingGemma2Error("EG2_OFFICIAL_PROMPT_MISMATCH")
            if self.model.get_sentence_embedding_dimension() != DIMENSION:
                self.close()
                raise EmbeddingGemma2Error("EG2_DIMENSION_MISMATCH")
            names = [name for name, _ in self.model.named_parameters()]
            if any("audio" in name for name in names):
                self.close()
                raise EmbeddingGemma2Error("EG2_AUDIO_ENCODER_LOADED")
            if any(parameter.dtype == torch.float16 for parameter in self.model.parameters()):
                self.close()
                raise EmbeddingGemma2Error("EG2_FP16_PROHIBITED")
            if any(parameter.device.type != "cuda" for parameter in self.model.parameters()):
                self.close()
                raise EmbeddingGemma2Error("EG2_CUDA_NOT_USED")
            self.load_count += 1
            properties = torch.cuda.get_device_properties(0)
            self.metrics = {"GPU": properties.name, "VRAM_total_bytes": properties.total_memory,
                "CUDA": torch.version.cuda, "PyTorch": torch.__version__, "dtype": self.config["dtype"],
                "BF16_supported": bool(dtype == torch.bfloat16),
                "BF16_including_emulation": torch.cuda.is_bf16_supported(), "audio_encoder_loaded": False,
                "cold_load_seconds": time.perf_counter() - started, "model_load_RAM_bytes": process.memory_info().rss,
                "model_load_RAM_delta_bytes": process.memory_info().rss - ram_before,
                "model_load_VRAM_bytes": torch.cuda.memory_allocated(), "model_load_peak_VRAM_bytes": torch.cuda.max_memory_allocated(),
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "modules": [type(m).__name__ for m in self.model], "load_count": self.load_count}
            return self

    def _encode(self, inputs, **kwargs):
        with self.lock:
            self.load()
            import psutil
            import torch
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    result = self.model.encode(inputs, batch_size=1, show_progress_bar=False,
                        normalize_embeddings=True, convert_to_numpy=True, **kwargs)
                torch.cuda.synchronize()
                vector = normalized(result).reshape(-1)
            except Exception as exc:
                raise EmbeddingGemma2Error("EG2_ENCODE_FAILED", repr(exc)) from exc
            return vector, {"seconds": time.perf_counter() - started,
                "peak_VRAM_bytes": torch.cuda.max_memory_allocated(), "RAM_bytes": psutil.Process().memory_info().rss,
                "norm": float(np.linalg.norm(vector)), "finite": True}

    def encode_text(self, query):
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ValueError("INVALID_QUERY")
        return self._encode(query, prompt_name="SearchQuery")

    def encode_video(self, path, config=None):
        config = validate_config(config or self.config)
        for key in ("revision", "dtype", "dimension", "query_prefix"):
            if config[key] != self.config[key]:
                raise EmbeddingGemma2Error("EG2_CONFIG_MISMATCH", key)
        started = time.perf_counter()
        if config["input_kind"] == "native":
            item = {"video": str(path)}
            timestamps = [0., 1., 2., 3., 4.]
            processing = {"video": {"fps": 1, "max_frames": 5, "max_soft_tokens": config["vision_tokens"]}}
        else:
            frames, timestamps = decode_frames(path, config["frame_count"])
            item = {"video": frames}
            processing = {"video": {"do_sample_frames": False, "max_soft_tokens": config["vision_tokens"]}}
        decode = time.perf_counter() - started
        vector, metrics = self._encode(item, processing_kwargs=processing)
        metrics.update(decode_seconds=decode, total_seconds=time.perf_counter() - started, timestamps=timestamps,
                       input_kind=config["input_kind"], representation=config["representation"])
        return vector, metrics

    def close(self):
        self.model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class EmbeddingGemma2Worker:
    """One persistent optional model process; bounded requests and offline loading."""
    def __init__(self, config, *, timeout=180):
        self.config = validate_config(config)
        self.timeout = timeout
        self.process = None
        self.lock = threading.RLock()
        self.sequence = 0

    def load(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return self
            started = time.perf_counter()
            python = worker_python_path()
            cached_model_path()
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
            log = ROOT / "outputs/embeddinggemma2/logs/worker.stderr.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as stderr:
                self.process = subprocess.Popen([str(python), "-B", "-m", "hybrid_v2.embeddinggemma2_worker"],
                    cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                    text=True, encoding="utf-8", bufsize=1,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                self.metrics = self.request("load", config=self.config)
                self.metrics["cold_worker_start_seconds"] = time.perf_counter() - started
            except Exception:
                self.close()
                raise
            return self

    def request(self, operation, **values):
        import queue
        with self.lock:
            if self.process is None:
                self.load()
            self.sequence += 1
            request_id = self.sequence
            self.process.stdin.write(json.dumps({"id": request_id, "operation": operation, **values}) + "\n")
            self.process.stdin.flush()
            output = queue.Queue(maxsize=1)
            def receive():
                try:
                    output.put(self.process.stdout.readline())
                except Exception as exc:
                    output.put(exc)
            threading.Thread(target=receive, daemon=True).start()
            try:
                line = output.get(timeout=self.timeout)
                if isinstance(line, Exception) or not line:
                    raise EmbeddingGemma2Error("EG2_WORKER_EXITED", str(line))
                response = json.loads(line)
                if response.get("id") != request_id:
                    raise EmbeddingGemma2Error("EG2_WORKER_PROTOCOL_ERROR")
                if "error" in response:
                    raise EmbeddingGemma2Error(response["error"]["code"], response["error"]["detail"])
                return response["result"]
            except queue.Empty as exc:
                self.close()
                raise EmbeddingGemma2Error("EG2_WORKER_TIMEOUT", operation) from exc

    def encode_text(self, query):
        result = self.request("text", query=query)
        return normalized(result["embedding"]), result["metrics"]

    def encode_video(self, path, config=None):
        result = self.request("video", path=str(path), config=config or self.config)
        return normalized(result["embedding"]), result["metrics"]

    def close(self):
        with self.lock:
            process, self.process = self.process, None
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                for stream in (process.stdin, process.stdout):
                    if stream:
                        stream.close()
