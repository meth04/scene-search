"""OpenAI-compatible external multimodal client (llama-server or vLLM)."""
from __future__ import annotations

import time
import requests
from urllib.parse import urlsplit
from .common import extract_json


class VLMError(RuntimeError):
    pass


def require_caption_endpoint(client):
    """Captioning fails promptly with the exact conservative local start command."""
    try:
        return client.health()
    except VLMError as exc:
        parsed = urlsplit(client.base_url)
        port = parsed.port or 8080
        command = (f"llama-server -hf Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M "
                   f"--host 127.0.0.1 --port {port} --ctx-size 4096 --parallel 1 "
                   "--n-gpu-layers 0 --no-mmproj-offload")
        raise VLMError(f"{exc}\nStart the existing local server: {command}\n"
                       "Use your installed llama-server.exe path if it is not on PATH.") from exc


class VLMClient:
    def __init__(self, base_url="http://127.0.0.1:8080/v1", model=None, timeout=60, retries=3):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.json_mode = True
        self.sampling = {}

    def health(self):
        try:
            r = requests.get(self.base_url + "/models", timeout=min(5, self.timeout))
            r.raise_for_status()
            data = r.json()
            ids = [x["id"] for x in data.get("data", []) if x.get("id")]
            if not ids:
                raise VLMError("VLM endpoint is reachable but no model is loaded (/models is empty)")
            if self.model is None:
                self.model = ids[0]
            elif self.model not in ids:
                raise VLMError(f"VLM model alias {self.model!r} is not loaded; served IDs: {ids}")
            return data
        except VLMError:
            raise
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            raise VLMError(f"VLM endpoint unavailable or invalid at {self.base_url}: {exc}") from exc

    def chat(self, prompt, image_urls=None, max_tokens=700, temperature=.05):
        if self.model is None:
            self.health()
        content = [{"type": "text", "text": prompt}]
        content += [{"type": "image_url", "image_url": {"url": u}} for u in image_urls or []]
        payload = {
            "model": self.model, "messages": [{"role": "user", "content": content}],
            "temperature": float(temperature), "max_tokens": int(max_tokens),
        }
        allowed_sampling = {"repeat_penalty", "frequency_penalty", "presence_penalty", "top_p"}
        if set(self.sampling)-allowed_sampling:
            raise ValueError("Unsupported VLM sampling option")
        payload.update(self.sampling)
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        last = None
        for attempt in range(self.retries):
            try:
                r = requests.post(self.base_url + "/chat/completions", json=payload, timeout=self.timeout)
                if r.status_code in (400, 422) and "response_format" in payload:
                    error = r.text.lower()
                    if any(x in error for x in ("response_format", "json_object", "json mode", "grammar")):
                        payload.pop("response_format")
                        self.json_mode = False
                        r = requests.post(self.base_url + "/chat/completions", json=payload, timeout=self.timeout)
                if r.status_code in (400, 404, 422):
                    message = r.text[:500]
                    if image_urls and any(x in message.lower() for x in ("vision", "image", "projector", "multimodal")):
                        raise VLMError(f"Vision/projector unsupported by served model: {message}")
                    raise VLMError(f"VLM model/request rejected (HTTP {r.status_code}): {message}")
                r.raise_for_status()
                value = r.json()["choices"][0]["message"]["content"]
                if not isinstance(value, str) or not value.strip():
                    raise VLMError("Malformed multimodal response: message.content must be nonempty text")
                return value
            except VLMError:
                raise
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise VLMError(f"Malformed multimodal response: {exc}") from exc
            except requests.RequestException as exc:
                last = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(4, 1.5 * 2**attempt))
        raise VLMError(f"VLM endpoint unavailable after {self.retries} transport attempts: {last}")

    def chat_json(self, *args, **kwargs):
        last = None
        for _ in range(2):
            text = self.chat(*args, **kwargs)
            try:
                value = extract_json(text)
                if not isinstance(value, dict):
                    raise ValueError("Expected a JSON object")
                return value
            except ValueError as exc:
                last = exc
        raise VLMError(f"Malformed multimodal JSON response: {last}")


# Preserve the historical public import for dataset/server scripts.
VLLMClient = VLMClient
