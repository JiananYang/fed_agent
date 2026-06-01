from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class SLMClient(Protocol):
    name: str

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Generate text from a small language model backend."""


@dataclass
class MockSLM:
    """Dependency-free SLM stand-in for tests and offline demos."""

    name: str = "mock-slm"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        if "Choose exactly one tool" in prompt:
            lowered = prompt.lower()
            if any(term in lowered for term in ["contract", "clause", "risk", "liability"]):
                return "contract_risk"
            if any(term in lowered for term in ["search", "find", "latest", "policy"]):
                return "search"
            if any(term in lowered for term in ["calculate", "sum", "price", "margin"]):
                return "calculator"
            return "general"

        return "SLM summary: " + prompt.strip().replace("\n", " ")[:220]


@dataclass
class OllamaSLM:
    """Ollama-backed local SLM.

    Example:
        ollama pull qwen2.5:1.5b
        set SLM_BACKEND=ollama
        set OLLAMA_MODEL=qwen2.5:1.5b
    """

    model: str = "qwen2.5:1.5b"
    base_url: str = "http://localhost:11434"
    name: str = "ollama"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. "
                "Start Ollama or use SLM_BACKEND=mock."
            ) from exc

        return data["message"]["content"].strip()


def build_slm_from_env() -> SLMClient:
    backend = os.getenv("SLM_BACKEND", "mock").lower()
    if backend == "ollama":
        return OllamaSLM(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    if backend == "mock":
        return MockSLM()
    raise ValueError(f"Unsupported SLM_BACKEND={backend!r}")

