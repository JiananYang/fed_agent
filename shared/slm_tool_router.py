from __future__ import annotations

from shared.slm import SLMClient
from shared.tool_router import LABELS, RouterPrediction


class SLMToolRouter:
    """Non-federated SLM baseline for tool selection.

    Use this to compare learned/federated routing against direct SLM routing.
    The trainable `ToolRouter` remains the component aggregated by Flower.
    """

    def __init__(self, slm: SLMClient) -> None:
        self.slm = slm

    def predict(self, text: str) -> RouterPrediction:
        labels = ", ".join(LABELS)
        prompt = (
            "Choose exactly one tool for the user query.\n"
            f"Available tools: {labels}\n"
            f"User query: {text}\n"
            "Return only the tool name."
        )
        raw = self.slm.generate(
            prompt,
            system="You are a precise tool-routing model. Return one valid tool name only.",
        )
        label = self._parse_label(raw)
        return RouterPrediction(label=label, confidence=1.0, top_k=[(label, 1.0)])

    def _parse_label(self, raw: str) -> str:
        lowered = raw.strip().lower()
        for label in LABELS:
            if label in lowered:
                return label
        return "general"

