from __future__ import annotations

from shared.schemas import TrainingExample
from shared.tool_router import ToolRouter


def train_tool_router(router: ToolRouter, examples: list[TrainingExample]) -> float:
    pairs = [(example.query, example.label_tool) for example in examples]
    return router.train(pairs, epochs=40, lr=0.25)

