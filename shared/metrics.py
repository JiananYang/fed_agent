from __future__ import annotations

from shared.tool_router import ToolRouter


def tool_accuracy(router: ToolRouter, examples: list[tuple[str, str]]) -> float:
    if not examples:
        return 0.0
    correct = 0
    for query, label in examples:
        correct += int(router.predict(query).label == label)
    return correct / len(examples)

