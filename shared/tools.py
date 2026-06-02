from __future__ import annotations

import re
from time import perf_counter

from shared.schemas import ToolResult


def calculator_tool(query: str) -> str:
    numbers = [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?", query)]
    if not numbers:
        return "No numbers found."
    return f"sum={sum(numbers):.2f}"


def search_tool(query: str) -> str:
    return f"paper_search_result: found candidate scientific papers for '{query}'"


def claim_verification_tool(query: str) -> str:
    return (
        "claim_verification_result: retrieve abstracts, identify supporting "
        "or refuting evidence, and return support/refute/not-enough-info."
    )


def general_tool(query: str) -> str:
    return f"general_answer: {query}"


TOOLS = {
    "calculator": calculator_tool,
    "paper_search": search_tool,
    "claim_verification": claim_verification_tool,
    "general": general_tool,
}


def call_tool(tool_name: str, query: str) -> ToolResult:
    start = perf_counter()
    if tool_name not in TOOLS:
        return ToolResult(tool_name=tool_name, output=f"Unknown tool: {tool_name}", success=False)

    try:
        output = TOOLS[tool_name](query)
        return ToolResult(
            tool_name=tool_name,
            output=output,
            success=True,
            latency_ms=(perf_counter() - start) * 1000,
        )
    except Exception as exc:  # pragma: no cover
        return ToolResult(
            tool_name=tool_name,
            output=str(exc),
            success=False,
            latency_ms=(perf_counter() - start) * 1000,
        )
