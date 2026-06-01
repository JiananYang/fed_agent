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
    return f"mock_search_result: found relevant notes for '{query}'"


def contract_risk_tool(query: str) -> str:
    return (
        "risk_report: review termination, liability, indemnity, payment, "
        "and unilateral change clauses."
    )


def general_tool(query: str) -> str:
    return f"general_answer: {query}"


TOOLS = {
    "calculator": calculator_tool,
    "search": search_tool,
    "contract_risk": contract_risk_tool,
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

