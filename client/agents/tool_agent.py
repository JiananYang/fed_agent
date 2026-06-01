from __future__ import annotations

from client.explainability.recorder import ExplanationRecorder
from shared.schemas import AgentTask, AgentTrace
from shared.tools import call_tool


class ToolAgent:
    def run(self, task: AgentTask, trace: AgentTrace, recorder: ExplanationRecorder) -> None:
        if not trace.plan or not trace.plan.selected_tool:
            raise ValueError("ToolAgent requires a plan with selected_tool.")

        result = call_tool(trace.plan.selected_tool, task.query)
        trace.tool_result = result
        recorder.record(
            layer="tool",
            event_type="tool_call",
            decision=result.tool_name,
            reason="ToolAgent executed the tool selected by the planning step.",
            confidence=trace.plan.confidence,
            evidence=[result.output],
            metrics={"success": result.success, "latency_ms": result.latency_ms},
        )

