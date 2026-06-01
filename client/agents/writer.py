from __future__ import annotations

from client.explainability.recorder import ExplanationRecorder
from shared.schemas import AgentTrace
from shared.slm import SLMClient


class WriterAgent:
    def __init__(self, slm: SLMClient) -> None:
        self.slm = slm

    def run(self, trace: AgentTrace, recorder: ExplanationRecorder) -> str:
        if not trace.tool_result:
            return "Unable to complete task."

        prompt = (
            "Write a concise final answer for the user.\n"
            f"User query: {trace.task.query}\n"
            f"Selected tool: {trace.tool_result.tool_name}\n"
            f"Tool result: {trace.tool_result.output}\n"
            "Keep the answer grounded in the tool result."
        )
        answer = self.slm.generate(
            prompt,
            system="You are the writer agent in a tool-using multi-agent system.",
        )
        recorder.record(
            layer="slm",
            event_type="final_answer_generation",
            decision=self.slm.name,
            reason="WriterAgent used the configured SLM backend to produce the final answer.",
            evidence=[trace.tool_result.output],
            metrics={"backend": self.slm.name},
        )
        return answer

