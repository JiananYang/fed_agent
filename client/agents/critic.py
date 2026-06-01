from __future__ import annotations

from client.explainability.recorder import ExplanationRecorder
from shared.schemas import AgentTrace


class CriticAgent:
    def run(self, trace: AgentTrace, recorder: ExplanationRecorder) -> bool:
        success = bool(trace.tool_result and trace.tool_result.success)
        recorder.record(
            layer="critic",
            event_type="validation",
            decision="pass" if success else "fail",
            reason="Critic checks whether the selected tool produced a usable result.",
            metrics={"tool_success": success},
        )
        return success

