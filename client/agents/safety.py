from __future__ import annotations

from client.explainability.recorder import ExplanationRecorder
from shared.schemas import AgentTask


class SafetyAgent:
    def run(self, task: AgentTask, recorder: ExplanationRecorder) -> bool:
        blocked_terms = ["bypass password", "steal", "exfiltrate"]
        lowered = task.query.lower()
        allowed = not any(term in lowered for term in blocked_terms)
        recorder.record(
            layer="safety",
            event_type="safety_check",
            decision="allow" if allowed else "block",
            reason="Rule-based safety check for the prototype.",
            metrics={"allowed": allowed},
        )
        return allowed

