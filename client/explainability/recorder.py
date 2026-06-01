from __future__ import annotations

from shared.schemas import AgentTrace, ExplanationEvent, new_id


class ExplanationRecorder:
    def __init__(self, trace: AgentTrace) -> None:
        self.trace = trace

    def record(
        self,
        *,
        layer: str,
        event_type: str,
        decision: str,
        reason: str,
        confidence: float | None = None,
        evidence: list[str] | None = None,
        metrics: dict | None = None,
    ) -> None:
        self.trace.events.append(
            ExplanationEvent(
                event_id=new_id("evt"),
                task_id=self.trace.task.task_id,
                client_id=self.trace.task.client_id,
                layer=layer,
                event_type=event_type,
                decision=decision,
                reason=reason,
                confidence=confidence,
                evidence=evidence or [],
                metrics=metrics or {},
            )
        )

