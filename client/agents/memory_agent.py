from __future__ import annotations

from client.explainability.recorder import ExplanationRecorder
from client.memory.local_memory import LocalMemory
from shared.schemas import AgentTask, AgentTrace, MemoryRecord


class MemoryAgent:
    """Client-local memory agent.

    The memory agent keeps private client memory local. It can be improved later
    with a trainable memory selector, but raw memory should not be sent to the
    server or Flower aggregator.
    """

    def __init__(self, memory: LocalMemory) -> None:
        self.memory = memory

    def retrieve(
        self,
        task: AgentTask,
        trace: AgentTrace,
        recorder: ExplanationRecorder,
        limit: int = 3,
    ) -> list[MemoryRecord]:
        memories = self.memory.search(task.query, limit=limit)
        trace.task.metadata["memory_ids"] = [record.memory_id for record in memories]
        recorder.record(
            layer="memory",
            event_type="memory_read",
            decision=f"{len(memories)} records",
            reason="MemoryAgent retrieved local private memories before planning.",
            evidence=[record.text for record in memories],
            metrics={
                "memory_count": len(memories),
                "memory_ids": [record.memory_id for record in memories],
            },
        )
        return memories

    def update_after_task(
        self,
        trace: AgentTrace,
        recorder: ExplanationRecorder,
    ) -> MemoryRecord | None:
        if trace.user_feedback != "accepted":
            return None
        if not trace.plan or not trace.plan.selected_tool:
            return None
        if not trace.tool_result or not trace.tool_result.success:
            return None

        record = self.memory.write(
            text=(
                f"For similar query '{trace.task.query}', "
                f"tool '{trace.plan.selected_tool}' worked."
            ),
            memory_type="episodic",
            tags=[trace.plan.task_type, trace.plan.selected_tool],
            source_task_id=trace.task.task_id,
        )
        recorder.record(
            layer="memory",
            event_type="memory_write",
            decision=record.memory_id,
            reason="MemoryAgent stored accepted task outcome as local episodic memory.",
            evidence=[record.text],
            metrics={
                "memory_type": record.memory_type,
                "tags": record.tags,
                "sensitivity": record.sensitivity,
            },
        )
        return record

