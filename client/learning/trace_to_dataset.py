from __future__ import annotations

from shared.schemas import AgentTrace, TrainingExample


def traces_to_tool_examples(traces: list[AgentTrace]) -> list[TrainingExample]:
    examples = []
    for trace in traces:
        if trace.user_feedback != "accepted":
            continue
        if not trace.plan or not trace.plan.selected_tool:
            continue
        if not trace.tool_result:
            continue
        label_tool = trace.task.metadata.get("label_tool", trace.plan.selected_tool)

        examples.append(
            TrainingExample(
                query=trace.task.query,
                label_tool=label_tool,
                source_task_id=trace.task.task_id,
            )
        )
    return examples
