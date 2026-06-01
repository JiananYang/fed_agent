from __future__ import annotations

from client.learning.local_train import train_tool_router
from client.learning.trace_to_dataset import traces_to_tool_examples
from client.runtime import ClientRuntime


class LearningController:
    def __init__(self, runtime: ClientRuntime) -> None:
        self.runtime = runtime

    def update_from_traces(self) -> dict:
        examples = traces_to_tool_examples(self.runtime.traces)
        loss = train_tool_router(self.runtime.router, examples)
        return {
            "num_examples": len(examples),
            "tool_router_loss": loss,
            "updated_components": ["tool_router"] if examples else [],
        }

