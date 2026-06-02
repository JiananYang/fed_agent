from __future__ import annotations

from client.runtime import ClientRuntime
from server.memory_agent import ServerMemoryAgent
from shared.schemas import AgentTask, ClientProfile, ClientSelection
from shared.tool_router import ToolRouter


CAPABILITY_BY_TOOL = {
    "calculator": "quantitative_analysis",
    "paper_search": "scientific_literature",
    "claim_verification": "scientific_verification",
    "general": "general",
}


class ClientSelector:
    """Arranges and forwards tasks to selected clients."""

    def __init__(self, router: ToolRouter) -> None:
        self.router = router

    def select(self, task: AgentTask, clients: list[ClientProfile]) -> ClientSelection:
        prediction = self.router.predict(task.query)
        required_capability = CAPABILITY_BY_TOOL.get(prediction.label, "general")
        candidates = []

        for profile in clients:
            capability_match = 1.0 if required_capability in profile.capabilities else 0.0
            tool_match = 1.0 if prediction.label in profile.tools else 0.0
            availability = 1.0 if profile.availability == "online" else 0.0
            load_score = 1.0 - profile.load
            score = (
                0.40 * capability_match
                + 0.25 * tool_match
                + 0.20 * availability
                + 0.10 * load_score
                + 0.05 * profile.trust_score
            )
            candidates.append(
                {
                    "client_id": profile.client_id,
                    "score": round(score, 4),
                    "capability_match": capability_match,
                    "tool_match": tool_match,
                    "availability": availability,
                    "load": profile.load,
                    "trust_score": profile.trust_score,
                }
            )

        if not candidates:
            raise ValueError("No clients are registered.")

        candidates.sort(key=lambda item: item["score"], reverse=True)
        selected = candidates[0]
        return ClientSelection(
            task_id=task.task_id,
            selected_client_id=selected["client_id"],
            required_capability=required_capability,
            score=selected["score"],
            reason=(
                f"Selected best client for predicted tool '{prediction.label}' "
                f"and capability '{required_capability}'."
            ),
            candidates=candidates,
        )

    def arrange(self, task: AgentTask, memory: ServerMemoryAgent) -> ClientSelection:
        return self.select(task, memory.list_clients())

    def forward(
        self,
        task: AgentTask,
        selection: ClientSelection,
        runtimes: dict[str, ClientRuntime],
        memory: ServerMemoryAgent,
    ):
        runtime = runtimes[selection.selected_client_id]
        trace = runtime.run_task(task.query)
        memory.remember_activity(selection.selected_client_id, trace)
        return trace
