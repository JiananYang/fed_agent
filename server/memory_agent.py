from __future__ import annotations

from server.client_registry import ClientRegistry
from shared.schemas import AgentTrace, ClientActivity, ClientProfile


class ServerMemoryAgent:
    """Server-side memory for client states, histories, and specialities.

    This memory stores metadata needed for orchestration. It should not store
    raw private client data, documents, or local client memories.
    """

    def __init__(self) -> None:
        self.registry = ClientRegistry()
        self.activities: list[ClientActivity] = []

    def remember_client(self, profile: ClientProfile) -> None:
        self.registry.register(profile)

    def list_clients(self) -> list[ClientProfile]:
        return self.registry.list_clients()

    def get_client(self, client_id: str) -> ClientProfile:
        return self.registry.get(client_id)

    def update_client_state(
        self,
        client_id: str,
        *,
        availability: str | None = None,
        load: float | None = None,
        trust_score: float | None = None,
    ) -> None:
        profile = self.registry.get(client_id)
        if availability is not None:
            profile.availability = availability
        if load is not None:
            profile.load = max(0.0, min(1.0, load))
        if trust_score is not None:
            profile.trust_score = max(0.0, min(1.0, trust_score))

    def remember_activity(self, client_id: str, trace: AgentTrace) -> ClientActivity:
        activity = ClientActivity(
            client_id=client_id,
            task_id=trace.task.task_id,
            query=trace.task.query,
            selected_tool=trace.plan.selected_tool if trace.plan else None,
            success=bool(trace.tool_result and trace.tool_result.success),
            final_answer=trace.final_answer,
        )
        self.activities.append(activity)
        self._update_trust_from_activity(activity)
        return activity

    def history_for_client(self, client_id: str) -> list[ClientActivity]:
        return [activity for activity in self.activities if activity.client_id == client_id]

    def _update_trust_from_activity(self, activity: ClientActivity) -> None:
        profile = self.registry.get(activity.client_id)
        delta = 0.02 if activity.success else -0.05
        profile.trust_score = max(0.0, min(1.0, profile.trust_score + delta))

