from __future__ import annotations

from shared.schemas import AgentTask, AgentTrace, Plan
from shared.tool_router import ToolRouter


class PlanningAgent:
    def __init__(self, router: ToolRouter) -> None:
        self.router = router

    def plan(self, task: AgentTask, trace: AgentTrace) -> Plan:
        prediction = self.router.predict(task.query)
        task_type = self._task_type_from_tool(prediction.label)
        selected_agents = ["memory", "planner", "tool_agent", "critic", "safety"]
        plan = Plan(
            task_type=task_type,
            selected_agents=selected_agents,
            selected_tool=prediction.label,
            confidence=prediction.confidence,
            reason="Planner used the trainable ToolRouter to select the first tool.",
        )
        trace.plan = plan
        return plan

    def _task_type_from_tool(self, tool_name: str) -> str:
        mapping = {
            "calculator": "calculation",
            "search": "information_retrieval",
            "contract_risk": "contract_review",
            "general": "general_assistance",
        }
        return mapping.get(tool_name, "general_assistance")
