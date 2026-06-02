from __future__ import annotations

from client.agents.critic import CriticAgent
from client.agents.memory_agent import MemoryAgent
from client.agents.planner import PlanningAgent
from client.agents.safety import SafetyAgent
from client.agents.tool_agent import ToolAgent
from client.agents.writer import WriterAgent
from client.explainability.recorder import ExplanationRecorder
from client.memory.local_memory import LocalMemory
from shared.schemas import AgentTask, AgentTrace
from shared.slm import SLMClient
from shared.tool_router import ToolRouter


class ClientRuntime:
    def __init__(
        self,
        client_id: str,
        router: ToolRouter | None = None,
        slm: SLMClient | None = None,
    ) -> None:
        self.client_id = client_id
        self.router = router or ToolRouter()
        self.slm = slm
        self.memory = LocalMemory(client_id)
        self.memory_agent = MemoryAgent(self.memory)
        self.planner = PlanningAgent(self.router)
        self.tool_agent = ToolAgent()
        self.critic = CriticAgent()
        self.safety = SafetyAgent()
        self.writer = WriterAgent(slm) if slm else None
        self.traces: list[AgentTrace] = []

    def run_task(
        self,
        query: str,
        user_feedback: str | None = "accepted",
        label_tool: str | None = None,
    ) -> AgentTrace:
        metadata = {"label_tool": label_tool} if label_tool else {}
        task = AgentTask(query=query, client_id=self.client_id, metadata=metadata)
        trace = AgentTrace(task=task, user_feedback=user_feedback)
        recorder = ExplanationRecorder(trace)

        self.memory_agent.retrieve(task, trace, recorder)

        if not self.safety.run(task, recorder):
            trace.final_answer = "Request blocked by safety policy."
            self.traces.append(trace)
            return trace

        plan = self.planner.plan(task, trace)
        recorder.record(
            layer="agent",
            event_type="planning",
            decision=plan.task_type,
            reason=plan.reason,
            confidence=plan.confidence,
            evidence=[f"{label}: {score:.3f}" for label, score in self.router.predict(query).top_k],
            metrics={"selected_agents": plan.selected_agents, "selected_tool": plan.selected_tool},
        )

        self.tool_agent.run(task, trace, recorder)
        passed = self.critic.run(trace, recorder)
        if passed and trace.tool_result:
            if self.writer:
                trace.final_answer = self.writer.run(trace, recorder)
            else:
                trace.final_answer = trace.tool_result.output
        else:
            trace.final_answer = "Unable to complete task."

        self.memory_agent.update_after_task(trace, recorder)

        self.traces.append(trace)
        return trace
