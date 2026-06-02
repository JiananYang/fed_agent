from __future__ import annotations

from client.agents.memory_agent import MemoryAgent
from client.agents.planner import PlanningAgent
from client.agents.tool_agent import ToolAgent
from client.explainability.recorder import ExplanationRecorder
from client.memory.local_memory import LocalMemory
from shared.schemas import AgentTask, AgentTrace
from shared.slm import SLMClient, build_slm_from_env
from shared.tool_router import ToolRouter


class ClientRuntime:
    """Client-side runtime with static memory/tool agents and an SLM.

    The collaboratively trained component is the planner's routing policy
    (`ToolRouter`). The memory agent, tool agent, and SLM backend are execution
    components held locally by each client.
    """

    def __init__(
        self,
        client_id: str,
        router: ToolRouter | None = None,
        slm: SLMClient | None = None,
    ) -> None:
        self.client_id = client_id
        self.router = router or ToolRouter()
        self.slm = slm or build_slm_from_env()
        self.memory = LocalMemory(client_id)
        self.memory_agent = MemoryAgent(self.memory)
        self.planner = PlanningAgent(self.router)
        self.tool_agent = ToolAgent()
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
        trace.final_answer = self._generate_final_answer(trace, recorder)

        self.memory_agent.update_after_task(trace, recorder)

        self.traces.append(trace)
        return trace

    def _generate_final_answer(
        self,
        trace: AgentTrace,
        recorder: ExplanationRecorder,
    ) -> str:
        if not trace.tool_result or not trace.tool_result.success:
            return "Unable to complete task."

        prompt = (
            "Write a concise final answer for the user.\n"
            f"User query: {trace.task.query}\n"
            f"Selected tool: {trace.tool_result.tool_name}\n"
            f"Tool result: {trace.tool_result.output}\n"
            "Keep the answer grounded in the tool result."
        )
        answer = self.slm.generate(
            prompt,
            system="You are the local SLM inside a federated agent client.",
        )
        recorder.record(
            layer="slm",
            event_type="final_answer_generation",
            decision=self.slm.name,
            reason="ClientRuntime used the client's local SLM to produce the final answer.",
            evidence=[trace.tool_result.output],
            metrics={"backend": self.slm.name},
        )
        return answer
