from __future__ import annotations

from client.runtime import ClientRuntime
from server.client_selector import ClientSelector
from server.memory_agent import ServerMemoryAgent
from shared.schemas import AgentTask, ClientProfile, TaskDispatchResult
from shared.tool_router import ToolRouter


class TaskGateway:
    """Receives server tasks and dispatches them to selected clients.

    This is intentionally separate from Flower. Flower aggregates trainable
    components; the gateway handles runtime task orchestration.
    """

    def __init__(self, router: ToolRouter | None = None) -> None:
        self.router = router or ToolRouter()
        self.memory_agent = ServerMemoryAgent()
        self.selector = ClientSelector(self.router)
        self.runtimes: dict[str, ClientRuntime] = {}

    def register_client(self, profile: ClientProfile, runtime: ClientRuntime) -> None:
        self.memory_agent.remember_client(profile)
        self.runtimes[profile.client_id] = runtime

    def dispatch(self, query: str) -> TaskDispatchResult:
        server_task = AgentTask(query=query, client_id="server")
        selection = self.selector.arrange(server_task, self.memory_agent)
        trace = self.selector.forward(
            server_task,
            selection,
            self.runtimes,
            self.memory_agent,
        )
        return TaskDispatchResult(selection=selection, trace=trace)
