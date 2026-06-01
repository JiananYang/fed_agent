from __future__ import annotations

from client.runtime import ClientRuntime
from server.client_registry import ClientRegistry
from server.client_selector import ClientSelector
from shared.schemas import AgentTask, ClientProfile, TaskDispatchResult
from shared.tool_router import ToolRouter


class TaskGateway:
    """Receives server tasks and dispatches them to selected clients.

    This is intentionally separate from Flower. Flower aggregates trainable
    components; the gateway handles runtime task orchestration.
    """

    def __init__(self, router: ToolRouter | None = None) -> None:
        self.router = router or ToolRouter()
        self.registry = ClientRegistry()
        self.selector = ClientSelector(self.router)
        self.runtimes: dict[str, ClientRuntime] = {}

    def register_client(self, profile: ClientProfile, runtime: ClientRuntime) -> None:
        self.registry.register(profile)
        self.runtimes[profile.client_id] = runtime

    def dispatch(self, query: str) -> TaskDispatchResult:
        server_task = AgentTask(query=query, client_id="server")
        selection = self.selector.select(server_task, self.registry.list_clients())
        runtime = self.runtimes[selection.selected_client_id]
        trace = runtime.run_task(query)
        return TaskDispatchResult(selection=selection, trace=trace)

