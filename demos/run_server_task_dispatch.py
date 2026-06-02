from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from client.learning.controller import LearningController
from client.runtime import ClientRuntime
from server.task_gateway import TaskGateway
from shared.scientific_dataset import SCIENTIFIC_CLIENT_DATA
from shared.schemas import ClientProfile
from shared.tool_router import ToolRouter, average_parameters


SEED_CLIENTS = {
    "client_quant": {
        "capabilities": ["quantitative_analysis", "general"],
        "tools": ["calculator", "general"],
        "examples": [(example.query, example.label_tool) for example in SCIENTIFIC_CLIENT_DATA[0]],
    },
    "client_literature": {
        "capabilities": ["scientific_literature", "general"],
        "tools": ["paper_search", "general"],
        "examples": [(example.query, example.label_tool) for example in SCIENTIFIC_CLIENT_DATA[1]],
    },
    "client_verifier": {
        "capabilities": ["scientific_verification", "general"],
        "tools": ["claim_verification", "general"],
        "examples": [(example.query, example.label_tool) for example in SCIENTIFIC_CLIENT_DATA[2]],
    },
}


def build_trained_router_and_clients() -> tuple[ToolRouter, dict[str, ClientRuntime]]:
    client_parameters = []
    runtimes = {}
    for client_id, config in SEED_CLIENTS.items():
        runtime = ClientRuntime(client_id=client_id)
        for query, label_tool in config["examples"]:
            runtime.run_task(query, label_tool=label_tool)
        LearningController(runtime).update_from_traces()
        client_parameters.append(runtime.router.get_parameters())
        runtimes[client_id] = runtime

    router = ToolRouter()
    router.set_parameters(average_parameters(client_parameters))
    return router, runtimes


def main() -> None:
    router, runtimes = build_trained_router_and_clients()
    gateway = TaskGateway(router=router)

    for client_id, config in SEED_CLIENTS.items():
        gateway.register_client(
            ClientProfile(
                client_id=client_id,
                capabilities=config["capabilities"],
                tools=config["tools"],
                load=0.2 if client_id != "client_verifier" else 0.1,
            ),
            runtimes[client_id],
        )

    result = gateway.dispatch("verify whether the paper supports the vaccine efficacy claim")
    selected_profile = gateway.memory_agent.get_client(result.selection.selected_client_id)
    selected_history = gateway.memory_agent.history_for_client(result.selection.selected_client_id)
    print(json.dumps({
        "selected_client": result.selection.selected_client_id,
        "selection_reason": result.selection.reason,
        "required_capability": result.selection.required_capability,
        "candidate_scores": result.selection.candidates,
        "selected_client_activity_count": len(selected_history),
        "selected_client_trust_score": selected_profile.trust_score,
        "client_selected_tool": result.trace.plan.selected_tool if result.trace.plan else None,
        "answer": result.trace.final_answer,
    }, indent=2))


if __name__ == "__main__":
    main()
