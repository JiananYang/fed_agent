from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from client.learning.controller import LearningController
from client.runtime import ClientRuntime
from shared.metrics import tool_accuracy
from shared.tool_router import ToolRouter, average_parameters


CLIENT_QUERIES = {
    "client_legal": [
        ("contract termination risk", "contract_risk"),
        ("review liability clause", "contract_risk"),
        ("check contract indemnity clause", "contract_risk"),
    ],
    "client_research": [
        ("search latest policy", "search"),
        ("find latest regulation", "search"),
        ("search market update", "search"),
    ],
    "client_finance": [
        ("calculate 10 and 25", "calculator"),
        ("sum price 30 and 5", "calculator"),
        ("calculate margin 100 and 40", "calculator"),
    ],
}

EVAL = [
    ("check contract termination clause", "contract_risk"),
    ("search latest compliance policy", "search"),
    ("calculate 9 and 11", "calculator"),
]


def main() -> None:
    before_router = ToolRouter()
    print(f"global_accuracy_before={tool_accuracy(before_router, EVAL):.3f}")

    client_parameters = []
    for client_id, queries in CLIENT_QUERIES.items():
        runtime = ClientRuntime(client_id=client_id)
        for query, label_tool in queries:
            runtime.run_task(query, label_tool=label_tool)

        result = LearningController(runtime).update_from_traces()
        client_parameters.append(runtime.router.get_parameters())
        print(f"{client_id} local_update={result}")

    global_router = ToolRouter()
    global_router.set_parameters(average_parameters(client_parameters))
    print(f"global_accuracy_after={tool_accuracy(global_router, EVAL):.3f}")

    demo_runtime = ClientRuntime(client_id="client_demo", router=global_router)
    trace = demo_runtime.run_task("please review liability risk in this contract")
    print(f"selected_tool={trace.plan.selected_tool if trace.plan else None}")
    print(f"answer={trace.final_answer}")


if __name__ == "__main__":
    main()
