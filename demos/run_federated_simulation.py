from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from client.learning.controller import LearningController
from client.runtime import ClientRuntime
from shared.metrics import tool_accuracy
from shared.scientific_dataset import SCIENTIFIC_CLIENT_DATA, SCIENTIFIC_EVAL
from shared.tool_router import ToolRouter, average_parameters


CLIENT_QUERIES = {
    "client_quant": [
        (example.query, example.label_tool)
        for example in SCIENTIFIC_CLIENT_DATA[0]
    ],
    "client_literature": [
        (example.query, example.label_tool)
        for example in SCIENTIFIC_CLIENT_DATA[1]
    ],
    "client_verifier": [
        (example.query, example.label_tool)
        for example in SCIENTIFIC_CLIENT_DATA[2]
    ],
}

EVAL = SCIENTIFIC_EVAL


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
    trace = demo_runtime.run_task("verify whether the paper supports the vaccine efficacy claim")
    print(f"selected_tool={trace.plan.selected_tool if trace.plan else None}")
    print(f"answer={trace.final_answer}")


if __name__ == "__main__":
    main()
