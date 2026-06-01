from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from client.learning.controller import LearningController
from client.runtime import ClientRuntime
from shared.slm import build_slm_from_env
from shared.slm_tool_router import SLMToolRouter


def main() -> None:
    slm = build_slm_from_env()
    runtime = ClientRuntime(client_id="client_slm", slm=slm)

    labeled_examples = [
        ("calculate 20 and 22", "calculator"),
        ("search latest procurement policy", "search"),
        ("check contract liability risk", "contract_risk"),
        ("hello explain what you can do", "general"),
    ]
    for query, label_tool in labeled_examples:
        runtime.run_task(query, label_tool=label_tool)

    result = LearningController(runtime).update_from_traces()
    trace = runtime.run_task("please review termination risk in this contract")

    slm_router = SLMToolRouter(slm)
    slm_prediction = slm_router.predict("please review termination risk in this contract")

    print(json.dumps({
        "slm_backend": slm.name,
        "local_training": result,
        "federated_router_plan": trace.plan.__dict__ if trace.plan else None,
        "slm_router_baseline": slm_prediction.__dict__,
        "final_answer": trace.final_answer,
        "slm_events": [
            event.__dict__
            for event in trace.events
            if event.layer == "slm"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()

