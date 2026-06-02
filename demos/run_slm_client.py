from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from client.learning.controller import LearningController
from client.runtime import ClientRuntime
from shared.scientific_dataset import SCIENTIFIC_CLIENT_DATA
from shared.slm import build_slm_from_env
from shared.slm_tool_router import SLMToolRouter


def main() -> None:
    slm = build_slm_from_env()
    runtime = ClientRuntime(client_id="client_slm", slm=slm)

    labeled_examples = [
        ("hello explain what you can do", "general"),
        *[(example.query, example.label_tool) for examples in SCIENTIFIC_CLIENT_DATA.values() for example in examples[:1]],
    ]
    for query, label_tool in labeled_examples:
        runtime.run_task(query, label_tool=label_tool)

    result = LearningController(runtime).update_from_traces()
    query = "verify whether the paper supports the vaccine efficacy claim"
    trace = runtime.run_task(query)

    slm_router = SLMToolRouter(slm)
    slm_prediction = slm_router.predict(query)

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
