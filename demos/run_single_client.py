from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from client.learning.controller import LearningController
from client.runtime import ClientRuntime
from shared.scientific_dataset import SCIENTIFIC_CLIENT_DATA


def main() -> None:
    runtime = ClientRuntime(client_id="client_a")

    queries = [
        ("hello can you help explain this", "general"),
        *[
            (example.query, example.label_tool)
            for examples in SCIENTIFIC_CLIENT_DATA.values()
            for example in examples[:1]
        ],
    ]

    for query, label_tool in queries:
        trace = runtime.run_task(query, label_tool=label_tool)
        print(f"query={query!r} answer={trace.final_answer!r}")

    learning = LearningController(runtime)
    result = learning.update_from_traces()

    after = runtime.run_task(
        "verify whether the paper supports the vaccine efficacy claim",
        label_tool="claim_verification",
    )
    print(json.dumps(result, indent=2))
    print(json.dumps({
        "final_answer": after.final_answer,
        "plan": after.plan.__dict__ if after.plan else None,
        "events": [event.__dict__ for event in after.events],
    }, indent=2))


if __name__ == "__main__":
    main()
