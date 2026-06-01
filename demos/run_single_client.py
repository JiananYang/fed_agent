from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from client.learning.controller import LearningController
from client.runtime import ClientRuntime


def main() -> None:
    runtime = ClientRuntime(client_id="client_a")

    queries = [
        ("hello can you help explain this", "general"),
        ("calculate 25 and 17", "calculator"),
        ("please check contract termination risk", "contract_risk"),
        ("search latest policy update", "search"),
    ]

    for query, label_tool in queries:
        trace = runtime.run_task(query, label_tool=label_tool)
        print(f"query={query!r} answer={trace.final_answer!r}")

    learning = LearningController(runtime)
    result = learning.update_from_traces()

    after = runtime.run_task("review liability clause in this contract", label_tool="contract_risk")
    print(json.dumps(result, indent=2))
    print(json.dumps({
        "final_answer": after.final_answer,
        "plan": after.plan.__dict__ if after.plan else None,
        "events": [event.__dict__ for event in after.events],
    }, indent=2))


if __name__ == "__main__":
    main()
