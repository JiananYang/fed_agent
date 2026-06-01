from __future__ import annotations

try:
    import flwr as fl
    from flwr.client import ClientApp, NumPyClient
    from flwr.common import Context
except ImportError:  # Allows local demos to run without Flower installed.
    fl = None
    ClientApp = None
    NumPyClient = object
    Context = object

from client.learning.local_train import train_tool_router
from shared.schemas import TrainingExample
from shared.tool_router import ToolRouter


LOCAL_DATA = {
    0: [
        TrainingExample("calculate 10 and 25", "calculator", "seed_0"),
        TrainingExample("sum price margin 3 and 7", "calculator", "seed_1"),
    ],
    1: [
        TrainingExample("search latest policy", "search", "seed_2"),
        TrainingExample("find latest regulation", "search", "seed_3"),
    ],
    2: [
        TrainingExample("contract termination risk", "contract_risk", "seed_4"),
        TrainingExample("review liability clause", "contract_risk", "seed_5"),
    ],
}


class FlowerToolRouterClient(NumPyClient):
    def __init__(self, client_id: int) -> None:
        self.client_id = client_id
        self.router = ToolRouter()
        self.examples = LOCAL_DATA[client_id]

    def get_parameters(self, config):
        return self.router.get_parameters()

    def fit(self, parameters, config):
        self.router.set_parameters(parameters)
        loss = train_tool_router(self.router, self.examples)
        return self.router.get_parameters(), len(self.examples), {
            "client_id": self.client_id,
            "loss": loss,
        }

    def evaluate(self, parameters, config):
        self.router.set_parameters(parameters)
        correct = 0
        for example in self.examples:
            correct += int(self.router.predict(example.query).label == example.label_tool)
        accuracy = correct / len(self.examples)
        return 1.0 - accuracy, len(self.examples), {
            "client_id": self.client_id,
            "accuracy": accuracy,
        }


def client_fn(context: Context):
    partition_id = int(context.node_config.get("partition-id", 0))
    client_id = partition_id % len(LOCAL_DATA)
    return FlowerToolRouterClient(client_id).to_client()


app = ClientApp(client_fn=client_fn) if ClientApp else None

