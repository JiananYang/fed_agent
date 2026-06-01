from __future__ import annotations

try:
    from flwr.common import Context
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.server.strategy import FedAvg
except ImportError:
    Context = object
    ServerApp = None
    ServerAppComponents = None
    ServerConfig = None
    FedAvg = None


def server_fn(context: Context):
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn) if ServerApp else None

