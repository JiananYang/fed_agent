from __future__ import annotations

try:
    from flwr.common import Context
    from flwr.common.typing import Metrics
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.server.strategy import FedAvg
except ImportError:
    Context = object
    Metrics = dict
    ServerApp = None
    ServerAppComponents = None
    ServerConfig = None
    FedAvg = None


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated = {}
    metric_names = set().union(*(metric.keys() for _, metric in metrics))
    for name in metric_names:
        if name == "client_id":
            continue
        values = [
            num_examples * float(metric[name])
            for num_examples, metric in metrics
            if name in metric and isinstance(metric[name], (int, float))
        ]
        if values:
            aggregated[name] = sum(values) / total_examples
    return aggregated


def server_fn(context: Context):
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn) if ServerApp else None
