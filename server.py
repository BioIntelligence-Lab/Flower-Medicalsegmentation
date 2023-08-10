"""Flower server example."""
import flwr as fl
from collections import OrderedDict
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, FitRes, Scalar

from typing import Dict, List, Tuple, Optional, Union
if __name__ == "__main__":
    class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
        def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
            """Aggregate evaluation accuracy using weighted average."""

            if not results:
                return None, {}

            # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
            aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

            # Weigh accuracy of each client by number of examples used
            accuracies = [r.metrics["Dice"] * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]

            # Aggregate and print custom metric
            aggregated_accuracy = sum(accuracies) / sum(examples)
            print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

            # Return aggregated loss and metrics (i.e., aggregated accuracy)
            return aggregated_loss, {"Dice": aggregated_accuracy}

    # Create strategy and run server
    strategy = AggregateCustomMetricStrategy(
        # (same arguments as FedAvg here)
)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=500),
        strategy=strategy,
    )