"""Flower server example."""
from time import sleep
from logging import INFO
import pickle
from pathlib import Path
from collections import OrderedDict

import flwr as fl
from flwr.common import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateIns, EvaluateRes, FitRes, Parameters, Scalar, parameters_to_ndarrays

from typing import Dict, List, Tuple, Optional, Union

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):

    def __init__(self, save_global_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.best_dice_so_far = - float("inf")
        self.save_global_path = Path(save_global_path)

        # Create path if it doesn't exist
        self.save_global_path.mkdir(exist_ok=True, parents=True)

    def save_model(self, server_round: int, avg_dice: float):
        """Save global parameters to disk as list of NumPy arrays."""

        ndarrays = parameters_to_ndarrays(self.global_model)
        filename = self.save_global_path/f"global_model_round_{server_round}.pkl"

        # construct artifact to save
        to_save = {'ndarrays': ndarrays, 'avg_dice': avg_dice}

        with open(filename, 'wb') as h:
            pickle.dump(to_save, h, protocol=pickle.HIGHEST_PROTOCOL)
        
        log(INFO, f"Saved new model into: {filename}")

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        # Configure as usual
        proxies_and_instructions = super().configure_evaluate(server_round, parameters, client_manager)

        # Now keep a local copy of the parameters sent to the clients
        # This is what we'll save to disk if a new best average dice metric is achieved.
        self.global_model = parameters

        return proxies_and_instructions

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
        log(INFO, f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

        if aggregated_accuracy > self.best_dice_so_far:
            log(INFO, f"New best average dice achieved (round {server_round})")
            self.save_model(server_round, aggregated_accuracy)
            self.best_dice_so_far = aggregated_accuracy

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"Dice": aggregated_accuracy}
    

def get_evaluate_fn(server_dataset):
    """This function returns a function that will be executed by the 
    strategy after aggregation when invoking its evaluate() method. It
    can be used to evalute the global model on a dataset hosted by
    the server."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        """Use the entire CIFAR-10 test set for evaluation."""

        log(INFO,"Evaluating global model on a dataset held by the server")
        log(INFO," --------------------------- WARNING ----------------------")
        sleep(10)
        log(INFO," --------------------------- MUST IMPLEMENT ---------------")

        # model = # Construct your model
        # set_params(model, parameters) # Appply `parameters` (just how clients do when they receive the model from the server)
        # model.to(device)

        # construct dataloader if needed
        # testloader = DataLoader(server_dataset, batch_size=50)
        # loss, accuracy = test(model, testloader, device=device) # evaluate your global model

        # return loss, {"accuracy": accuracy} report metrics
        return 0.0, {}

    return evaluate

def main():

    log(INFO, "PLEASE LOAD YOUR SERVER-SIDE dataset")
    server_dataset = None # load dataset/dataloader

    # Create strategy and run server
    strategy = AggregateCustomMetricStrategy(
        save_global_path='global_models',
        evaluate_fn=get_evaluate_fn(server_dataset)) # pass your dataset here
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=500),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()