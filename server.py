"""Flower server example."""
from time import sleep
from logging import INFO
import pickle
from pathlib import Path
from flwr.server import ServerApp, ServerConfig
import torch
import flwr as fl
from flwr.common import FitIns, log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, FitRes, Parameters, Scalar, parameters_to_ndarrays

from typing import Dict, List, Tuple, Optional, Union

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):

    def __init__(self, save_global_path: str, total_rounds: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.best_dice_so_far = - float("inf")
        self.save_global_path = Path(save_global_path)
        self.total_rounds = total_rounds
        # will be set to true if a new best dice is found
        # and when commencing the last round
        self.signal_clients_to_save_model = False

        # Create path if it doesn't exist
        self.save_global_path.mkdir(exist_ok=True, parents=True)

    def save_model(self, server_round: int, avg_dice: float):
        """Save global parameters to disk as list of NumPy arrays."""

        ndarrays = parameters_to_ndarrays(self.global_model)
        state_dict = {f"param_{i}": torch.tensor(ndarray) for i, ndarray in enumerate(ndarrays)}
        # Ignore saving the average dice metric for now
        # state_dict["avg_dice"] = avg_dice
        filename = self.save_global_path/f"global_model_round_{server_round}.pth"

        torch.save(state_dict, filename)
        
        log(INFO, f"Saved new model into: {filename}")

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy | FitIns]]:
        configure_fit =  super().configure_fit(server_round, parameters, client_manager)

        # here we simply insert an element in the config dictionary
        # to signal whether the client receiving the instrcutions should
        # save the model or not (all clients receive the same instructions)
        # if it's the last round, force to save
        if server_round == self.total_rounds:
            self.signal_clients_to_save_model = True
            print("Last round, ensuring all clients save model.")
        for _, fitins in configure_fit:
            fitins.config['save_model'] = self.signal_clients_to_save_model

        self.signal_clients_to_save_model = False
        return configure_fit

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
            # signal that clients must save model before training in a new round
            self.signal_clients_to_save_model = True

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"Dice": aggregated_accuracy}

def get_on_fit_config_fn():
    def on_fit_config(server_round: int):
        return {"current_round": server_round}
    return on_fit_config

def get_evaluate_fn(server_dataset):
    """This function returns a function that will be executed by the 
    strategy after aggregation when invoking its evaluate() method. It
    can be used to evalute the global model on a dataset hosted by
    the server."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        """Use the a test set for centralized evaluation."""

        log(INFO,"Evaluating global model on a dataset held by the server")
        log(INFO," --------------------------- WARNING ----------------------")
        log(INFO, "\t\t Global Model evaluation is not implemented")
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

rounds = 100
server_dataset = None # load dataset/dataloader
config = ServerConfig(num_rounds=rounds)

# Create strategy and run server
strategy = AggregateCustomMetricStrategy(
    total_rounds=rounds,
    save_global_path='global_models',
    on_fit_config_fn=get_on_fit_config_fn(),
    evaluate_fn=get_evaluate_fn(server_dataset)) # pass your dataset here

# Flower ServerApp
# Launch via `flower-server-app server:app`
app = ServerApp(
    config=config,
    strategy=strategy,
)

# Legacy code
# Launch via `python server.py`
def main():

    log(INFO, "PLEASE LOAD YOUR SERVER-SIDE dataset")
    server_dataset = None # load dataset/dataloader

    rounds = 100

    # Create strategy and run server
    strategy = AggregateCustomMetricStrategy(
        total_rounds=rounds,
        save_global_path='global_models',
        evaluate_fn=get_evaluate_fn(server_dataset)) # pass your dataset here
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )


if __name__ == "__main__":
    main()