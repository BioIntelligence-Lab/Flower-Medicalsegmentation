from pathlib import Path

import hydra
from hydra.utils import instantiate, call
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple
from model import MONAIUNet, train, validate

USE_FEDBN = False
class MedSegClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader):
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = MONAIUNet()

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        self.model.train()
        if USE_FEDBN:
            # Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
            return [val.cpu().numpy() for name, val in self.model.state_dict().items() if "model.2" not in name]
        else:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        if USE_FEDBN:
            keys = [name for name, val in self.model.state_dict().items() if "model.2" not in name]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        learning_rate = config["learning_rate"]
        max_epochs = config["max_epochs"]
        local_epochs = config["num_epochs"]

        # do local training

        train(self.model, self.trainloader, max_epochs, learning_rate, torch.device("cuda"))

        return self.get_parameters(), len(self.trainloader), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # do local evaluation
        learning_rate = config["learning_rate"]
        max_epochs = config["max_epochs"]
        loss, dice = validate(self.model, self.valloader, learning_rate, max_epochs, torch.device("cuda"))
        return float(loss), len(self.valloader), {'dice': float(dice)}
    
def generate_client_fn(trainloader, valloader, class_id):
    def get_client():
        return MedSegClient(trainloader, valloader)
    return get_client
