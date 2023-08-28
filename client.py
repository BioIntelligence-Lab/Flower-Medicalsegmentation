import pickle
from pathlib import Path

import hydra
from hydra.utils import instantiate, call
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evalulate_fn

