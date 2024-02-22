from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    RandAffined,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    Resized,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet, BasicUNet
from monai.networks.layers import Norm
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.metrics import DiceMetric, ROCAUCMetric, MSEMetric
from monai.networks.utils import copy_model_state
from monai.optimizers import generate_param_groups
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import numpy as np
import wandb
import copy
import nibabel as nib

import msd

import os
import sys
import timeit
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl

root_dir = '/home/adwaykanhere/Documents/SegViz/'

USE_FEDBN: bool = True

# pylint: disable=no-member
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member
config = {
    # data
    "cache_rate": 1.0,
    "num_workers": 0,


    # train settings
    "train_batch_size": 2,
    "val_batch_size": 1,
    "learning_rate": 1e-4,
    "max_epochs": 1000,
    "val_interval": 2, # check validation score after n epochs
    "lr_scheduler": "cosine_decay", # just to keep track




    # Unet model (you can even use nested dictionary and this will be handled by W&B automatically)
    "model_type": "unet", # just to keep track
    "model_params": dict(spatial_dims=3,
                  in_channels=1,
                  out_channels=2,
                  channels=(16, 32, 64, 128, 256),
                  strides=(2, 2, 2, 2),
                  num_res_units=2,
                  norm=Norm.BATCH,),
}

# Flower Client
class MSDClient(fl.client.NumPyClient):
    """Flower client implementing MSD medical image segmentation using
    PyTorch and MONAI."""

    def __init__(
        self,
        model: UNet(**config['model_params']),
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

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

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        msd.train(self.model, self.trainloader, max_epochs=2, device=DEVICE)
        
        #torch.save(self.model.state_dict(), os.path.join(root_dir, "best_metric_model_spleen_128_segviz_flwr.pth"))        
        
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        accuracy = msd.validate(self.model, self.testloader, device=DEVICE)
        return float(accuracy), self.num_examples["valset"], {"Dice": float(accuracy)}


def main() -> None:
    """Load data, start MSDClient."""

    data_dir_spleen = '/mnt/hdd1/Task09_Spleen' # Local path to data. Should contain imagesTr and labelsTr subdirs
    # Load data
    trainloader, testloader, num_examples = msd.load_data(data_dir_spleen, -57, 164)

    # Load model
    model = UNet(**config['model_params']).to(DEVICE).train()

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(first(trainloader)["image"].to(DEVICE))


    # Start client
    client = MSDClient(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
