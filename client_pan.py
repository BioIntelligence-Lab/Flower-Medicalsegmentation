import argparse
from pathlib import Path
from datetime import datetime
from monai.utils import first
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import torch
import numpy as np
from flwr.client import NumPyClient, ClientApp
from flwr.client.mod import secaggplus_mod

import msd
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl

USE_FEDBN: bool = True

# pylint: disable=no-member
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Flower Pancrease Client for Medical Segmentation Decathlon")
parser.add_argument(
    "--pancreas-path",
    required=True,
    type=str,
    help="Path to the Pancreas dataset (e.g. datasets/Task07_Pancreas). Download from medicaldecathlon.com.",
)
parser.add_argument(
    "--save-path",
    required=True,
    type=str,
    help="Path where this client will save local models (if doesn't exist, directory will be created)",
)

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
        save_path: str,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

        # prepare directory where models will be saved
        self.path = Path(save_path)/datetime.now().strftime('%d-%m-%Y/%H-%M-%S')
        self.path.mkdir(parents=True, exist_ok=True)
        print(f"Client will save to: {str(self.path)}")

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
        if config['save_model']:
            print("SAVE MODEL!! -- must implement")

            #TODO: save model into self.path
            torch.save(self.model.state_dict(), str(self.path / f"pan_local_model_round_{config['current_round']}.pth"))

        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        msd.train(self.model, self.trainloader, max_epochs=2, device=DEVICE)
        #torch.save(self.model.state_dict(), os.path.join(root_dir, "best_metric_model_pan_128_segviz_flwr.pth"))
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        accuracy = msd.validate(self.model, self.testloader, device=DEVICE)
        return float(accuracy), self.num_examples["valset"], {"Dice": float(accuracy)}

# Flower Next API
def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    args = parser.parse_args()
    data_dir_pan = args.pancreas_path # Local path to data. Should contain imagesTr and labelsTr subdirs
    # Load data
    trainloader, testloader, num_examples = msd.load_data(data_dir_pan, -87, 199)

    # Load model
    model = UNet(**config['model_params']).to(DEVICE).train()

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(first(trainloader)["image"].to(DEVICE))
    return MSDClient(model, trainloader, testloader,
                       num_examples, save_path=args.save_path).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[
        secaggplus_mod,
    ],
)

# Legacy code
# def main() -> None:
#     """Load data, start MSDClient."""

#     args = parser.parse_args()
#     data_dir_pan = args.pancreas_path # Local path to data. Should contain imagesTr and labelsTr subdirs
#     # Load data
#     trainloader, testloader, num_examples = msd.load_data(data_dir_pan, -87, 199)

#     # Load model
#     model = UNet(**config['model_params']).to(DEVICE).train()

#     # Perform a single forward pass to properly initialize BatchNorm
#     _ = model(first(trainloader)["image"].to(DEVICE))

#     # Start client
#     client = MSDClient(model, trainloader, testloader,
#                        num_examples, save_path=args.save_path).to_client()
#     # Legacy code
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


# if __name__ == "__main__":
#     main()
