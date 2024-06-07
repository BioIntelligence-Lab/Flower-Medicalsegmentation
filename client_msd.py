import argparse
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from typing import Dict, List, Tuple
import os
import glob
import torch
import numpy as np
import flwr as fl
from flwr.client import ClientApp
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, ScaleIntensityRanged,
    ScaleIntensityd, Orientationd, Spacingd, Resized
)
from monai.utils import first
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import DataLoader, Dataset

import msd

USE_FEDBN = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "output"

dataset_info = {
    "01": {"name": "BrainTumour", "type": "MR", "intensity_min": 0, "intensity_max": 200},
    "02": {"name": "Heart", "type": "MR", "intensity_min": 0, "intensity_max": 200},
    "03": {"name": "Liver", "type": "CT", "intensity_min": -17, "intensity_max": 201},
    "04": {"name": "Hippocampus", "type": "MR", "intensity_min": 0, "intensity_max": 200},
    "05": {"name": "Prostate", "type": "MR", "intensity_min": 0, "intensity_max": 200},
    "06": {"name": "Lung", "type": "CT", "intensity_min": -1024, "intensity_max": 325},
    "07": {"name": "Pancreas", "type": "CT", "intensity_min": -96.0, "intensity_max": 215.0},
    "08": {"name": "HepaticVessel", "type": "CT", "intensity_min": -3, "intensity_max": 243},
    "09": {"name": "Spleen", "type": "CT", "intensity_min": -41, "intensity_max": 176},
    "10": {"name": "Colon", "type": "CT", "intensity_min": -30.0, "intensity_max": 165.82}
}

config = {
    "cache_rate": 1.0,
    "num_workers": 0,
    "train_batch_size": 2,
    "val_batch_size": 1,
    "learning_rate": 1e-4,
    "max_epochs": 1000,
    "val_interval": 2,
    "lr_scheduler": "cosine_decay",
    "model_type": "unet",
    "model_params": dict(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ),
}

class MSDClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, num_examples, save_path):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples
        self.path = Path(save_path) / datetime.now().strftime('%d-%m-%Y/%H-%M-%S')
        self.path.mkdir(parents=True, exist_ok=True)
        print(f"Client will save to: {str(self.path)}")

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        self.model.train()
        if USE_FEDBN:
            return [val.cpu().numpy() for name, val in self.model.state_dict().items() if "model.2" not in name]
        else:
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
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

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        if config['save_model']:
            torch.save(self.model.state_dict(), str(self.path / f"local_model_round_{config['current_round']}.pth"))
        self.set_parameters(parameters)
        msd.train(self.model, self.trainloader, max_epochs=10, device=DEVICE)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        accuracy = msd.validate(self.model, self.testloader, device=DEVICE)
        return float(accuracy), self.num_examples["valset"], {"Dice": float(accuracy)}

def load_dataset(data_dir, dataset_type, dataset_name, min_intensity, max_intensity):
    if dataset_name in ["Prostate", "BrainTumour", "Heart"]:
        train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*_0000.nii.gz")))
    else:
        train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    
    data_dicts = [{"image": image_name} for image_name in train_images]
    transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
    ]
    
    if dataset_type == "MR":
        transforms.append(NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True))
    elif dataset_type == "CT":
        if dataset_name == "Liver":
            transforms.extend([
                ScaleIntensityRanged(keys=["image"], a_min=-17, a_max=201, b_min=0.0, b_max=1.0, clip=True),
                ScaleIntensityd(keys=["image"], factor=39.36, shift=-99.40),
            ])
        elif dataset_name == "Lung":
            transforms.extend([
                ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=325, b_min=0.0, b_max=1.0, clip=True),
                ScaleIntensityd(keys=["image"], factor=324.70, shift=-(-158.58)),
            ])
        elif dataset_name == "Pancreas":
            transforms.extend([
                ScaleIntensityRanged(keys=["image"], a_min=-96.0, a_max=215.0, b_min=0.0, b_max=1.0, clip=True),
                ScaleIntensityd(keys=["image"], factor=75.40, shift=-77.99),
            ])
        elif dataset_name == "HepaticVessel":
            transforms.extend([
                ScaleIntensityRanged(keys=["image"], a_min=-3, a_max=243, b_min=0.0, b_max=1.0, clip=True),
                ScaleIntensityd(keys=["image"], factor=52.62, shift=-104.37),
            ])
        elif dataset_name == "Spleen":
            transforms.extend([
                ScaleIntensityRanged(keys=["image"], a_min=-41, a_max=176, b_min=0.0, b_max=1.0, clip=True),
                ScaleIntensityd(keys=["image"], factor=39.47, shift=-99.29),
            ])
        elif dataset_name == "Colon":
            transforms.extend([
                ScaleIntensityRanged(keys=["image"], a_min=-30.0, a_max=165.82, b_min=0.0, b_max=1.0, clip=True),
                ScaleIntensityd(keys=["image"], factor=32.65, shift=-62.18),
            ])

    transforms.extend([
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
        Resized(keys=["image"], spatial_size=(128, 128, 32), mode=["nearest"])
    ])

    dataset = Dataset(data=data_dicts, transform=transforms)
    return DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4), len(dataset)

def client_fn(cid: str, dataset_id: str):
    dataset_name = dataset_info[dataset_id]["name"]
    dataset_type = dataset_info[dataset_id]["type"]
    data_dir = f"dataset/Task{dataset_id}_{dataset_name}"
    min_intensity = dataset_info[dataset_id]["intensity_min"]
    max_intensity = dataset_info[dataset_id]["intensity_max"]

    trainloader, num_train = load_dataset(data_dir, dataset_type, dataset_name, min_intensity, max_intensity)
    testloader, num_test = load_dataset(data_dir, dataset_type, dataset_name, min_intensity, max_intensity)

    model = UNet(**config['model_params']).to(DEVICE).train()
    _ = model(first(trainloader)["image"].to(DEVICE))

    num_examples = {"trainset": num_train, "valset": num_test}
    return MSDClient(model, trainloader, testloader, num_examples, save_path=SAVE_PATH).to_client()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Client for MSD")
    parser.add_argument("--dataset_id", type=str, required=True, help="Dataset ID from MSD (e.g., 01 for BrainTumour)")
    args = parser.parse_args()

    app = ClientApp(client_fn=lambda cid: client_fn(cid, args.dataset_id))
    app.start()
