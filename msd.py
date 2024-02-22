"""Flower code for FedBN on MSD
"""

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
# import wandb
import copy
import nibabel as nib

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

def load_data(data_dir, a_min, a_max):
    """Loads the MSD dataset
    """

    train_images = sorted(
    glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    train_files, val_files = data_dicts[:-round(len(data_dicts)*0.2)], data_dicts[-round(len(data_dicts)*0.2):]

    train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"], a_min=a_min, a_max=a_max,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        Resized(keys=["image"], spatial_size=(256,256,128)),   
        Resized(keys=["label"], spatial_size=(256,256,128), mode='nearest'),   
        # CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(
            1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128,128,32),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=a_min, a_max=a_max,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ]
    )

    train_ds = CacheDataset(
    data=train_files, transform=train_transforms,
    cache_rate=config['cache_rate'], num_workers=config['num_workers'])
    # train_ds = Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=config['train_batch_size'], shuffle=True, num_workers=config['num_workers'])

    val_ds = CacheDataset(
        data=val_files, transform=val_transform, cache_rate=config['cache_rate'], num_workers=config['num_workers'])
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=config['val_batch_size'], num_workers=config['num_workers'])

    num_examples = {"trainset": len(train_ds), "valset": len(val_ds)}

    return train_loader, val_loader, num_examples



def train( model: UNet(**config['model_params']),
    train_loader: torch.utils.data.DataLoader,
    max_epochs: int,
    device: torch.device,):

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    epoch_loss_values = []
    
    model.to(device)
    
    for epoch in range(max_epochs):
        epoch_loss = 0

        step_0 = 0
        
        # For one epoch
        
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        
        model.train()
        
        
        # One forward pass of the spleen data through the spleen UNet
        for batch_data in train_loader:
            step_0 += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = loss_function(outputs, labels)
            loss.backward()
            
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"train_loss: {loss.item():.4f}")
            #wandb.log({"train/loss: ": loss.item()})
        epoch_loss /= step_0
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")


def validate( model: UNet(**config['model_params']),
    val_loader: torch.utils.data.DataLoader,
    device: torch.device, ):
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['max_epochs'], eta_min=1e-9)

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    
    metric_values = []

    model.to(device)
    model.eval()

    with torch.no_grad():

        # Validation forward spleen
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model)
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)

        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        scheduler.step(metric)
        # reset the status for next validation round
        dice_metric.reset()

        metric_values.append(metric)

        return metric
    
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Centralized PyTorch training")
    print("Load data")
    
    trainloader_spleen, testloader_spleen, _ = load_data('/mnt/hdd1/Task09_Spleen') # Change path to spleen data
    trainloader_pan, testloader_pan, _ = load_data('/mnt/hdd1/Task07_Pancreas')    # Change path to pancreas data
    
    net_spleen = UNet(**config['model_params']).to(DEVICE)
    net_spleen.eval()

    net_pan = UNet(**config['model_params']).to(DEVICE)
    net_pan.eval()

    print("Start training Spleen")
    train(model=net_spleen, train_loader=trainloader_spleen, max_epochs=100, device=DEVICE)
    print("Validate model Spleen")
    dice_spleen = validate(model=net_spleen, val_loader=testloader_spleen, device=DEVICE)
    print("Dice metric Spleen: ", dice_spleen)

    print("Start training Pancreas")
    train(model=net_pan, train_loader=trainloader_pan, max_epochs=100, device=DEVICE)
    print("Validate model Pancreas")
    dice_liver = validate(model=net_pan, val_loader=testloader_pan, device=DEVICE)
    print("Dice metric Liver: ", dice_liver)


if __name__ == "__main__":
    main()

