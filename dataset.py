import torch
import torch.nn as nn
import os
import numpy as np
import glob as glob
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    RandScaleIntensityd,
    Compose,
    CropForegroundd,
    LoadImaged,
    RandShiftIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    RandAffined,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    Resized,
)
from monai.data import DataLoader, Dataset
from hydra.utils import call, instantiate
from hydra.core.hydra_config import HydraConfig

def get_data(data_path: str, img_min: int = -57, img_max: int = 164, train_batch_size: int = 2, val_batch_size: int =1, num_workers: int = 4):

    # Transforms for CT images
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min = img_min, a_max=img_max,
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
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            # Feel free to add more transforms here.
            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=1.0, spatial_size=(128,128,32),
                rotate_range=(0, 0, np.pi/15),
                scale_range=(0.1, 0.1, 0.1)),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ]
    )
    train_images = sorted(
    glob.glob(os.path.join(data_path, "imagesTr", "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(data_path, "labelsTr", "*.nii.gz")))
    data_dicts_spleen = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    train_files, val_files = data_dicts_spleen[:-(0.2*(len(data_dicts_spleen)))], data_dicts_spleen[-(0.2*(len(data_dicts_spleen))):]
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, num_workers=num_workers)

    return train_loader, val_loader

