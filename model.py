import torch
import torch.nn as nn
import torch.nn.functional as F

from flwr

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.metrics import DiceMetric,
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import Compose, AsDiscrete



def MONAIUNet():
    return UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
    )

def train( model,
    train_loader: torch.utils.data.DataLoader,
    max_epochs: int,
    learning_rate: float,
    device: torch.device,):

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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


def validate( model,
    val_loader: torch.utils.data.DataLoader,
    learning_rate: float,
    num_epochs: int,
    device: torch.device, ):
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-9)

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