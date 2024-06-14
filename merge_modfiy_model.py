import torch
import os
import argparse
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Norm
from collections import OrderedDict

def create_model(out_channels):
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

def replace_task_block(model, out_channels):
    task_block = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    # Copy task block parameters
    new_state_dict = model.state_dict()
    task_block_state_dict = task_block.state_dict()
    
    for name, param in task_block_state_dict.items():
        if "model.2" in name:
            new_state_dict[name] = param
    
    model.load_state_dict(new_state_dict)
    return model

def create_modified_model(repr_block, out_channels):
    new_model = create_model(out_channels=out_channels)
    
    # Replace representation block
    new_state_dict = new_model.state_dict()
    for name, param in repr_block.items():
        new_state_dict[name] = param

    new_model.load_state_dict(new_state_dict)
    
    # Replace task block with new out_channels
    new_model = replace_task_block(new_model, out_channels)
    return new_model

def main(args):
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load models
    model_1 = create_model(out_channels=2)
    model_2 = create_model(out_channels=2)

    model_1.load_state_dict(torch.load(args.model_1_path))
    model_2.load_state_dict(torch.load(args.model_2_path))

    # Extract representation block from model_1
    repr_block_1 = OrderedDict()
    for name, param in model_1.named_parameters():
        if "model.2" not in name:
            repr_block_1[name] = param

    # Create new models with different out_channels
    new_model = create_modified_model(repr_block_1, out_channels=args.out_channels)

    # Save the new model
    torch.save(new_model.state_dict(), args.save_path)

    print(f"New model with out_channels={args.out_channels} saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new model with different task block out_channels")
    parser.add_argument("--model_1_path", type=str, required=True, help="Path to the first model state dict")
    parser.add_argument("--model_2_path", type=str, required=True, help="Path to the second model state dict")
    parser.add_argument("--out_channels", type=int, required=True, help="Number of out_channels for the new task block")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the new model state dict")

    args = parser.parse_args()
    main(args)
