# Flower for Federated 3D Medical Image Segmentation 

This repository is maintained to develop and test the [Flower framework](https://flower.ai/) for 3D medical image segmentation using the data from the Medical Segmentation Decathalon challenge.

This code is inspired by the official [Flower FedBN tutorial](https://flower.dev/docs/fedbn-example-pytorch-from-centralized-to-federated.html) 

# 🚀 Live Demo Instructions Coming Soon! 📢

We are excited to announce our paper was accepted at the DCAMI workshop at CVPR 2024! We will present a live demo with the ability to connect your own server and supernodes and be a part of an international federation! 🎉
Please stay tuned for detailed instructions on connecting to the Flower SuperLink, which will be posted here shortly. 🙌

In the meantime, feel free to explore the repository and familiarize yourself with the codebase. 🔍
If you have any questions or concerns, please don't hesitate to reach out by creating an issue or contacting us directly. 📩

We look forward to showcasing our work and engaging with the community at CVPR 2024! 🤖🌟

## Environment Setup

```bash
# Create a conda environment
conda create -n um2ii-flower python=3.10 -y

# Activate the environment
conda activate um2ii-flower # or source activate um2ii-flower

# Install requirements
pip install -r requirements.txt
```
To download a particular organ type from the MSD Data, please refer the following addresses for each dataset.
* Brain Tumors - https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
* Heart - https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar
* Liver - https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar
* Hippocampus - https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar
* Prostate - https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar
* Lung - https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar
* Pancreas - https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar
* Hepatic Vessel - https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar
* Spleen - https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar
* Colon - https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar

Before you start using the scripts, please ensure that you have the Pancreas and Spleen data accessible locally and in the right format as detailed in our framework [SegViz](https://github.com/UM2ii/SegViz). The direct commands to download and extract the dataset are:

```bash
# let's create first a dataset directory
mkdir dataset
cd dataset

# Download Spleen dataset (~1.5GB)
wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar # Change as per requirement
# now extract it
tar -xvf Task09_Spleen.tar

# Download Pancreas dataset (~11.5 GB)
wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar # Change as per requirement
# now extract it
tar -xvf Task07_Pancreas.tar
```

The Pancreas dataset has normal and tumor labels, so please use the script below to convert the dataset into single-label (organ) only. 

```python
import nibabel as nib
data_dir_pan = 'path_to_dir_containing_images'
for image_path in sorted(glob.glob(os.path.join(data_dir_pan, "labelsTr", "*.nii.gz"))):
    image_file = nib.load(image_path)
    image_file_array = nib.load(image_path).get_fdata()
    image_file_array[image_file_array > 1 ] = 1
    image_file_final = nib.Nifti1Image(image_file_array, image_file.affine)
    nib.save(image_file_final, image_path)  
```

## Run the experiment (Centralized)

```bash
python3 msd.py --spleen-path <path/to/spleen/dataset> --pancreas-path <path/to/spleen/dataset>
```

## Run the experiment (Federated with `Flower Next`)

You will need to run the following scripts in separate terminal windows.

Flower Next integrates a collection of new featuers that will be gradually incorporated in the usual `flwr` package. But you can start taking advantage of them by running your experiments making use of `ClientApp` and `ServerApp`. You can find a guide on how to upgrade to Flower Next style in the [Flower Documentation](https://flower.ai/docs/framework/how-to-upgrade-to-flower-next.html). For the purpose of this project:

You'll need to run first the `SuperLink`, to which you can pass certificates if you wish. Check the documentation for that. The lines below show how to do this w/o certificates.

```bash
# start the superlink
flower-superlink --insecure
```

Next, you'll need your `SuperNode` (i.e. nodes containing the data that will eventually do training) to the `SuperLink`. Repeat the below for as many nodes as you have, each pointing to their local data. 

```bash
# launches supernode wich will execute the the `ClientApp` in `client_spleen.py`
flower-client-app client_spleen:app --insecure --superlink=<SUPERLINK_SERVER_IP>

# launch supernode pointing to a `ClientApp` making use of the pancreas data
flower-client-app client_pan:app --insecure --superlink=<SUPERLINK_SERVER_IP>
```
When client save a model, they will follwow the directory structure: `save-path/date/time/<model>`

With the above done, you'll see nothing seems to happen. The `SuperNodes` periodically ping the `SuperLink` for messages that the `ServerApp` (which we haven't launched yet) is sending them. Without further due, let's launch the `ServerApp` to start the federation:

```bash
flower-server-app server:app --insecure --superlink=<SUPERLINK_SERVER_IP>
```

You'll notice once the N rounds finish, the `SuperLink` and `SuperNode` remain idle. You can launch another `ServerApp` to start a new experiment (yes, without having to restart the `SuperNode` or `SuperLink`)

## Seamless Collaboration Across Institutions Using Modifiable Task Blocks in Superlink Models

When multiple institutions collaborate on a project using the superlink, each institution may focus on training different tasks. As a result, they will have separate models with the same representation block but different task blocks tailored to their specific tasks.
To facilitate seamless collaboration, each institution will have a copy of the global model on their server. 

We have provided a Python script that allows an institution to modify the task block from another institution's model, enabling them to run inference for the new task, even if they have never trained on that particular data.

This approach ensures that institutions can easily collaborate and leverage each other's work, even if they have focused on different tasks during training.

Execute the script from the command line with the necessary arguments. The script requires the paths to two existing model state dictionaries, the desired number of output channels for the new task block, and the path to save the new model.

```sh
python merge_modify_model.py --model_1_path /path/to/model_1.pth --model_2_path /path/to/model_2.pth --out_channels 4 --save_path /path/to/save_new_model.pth
