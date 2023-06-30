# Flower for 3D medical image segmentation 

This repository is maintained to develop and test the Flower framework for 3D medical image segmentation using the data from the Medical Segmentation Decathalon.

This code is inspired by the official [Flower FedBN tutorial](https://flower.dev/docs/fedbn-example-pytorch-from-centralized-to-federated.html) 

Before you start using the scripts, please ensure that you have the Pancreas and Spleen data accessible locally and in the right format as detailed in our framework [SegViz](https://github.com/UM2ii/SegViz)

The Pancreas dataset has normal and tumor labels, so please use the script below to convert the dataset into single-label (organ) only. 

```
import nibabel as nib
data_dir_pan = 'path_to_dir_containing_images'
for image_path in sorted(glob.glob(os.path.join(data_dir_pan, "labelsTr", "*.nii.gz"))):
    image_file = nib.load(image_path)
    image_file_array = nib.load(image_path).get_fdata()
    image_file_array[image_file_array > 1 ] = 1
    image_file_final = nib.Nifti1Image(image_file_array, image_file.affine)
```

You will need to run the following scripts in separate terminal windows.

```
# Start the Flower central server using 
python3 msd.py
```

```
# Start the Flower server using 
python3 server.py
```

```
# Start the Flower Liver client using 
python3 client_spleen.py
```
```
# Start the Flower Pancreas client using 
python3 client_pan.py
```


