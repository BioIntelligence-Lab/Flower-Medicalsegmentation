import os
import glob
import nibabel as nib
import argparse

def process_images(data_dir):
    for image_path in sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz"))):
        image_file = nib.load(image_path)
        image_file_array = nib.load(image_path).get_fdata()
        image_file_array[image_file_array > 1] = 1
        image_file_final = nib.Nifti1Image(image_file_array, image_file.affine)
        nib.save(image_file_final, image_path)

def main():
    parser = argparse.ArgumentParser(description='Process .nii.gz images in the specified directory.')
    parser.add_argument('data_dir', type=str, help='Path to the directory containing the images')
    args = parser.parse_args()

    process_images(args.data_dir)

if __name__ == '__main__':
    main()
