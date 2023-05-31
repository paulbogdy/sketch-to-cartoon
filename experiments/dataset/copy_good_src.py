import os
import shutil
from pathlib import Path

dataset_dir = os.path.join(Path(__file__).parent.parent.parent, 'dataset', 'synthetic_dataset_cartoon_faces')

# File with the valid images
valid_images_file = os.path.join(dataset_dir, 'valid_images.txt')

# Source and destination directories
src_dir = os.path.join(dataset_dir, 'src')  # replace with your source directory
dst_dir = os.path.join(dataset_dir, 'src_good')  # replace with your destination directory

# Ensure destination directory exists
os.makedirs(dst_dir, exist_ok=True)

# Read valid image names from file
with open(valid_images_file, 'r') as f:
    valid_images = [line.strip().split('.')[0] for line in f.readlines()]


# Iterate over valid images
for image in valid_images:
    # Add the file extension to the image name
    image = f'{image}.png'

    # Define source and destination file paths
    src_file = os.path.join(src_dir, image)
    dst_file = os.path.join(dst_dir, image)

    # Check if source file exists and is a file
    if os.path.isfile(src_file):
        # Copy the file
        shutil.copy(src_file, dst_file)
    else:
        print(f'Source file not found: {src_file}')
