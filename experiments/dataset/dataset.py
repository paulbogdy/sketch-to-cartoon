import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class Donarobu128Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sketch_paths = []
        self.src_paths = []
        for color_or_gray in ['color', 'gray']:
            sketch_dir = os.path.join(self.root_dir, color_or_gray, 'sketch')
            src_dir = os.path.join(self.root_dir, color_or_gray, 'src')
            for i in range(151):
                folder = f'0{i:03}'
                for filename in os.listdir(os.path.join(sketch_dir, folder)):
                    if filename.endswith('.png') or filename.endswith('.jpg'):
                        sketch_path = os.path.join(sketch_dir, folder, filename)
                        src_path = os.path.join(src_dir, folder, filename)
                        self.sketch_paths.append(sketch_path)
                        self.src_paths.append(src_path)

    def __len__(self):
        return len(self.sketch_paths)

    def __getitem__(self, idx):
        sketch_path = self.sketch_paths[idx]
        src_path = self.src_paths[idx]
        try:
            sketch_image = Image.open(sketch_path)
            src_image = Image.open(src_path)
            valid_image = True
        except (ValueError, IOError):
            sketch_image = Image.new('L', (128, 128), color='black')
            src_image = Image.new('RGB', (128, 128), color='black')
            valid_image = False
        if self.transform:
            sketch_image = self.transform(sketch_image)
            src_image = self.transform(src_image)
        return sketch_image, src_image, valid_image


class SketchInverterDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=(512, 512), z_dim=512):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.z_dim = z_dim
        self.sketch_paths = []
        self.src_paths = []
        self.point_paths = []
        sketch_dir = os.path.join(self.root_dir, 'sketch')
        src_dir = os.path.join(self.root_dir, 'src')
        point_dir = os.path.join(self.root_dir, 'points')
        for filename in os.listdir(src_dir):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                sketch_path = os.path.join(sketch_dir, filename)
                src_path = os.path.join(src_dir, filename)
                point_path = os.path.join(point_dir, filename.split('.')[0] + '.pt')
                self.sketch_paths.append(sketch_path)
                self.src_paths.append(src_path)
                self.point_paths.append(point_path)

    def preprocess_images(self):
        valid_src_paths = []
        valid_sketch_paths = []
        valid_point_paths = []

        for src_path, sketch_path, point_path in tqdm(zip(self.src_paths, self.sketch_paths, self.point_paths),
                                                      desc="Checking images"):
            try:
                Image.open(src_path)
                Image.open(sketch_path)
                torch.load(point_path)
                valid_src_paths.append(src_path)
                valid_sketch_paths.append(sketch_path)
                valid_point_paths.append(point_path)
            except (IOError, ValueError):
                print(f"Invalid data at: {src_path}, {sketch_path}, {point_path}")

        self.src_paths = valid_src_paths
        self.sketch_paths = valid_sketch_paths
        self.point_paths = valid_point_paths

    def __len__(self):
        return len(self.src_paths)

    def __getitem__(self, idx):
        sketch_path = self.sketch_paths[idx]
        src_path = self.src_paths[idx]
        point_path = self.point_paths[idx]
        sketch_image = Image.open(sketch_path)
        src_image = Image.open(src_path)
        point_tensor = torch.load(point_path)

        if self.transform:
            sketch_image = self.transform(sketch_image)
            src_image = self.transform(src_image)

        return sketch_image[:1], src_image, point_tensor


class DatasetForTestingDiscriminator(Dataset):
    def __init__(self, root_dir, src_name="src", transform=None, image_size=(256, 256)):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.src_paths = []
        src_dir = os.path.join(self.root_dir, src_name)
        for filename in os.listdir(src_dir):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                src_path = os.path.join(src_dir, filename)
                self.src_paths.append(src_path)

    def __len__(self):
        return len(self.src_paths)

    def __getitem__(self, idx):
        src_path = self.src_paths[idx]
        src_image = Image.open(src_path)

        if self.transform:
            src_image = self.transform(src_image)

        return src_image

class CartoonImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.src_paths = []

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith('.png') or filename.endswith('.jpg'):
                        src_path = os.path.join(folder_path, filename)
                        self.src_paths.append(src_path)

    def __len__(self):
        return len(self.src_paths)

    def __getitem__(self, idx):
        src_path = self.src_paths[idx]
        try:
            src_image = Image.open(src_path)
            valid_image = True
        except (ValueError, IOError):
            print(f"Invalid {src_path}")
            src_image = Image.new('RGB', (512, 512), color='black')
            valid_image = False

        if self.transform:
            src_image = self.transform(src_image)

        return src_image, valid_image