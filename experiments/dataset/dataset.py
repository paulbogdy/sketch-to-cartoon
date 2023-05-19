import os

import torch
from PIL import Image
from torch.utils.data import Dataset


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
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
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

    def __len__(self):
        return len(self.src_paths)

    def __getitem__(self, idx):
        sketch_path = self.sketch_paths[idx]
        src_path = self.src_paths[idx]
        point_path = self.point_paths[idx]
        try:
            sketch_image = Image.open(sketch_path)
            src_image = Image.open(src_path)
            point_tensor = torch.load(point_path)
            valid_image = True
        except (ValueError, IOError):
            print(f"Invalid {src_path}")
            sketch_image = Image.new('L', (512, 512), color='black')
            src_image = Image.new('RGB', (512, 512), color='black')
            point_tensor = torch.zeros(1, 512)
            valid_image = False
        if self.transform:
            sketch_image = self.transform(sketch_image)
            src_image = self.transform(src_image)

        return sketch_image[:1], src_image, point_tensor, valid_image


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