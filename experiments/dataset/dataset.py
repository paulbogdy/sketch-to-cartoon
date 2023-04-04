import os
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
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



