import numpy as np
import torchvision.utils
from torchvision.transforms import ToTensor

from models.HEDNet import HDENet
import os
from PIL import Image
from tqdm import tqdm
import torch


def regenerate_sketches(sketcher, path, device):
    sketches = os.listdir(path)
    transform = ToTensor()
    os.makedirs(path.replace("sketch", "sketch_regen"), exist_ok=True)
    for sketch_path in tqdm(sketches, total=len(sketches)):
        full_path = os.path.join(path, sketch_path)
        img = Image.open(full_path).convert("RGB")
        img = transform(img).to(device)
        sketch = sketcher(img)
        torchvision.utils.save_image(sketch, full_path.replace("sketch", "sketch_regen"))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

S = HDENet().to(device)

regenerate_sketches(S, "../../dataset/synthetic_dataset_cartoon_faces/sketch", device)
