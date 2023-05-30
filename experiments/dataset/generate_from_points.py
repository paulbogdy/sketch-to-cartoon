import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import numpy as np
import torchvision.utils
from torchvision.transforms import ToTensor
from pretrained.PretrainedGenerator import CartoonGenerator
from models.HEDNet import HDENet
import os
from PIL import Image
from tqdm import tqdm
import torch


def from_points(generator, sketcher, path, device):
    points_paths = os.listdir(path)[:1000]
    src_dir = path.replace("points", "src_test")
    os.makedirs(src_dir, exist_ok=True)
    sketch_dir = path.replace("points", "sketch_test")
    os.makedirs(sketch_dir, exist_ok=True)
    generator.eval()
    sketcher.eval()
    for point_path in tqdm(points_paths, total=len(points_paths)):
        point = torch.load(os.path.join(path, point_path)).to(device)
        point = point.unsqueeze(0)
        path_name = point_path.split('.')[0]
        with torch.no_grad():
            fake_image = generator(point)
            torchvision.utils.save_image(fake_image[0], os.path.join(src_dir, f"{path_name}.png"))
            fake_image = sketcher(fake_image)
            torchvision.utils.save_image(fake_image[0], os.path.join(sketch_dir, f"{path_name}.png"))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_path = Path(__file__).parent.parent

S = HDENet().to(device)
G = CartoonGenerator(root_path).to(device)

from_points(G, S, "../../dataset/synthetic_dataset_cartoon_faces/points", device)

# valid_images = os.listdir("../../dataset/synthetic_dataset_cartoon_faces/src")

# # take the valid images names, and convert *.png to *.pt and write them in a file
# with open("../../dataset/synthetic_dataset_cartoon_faces/valid_images.txt", "w") as f:
#     for image in valid_images:
#         f.write(image.replace("png", "pt") + "\n")
