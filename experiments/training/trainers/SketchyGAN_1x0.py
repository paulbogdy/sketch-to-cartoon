from dataset.dataset import CartoonImageDataset
from torchvision.transforms import ToTensor
from training.SketchyGanTrainer import SketchyGanTrainer
from models.HEDNet import HDENet
import torch
import random
from torch.utils.data import Subset

batch_size = 8
num_epochs = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = CartoonImageDataset(root_dir="../../../../../naver-webtoon-data/faces/", transform=ToTensor())

sketcher = HDENet().to(device)

trainer = SketchyGanTrainer(device, dataset, "SketchyGAN_1x0", sketcher)

# trainer.save_for_QT(checkpoint_path="checkpoints/SketchyGAN_1x0/checkpoint_epoch_8.pt")

trainer.continue_training(batch_size, num_epochs)
