from dataset.dataset import MyDataset
from torchvision.transforms import ToTensor
from training.SketchyGanTrainer import SketchyGanTrainer
import torch
import random
from torch.utils.data import Subset

batch_size = 16
num_epochs = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = MyDataset(root_dir="../../../dataset/archive/danbooru-sketch-pair-128x/", transform=ToTensor())


subset_size = int(len(dataset) * 0.05)
indices = random.sample(range(len(dataset)), subset_size)
subset = Subset(dataset, indices)

trainer = SketchyGanTrainer(device, subset, "SketchyGAN_3x0")

trainer.continue_training(batch_size, num_epochs)
