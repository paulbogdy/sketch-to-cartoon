from dataset.dataset import Donarobu128Dataset
from torchvision.transforms import ToTensor
from training.SketchyGanTrainer import SketchyGanTrainer
import torch
import random
from torch.utils.data import Subset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = Donarobu128Dataset(root_dir="../dataset/archive/danbooru-sketch-pair-128x/", transform=ToTensor())
subset_size = int(len(dataset) * 0.01)

# Create a list of random unique indices
indices = random.sample(range(len(dataset)), subset_size)

# Create a subset of the dataset using the random indices
subset = Subset(dataset, indices)

trainer = SketchyGanTrainer(device, subset)
start_epoch = trainer.load_checkpoint("checkpoints/checkpoint_epoch_0.pt")

trainer.train(8, 10, start_epoch=start_epoch)
