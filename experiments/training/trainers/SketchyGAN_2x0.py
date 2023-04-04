from dataset.dataset import MyDataset
from torchvision.transforms import ToTensor
from training.SketchyGanTrainer import SketchyGanTrainer
from models.SketchyGan import SketchyDiscriminatorInstaNorm, SketchyGeneratorInstaNorm
import torch
import random
import torch.optim as optim
from torch.utils.data import Subset


class AdjustedSketchyGanTrainer(SketchyGanTrainer):
    def __init__(self, device, dataset, model_name):
        super().__init__(device, dataset, model_name)
        self.generator = SketchyGeneratorInstaNorm(input_channels=self.noise_channels)
        self.generator.to(self.device)
        self.discriminator = SketchyDiscriminatorInstaNorm()
        self.discriminator.to(self.device)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0004, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


batch_size = 8
num_epochs = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = MyDataset(root_dir="../../../dataset/archive/danbooru-sketch-pair-128x/", transform=ToTensor())


subset_size = int(len(dataset) * 0.05)
indices = random.sample(range(len(dataset)), subset_size)
subset = Subset(dataset, indices)

trainer = AdjustedSketchyGanTrainer(device, subset, "SketchyGAN_2x0")

trainer.continue_training(batch_size, num_epochs)
