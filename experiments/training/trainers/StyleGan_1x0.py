from dataset.dataset import CartoonImageDataset
from torchvision.transforms import ToTensor
from training.StyleGanTrainer import StyleGanTrainer
from models.StyleGan2_new import Generator, Discriminator
import torch
import random
from torch.utils.data import Subset

batch_size = 32
num_epochs = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = CartoonImageDataset(root_dir="../../../../../naver-webtoon-data/faces/", transform=ToTensor())

generator = Generator(512).to(device)
gen_param = sum(p.numel() for p in generator.parameters())
discriminator = Discriminator(8).to(device)
disc_param = sum(p.numel() for p in discriminator.parameters())

print("Generator parameters: ", gen_param)
print("Discriminator parameters: ", disc_param)

trainer = StyleGanTrainer(device, dataset, "StyleGan_1x0", generator, discriminator)

# trainer.save_for_QT(checkpoint_path="checkpoints/SketchyGAN_1x0/checkpoint_epoch_8.pt")

trainer.continue_training(batch_size, num_epochs)
