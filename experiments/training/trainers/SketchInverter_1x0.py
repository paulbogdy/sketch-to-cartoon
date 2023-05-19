import random

from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from dataset.dataset import SketchInverterDataset
from models.SketchInverter import Encoder, Sketcher
from training.SketchInverterTrainer import SketchInverterTrainer
import pickle
import torch

batch_size = 8
num_epochs = 1
model_size = 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))
dataset = SketchInverterDataset(root_dir="../../../dataset/synthetic_dataset_anime_faces/", transform=ToTensor())


subset_size = int(len(dataset) * 0.01)
indices = random.sample(range(len(dataset)), subset_size)
subset = Subset(dataset, indices)


with open('network-snapshot-000880.pkl', 'rb') as f:
    generator = pickle.load(f)['G_ema']

sketcher = Sketcher(model_size=model_size)
encoder = Encoder(generator.z_dim, model_size=model_size)

trainer = SketchInverterTrainer(encoder, generator, sketcher, device, subset, "SketchInverter_1x0")
#trainer.continue_training(batch_size, num_epochs)

trainer.save_for_QT()
