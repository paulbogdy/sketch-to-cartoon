from models.HEDNet import HDENet
from models.SketchInverter import Encoder
from training.GanInversionTrainer import GanInversionTrainer
import pickle
import torch

batch_size = 8
num_epochs = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('network-snapshot-000880.pkl', 'rb') as f:
    generator = pickle.load(f)['G_ema']
sketcher = HDENet()
encoder = Encoder(generator.z_dim)

# print(sum(p.numel() for p in generator.parameters()))
# print(sum(p.numel() for p in sketcher.parameters()))
print(sum(p.numel() for p in encoder.parameters()))

trainer = GanInversionTrainer(device, "GanInv_1x0", generator, sketcher, encoder)
trainer.continue_training(batch_size, num_epochs)

# trainer.continue_training(batch_size, num_epochs)

