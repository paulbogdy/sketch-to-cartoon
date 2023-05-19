from models.HEDNet import HDENet
from models.SketchInverter import Encoder
from training.GanInversionExperiment import GanInversionExperiment
import pickle
import torch

batch_size = 32
num_epochs = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from models.StyleGan2 import Generator

class Generator2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.z_dim = 512
        self.g_ema = Generator(
            256, 512, 8, channel_multiplier=2
        ).cuda()
        checkpoint = torch.load('../../pretrained/NaverWebtoon-040000.pt')
        self.g_ema.load_state_dict(checkpoint['g_ema'])

    def forward(self, x):
        return self.g_ema([x], 0, None)[0]


generator = Generator2()
sketcher = HDENet()
encoder = Encoder(generator.z_dim)

# print(sum(p.numel() for p in generator.parameters()))
# print(sum(p.numel() for p in sketcher.parameters()))
print(sum(p.numel() for p in encoder.parameters()))

trainer = GanInversionExperiment("GanInv_2x0", encoder, generator, sketcher, ".")

trainer.load_experiment("test2", checkpoint_nr=180)
print("FID 180: ", trainer.compute_fid2(num_samples=1000))

trainer.load_experiment("test2")
print("FID Final: ", trainer.compute_fid2(num_samples=1000))
#
# trainer.load_experiment("test")
# fid = trainer.compute_fid2(num_samples=1000)
# print(fid)

