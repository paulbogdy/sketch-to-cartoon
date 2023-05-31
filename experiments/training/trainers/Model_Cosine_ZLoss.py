import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.HEDNet import HDENet
from models.SketchInverter import Encoder
from training.SimpleExperiment import SimpleExperiment
from pathlib import Path
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from pretrained.PretrainedGenerator import CartoonGenerator

root_path = Path(__file__).parent.parent.parent

generator = CartoonGenerator(root_path)
sketcher = HDENet()
encoder = Encoder(generator.z_dim, (1, 256, 256))

SimpleExperiment("Model_Cosine_ZLoss",
                 SimpleExperiment.Datasets.CARTOON,
                 encoder,
                 generator,
                 sketcher,
                 root_path).run_experiment(
    batch_size=32,
    num_epochs=50,
    accumulation_steps=1,
    save_every_n_batches=1000,
    show_every_n_steps=1000,
)
