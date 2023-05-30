import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.HEDNet import HDENet
from models.SketchInverter import Encoder
from training.ExperimentFramework import Experiment
from pathlib import Path
import torch

batch_size = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from pretrained.PretrainedGenerator import CartoonGenerator

root_path = Path(__file__).parent.parent.parent

generator = CartoonGenerator(root_path)
sketcher = HDENet()
encoder = Encoder(generator.z_dim, (1, 256, 256))

Experiment("Model_Simple",
           Experiment.Datasets.CARTOON,
           encoder,
           generator,
           sketcher,
           root_path,
           encoder_hyper_params={'lr': 0.001}).run_experiment(
    batch_size,
    accumulation_steps=8,
)


