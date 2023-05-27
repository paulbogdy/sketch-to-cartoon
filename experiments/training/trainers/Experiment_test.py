from models.HEDNet import HDENet
from models.SketchInverter import Encoder
from training.ExperimentFramework import Experiment
from pathlib import Path
import torch

batch_size = 32
num_epochs = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from pretrained.PretrainedGenerator import CartoonGenerator

root_path = Path(__file__).parent.parent.parent

generator = CartoonGenerator(root_path)
sketcher = HDENet()
encoder = Encoder(generator.z_dim, (1, 256, 256))

Experiment("without_torch_no_grad",
           Experiment.Datasets.CARTOON,
           encoder,
           generator,
           sketcher,
           root_path).run_experiment(
    batch_size,
    num_epochs=2,
    accumulation_steps=8
)
