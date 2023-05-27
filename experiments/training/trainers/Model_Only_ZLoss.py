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

Experiment("Model_Only_ZLoss",
           Experiment.Datasets.CARTOON,
           encoder,
           generator,
           sketcher,
           root_path,
           content_loss_alpha=0,
           shape_loss_alpha=0).run_experiment(
    batch_size,
    num_epochs=10,
    accumulation_steps=1,
)


