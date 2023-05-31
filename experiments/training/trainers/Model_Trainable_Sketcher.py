import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.SketchInverter import Encoder, Sketcher
from training.TrainableSketcherExperiment import TrainableSketcherExperiment
from pathlib import Path
import torch

batch_size = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from pretrained.PretrainedGenerator import CartoonGenerator

root_path = Path(__file__).parent.parent.parent

generator = CartoonGenerator(root_path)
sketcher = Sketcher()
encoder = Encoder(generator.z_dim, (1, 256, 256))

TrainableSketcherExperiment("Model_Trainable_Sketcher",
                            TrainableSketcherExperiment.Datasets.CARTOON,
                            encoder,
                            generator,
                            sketcher,
                            root_path).run_experiment(
    batch_size,
    num_epochs=50,
    accumulation_steps=8,
    save_every_n_batches=1000,
    show_every_n_steps=200,
)
