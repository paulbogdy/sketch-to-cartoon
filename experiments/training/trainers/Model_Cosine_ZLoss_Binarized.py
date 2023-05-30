import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.HEDNet import HDENet
from models.SketchInverter import Encoder, Sketcher
from training.ExperimentFramework import Experiment
from pathlib import Path
import torch
from models.StyleGan2 import Discriminator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from pretrained.PretrainedGenerator import CartoonGenerator

root_path = Path(__file__).parent.parent.parent

generator = CartoonGenerator(root_path)
sketcher = HDENet()
encoder = Encoder(generator.z_dim, (1, 256, 256))

Experiment("Model_Cosine_ZLoss_Binarized",
           Experiment.Datasets.CARTOON,
           encoder,
           generator,
           sketcher,
           root_path,
           binarize_sketch=True,
           use_cosine_for_z=True,
           content_loss_alpha=0,
           shape_loss_alpha=0).run_experiment(
    batch_size=32,
    num_epochs=10,
    accumulation_steps=1,
    save_every_n_batches=1000,
    show_every_n_steps=250,
)
