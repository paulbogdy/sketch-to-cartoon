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

# function to get number of params of a model
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters())
#
#
# print("Number of parameters of the generator: ", count_parameters(generator))
# print("Number of parameters of the sketcher: ", count_parameters(sketcher))
# print("Number of parameters of the encoder: ", count_parameters(encoder))
# print("Number of parameters of the trainable sketcher: ", count_parameters(trainable_sketcher))

Experiment("Model_Only_ZLoss",
           Experiment.Datasets.CARTOON,
           encoder,
           generator,
           sketcher,
           root_path,
           content_loss_alpha=0,
           shape_loss_alpha=0).run_experiment(
    batch_size=32,
    num_epochs=10,
    accumulation_steps=1,
    save_every_n_batches=1000,
)
