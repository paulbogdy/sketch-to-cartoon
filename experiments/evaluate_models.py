import os

import numpy as np
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from models.SketchInverter import Encoder
from pretrained.PretrainedGenerator import CartoonGenerator

from dataset.dataset import SketchInverterDataset
from eval.Evaluator import Evaluator

root_dir = os.path.dirname(os.path.abspath(__file__))

dataset = SketchInverterDataset(
                root_dir=os.path.join(os.path.dirname(root_dir), 'dataset', 'synthetic_dataset_cartoon_faces_test'),
                transform=ToTensor(),
                image_size=(256, 256)
            )

# percentage = 10
# num_examples = len(dataset)
# num_subset = int(num_examples * (percentage / 100))
#
# # Generate a list of indices without replacement.
# indices = np.random.choice(num_examples, num_subset, replace=False)
#
# dataset = Subset(dataset, indices)

generator = CartoonGenerator(root_dir)
encoder = Encoder(generator.z_dim, (1, 256, 256), scale=2)

evaluator = Evaluator(root_dir, os.path.join(root_dir, 'trained_models'), dataset, generator, encoder)

evaluator.evaluate_model_FID2('Model_Cosine_ZLoss_500.pt')
evaluator.evaluate_model_FID2('Model_L1_ZLoss_500.pt')