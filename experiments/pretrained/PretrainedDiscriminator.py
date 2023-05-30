import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data import Subset
from tqdm import tqdm

from models.StyleGan2 import Discriminator
from dataset.dataset import DatasetForTestingDiscriminator


class CartoonDiscriminator(torch.nn.Module):
    def __init__(self, root_path):
        super().__init__()
        self.z_dim = 512
        self.d = Discriminator(
            256, channel_multiplier=2
        )
        checkpoint = torch.load(os.path.join(root_path, 'pretrained', 'NaverWebtoon-040000.pt'))
        self.d.load_state_dict(checkpoint['d'])

    def forward(self, x):
        return self.d(x)

# root_path = Path(__file__).parent.parent
#
# discriminator = CartoonDiscriminator(root_path=root_path)
# discriminator.eval()
#
# dataset = DatasetForTestingDiscriminator(
#     root_dir=os.path.join(root_path.parent, 'dataset', 'synthetic_dataset_cartoon_faces'),
#     src_name='bad_src',
#     transform=ToTensor())
#
# percentage = 10
# num_examples = len(dataset)
# num_subset = int(num_examples * (percentage / 100))
#
# # Generate a list of indices without replacement.
# indices = np.random.choice(num_examples, num_subset, replace=False)
#
# subset = Subset(dataset, indices)
#
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
#
# accuracy = 0
#
# for data in tqdm(dataloader, total=len(dataloader)):
#     with torch.no_grad():
#         data = data.cuda()
#
#         result = discriminator(data)
#         accuracy += (result > -1.427).float().mean()
#
# print(accuracy / len(dataloader))