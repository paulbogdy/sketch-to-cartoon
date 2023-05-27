import os

import torch
from models.StyleGan2 import Generator


class CartoonGenerator(torch.nn.Module):
    def __init__(self, root_path):
        super().__init__()
        self.z_dim = 512
        self.g_ema = Generator(
            256, 512, 8, channel_multiplier=2
        ).cuda()
        checkpoint = torch.load(os.path.join(root_path, 'pretrained', 'NaverWebtoon-040000.pt'))
        self.g_ema.load_state_dict(checkpoint['g_ema'])

    def forward(self, x):
        return self.g_ema([x], 0, None)[0]


class CartoonGeneratorStyleLatent(torch.nn.Module):
    def __init__(self, root_path):
        super().__init__()
        self.z_dim = 512
        self.g_ema = Generator(
            256, 512, 8, channel_multiplier=2
        ).cuda()
        checkpoint = torch.load(os.path.join(root_path, 'pretrained', 'NaverWebtoon-040000.pt'))
        self.g_ema.load_state_dict(checkpoint['g_ema'])

    def forward(self, x):
        return self.g_ema([x], 0, None, input_is_latent=True)[0]
