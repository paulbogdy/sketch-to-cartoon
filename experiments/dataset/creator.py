import pickle

from models.HEDNet import HDENet
import os
import torch
import torchvision


def normalize(x):
    x_min = x.min()
    x_max = x.max()
    return ((x - x_min) / (x_max - x_min)) * 2 - 1


def generate_from_gan(generator, sketcher, dst_path, device, num_images=1000, batch_size=8):
    os.makedirs(dst_path, exist_ok=True)
    src_images_path = os.path.join(dst_path, "src")
    os.makedirs(src_images_path, exist_ok=True)
    sketch_images_path = os.path.join(dst_path, "sketch")
    os.makedirs(sketch_images_path, exist_ok=True)
    z_points_path = os.path.join(dst_path, 'points')
    os.makedirs(z_points_path, exist_ok=True)
    for i in range(num_images//batch_size):
        z = torch.randn([batch_size, generator.z_dim]).to(device)
        with torch.no_grad():
            fake_images = generator(z)
            for j in range(batch_size):
                torch.save(z[j].cpu(), os.path.join(z_points_path, f"{i * batch_size + j}.pt"))
                torchvision.utils.save_image(fake_images[j], os.path.join(src_images_path, f"{i * batch_size + j}.png"))
            del z
            fake_images = sketcher(fake_images)
            for j in range(batch_size):
                torchvision.utils.save_image(fake_images[j], os.path.join(sketch_images_path, f"{i * batch_size + j}.png"))
            del fake_images
        print('\r', end='')
        print(f"{i+1}/{num_images // batch_size}", end='', flush=True)


from models.StyleGan2 import Generator


class Generator2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.z_dim = 512
        self.g_ema = Generator(
            256, 512, 8, channel_multiplier=2
        ).cuda()
        checkpoint = torch.load('../pretrained/NaverWebtoon-040000.pt')
        self.g_ema.load_state_dict(checkpoint['g_ema'])

    def forward(self, x):
        return self.g_ema([x], 0, None)[0]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

G = Generator2().to(device)
S = HDENet().to(device)

generate_from_gan(G, S, "../../dataset/testing_faster_run_pls", device, num_images=1000, batch_size=8)
