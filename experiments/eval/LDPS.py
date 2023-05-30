import torch
import lpips
from torch.utils.data import DataLoader
from tqdm import tqdm

from pretrained.PretrainedDiscriminator import CartoonDiscriminator


class LDPS:
    def __init__(self, root_dir):
        self.model = CartoonDiscriminator(root_dir)

    def __call__(self, encoder, pre_generator, dataset, batch_size=8):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(device).eval()

        encoder.to(device).eval()
        pre_generator.to(device).eval()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        ldps_score = 0
        for data in tqdm(dataloader, total=len(dataloader)):
            with torch.no_grad():
                sketch, src, point = data
                sketch = sketch.to(device)
                src = src.to(device)

                fake_z = encoder(sketch)
                fake = pre_generator(fake_z)

                real_features = self.model(src)
                fake_features = self.model(fake)

                ldps_score += torch.abs(real_features - fake_features).mean()
        ldps_score /= len(dataloader)

        return ldps_score
