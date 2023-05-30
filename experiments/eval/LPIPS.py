import torch
import lpips
from torch.utils.data import DataLoader
from tqdm import tqdm


class LPIPS:
    def __init__(self):
        self.lpips = lpips.LPIPS(net='alex')

    def __call__(self, encoder, pre_generator, dataset, batch_size=8):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.lpips.to(device).eval()

        encoder.to(device).eval()
        pre_generator.to(device).eval()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        lpips_score = 0
        for data in tqdm(dataloader, total=len(dataloader)):
            with torch.no_grad():
                sketch, src, point = data
                sketch = sketch.to(device)
                src = src.to(device)

                fake_z = encoder(sketch)
                fake = pre_generator(fake_z)

                lpips_score += self.lpips(fake, src).mean()

        return lpips_score / len(dataloader)
