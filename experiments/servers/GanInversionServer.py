import socket
import os
import sys
import io
import struct
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import pickle
from models.SketchInverter import Encoder, Sketcher
from servers.StrategyServer import StrategyServer
from typing import List, Tuple
import time


class GanInversionServer(StrategyServer):
    def __init__(self, diversity_rate: int = 0.1, socket_name: str = "gan_inversion_server"):
        super().__init__(socket_name)
        self.diversity_rate = diversity_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Running on device: {}".format(self.device))
        print("Loading pretrained generator...")

        resource_path = os.path.join(sys._MEIPASS, 'resources') if getattr(sys, 'frozen', False) else 'resources'
        network_snapshot_path = os.path.join(resource_path, 'network-snapshot-000880.pkl')
        sketch_inverter_path = os.path.join(resource_path, 'sketchInverter.pt')

        with open(network_snapshot_path, 'rb') as f:
            self.generator = pickle.load(f)['G_ema']

        self.generator.to(self.device).eval()

        sample = torch.randn(1, 512).to(self.device)
        self.generator(sample, None)

        print("Loading trained models...", end=' ')
        checkpoint = torch.load(sketch_inverter_path)
        self.encoder = Encoder(self.generator.z_dim, model_size=8)
        self.sketcher = Sketcher(model_size=8)

        self.encoder.to(self.device).eval()
        self.sketcher.to(self.device).eval()

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.sketcher.load_state_dict(checkpoint['sketcher_state_dict'])
        print("Done")

    def generate_images(self, sketch: Image.Image, num_samples: int) -> List[Image.Image]:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        sketch_tensor = transform(sketch).unsqueeze(0).to(self.device)

        # Get the latent vector
        with torch.no_grad():
            latent_vector = self.encoder(sketch_tensor)
            generated_images = [self.generate_one_image(latent_vector)]
            for _ in range(1, num_samples):
                latent_noise = torch.randn(1, self.generator.z_dim, device=self.device, requires_grad=False)
                latent_neighbour = (1 - self.diversity_rate) * latent_vector + latent_noise * self.diversity_rate
                torch.clamp(latent_neighbour, -1, 1, out=latent_neighbour)
                generated_images.append(self.generate_one_image(latent_neighbour))

            return generated_images

    def generate_one_image(self, latent_vector: torch.Tensor) -> Image.Image:
        with torch.no_grad():
            result = self.generator(latent_vector, None)
            return self.tensor_to_img(result)

    def generate_shadow(self, sketch: Image.Image, num_samples: int) -> Image.Image:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        sketch_tensor = transform(sketch).unsqueeze(0).to(self.device)

        shadow = torch.zeros(1, 1, 512, 512, device=self.device)

        with torch.no_grad():
            latent_vector = self.encoder(sketch_tensor)
            start_time = time.time()
            for _ in range(num_samples//8):
                latent_noise = torch.randn(8, self.generator.z_dim, device=self.device, requires_grad=False)
                latent_neighbour = latent_vector + latent_noise * 0.01
                generated_img = self.generator(latent_neighbour, None)
                generated_sketch = self.sketcher(generated_img)
                shadow += torch.sum(generated_sketch, dim=0)
                print("Time taken: {}".format(time.time() - start_time))
                start_time = time.time()

        shadow /= (num_samples//8 * 8)

        return self.tensor_to_img(shadow, grayscale=True)

    def tensor_to_img(self, tensor: torch.Tensor, grayscale=False) -> Image.Image:
        torch.clamp(tensor, -1, 1, out=tensor)
        img_np = tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        # Check value range and scale to [0, 255] if necessary
        min_val = np.min(img_np)
        max_val = np.max(img_np)
        if min_val < 0 or max_val > 1:
            img_np = (img_np - min_val) / (max_val - min_val)
        # Convert to PIL image
        if grayscale:
            img_np = img_np.squeeze(2)
        img = Image.fromarray((img_np * 255).astype(np.uint8))
        return img


server: GanInversionServer = GanInversionServer()
server.run_server()
