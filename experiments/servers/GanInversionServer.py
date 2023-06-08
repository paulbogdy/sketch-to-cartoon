import socket
import os
import sys
import io
import struct
import uuid
from pathlib import Path

import PIL.ImageShow
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import pickle
from models.SketchInverter import Encoder, Sketcher
from models.HEDNet import HDENet
from pretrained.PretrainedGenerator import CartoonGenerator
from servers.StrategyServer import StrategyServer
from typing import List, Tuple
import time
from scipy.spatial import distance
from operator import itemgetter


class Binarization:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, x):
        return (x > self.threshold).float()


class GanInversionServer(StrategyServer):
    def __init__(self, root_dir, diversity_rate: int = 0.1, socket_name: str = "gan_inversion_server"):
        super().__init__(socket_name)
        self.diversity_rate = diversity_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Running on device: {}".format(self.device))
        print("Loading pretrained generator...")

        resource_path = os.path.join(sys._MEIPASS, 'resources') if getattr(sys, 'frozen', False) else 'resources'
        sketch_inverter_path = os.path.join(resource_path, 'Model_Cosine_ZLoss_500.pt')

        self.generator = CartoonGenerator(root_dir)

        self.generator.to(self.device).eval()

        sample = torch.randn(1, 512).to(self.device)
        self.generator(sample)

        print("Loading trained models...", end=' ')
        checkpoint = torch.load(sketch_inverter_path)
        self.encoder = Encoder(self.generator.z_dim, (1, 256, 256), scale=2)
        self.sketcher = HDENet()

        self.encoder.to(self.device).eval()
        self.sketcher.to(self.device).eval()

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print("Done")

        self.cache = {}
        self.cache_limit = 1000

    def get_latent_vector(self, sketch: Image.Image) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        sketch_tensor = transform(sketch).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent_vector = self.encoder(sketch_tensor)
            max_val = torch.max(torch.abs(latent_vector))
            latent_vector /= max_val

        return latent_vector

    def generate_images(self, sketch: Image.Image, num_samples: int, batch_size: int = 8) -> List[Image.Image]:
        # Get the latent vector
        latent_vector = self.get_latent_vector(sketch)

        with torch.no_grad():
            # Create a list to store the generated images
            generated_images = []

            # Calculate how many batches we will have
            num_batches = num_samples // batch_size
            remainder = num_samples % batch_size

            # Generate the initial batch of latent noise
            latent_noise = torch.randn(batch_size, self.generator.z_dim, device=self.device, requires_grad=False)
            latent_noise[0] = torch.zeros_like(latent_noise[0])  # ensure first image uses only original latent_vector

            # Compute the latent neighbours for the first batch
            latent_neighbours = latent_vector + latent_noise * self.diversity_rate
            latent_neighbours.clamp_(-1, 1)

            # Generate a batch of images
            batch_generated_img_tensors = self.generator(latent_neighbours)

            # Convert image tensors to images and store in generated_images list
            for img_tensor in batch_generated_img_tensors:
                generated_images.append(self.tensor_to_img(img_tensor))

            # Store each generated image tensor and its latent neighbour to the cache
            for i in range(batch_size):
                self.store_to_cache(latent_neighbours[i], batch_generated_img_tensors[i])

            # For the remaining batches
            for _ in range(1, num_batches):
                # Generate a batch of latent noise
                latent_noise = torch.randn(batch_size, self.generator.z_dim, device=self.device, requires_grad=False)

                # Compute the latent neighbours
                latent_neighbours = latent_vector + latent_noise * self.diversity_rate
                latent_neighbours.clamp_(-1, 1)

                # Generate a batch of images
                batch_generated_img_tensors = self.generator(latent_neighbours)

                # Convert image tensors to images and store in generated_images list
                for img_tensor in batch_generated_img_tensors:
                    generated_images.append(self.tensor_to_img(img_tensor))

                # Store each generated image tensor and its latent neighbour to the cache
                for i in range(batch_size):
                    self.store_to_cache(latent_neighbours[i], batch_generated_img_tensors[i])

            # Handle the remainder
            if remainder > 0:
                latent_noise = torch.randn(remainder, self.generator.z_dim, device=self.device, requires_grad=False)
                latent_neighbours = latent_vector + latent_noise * self.diversity_rate
                latent_neighbours.clamp_(-1, 1)

                remainder_generated_img_tensors = self.generator(latent_neighbours)

                # Convert image tensors to images and store in generated_images list
                for img_tensor in remainder_generated_img_tensors:
                    generated_images.append(self.tensor_to_img(img_tensor))

                # Store each generated image tensor and its latent neighbour to the cache
                for i in range(remainder):
                    self.store_to_cache(latent_neighbours[i], remainder_generated_img_tensors[i])

            return generated_images

    def store_to_cache(self, latent_vector: torch.Tensor, img_tensor: torch.Tensor, sketch_tensor: torch.Tensor = None):
        # Generate a unique identifier for the image
        img_id = str(uuid.uuid4())

        # Ensure the cache does not exceed limit
        if len(self.cache) >= self.cache_limit:
            # Remove the oldest item from the cache
            self.cache.popitem(last=False)

        # Convert latent vector to numpy array and store in cache
        latent_np = latent_vector.cpu().numpy()

        # Add the image tensor and latent vector to the cache
        self.cache[img_id] = {
            'latent_vector': latent_np,
            'img_tensor': img_tensor,
            'sketch_tensor': sketch_tensor
        }

    def compute_similarity(self, latent_vector1_np, latent_vector2_np):
        # Compute cosine similarity and L1 distance
        cosine_sim = 1 - distance.cosine(latent_vector1_np, latent_vector2_np)
        l1_dist = distance.cityblock(latent_vector1_np,
                                     latent_vector2_np)  # L1 distance is also known as Manhattan or cityblock distance

        # Normalize L1 distance to [0, 1] by dividing by the maximum possible L1 distance
        # For example, if each element in the latent vector can range from -1 to 1, then the maximum L1 distance is 2 * the length of the vector
        max_l1_dist = 2 * len(latent_vector1_np)
        norm_l1_dist = l1_dist / max_l1_dist

        # Invert normalized L1 distance so that higher values mean higher similarity
        inv_l1_dist = 1 - norm_l1_dist

        # Compute combined similarity as the average of cosine similarity and inverted L1 distance
        combined_sim = (cosine_sim + inv_l1_dist) / 2

        return combined_sim

    def is_in_cache(self, latent_vector, threshold=0.0001):
        latent_vector_np = latent_vector.squeeze(0).cpu().numpy()

        for _, img_data in self.cache.items():
            dist = self.compute_similarity(latent_vector_np, img_data['latent_vector'])
            if 1 - dist < threshold:
                return True

        return False

    def get_top_similar(self, latent_vector, top_n):
        similarities = []
        latent_vector_np = latent_vector.squeeze(0).cpu().numpy()
        for img_id, img_data in self.cache.items():
            sim = self.compute_similarity(latent_vector_np, img_data['latent_vector'])
            similarities.append((img_id, sim))

        # Sort by similarity and select the top N
        top_similar = sorted(similarities, key=itemgetter(1), reverse=True)[:top_n]

        return top_similar

    def generate_image_sketch_pair(self, latent_vector: torch.Tensor, num_samples: int, batch_size: int = 8) -> None:
        with torch.no_grad():
            num_batches = num_samples // batch_size
            remainder = num_samples % batch_size

            # Generate the initial batch of latent noise
            latent_noise = torch.randn(batch_size, self.generator.z_dim, device=self.device, requires_grad=False)
            latent_noise[0] = torch.zeros_like(latent_noise[0])  # ensure first image uses only original latent_vector

            # Compute the latent neighbours for the first batch
            latent_neighbours = latent_vector + latent_noise * self.diversity_rate
            latent_neighbours.clamp_(-1, 1)

            # Generate a batch of images
            batch_generated_img_tensors = self.generator(latent_neighbours)
            batch_generated_sketch_tensor = self.sketcher(batch_generated_img_tensors)
            batch_generated_sketch_tensor = torch.nn.functional.interpolate(batch_generated_sketch_tensor,
                                                                            size=(512, 512),
                                                                            mode='bilinear',
                                                                            align_corners=False)

            # Add images and sketches to the cache
            for i in range(batch_size):
                self.store_to_cache(latent_neighbours[i], batch_generated_img_tensors[i], batch_generated_sketch_tensor[i])

            # For the remaining batches
            for _ in range(1, num_batches):
                # Generate a batch of latent noise
                latent_noise = torch.randn(batch_size, self.generator.z_dim, device=self.device,
                                           requires_grad=False)

                # Compute the latent neighbours
                latent_neighbours = latent_vector + latent_noise * self.diversity_rate
                latent_neighbours.clamp_(-1, 1)

                # Generate a batch of images
                batch_generated_img_tensors = self.generator(latent_neighbours)
                batch_generated_sketch_tensor = self.sketcher(batch_generated_img_tensors)
                batch_generated_sketch_tensor = torch.nn.functional.interpolate(batch_generated_sketch_tensor,
                                                                                    size=(512, 512),
                                                                                    mode='bilinear',
                                                                                    align_corners=False)

                # Store each generated image tensor and its latent neighbour to the cache
                for i in range(batch_size):
                    self.store_to_cache(latent_neighbours[i], batch_generated_img_tensors[i],
                                        batch_generated_sketch_tensor[i])

            # Handle the remainder
            if remainder > 0:
                latent_noise = torch.randn(remainder, self.generator.z_dim, device=self.device, requires_grad=False)
                latent_neighbours = latent_vector + latent_noise * self.diversity_rate
                latent_neighbours.clamp_(-1, 1)

                remainder_generated_img_tensors = self.generator(latent_neighbours)
                remainder_generated_sketch_tensor = self.sketcher(remainder_generated_img_tensors)
                remainder_generated_sketch_tensor = torch.nn.functional.interpolate(remainder_generated_sketch_tensor,
                                                                                    size=(512, 512),
                                                                                    mode='bilinear',
                                                                                    align_corners=False)

                # Store each generated image tensor and its latent neighbour to the cache
                for i in range(remainder):
                    self.store_to_cache(latent_neighbours[i], remainder_generated_img_tensors[i],
                                        remainder_generated_sketch_tensor[i])

    def number_of_close_points_cached(self, latent_vector, threshold=0.7):
        latent_vector_np = latent_vector.squeeze(0).cpu().numpy()
        count = 0
        for _, img_data in self.cache.items():
            dist = self.compute_similarity(latent_vector_np, img_data['latent_vector'])
            if dist > threshold:
                count += 1

        print(f'How many close points cached: {count}')

        return count

    def get_batches(self, list_to_split, batch_size):
        return [list_to_split[i:i + batch_size] for i in range(0, len(list_to_split), batch_size)]

    def generate_shadow(self, sketch: Image.Image, num_samples: int, batch_size: int = 8) -> Image.Image:
        latent_vector = self.get_latent_vector(sketch)
        shadow = torch.zeros(1, 1, 512, 512, device=self.device)

        with torch.no_grad():
            if len(self.cache) < num_samples:
                self.generate_image_sketch_pair(latent_vector,
                                                num_samples - self.number_of_close_points_cached(latent_vector),
                                                batch_size)
            elif not self.is_in_cache(latent_vector):
                self.generate_image_sketch_pair(latent_vector,
                                                max(1, num_samples - self.number_of_close_points_cached(latent_vector)),
                                                batch_size)

            top_similar = self.get_top_similar(latent_vector, num_samples)

            # Get the images and the sketches in two lists
            images_to_compute_sketch = []
            image_ids_to_compute_sketch = []
            for img_id, _ in top_similar:
                if self.cache[img_id]['sketch_tensor'] is None:
                    images_to_compute_sketch.append(self.cache[img_id]['img_tensor'])
                    image_ids_to_compute_sketch.append(img_id)

            # Compute the sketches in batches
            if images_to_compute_sketch:
                image_batches = self.get_batches(images_to_compute_sketch, batch_size)
                for i, image_batch in enumerate(image_batches):
                    image_batch = torch.stack(image_batch).to(self.device)
                    sketch_batch = self.sketcher(image_batch)
                    sketch_batch = torch.nn.functional.interpolate(sketch_batch, size=(512, 512), mode='bilinear',
                                                                   align_corners=False)
                    # Put the sketches back into the cache
                    for j, sketch_tensor in enumerate(sketch_batch):
                        self.cache[image_ids_to_compute_sketch[i * batch_size + j]]['sketch_tensor'] = sketch_tensor

            # Now all the necessary sketches should be in the cache
            # So, we can compute the shadow as the weighted sum of the sketches
            weights_sum = 0
            for img_id, sim in top_similar:
                sketch_tensor = self.cache[img_id]['sketch_tensor']
                weight = sim
                shadow += weight * sketch_tensor
                weights_sum += weight

            # Normalize the shadow by the sum of the weights
            shadow /= weights_sum

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
            img_np = 1 - img_np.repeat(3, axis=2)
        img = Image.fromarray((img_np * 255).astype(np.uint8))
        return img


root_dir = Path(__file__).parent.parent
server: GanInversionServer = GanInversionServer(root_dir)
server.run_server()
