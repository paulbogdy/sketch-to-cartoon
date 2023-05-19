import torchvision
from torch.utils.tensorboard import SummaryWriter

from models.CoolGan import Discriminator, DCGANGenerator
from training.Trainer import Trainer
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

class StyleGanTrainer(Trainer):
    def __init__(self, device, dataset, model_name, generator, discriminator, lr=0.001, beta1=0, beta2=0.99):
        super(StyleGanTrainer, self).__init__(device, dataset, model_name)
        self.g_last_loss = None
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)

        self.optimizer_mapping = optim.Adam(self.generator.mapping_network.parameters(), lr=lr/100, betas=(beta1, beta2))
        self.optimizer_synthesis = optim.Adam(self.generator.synthesis_network.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

        # Gradient accumulator
        self.gradient_accumulation_steps = 8
        self.steps_since_last_optim_step = 0

        self.writer_fake = SummaryWriter(f"runs/CoolGan2/fake")
        self.writer_real = SummaryWriter(f"runs/CoolGan2/real")
        self.writer_loss = SummaryWriter(f"runs/CoolGan2/loss")

        self.criterion = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

        self.step = 0
        self.loss_step = 0
        self.n_critic = 5
        self.weight_clip = 0.01
        self.device = device

        self.style_mix_prob = 0.9

    def get_w(self, batch_size):
        if torch.rand(()).item() < self.style_mix_prob:
            cross_over_poit = int(self.generator.z_dim * torch.rand(()).item())

            z1 = torch.randn(batch_size, self.generator.z_dim, device=self.device)
            z2 = torch.randn(batch_size, self.generator.z_dim, device=self.device)

            w1 = self.generator.mapping_network(z1)
            w2 = self.generator.mapping_network(z2)

            w1[:, :cross_over_poit] = w2[:, :cross_over_poit]
            return w1
        else:
            z = torch.randn(batch_size, self.generator.z_dim, device=self.device)
            return self.generator.mapping_network(z)

    def get_noise(self, batch_size):
        noise = []
        resolution = 4
        for i in range(8):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)

            n2 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)
            noise.append((n1, n2))
            resolution *= 2

        return noise

    def train_step(self, batch_idx, data, batch_size):
        real_images, _ = data
        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)
        mini_batch_size = batch_size // self.gradient_accumulation_steps

        all_fake_images = []
        all_w = [self.get_w(mini_batch_size) for _ in range(self.gradient_accumulation_steps)]

        # Pre-generate fake images for all mini-batches
        for w in all_w:
            fake_images = self.generator.synthesis_network(w, self.get_noise(mini_batch_size))
            all_fake_images.append(fake_images)

        # ---------------------
        # Train the discriminator
        # ---------------------

        self.optimizer_D.zero_grad()
        discriminator_loss = 0

        for i in range(self.gradient_accumulation_steps):
            mini_batch_start = i * mini_batch_size
            mini_batch_end = (i + 1) * mini_batch_size

            mini_batch_real_images = real_images[mini_batch_start:mini_batch_end]
            fake_images = all_fake_images[i].detach()  # Detach the fake images for discriminator training

            outputs_real = self.discriminator(mini_batch_real_images)
            outputs_fake = self.discriminator(fake_images)

            # Compute the gradient penalty
            alpha = torch.rand(mini_batch_real_images.size(0), 1, 1, 1, device=self.device)
            x_hat = (alpha * mini_batch_real_images + (1 - alpha) * fake_images).requires_grad_(True)
            out_x_hat = self.discriminator(x_hat)
            gradients = torch.autograd.grad(outputs=out_x_hat, inputs=x_hat,
                                            grad_outputs=torch.ones(out_x_hat.size(), device=self.device),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

            # Combine the losses and update the discriminator
            d_loss = outputs_fake.mean() - outputs_real.mean() + gradient_penalty
            discriminator_loss += d_loss.item()

            d_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)

        self.optimizer_D.step()
        discriminator_loss /= self.gradient_accumulation_steps

        # -----------------
        # Train the generator
        # -----------------

        self.optimizer_mapping.zero_grad()
        self.optimizer_synthesis.zero_grad()
        generator_loss = 0

        for i in range(self.gradient_accumulation_steps):
            fake_images = all_fake_images[i]  # Use non-detached fake images for generator training
            outputs_fake = self.discriminator(fake_images)

            g_loss = -outputs_fake.mean()
            generator_loss += g_loss.item()

            g_loss.backward()

        all_fake_images = torch.cat([img.detach().cpu() for img in all_fake_images], dim=0)

        torch.nn.utils.clip_grad_norm_(self.generator.mapping_network.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.generator.synthesis_network.parameters(), max_norm=1.0)

        self.optimizer_mapping.step()
        self.optimizer_synthesis.step()

        generator_loss /= self.gradient_accumulation_steps

        self.loss_step += 1
        self.writer_loss.add_scalar('Loss/Generator', generator_loss, self.loss_step)
        self.writer_loss.add_scalar('Loss/Discriminator', discriminator_loss, self.loss_step)

        if batch_idx % 10 == 0:
            with torch.no_grad():
                img_grid_fake = torchvision.utils.make_grid(all_fake_images, normalize=True)
                img_grid_real = torchvision.utils.make_grid(real_images, normalize=True)
                self.writer_fake.add_image(
                    "Fake Images", img_grid_fake, global_step=self.step
                )
                self.writer_real.add_image(
                    "Real Images", img_grid_real, global_step=self.step
                )
                self.step += 1

        return generator_loss, discriminator_loss

