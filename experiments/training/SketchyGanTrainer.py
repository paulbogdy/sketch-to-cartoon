from models.SketchyGan import SketchyGenerator, SketchyDiscriminator
from training.Trainer import Trainer
import torch
import torch.nn as nn
import torch.optim as optim
from utils.Losses import InceptionLoss
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.models
import os
from torchvision import transforms
from torch.autograd.variable import Variable


class SketchyGanTrainer(Trainer):
    def __init__(self, device, dataset, model_name, gamma_div=1, gamma_p=1):
        super().__init__(device, dataset, model_name)
        self.noise_channels = 1
        self.generator = SketchyGenerator(input_channels=self.noise_channels)
        self.generator.to(self.device)
        self.discriminator = SketchyDiscriminator()
        self.discriminator.to(self.device)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.ganLoss = nn.BCELoss()
        self.L1Loss = nn.L1Loss()
        self.pLoss = InceptionLoss(device)
        self.gamma_div = gamma_div
        self.gamma_p = gamma_p
        self.writer_sketch = SummaryWriter(f"runs/SketchyGAN/sketch")
        self.writer_fake = SummaryWriter(f"runs/SketchyGAN/fake")
        self.writer_real = SummaryWriter(f"runs/SketchyGAN/real")
        self.writer_loss = SummaryWriter(f"runs/SketchyGAN/loss")
        self.step = 0
        self.loss_step = 0

    def train_step(self, batch_idx, data):
        valid_indices = torch.nonzero(data[2]).squeeze()
        sketch_data = F.interpolate(data[0], size=(64, 64), mode='bilinear', align_corners=True).to(self.device)[valid_indices]
        real_data = F.interpolate(data[1], size=(64, 64), mode='bilinear', align_corners=True).to(self.device)[valid_indices]

        # Train the discriminator with real data
        self.d_optimizer.zero_grad()
        real_outputs = self.discriminator(real_data, sketch_data)
        real_loss = self.ganLoss(0.9 * real_outputs, torch.ones_like(real_outputs, device=self.device))

        # Generate fake data
        batch_size, _, height, width = real_data.shape
        noise = torch.randn(batch_size, self.noise_channels, height, width, device=self.device)
        fake_data = self.generator(noise, sketch_data)

        # Train the discriminator with fake data
        fake_outputs = self.discriminator(fake_data, sketch_data)
        fake_loss = self.ganLoss(fake_outputs, torch.zeros_like(fake_outputs, device=self.device))

        # Update the discriminator
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()

        # Train the generator
        self.g_optimizer.zero_grad()
        noise = torch.randn(batch_size, self.noise_channels, height, width, device=self.device)
        fake_data = self.generator(noise, sketch_data)
        fake_outputs = self.discriminator(fake_data, sketch_data)

        g_loss_gan = self.ganLoss(fake_outputs, torch.ones_like(fake_outputs).to(self.device))
        g_loss_sup = self.L1Loss(fake_data, real_data)

        noise2 = torch.randn(batch_size, self.noise_channels, height, width, device=self.device)
        fake_data2 = self.generator(noise2, sketch_data)

        g_loss_div = - self.gamma_div * self.L1Loss(fake_data, fake_data2)
        g_loss_p = self.gamma_p * self.pLoss(fake_data, real_data)

        # Update the generator
        g_loss = g_loss_gan + g_loss_sup + g_loss_div + g_loss_p
        g_loss.backward()
        self.g_optimizer.step()

        if batch_idx % 50 == 0:
            with torch.no_grad():
                img_grid_sketch = torchvision.utils.make_grid(sketch_data[:16], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake_data[:16], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real_data[:16], normalize=True)
                self.writer_sketch.add_image(
                    "Sketch Images", img_grid_sketch, global_step=self.step
                )
                self.writer_fake.add_image(
                    "Fake Images", img_grid_fake, global_step=self.step
                )
                self.writer_real.add_image(
                    "Real Images", img_grid_real, global_step=self.step
                )
                self.step += 1

        self.loss_step += 1
        self.writer_loss.add_scalar('Loss/Generator', g_loss.item(), self.loss_step)
        self.writer_loss.add_scalar('Loss/Discriminator', d_loss.item(), self.loss_step)

        return d_loss, g_loss

    def save_checkpoint(self, epoch):
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'epoch': epoch,
            'step': self.step
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.step = checkpoint['step']
        return checkpoint['epoch']

    def save_for_QT(self, checkpoint_path=None):
        print("Saving Model for QT...")
        if checkpoint_path is None:
            self.load_checkpoint(self.find_latest_checkpoint())
        else:
            self.load_checkpoint(checkpoint_path)

        # Create the noise input tensor and the input sketch tensor
        noise_example_input = torch.randn(1, 1, 64, 64, device=self.device)
        sketch_example_input = torch.randn(1, 1, 64, 64, device=self.device)
        generated_example_input = torch.randn(1, 3, 64, 64, device=self.device)

        # Trace the generator model with the example inputs
        traced_generator = torch.jit.trace(self.generator, (noise_example_input, sketch_example_input))
        traced_discriminator = torch.jit.trace(self.discriminator, (generated_example_input, sketch_example_input))

        # Save the traced model
        torch.jit.save(traced_generator, "SketchyGenerator.pt")
        torch.jit.save(traced_discriminator, "SketchyDiscriminator.pt")

        print("Model Saved")
