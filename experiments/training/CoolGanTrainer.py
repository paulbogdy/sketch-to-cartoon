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


class CoolGanTrainer(Trainer):
    def __init__(self, device, dataset, model_name, generator, discriminator, lr=0.00005, beta1=0.5, beta2=0.999):
        super(CoolGanTrainer, self).__init__(device, dataset, model_name)
        self.g_last_loss = None
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)

        self.optimizer_G = optim.RMSprop(self.generator.parameters(), lr=lr)
        self.optimizer_D = optim.RMSprop(self.discriminator.parameters(), lr=lr)

        self.writer_fake = SummaryWriter(f"runs/CoolGan2/fake")
        self.writer_real = SummaryWriter(f"runs/CoolGan2/real")
        self.writer_loss = SummaryWriter(f"runs/CoolGan2/loss")

        self.criterion = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

        self.step = 0
        self.loss_step = 0
        self.n_critic = 5
        self.weight_clip = 0.01

    def train_step(self, batch_idx, data, batch_size):
        real_images, _ = data
        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)

        # Labels for real and fake images
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # ---------------------
        # Train the discriminator
        # ---------------------

        self.optimizer_D.zero_grad()

        # Compute the loss for real images
        outputs_real = self.discriminator(real_images)
        d_loss_real = self.criterion(outputs_real, real_labels)

        # Generate fake images
        z = torch.randn(batch_size, z_dim, 1, 1, device=device)
        fake_images = self.generator(z)

        # Compute the loss for fake images
        outputs_fake = self.discriminator(fake_images.detach())
        d_loss_fake = self.criterion(outputs_fake, fake_labels)

        # Combine the losses and update the discriminator
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.optimizer_D.step()

        for p in self.discriminator.parameters():
            p.data.clamp_(-self.weight_clip, self.weight_clip)

        if batch_idx % self.n_critic == 1:
            # ---------------------
            # Train the generator
            # ---------------------

            self.optimizer_G.zero_grad()

            # Generate fake images
            z = torch.randn(batch_size, self.generator.z_dim, 1, 1, device=self.device)
            fake_images = self.generator(z)

            # Compute the loss for fake images, but using real labels
            outputs_fake = self.discriminator(fake_images)
            g_loss = self.criterion(outputs_fake, real_labels)
            g_loss += self.discriminator.loss(self.l1_loss, fake_images, real_images)

            # Update the generator
            g_loss.backward()
            self.optimizer_G.step()

            self.g_last_loss = g_loss

            self.loss_step += 1
            self.writer_loss.add_scalar('Loss/Generator', g_loss.item(), self.loss_step)
            self.writer_loss.add_scalar('Loss/Discriminator', d_loss.item(), self.loss_step)

        if batch_idx % 50 == 0:
            with torch.no_grad():
                img_grid_fake = torchvision.utils.make_grid(fake_images, normalize=True)
                img_grid_real = torchvision.utils.make_grid(real_images, normalize=True)
                self.writer_fake.add_image(
                    "Fake Images", img_grid_fake, global_step=self.step
                )
                self.writer_real.add_image(
                    "Real Images", img_grid_real, global_step=self.step
                )
                self.step += 1

        return d_loss, self.g_last_loss


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST dataset for training
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

z_dim = 100

G = DCGANGenerator(z_dim, 1)
D = Discriminator(1)
trainer = CoolGanTrainer(device, train_dataset, "CoolGan", G, D)

trainer.train(8, 100)


