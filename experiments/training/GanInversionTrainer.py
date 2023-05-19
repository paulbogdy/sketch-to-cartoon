import torch
import torch.nn as nn
import torch.optim as optim
from utils.Losses import InceptionLoss
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.models
import os
from training.Trainer import Trainer


class GanInversionTrainer(Trainer):
    def __init__(self, device, model_name, generator, sketcher, encoder):
        super(GanInversionTrainer, self).__init__(device, None, model_name)
        self.device = device
        self.generator = generator.to(self.device)
        for param in self.generator.parameters():
            param.requires_grad = False
        self.sketcher = sketcher.to(self.device)
        for param in self.sketcher.parameters():
            param.requires_grad = False
        self.encoder = encoder.to(self.device)
        self.optimizer = optim.RMSprop(self.encoder.parameters(), lr=0.00005)
        self.loss = nn.L1Loss()
        self.writer_sketch = SummaryWriter(f"runs/{model_name}/sketch")
        self.writer_fake = SummaryWriter(f"runs/{model_name}/fake")
        self.writer_real = SummaryWriter(f"runs/{model_name}/real")
        self.writer_loss = SummaryWriter(f"runs/{model_name}/loss")
        self.loss_step = 0
        self.step = 0
        self.update_freq = 5

    def train(self, batch_size, num_epochs, start_epoch=0):
        for epoch in range(start_epoch, num_epochs):
            try:
                loss = None
                z_loss_avg = 0
                gen_loss_avg = 0
                self.optimizer.zero_grad()
                sketch_list = []
                real_list = []
                fake_list = []
                for item_nr in range(batch_size):
                    z = torch.randn([2, self.generator.z_dim], device=self.device)
                    with torch.no_grad():
                        real = self.generator(z, None)
                        sketch = self.sketcher(real)
                        sketch = F.interpolate(sketch, size=(512, 512), mode='bilinear', align_corners=True)

                    fake_z = self.encoder(sketch)
                    z_loss = self.loss(z, fake_z)

                    with torch.no_grad():
                        fake_img = self.generator(fake_z, None)

                    gen_loss = self.loss(real, fake_img)

                    if loss is None:
                        loss = z_loss + gen_loss
                    else:
                        loss += z_loss + gen_loss

                    z_loss_avg += z_loss
                    gen_loss_avg += gen_loss
                    sketch_list.append(sketch)
                    real_list.append(real)
                    fake_list.append(fake_img)

                loss.backward()
                self.optimizer.step()

                if epoch % self.update_freq == 0:
                    with torch.no_grad():
                        # Concatenate tensors along the batch dimension
                        sketch_tensors = torch.cat(sketch_list[:16], dim=0)
                        fake_tensors = torch.cat(fake_list[:16], dim=0)
                        real_tensors = torch.cat(real_list[:16], dim=0)

                        img_grid_sketch = torchvision.utils.make_grid(sketch_tensors, normalize=True)
                        img_grid_fake = torchvision.utils.make_grid(fake_tensors, normalize=True)
                        img_grid_real = torchvision.utils.make_grid(real_tensors, normalize=True)
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
                self.writer_loss.add_scalar('Loss/Z_Loss', z_loss_avg/batch_size, self.loss_step)
                self.writer_loss.add_scalar('Loss/Gen_Loss', gen_loss_avg/batch_size, self.loss_step)
                self.writer_loss.add_scalar('Loss/loss', loss.item(), self.loss_step)

                print('\r', end='')
                print(f"Epoch [{epoch + 1}/{num_epochs}] "
                      f"z_loss: {z_loss_avg/batch_size} gen_loss: {gen_loss_avg/batch_size} ",
                      end='', flush=True)

                self.save_checkpoint(epoch)

            except KeyboardInterrupt:
                print("\nStopping training. Saving the model...")
                self.save_checkpoint(epoch)
                return

    def save_checkpoint(self, epoch):
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'epoch': epoch,
            'step': self.step,
            'loss_step': self.loss_step
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.step = checkpoint['step']
        self.loss_step = checkpoint['loss_step']
        return checkpoint['epoch']
