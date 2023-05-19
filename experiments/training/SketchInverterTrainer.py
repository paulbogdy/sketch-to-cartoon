from training.Trainer import Trainer
import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter


class SketchInverterTrainer(Trainer):
    def __init__(self,
                 encoder,
                 generator,
                 sketcher,
                 device,
                 dataset,
                 model_name,
                 checkpoint_dir="checkpoints"):
        super(SketchInverterTrainer, self).__init__(device, dataset, model_name, checkpoint_dir)

        self.accumulation_steps = 4
        self.encoder = encoder.to(device)
        self.generator = generator.to(device)
        self.sketcher = sketcher.to(device)

        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=0.0001)
        self.sketcher_optim = torch.optim.Adam(self.sketcher.parameters(), lr=0.0001)

        self.l1_loss = torch.nn.L1Loss()

        self.writer = SummaryWriter(f"runs/{model_name}")

        self.writer_sketch = SummaryWriter(f"runs/{model_name}/sketch")
        self.writer_fake = SummaryWriter(f"runs/{model_name}/fake")
        self.writer_real = SummaryWriter(f"runs/{model_name}/real")
        self.writer_loss = SummaryWriter(f"runs/{model_name}/loss")

        self.step = 0

    def train_step(self, batch_idx, data, batch_size):
        valid_indices = torch.nonzero(data[3]).squeeze()
        sketch_data = data[0].to(self.device)[valid_indices]
        real_data = data[1].to(self.device)[valid_indices]
        z_data = data[2].to(self.device)[valid_indices]

        encoded_z = self.encoder(sketch_data)
        loss = self.l1_loss(encoded_z, z_data)

        fake_img = []

        for i in range(batch_size):
            with torch.no_grad():
                single_fake_img = self.generator(encoded_z[i].unsqueeze(0), None)
            fake_img.append(single_fake_img)

        # Stack the list of individual fake images to create a batch tensor
        fake_img = torch.cat(fake_img, dim=0)

        loss.add_(self.l1_loss(fake_img, real_data))

        sketch_fake = self.sketcher(fake_img)
        sketch_real = self.sketcher(real_data)

        loss.add_(self.l1_loss(sketch_fake, sketch_data))
        loss.add_(self.l1_loss(sketch_real, sketch_data))

        loss.div_(self.accumulation_steps)

        loss.backward()

        if (batch_idx + 1) % self.accumulation_steps == 0:
            self.encoder_optim.step()
            self.sketcher_optim.step()

            # Zero the gradients after updating the model parameters
            self.encoder_optim.zero_grad()
            self.sketcher_optim.zero_grad()

        self.writer.add_image("images/sketch", sketch_data[-1], self.step)
        self.writer.add_image("images/fake", fake_img[-1], self.step)
        self.writer.add_image("images/real", real_data[-1], self.step)

        self.writer_loss.add_scalar("loss", loss.item(), self.step)

        self.step += 1

    def save_checkpoint(self, epoch):
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'sketcher_state_dict': self.sketcher.state_dict(),
            'epoch': epoch,
            'step': self.step
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.sketcher.load_state_dict(checkpoint['sketcher_state_dict'])
        self.step = checkpoint['step']
        return checkpoint['epoch']

    def each_epoch(self):
        self.encoder_optim.zero_grad()
        self.sketcher_optim.zero_grad()

    def save_for_QT(self, checkpoint_path=None):
        print("Saving Model for QT...")
        if checkpoint_path is None:
            self.load_checkpoint(self.find_latest_checkpoint())
        else:
            self.load_checkpoint(checkpoint_path)

        # Create the noise input tensor and the input sketch tensor
        noise_example_input = torch.randn(1, 512, device=self.device)
        sketch_example_input = torch.randn(1, 1, 512, 512, device=self.device)
        generated_example_input = torch.randn(1, 3, 512, 512, device=self.device)

        # class GeneratorWrapper(nn.Module):
        #     def __init__(self, generator):
        #         super(GeneratorWrapper, self).__init__()
        #         self.generator = generator
        #
        #     def forward(self, x):
        #         return self.generator(x, None)
        #
        # generator_wrapper = GeneratorWrapper(self.generator).to(self.device)
        # generator_wrapper.eval()
        self.generator.eval()
        # Trace the generator model with the example inputs
        scripted_generator = torch.jit.script(self.generator)

        self.encoder.eval()
        self.sketcher.eval()
        scripted_encoder = torch.jit.trace(self.encoder, sketch_example_input)
        traced_sketcher = torch.jit.trace(self.sketcher, generated_example_input)

        # Save the traced model
        torch.jit.save(scripted_generator, "SketchInverterGenerator.pt")
        torch.jit.save(scripted_encoder, "SketchyInverterEncoder.pt")
        torch.jit.save(traced_sketcher, "SketchyInverterSketcher.pt")

        print("Model Saved")


