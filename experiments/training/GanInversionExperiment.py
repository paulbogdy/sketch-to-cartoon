import time
from collections import deque

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import os
from torchmetrics.image.fid import FrechetInceptionDistance
from losses.AlexNetLoss import AlexNetLoss
from losses.InceptionV3 import InceptionV3
from scipy import linalg
import numpy as np


class GanInversionExperiment:
    def __init__(self,
                 model_name,
                 encoder,
                 pre_generator,
                 pre_sketcher,
                 root_dir,
                 sketcher=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.root_dir = os.path.join(root_dir, 'experiment', model_name)

        self.pre_generator = pre_generator.to(self.device)
        for param in self.pre_generator.parameters():
            param.requires_grad = False
        self.pre_sketcher = pre_sketcher.to(self.device)
        for param in self.pre_sketcher.parameters():
            param.requires_grad = False
        self.encoder = encoder.to(self.device)
        self.sketcher = sketcher.to(self.device) if sketcher else None

        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=0.0001)
        self.sketcher_optim = None
        if self.sketcher:
            self.sketcher_optim = torch.optim.Adam(self.sketcher.parameters(), lr=0.0001)

        self.z_loss = torch.nn.L1Loss()
        self.gen_loss = torch.nn.L1Loss()
        self.conceptual_loss = AlexNetLoss(self.device)

        self.writer_images = SummaryWriter(f'runs/{model_name}/images')
        self.writer_loss = SummaryWriter(f'runs/{model_name}/loss')

    def load_experiment(self, experiment_name, checkpoint_nr=None):
        self.experiment_dir = os.path.join(self.root_dir, experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if checkpoint_nr is not None:
            checkpoint = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{checkpoint_nr}.pt')
            self._load_checkpoint(checkpoint)
        else:
            latest_checkpoint = self._find_latest_checkpoint()
            if latest_checkpoint:
                self._load_checkpoint(latest_checkpoint)

    def run_experiment(self,
                       experiment_name,
                       batch_size=32,
                       num_epochs=1000,
                       accumulation_steps=4,
                       save_every=10,
                       max_img_to_show=8):
        self.experiment_dir = os.path.join(self.root_dir, experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        latest_checkpoint = self._find_latest_checkpoint()
        self.start_epoch = 0
        if latest_checkpoint:
            self._load_checkpoint(latest_checkpoint)

        time_history = deque(maxlen=10)

        for epoch in range(self.start_epoch, num_epochs):
            try:
                self.encoder_optim.zero_grad()
                if self.sketcher_optim:
                    self.sketcher_optim.zero_grad()

                sketch_list = []
                real_list = []
                fake_list = []
                encoder_loss_avg = 0

                start_time = time.time()

                for step in range(accumulation_steps):
                    encoder_loss = None

                    for item in range(batch_size//accumulation_steps):
                        z = torch.randn([1, self.pre_generator.z_dim], device=self.device)
                        with torch.no_grad():
                            real = self._generate_image(z)
                            sketch = self._generate_sketch(real)

                        fake_z = self.encoder(sketch)
                        z_loss = self.z_loss(z, fake_z)

                        with torch.no_grad():
                            fake = self._generate_image(fake_z)

                        gen_loss = self.gen_loss(real, fake)
                        gen_loss.add_(self.conceptual_loss(real, fake))

                        if encoder_loss is None:
                            encoder_loss = z_loss + gen_loss
                        else:
                            encoder_loss.add_(z_loss)
                            encoder_loss.add_(gen_loss)

                        if self.sketcher is None:
                            with torch.no_grad():
                                fake_sketch = self._generate_sketch(fake)
                            sketch_loss = self.gen_loss(sketch, fake_sketch)
                            sketch_loss.add_(self.conceptual_loss(sketch, fake_sketch))
                            encoder_loss.add_(sketch_loss)
                        else:
                            fake_sketch = self.sketcher(fake)
                            sketch_loss = self.gen_loss(sketch, fake_sketch)
                            sketch_loss.add_(self.conceptual_loss(sketch, fake_sketch))
                            encoder_loss.add_(sketch_loss)

                        sketch_list.append(sketch.detach().cpu())
                        real_list.append(real.detach().cpu())
                        fake_list.append(fake.detach().cpu())

                    encoder_loss.div_(batch_size)
                    encoder_loss_avg += encoder_loss.item()
                    encoder_loss.backward()

                self.encoder_optim.step()
                if self.sketcher_optim:
                    self.sketcher_optim.step()

                self.writer_loss.add_scalar('encoder_loss', encoder_loss_avg, epoch)

                if epoch % save_every == 0:
                    self._save_checkpoint(epoch)
                    with torch.no_grad():
                        # Concatenate tensors along the batch dimension
                        sketch_tensors = torch.cat(sketch_list[:max_img_to_show], dim=0)
                        fake_tensors = torch.cat(fake_list[:max_img_to_show], dim=0)
                        real_tensors = torch.cat(real_list[:max_img_to_show], dim=0)

                        img_grid_sketch = torchvision.utils.make_grid(sketch_tensors, normalize=True)
                        img_grid_fake = torchvision.utils.make_grid(fake_tensors, normalize=True)
                        img_grid_real = torchvision.utils.make_grid(real_tensors, normalize=True)
                        self.writer_images.add_image(
                            "Sketch Images", img_grid_sketch, global_step=epoch//save_every
                        )
                        self.writer_images.add_image(
                            "Fake Images", img_grid_fake, global_step=epoch//save_every
                        )
                        self.writer_images.add_image(
                            "Real Images", img_grid_real, global_step=epoch//save_every
                        )

                end_time = time.time()
                elapsed_time = end_time - start_time
                time_history.append(elapsed_time)

                progress = epoch / num_epochs * 100
                avg_time = sum(time_history) / len(time_history)
                remaining_batches = num_epochs - epoch
                remaining_time = remaining_batches * avg_time

                mins, secs = divmod(remaining_time, 60)
                hours, mins = divmod(mins, 60)

                print('\r', end='')
                print(f"Epoch [{epoch + 1}/{num_epochs}] "
                      f"Progress: {progress:.2f}% "
                      f"Remaining time: {int(hours)}:{int(mins)}:{int(secs)}",
                      end='', flush=True)
            except KeyboardInterrupt:
                print()
                print("Stopping training... Saving checkpoint...")
                self._save_checkpoint(epoch)
                print("Checkpoint saved.")
                return

    def inception_v3_forward_with_features(self, model, x):
        x = model.Conv2d_1a_3x3(x)
        x = model.Conv2d_2a_3x3(x)
        x = model.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = model.Conv2d_3b_1x1(x)
        x = model.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = model.Mixed_5b(x)
        x = model.Mixed_5c(x)
        x = model.Mixed_5d(x)
        x = model.Mixed_6a(x)
        x = model.Mixed_6b(x)
        x = model.Mixed_6c(x)
        x = model.Mixed_6d(x)
        x = model.Mixed_6e(x)
        x = model.Mixed_7a(x)
        x = model.Mixed_7b(x)
        x = model.Mixed_7c(x)
        return x

    def compute_inception_features(self, model, images):
        # Resize images to at least 299x299 pixels
        images_resized = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        with torch.no_grad():
            features = self.inception_v3_forward_with_features(model, images_resized)
            features = features.mean([2, 3])  # Global average pooling
        return features

    def preprocess_inception(self, images):
        images.clamp(-1, 1)
        images = images.clamp(-1, 1)
        return images

    def update_mean_cov(self, features, prev_mean, prev_cov, batch_num):
        batch_size = features.shape[0]
        mean = features.mean(0)

        if batch_num == 1:
            cov = (features - mean).T @ (features - mean) / (batch_size - 1)
        else:
            cov = prev_cov * (batch_num - 1)
            cov += (features - mean).T @ (features - mean) / (batch_size - 1)
            cov += (mean - prev_mean).unsqueeze(1) @ (mean - prev_mean).unsqueeze(0) * batch_size * (batch_num - 1) / batch_num
            cov /= batch_num

        return mean, cov

    def compute_fid2(self, batch_size=4, num_samples=10000):
        print("Computing FID...")

        num_batches = num_samples // batch_size

        self.encoder.eval()
        self.pre_generator.eval()
        self.pre_sketcher.eval()

        inception_model = InceptionV3(output_blocks=[3], normalize_input=True).to(self.device)
        inception_model.eval()

        all_real_features = []
        all_fake_features = []

        for i in range(num_batches):
            with torch.no_grad():
                real_batch = self._generate_image(
                    torch.randn([batch_size, self.pre_generator.z_dim], device=self.device))
                generated_batch = self._generate_image(self.encoder(self._generate_sketch(real_batch)))

                with torch.no_grad():
                    real_batch = self.preprocess_inception(real_batch)
                    generated_batch = self.preprocess_inception(generated_batch)

                real_features = inception_model(real_batch)[-1].squeeze()
                fake_features = inception_model(generated_batch)[-1].squeeze()

                all_real_features.append(real_features)
                all_fake_features.append(fake_features)

            print("\r", end="")
            print(f"Batch {i + 1}/{num_batches}", end="", flush=True)
        print()

        all_real_features = torch.cat(all_real_features, dim=0)
        all_fake_features = torch.cat(all_fake_features, dim=0)

        real_mean = all_real_features.mean(dim=0)
        real_cov = torch.matmul((all_real_features - real_mean).T, (all_real_features - real_mean)) / (num_samples - 1)

        fake_mean = all_fake_features.mean(dim=0)
        fake_cov = torch.matmul((all_fake_features - fake_mean).T, (all_fake_features - fake_mean)) / (num_samples - 1)

        fid_score = self.calculate_frechet_distance(real_mean, real_cov, fake_mean, fake_cov)
        return fid_score

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        # convert pytorch tensors to numpy arrays
        mu1 = mu1.cpu().numpy()
        mu2 = mu2.cpu().numpy()
        sigma1 = sigma1.cpu().numpy()
        sigma2 = sigma2.cpu().numpy()

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)

    def compute_fid(self, batch_size=4, num_samples=10000):
        print("Computing FID...")

        num_batches = num_samples // batch_size

        self.encoder.eval()
        self.pre_generator.eval()
        self.pre_sketcher.eval()

        real_images = []
        generated_images = []

        for i in range(num_batches):
            with torch.no_grad():
                real_batch = self._generate_image(
                    torch.randn([batch_size, self.pre_generator.z_dim], device=self.device))
                generated_batch = self._generate_image(self.encoder(self._generate_sketch(real_batch)))

                real_images.append(real_batch)
                generated_images.append(generated_batch)

            if i % 10 == 0:
                print("\r", end="")
                print(f"Batch {i + 1}/{num_batches}", end="", flush=True)

        real_images = torch.cat(real_images, 0).mul(255).clamp(0, 255).byte()
        generated_images = torch.cat(generated_images, 0).mul(255).clamp(0, 255).byte()

        fid_metric = FrechetInceptionDistance(num_features=2048).to(self.device)
        fid_metric.update(real_images, real=True)
        fid_metric.update(generated_images, real=False)
        fid = fid_metric.compute()

        return fid

    def _generate_image(self, z):
        with torch.no_grad():
            fake_img = self.pre_generator(z)
        return fake_img

    def _generate_sketch(self, img):
        with torch.no_grad():
            sketch = self.pre_sketcher(img)
        return sketch

    def _find_latest_checkpoint(self):
        checkpoint_files = glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pt"))
        if not checkpoint_files:
            return None

        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        return latest_checkpoint

    def _save_checkpoint(self, epoch):
        checkpoint = {
            'has_sketcher': self.sketcher is not None,
            'encoder_state_dict': self.encoder.state_dict(),
            'epoch': self.start_epoch
        }

        if self.sketcher is not None:
            checkpoint['sketcher_state_dict'] = self.sketcher.state_dict()

        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        if checkpoint['has_sketcher']:
            self.sketcher.load_state_dict(checkpoint['sketcher_state_dict'])
        self.start_epoch = checkpoint['epoch']
