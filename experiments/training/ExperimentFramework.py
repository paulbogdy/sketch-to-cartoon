import glob
import json
import os
import random
from enum import Enum
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from tqdm import tqdm
from dataset.dataset import SketchInverterDataset
from eval.FID import FID
from eval.LDPS import LDPS
from eval.LPIPS import LPIPS

from losses.AlexNetLoss import AlexNetLoss
from losses.CosineSimilarityLoss import CosineSimilarityLoss


class Experiment:

    class Datasets(Enum):
        CARTOON = 1

    class Optimizers(Enum):
        ADAM = 1

    def __init__(self,
                 experiment_name: str,
                 dataset_choice: Datasets,
                 encoder,
                 pre_generator,
                 pre_sketcher,
                 root_dir,
                 sketcher=None,
                 no_memory_optimization=False,
                 z_loss_alpha=1,
                 use_cosine_for_z=False,
                 content_loss_alpha=1,
                 shape_loss_alpha=1,
                 use_conceptual_loss=False,
                 binarize_sketch=False,
                 encoder_optimizer: Optimizers = Optimizers.ADAM,
                 encoder_hyper_params={'lr': 0.001},
                 sketcher_optimizer: Optimizers = Optimizers.ADAM,
                 sketcher_hyper_params={'lr': 0.001},
                 experiment_description='',
                 experiment_seed=42):
        self.experiment_name = experiment_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.root_dir = os.path.join(root_dir, 'experiments')
        self.datasets_dir = os.path.join(os.path.dirname(root_dir), 'dataset')
        self.experiment_dir = os.path.join(self.root_dir, experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create a file handler for output filename
        handler = logging.FileHandler(os.path.join(self.experiment_dir, 'experiment.log'))

        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(handler)

        self.experiment_seed = experiment_seed

        self.dataset_choice = dataset_choice

        self.no_memory_optimization = no_memory_optimization

        self.pre_generator = pre_generator.to(self.device)
        for param in self.pre_generator.parameters():
            param.requires_grad = False
        self.pre_sketcher = pre_sketcher.to(self.device)
        for param in self.pre_sketcher.parameters():
            param.requires_grad = False
        self.encoder = encoder.to(self.device)
        self.sketcher = sketcher.to(self.device) if sketcher else None

        self.binarize_sketch = binarize_sketch

        self.encoder_optimizer = encoder_optimizer
        self.encoder_hyper_params = encoder_hyper_params

        if sketcher:
            self.sketcher_optimizer = sketcher_optimizer
            self.sketcher_hyper_params = sketcher_hyper_params
        else:
            self.sketcher_optimizer = None
            self.sketcher_hyper_params = None

        if use_cosine_for_z:
            self.z_loss = CosineSimilarityLoss()
        else:
            self.z_loss = torch.nn.L1Loss()
        self.img_loss = torch.nn.L1Loss()
        self.conceptual_loss = AlexNetLoss(self.device)

        self.z_loss_alpha = z_loss_alpha
        self.content_loss_alpha = content_loss_alpha
        self.shape_loss_alpha = shape_loss_alpha
        self.use_conceptual_loss = use_conceptual_loss

        self.writer_images = SummaryWriter(os.path.join(self.experiment_dir, 'runs', 'images'))
        self.writer_loss = SummaryWriter(os.path.join(self.experiment_dir, 'runs', 'loss'))

        self.experiment_description = experiment_description

        self.start_epoch = 0
        self.start_batch_idx = 0

        self.data_loader = None

        # Check if config exists
        config_path = os.path.join(self.experiment_dir, 'config.json')
        if os.path.exists(config_path):
            self.logger.info('Found existing config file. Loading config.')
            self.load_experiment_config()

        else:
            self.logger.info('No config file found. Saving current config.')
            self.save_experiment_config()

        random.seed(self.experiment_seed)
        torch.manual_seed(self.experiment_seed)
        np.random.seed(self.experiment_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if encoder_optimizer == self.Optimizers.ADAM:
            self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), **self.encoder_hyper_params)

        if sketcher and sketcher_optimizer == self.Optimizers.ADAM:
            self.sketcher_optim = torch.optim.Adam(self.sketcher.parameters(), **self.sketcher_hyper_params)

    def run_experiment(self,
                       batch_size=32,
                       num_epochs=1,
                       accumulation_steps=4,
                       save_every_n_batches=10,
                       show_every_n_steps=10):
        self.logger.info('Starting experiment.')
        self.data_loader = self.load_dataset(batch_size=batch_size)

        # Load latest checkpoint if it exists
        latest_checkpoint = self.find_latest_checkpoint()
        if latest_checkpoint is not None:
            self.logger.info(f'Found latest checkpoint: {latest_checkpoint}. Loading checkpoint.')
            self.load_checkpoint(latest_checkpoint, batch_size)
            self.logger.info('Checkpoint loaded.')
        else:
            self.logger.info('No checkpoint found.')

        self.logger.info(f'Num epochs: {self.start_epoch}')

        for epoch in range(num_epochs):
            if epoch < self.start_epoch:
                continue
            print(f'Epoch: {epoch}/{num_epochs}')
            for batch_idx, data in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
                if epoch == self.start_epoch and batch_idx < self.start_batch_idx:
                    continue
                try:
                    self.train_step(epoch, batch_idx, data, batch_size, accumulation_steps, show_every_n_steps)
                except KeyboardInterrupt:
                    self.logger.info("Keyboard interrupt detected!")
                    self.save_checkpoint(epoch, batch_idx, batch_size)
                    self.logger.info("Checkpoint saved!")
                    return
                except Exception as e:
                    self.logger.info("Something went wrong: ", e)
                    self.logger.info("Trying to save checkpoint...")
                    self.save_checkpoint(epoch, batch_idx, batch_size)
                    self.logger.info("Checkpoint saved!")
                    return
                if batch_idx % save_every_n_batches == 0:
                    self.save_checkpoint(epoch, batch_idx, batch_size)
        self.save_checkpoint(num_epochs, 0, batch_size)

    def train_step(self,
                   epoch,
                   batch_idx,
                   data,
                   batch_size,
                   accumulation_steps,
                   show_every_n_steps=10):
        self.encoder_optim.zero_grad()
        if self.sketcher is not None:
            self.sketcher_optim.zero_grad()
        sketch, src, point = data
        if self.binarize_sketch:
            sketch = self.binarize(sketch)
        fake = []

        mini_batch_size = batch_size//accumulation_steps
        tensorboard_step = batch_idx + len(self.data_loader) * epoch

        encoder_loss_agg = 0
        z_loss_agg = 0
        content_loss_agg = 0
        shape_loss_agg = 0

        for step in range(accumulation_steps):
            mini_batch_src = src[step * mini_batch_size: (step + 1) * mini_batch_size].to(self.device)
            mini_batch_sketch = sketch[step * mini_batch_size: (step + 1) * mini_batch_size].to(self.device)
            mini_batch_point = point[step * mini_batch_size: (step + 1) * mini_batch_size].to(self.device)

            fake_z = self.encoder(mini_batch_sketch)
            
            z_loss = self.z_loss(mini_batch_point, fake_z)
            encoder_loss = z_loss * self.z_loss_alpha
            z_loss_agg += z_loss.item()

            if self.content_loss_alpha == 0 and self.shape_loss_alpha == 0:
                encoder_loss.backward()
                encoder_loss_agg += encoder_loss.item()
                if tensorboard_step % show_every_n_steps == 0:
                    if self.no_memory_optimization:
                        fake_src = self._generate_image(fake_z)
                        for i in range(mini_batch_size):
                            fake.append(fake_src[i].unsqueeze(0).cpu().detach())
                    else:
                        for i in range(mini_batch_size):
                            z_i = fake_z[i].unsqueeze(0)
                            with torch.no_grad():
                                fake.append(self._generate_image(z_i).cpu().detach())
                continue

            if self.no_memory_optimization:
                fake_src = self._generate_image(fake_z)
                if tensorboard_step % show_every_n_steps == 0:
                    for i in range(mini_batch_size):
                        fake.append(fake_src[i].unsqueeze(0).cpu().detach())
            else:
                fake_src = torch.zeros_like(mini_batch_src, device=self.device)
                for i in range(mini_batch_size):
                    z_i = fake_z[i].unsqueeze(0)
                    fake_src[i] = self._generate_image(z_i)
                    if tensorboard_step % show_every_n_steps == 0:
                        fake.append(fake_src[i].unsqueeze(0).cpu().detach())

            if self.content_loss_alpha != 0:
                content_loss = self.img_loss(mini_batch_src, fake_src)
                if self.use_conceptual_loss:
                    content_loss.add_(self.conceptual_loss(mini_batch_src, fake_src))
                encoder_loss.add_(content_loss * self.content_loss_alpha)
                content_loss_agg += content_loss.item()

            if self.shape_loss_alpha == 0:
                encoder_loss.backward()
                encoder_loss_agg += encoder_loss.item()
                continue

            if self.sketcher:
                fake_sketch = self.sketcher(fake_src)
                real_sketch = self.sketcher(mini_batch_src)
                shape_loss = self.img_loss(mini_batch_sketch, real_sketch)
                shape_loss.add_(self.img_loss(mini_batch_sketch, fake_sketch))
                if self.use_conceptual_loss:
                    shape_loss.add_(self.conceptual_loss(mini_batch_sketch, real_sketch))
                    shape_loss.add_(self.conceptual_loss(mini_batch_sketch, fake_sketch))
            else:
                if self.no_memory_optimization:
                    fake_sketch = self._generate_sketch(fake_src)
                else:
                    fake_sketch = torch.zeros_like(mini_batch_sketch, device=self.device)
                    for i in range(mini_batch_size):
                        fake_src_i = fake_src[i].unsqueeze(0)
                        fake_sketch[i] = self._generate_sketch(fake_src_i)

                shape_loss = self.img_loss(mini_batch_sketch, fake_sketch)
                if self.use_conceptual_loss:
                    shape_loss.add_(self.conceptual_loss(mini_batch_sketch, fake_sketch))

            encoder_loss.add_(shape_loss * self.shape_loss_alpha)
            shape_loss_agg += shape_loss.item()

            encoder_loss.backward()
            encoder_loss_agg += encoder_loss.item()

        self.encoder_optim.step()
        if self.sketcher is not None:
            self.sketcher_optim.step()
        encoder_loss_agg /= accumulation_steps
        z_loss_agg /= accumulation_steps
        content_loss_agg /= accumulation_steps
        shape_loss_agg /= accumulation_steps

        if tensorboard_step % show_every_n_steps == 0:
            fake = torch.cat(fake, dim=0)
            self.writer_images.add_images('images/fake', fake, tensorboard_step)
            self.writer_images.add_images('images/real', src, tensorboard_step)
            self.writer_images.add_images('images/sketch', sketch, tensorboard_step)

        self.writer_loss.add_scalar('losses/Encoder Loss', encoder_loss_agg, tensorboard_step)
        self.writer_loss.add_scalar('losses/Z Loss', z_loss_agg, tensorboard_step)
        self.writer_loss.add_scalar('losses/Content Loss', content_loss_agg, tensorboard_step)
        self.writer_loss.add_scalar('losses/Shape Loss', shape_loss_agg, tensorboard_step)

    def load_dataset(self, batch_size):
        if self.dataset_choice == self.Datasets.CARTOON:
            dataset = SketchInverterDataset(
                root_dir=os.path.join(self.datasets_dir, 'synthetic_dataset_cartoon_faces'),
                device=self.device,
                transform=ToTensor(),
                image_size=(256, 256)
            )
            return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:

            raise NotImplementedError('Dataset not implemented')

    def load_test_dataset(self):
        if self.dataset_choice == self.Datasets.CARTOON:
            dataset = SketchInverterDataset(
                root_dir=os.path.join(self.datasets_dir, 'synthetic_dataset_cartoon_faces_test'),
                transform=ToTensor(),
                device=self.device,
                image_size=(256, 256)
            )
            percentage = 10
            num_examples = len(dataset)
            num_subset = int(num_examples * (percentage / 100))

            # Generate a list of indices without replacement.
            indices = np.random.choice(num_examples, num_subset, replace=False)

            subset = Subset(dataset, indices)
            return subset
        else:

            raise NotImplementedError('Dataset not implemented')

    def binarize(self, x, threshold=0.5):
        return (x > threshold).float()

    def _generate_image(self, z):
        return self.pre_generator(z)

    def _generate_sketch(self, img):
        return self.pre_sketcher(img)

    def save_checkpoint(self, epoch, batch_idx, batch_size):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'epoch_{epoch}_batch_{batch_idx}.pt')
        total_data_processed = epoch * len(self.data_loader) * batch_size + batch_idx * batch_size
        checkpoint = {
            'epoch': epoch,
            'total_data_processed': total_data_processed,
            'encoder_state_dict': self.encoder.state_dict(),
            'sketcher_state_dict': self.sketcher.state_dict() if self.sketcher else None,
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, new_batch_size):
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        if checkpoint['sketcher_state_dict'] is not None:
            self.sketcher.load_state_dict(checkpoint['sketcher_state_dict'])
        total_data_processed = checkpoint['total_data_processed']
        self.start_epoch = checkpoint['epoch']
        self.start_batch_idx = (total_data_processed % (len(self.data_loader) * new_batch_size)) // new_batch_size

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        if checkpoint['sketcher_state_dict'] is not None:
            self.sketcher.load_state_dict(checkpoint['sketcher_state_dict'])

    def find_latest_checkpoint(self):
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, '*.pt'))
        if len(checkpoints) == 0:
            return None
        else:
            return max(checkpoints, key=os.path.getctime)

    def save_experiment_config(self):
        config = {
            'experiment_name': self.experiment_name,
            'dataset_choice': self.dataset_choice.name,  # save the enum name
            'z_loss_alpha': self.z_loss_alpha,
            'content_loss_alpha': self.content_loss_alpha,
            'shape_loss_alpha': self.shape_loss_alpha,
            'use_conceptual_loss': self.use_conceptual_loss,
            'encoder_optimizer': self.encoder_optimizer.name,  # save the enum name
            'encoder_hyper_params': self.encoder_hyper_params,
            'sketcher_exists': self.sketcher is not None,  # flag to indicate if sketcher exists
            'sketcher_optimizer': self.sketcher_optimizer.name if self.sketcher else None,
            'sketcher_hyper_params': self.sketcher_hyper_params if self.sketcher else None,
            'experiment_description': self.experiment_description,
            'experiment_seed': self.experiment_seed,
        }

        with open(os.path.join(self.experiment_dir, 'config.json'), 'w') as f:
            json.dump(config, f)

    def load_experiment_config(self):
        with open(os.path.join(self.experiment_dir, 'config.json'), 'r') as f:
            config = json.load(f)

        if config['experiment_name'] != self.experiment_name:
            logging.warning('Experiment name in config does not match the experiment name in the experiment dir')
            return

        # Load the parameters from the config dictionary
        self.dataset_choice = self.Datasets[config['dataset_choice']]  # load the enum from the name
        self.z_loss_alpha = config['z_loss_alpha']
        self.content_loss_alpha = config['content_loss_alpha']
        self.shape_loss_alpha = config['shape_loss_alpha']
        self.encoder_optimizer = self.Optimizers[config['encoder_optimizer']]  # load the enum from the name
        self.encoder_hyper_params = config['encoder_hyper_params']
        # check if sketcher exists before loading sketcher parameters
        if config['sketcher_exists']:
            self.sketcher_optimizer = self.Optimizers[config['sketcher_optimizer']]  # load the enum from the name
            self.sketcher_hyper_params = config['sketcher_hyper_params']
        else:
            self.sketcher_optimizer = None
            self.sketcher_hyper_params = None
        self.experiment_description = config['experiment_description']
        self.experiment_seed = config['experiment_seed']
        # load all other parameters...

    def evaluate(self):
        self.logger.info('Starting evaluation.')
        test_dataset = self.load_test_dataset()

        # Load latest checkpoint if it exists
        latest_checkpoint = self.find_latest_checkpoint()
        if latest_checkpoint is not None:
            self.logger.info(f'Found latest checkpoint: {latest_checkpoint}. Loading checkpoint.')
            self.load_model(latest_checkpoint)
            self.logger.info('Checkpoint loaded.')
        else:
            self.logger.info('No checkpoint found.')

        self.logger.info('Evaluating FID score.')
        print('Evaluating FID score.')
        fid = FID()
        fid_score = fid(self.encoder, self.pre_generator, test_dataset, batch_size=4)
        self.logger.info(f'FID score: {fid_score}')

        del fid

        self.logger.info('Evaluating LPIPS score.')
        print('Evaluating LPIPS score.')
        lpips = LPIPS()
        lpips_score = lpips(self.encoder, self.pre_generator, test_dataset, batch_size=4)
        self.logger.info(f'LPIPS score: {lpips_score}')

        del lpips

        self.logger.info('Evaluating LDPS score.')
        print('Evaluating LDPS score.')
        ldps = LDPS(os.path.dirname(self.root_dir))
        ldps_score = ldps(self.encoder, self.pre_generator, test_dataset, batch_size=4)
        self.logger.info(f'LDPS score: {ldps_score}')

        del ldps

        return {
            "FID": fid_score,
            "LPIPS": lpips_score,
            "LDPS": ldps_score,
        }
