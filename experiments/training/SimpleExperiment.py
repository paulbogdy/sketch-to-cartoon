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


class SimpleExperiment:

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
                 z_loss_alpha=1,
                 binarize_sketch=False,
                 encoder_optimizer: Optimizers = Optimizers.ADAM,
                 encoder_hyper_params={'lr': 0.001},
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

        self.pre_generator = pre_generator.to(self.device)
        for param in self.pre_generator.parameters():
            param.requires_grad = False
        self.pre_sketcher = pre_sketcher.to(self.device)
        for param in self.pre_sketcher.parameters():
            param.requires_grad = False
        self.encoder = encoder.to(self.device)

        self.binarize_sketch = binarize_sketch

        self.encoder_optimizer = encoder_optimizer
        self.encoder_hyper_params = encoder_hyper_params

        self.z_loss = CosineSimilarityLoss()

        self.z_loss_alpha = z_loss_alpha

        self.writer_images = SummaryWriter(os.path.join(self.experiment_dir, 'runs', 'images'))
        self.writer_loss = SummaryWriter(os.path.join(self.experiment_dir, 'runs', 'loss'))

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
        sketch, src, point = data
        sketch = sketch.to(self.device)
        src = src.to(self.device)
        point = point.to(self.device)

        tensorboard_step = batch_idx + len(self.data_loader) * epoch

        fake_z = self.encoder(sketch)
            
        encoder_loss = self.z_loss_alpha * self.z_loss(point, fake_z)

        encoder_loss.backward()
        encoder_loss_agg = encoder_loss.item()

        self.encoder_optim.step()

        if tensorboard_step % show_every_n_steps == 0:
            #self.writer_images.add_images('images/fake', fake, tensorboard_step)
            self.writer_images.add_images('images/real', src, tensorboard_step)
            self.writer_images.add_images('images/sketch', sketch, tensorboard_step)

        self.writer_loss.add_scalar('losses/Encoder Loss', encoder_loss_agg, tensorboard_step)

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
            'encoder_state_dict': self.encoder.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, new_batch_size):
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        total_data_processed = checkpoint['total_data_processed']
        self.start_epoch = checkpoint['epoch']
        self.start_batch_idx = (total_data_processed % (len(self.data_loader) * new_batch_size)) // new_batch_size

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])

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
            'encoder_optimizer': self.encoder_optimizer.name,  # save the enum name
            'encoder_hyper_params': self.encoder_hyper_params,
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
        self.encoder_optimizer = self.Optimizers[config['encoder_optimizer']]  # load the enum from the name
        self.encoder_hyper_params = config['encoder_hyper_params']
        # check if sketcher exists before loading sketcher parameters
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
