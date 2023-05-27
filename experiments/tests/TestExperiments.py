import unittest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import shutil
from unittest.mock import MagicMock, patch

import torch
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage

from training.ExperimentFramework import Experiment
from models.SketchInverter import Encoder
from models.HEDNet import HDENet
from pathlib import Path
import HtmlTestRunner


class TestExperiment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.encoder = Encoder(512, (1, 256, 256))
        cls.generator = MagicMock()
        cls.sketcher = HDENet()
        cls.experiment_dir = Path(__file__).parent.parent

    def setUp(self):
        self.experiment_name = "test_config"
        self.experiment_name_2 = "test_config_2"
        self.experiment_name_3 = "test_config_3"
        self.experiment_path = self.experiment_dir / 'experiments' / self.experiment_name
        self.experiment_path_2 = self.experiment_dir / 'experiments' / self.experiment_name_2
        self.experiment_path_3 = self.experiment_dir / 'experiments' / self.experiment_name_3

    def tearDown(self):
        if self.experiment_path.exists():
            shutil.rmtree(self.experiment_path)
        if self.experiment_path_2.exists():
            shutil.rmtree(self.experiment_path_2)
        if self.experiment_path_3.exists():
            shutil.rmtree(self.experiment_path_3)
        pass

    def test_experiment_config(self):
        experiment = Experiment(self.experiment_name,
                                Experiment.Datasets.CARTOON,
                                self.encoder,
                                self.generator,
                                self.sketcher,
                                self.experiment_dir)

        config_file_path = self.experiment_path / 'config.json'

        # Check if the config file was created
        self.assertTrue(config_file_path.exists())

        # Load the config file
        with open(config_file_path, 'r') as f:
            config_data = json.load(f)

        # Check that the config file contains the correct data
        self.assertEqual(config_data['experiment_name'], self.experiment_name)
        self.assertEqual(config_data['dataset_choice'], experiment.dataset_choice.name)
        self.assertEqual(config_data['z_loss_alpha'], experiment.z_loss_alpha)
        self.assertEqual(config_data['content_loss_alpha'], experiment.content_loss_alpha)
        self.assertEqual(config_data['shape_loss_alpha'], experiment.shape_loss_alpha)
        self.assertEqual(config_data['encoder_hyper_params'], experiment.encoder_hyper_params)
        self.assertEqual(config_data['experiment_seed'], experiment.experiment_seed)

        # Create the experiment again, the config file should not be overwritten
        experiment2 = Experiment(self.experiment_name,
                                 Experiment.Datasets.CARTOON,
                                 self.encoder,
                                 self.generator,
                                 self.sketcher,
                                 self.experiment_dir,
                                 z_loss_alpha=None,
                                 content_loss_alpha=None,
                                 shape_loss_alpha=None,
                                 encoder_optimizer=None,
                                 encoder_hyper_params=None,
                                 experiment_seed=None)

        with open(config_file_path, 'r') as f:
            config_data2 = json.load(f)

        # Check that the config file wasn't overwritten
        self.assertEqual(config_data, config_data2)

        self.assertEqual(experiment2.z_loss_alpha, experiment.z_loss_alpha)
        self.assertEqual(experiment2.content_loss_alpha, experiment.content_loss_alpha)
        self.assertEqual(experiment2.shape_loss_alpha, experiment.shape_loss_alpha)
        self.assertEqual(experiment2.encoder_optimizer, experiment.encoder_optimizer)
        self.assertEqual(experiment2.encoder_hyper_params, experiment.encoder_hyper_params)
        self.assertEqual(experiment2.sketcher_optimizer, experiment.sketcher_optimizer)
        self.assertEqual(experiment2.sketcher_hyper_params, experiment.sketcher_hyper_params)
        self.assertEqual(experiment2.experiment_seed, experiment.experiment_seed)
        self.assertEqual(experiment2.dataset_choice, experiment.dataset_choice)

    def test_save_checkpoint(self):
        # Create an Experiment object
        experiment = Experiment(
            experiment_name=self.experiment_name,
            dataset_choice=Experiment.Datasets.CARTOON,
            encoder=self.encoder,
            pre_generator=self.generator,
            pre_sketcher=self.sketcher,
            root_dir=self.experiment_dir,
            sketcher=None,
            experiment_description='',
            experiment_seed=42
        )

        experiment.data_loader = experiment.load_dataset(32)

        # Mock torch.save
        with patch('torch.save') as mock_save:
            # Call save_checkpoint
            experiment.save_checkpoint(epoch=0, batch_idx=0, batch_size=32)
            # Check that torch.save was called with the correct arguments
            mock_save.assert_called_once()

    def test_load_checkpoint(self):
        # Create an Experiment object
        experiment = Experiment(
            experiment_name=self.experiment_name,
            dataset_choice=Experiment.Datasets.CARTOON,
            encoder=self.encoder,
            pre_generator=self.generator,
            pre_sketcher=self.sketcher,
            root_dir=self.experiment_dir,
            sketcher=None,
            experiment_description='',
            experiment_seed=42
        )

        experiment.data_loader = experiment.load_dataset(32)

        # Mock torch.load to return a fake checkpoint
        fake_checkpoint = {
            'epoch': 0,
            'total_data_processed': 0,
            'encoder_state_dict': MagicMock(),
            'sketcher_state_dict': None,
        }
        with patch('torch.load', return_value=fake_checkpoint):
            # Replace the load_state_dict method of the Encoder with a MagicMock
            experiment.encoder.load_state_dict = MagicMock()
            # Call load_checkpoint
            experiment.load_checkpoint(checkpoint_path='fake_path', new_batch_size=32)
            # Check that the encoder's load_state_dict was called with the correct arguments
            experiment.encoder.load_state_dict.assert_called_once_with(fake_checkpoint['encoder_state_dict'])

    def test_load_dataset(self):
        # Create an Experiment object with dataset_choice=CARTOON
        experiment = Experiment(
            experiment_name=self.experiment_name,
            dataset_choice=Experiment.Datasets.CARTOON,
            encoder=self.encoder,
            pre_generator=self.generator,
            pre_sketcher=self.sketcher,
            root_dir=self.experiment_dir,
            sketcher=None,
            experiment_description='',
            experiment_seed=42
        )

        # Mock SketchInverterDataset and DataLoader
        with patch('training.ExperimentFramework.SketchInverterDataset', return_value=MagicMock()) as MockDataset:
            with patch('training.ExperimentFramework.DataLoader', return_value=MagicMock()) as MockDataLoader:
                # Call load_dataset
                result = experiment.load_dataset(batch_size=32)
                # Check that DataLoader was called with the correct arguments
                MockDataLoader.assert_called_once_with(MockDataset.return_value, batch_size=32, shuffle=True)
                # Check that the result is a DataLoader
                self.assertIsInstance(result, MagicMock)

    def test_train_step(self):
        # Create an Experiment object
        experiment = Experiment(
            experiment_name=self.experiment_name,
            dataset_choice=Experiment.Datasets.CARTOON,
            encoder=self.encoder,
            pre_generator=self.generator,
            pre_sketcher=self.sketcher,
            root_dir=self.experiment_dir,
            sketcher=None,
            experiment_description='',
            experiment_seed=42
        )

        # Mock the necessary methods and attributes
        experiment.encoder_optim.zero_grad = MagicMock()
        experiment.encoder_optim.step = MagicMock()
        experiment._generate_image = MagicMock(return_value=torch.randn(1, 3, 256, 256))
        experiment._generate_sketch = MagicMock(return_value=torch.randn(1, 1, 256, 256))
        experiment.z_loss = MagicMock(return_value=MagicMock())
        experiment.img_loss = MagicMock(return_value=MagicMock())
        experiment.conceptual_loss = MagicMock(return_value=MagicMock())
        experiment.writer_images.add_images = MagicMock()
        experiment.writer_loss.add_scalar = MagicMock()

        # Create fake data
        batch_idx = 0
        data = (torch.randn(32, 1, 256, 256), torch.randn(32, 3, 256, 256), torch.randn(32, 512))
        batch_size = 32
        accumulation_steps = 4

        # Call train_step
        experiment.train_step(batch_idx, data, batch_size, accumulation_steps)

        # Check that the necessary methods were called
        experiment.encoder_optim.zero_grad.assert_called_once()
        experiment.encoder_optim.step.assert_called_once()
        experiment.z_loss.assert_called()
        experiment.img_loss.assert_called()
        experiment.conceptual_loss.assert_called()
        experiment.writer_images.add_images.assert_called()
        experiment.writer_loss.add_scalar.assert_called()

    @staticmethod
    def display_image(image_tensor, title=None):
        """Display an image tensor."""
        image = ToPILImage()(image_tensor).convert("RGB")
        plt.imshow(image)
        if title:
            plt.title(title)
        plt.show()


if __name__ == "__main__":
    unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(output='test_reports'))

