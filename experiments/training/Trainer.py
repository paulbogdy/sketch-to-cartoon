import torch
import os
from torch.utils.data import DataLoader
from glob import glob
import time
from collections import deque


class Trainer:
    def __init__(self, device, dataset, model_name, checkpoint_dir="checkpoints"):
        self.device = device
        self.dataset = dataset
        self.model_name = model_name
        self.checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, batch_size, num_epochs, start_epoch=0):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        nr_batches = len(dataloader)
        total_batches = nr_batches * num_epochs
        completed_batches = 0
        time_history = deque(maxlen=10)

        update_frequency = 10  # Update console output every 10 batches

        for epoch in range(start_epoch, num_epochs):
            try:
                for batch_idx, data in enumerate(dataloader):
                    start_time = time.time()
                    d_loss, g_loss = self.train_step(batch_idx, data)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    time_history.append(elapsed_time)

                    completed_batches += 1

                    if completed_batches % update_frequency == 0:
                        progress = completed_batches / total_batches * 100
                        avg_time = sum(time_history) / len(time_history)
                        remaining_batches = total_batches - completed_batches
                        remaining_time = remaining_batches * avg_time

                        mins, secs = divmod(remaining_time, 60)
                        hours, mins = divmod(mins, 60)

                        print('\r', end='')
                        print(f"Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx + 1}/{nr_batches}] "
                              f"d_loss: {d_loss.item()} g_loss: {g_loss.item()} "
                              f"Progress: {progress:.2f}% "
                              f"Remaining time: {int(hours)}:{int(mins)}:{int(secs)}",
                              end='', flush=True)
            except KeyboardInterrupt:
                print("\nStopping training. Saving the model...")
                self.save_checkpoint(epoch)
                return

            self.save_checkpoint(epoch)

    def continue_training(self, batch_size, num_epochs, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()

        if checkpoint_path is not None:
            print(f"Loading checkpoint: {checkpoint_path}")
            start_epoch = self.load_checkpoint(checkpoint_path)
        else:
            start_epoch = 0

        self.train(batch_size, num_epochs, start_epoch=start_epoch)

    def find_latest_checkpoint(self):
        checkpoint_files = glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pt"))
        if not checkpoint_files:
            return None

        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        return latest_checkpoint

    def train_step(self, batch_idx, data):
        raise NotImplementedError("Implement train step.")

    def save_checkpoint(self, epoch):
        pass

    def load_checkpoint(self, checkpoint_path):
        pass
