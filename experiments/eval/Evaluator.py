import os

import torch

from eval.FID2 import FID2
from models.SketchInverter import Encoder

from eval.FID import FID
from eval.LPIPS import LPIPS
from eval.LDPS import LDPS
from eval.SketchFidelity import SketchFidelity

import csv

class Evaluator:
    def __init__(self, root_dir, models_folder, dataset, generator, encoder):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.root_dir = root_dir
        self.models_folder = models_folder
        self.dataset = dataset

        self.generator = generator.to(self.device)
        for param in self.generator.parameters():
            param.requires_grad = False
        self.encoder = encoder.to(self.device)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.fid = FID()
        self.fid2 = FID2()
        self.lpips = LPIPS()
        self.ldps = LDPS(root_dir)
        self.sf = SketchFidelity()

    def evaluate_all(self, batch_size=8):
        evaluation_results = []
        for model in os.listdir(self.models_folder):
            if model.endswith(".pt"):
                model_scores = self.evaluate_model(model, batch_size)
                model_scores["Model"] = model
                evaluation_results.append(model_scores)

        with open(os.path.join(self.root_dir, "evaluation_results.csv"), mode='w') as csv_file:
            fieldnames = ["Model", "FID", "LPIPS", "LDPS"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            for row in evaluation_results:
                writer.writerow(row)

    def evaluate_model(self, model, batch_size=8):
        model_path = os.path.join(self.models_folder, model)
        checkpoint = torch.load(model_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])

        print(f'Starting evaluation for {model}.')

        print('Evaluating FID score.')
        fid_score = self.fid(self.encoder, self.generator, self.dataset, batch_size=batch_size)
        print(f'M: {model} | FID score: {fid_score}')

        print('Evaluating LPIPS score.')
        lpips_score = self.lpips(self.encoder, self.generator, self.dataset, batch_size=batch_size)
        print(f'M: {model} | LPIPS score: {lpips_score}')

        print('Evaluating LDPS score.')
        ldps_score = self.ldps(self.encoder, self.generator, self.dataset, batch_size=batch_size)
        print(f'M: {model} | LDPS score: {ldps_score}')

        return {
            "FID": fid_score,
            "LPIPS": lpips_score,
            "LDPS": ldps_score,
        }

    def evaluate_model_SF(self, model_name, batch_size=8):
        model_path = os.path.join(self.models_folder, model_name)
        checkpoint = torch.load(model_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])

        print(f'Starting evaluation for {model_name}.')
        fidelity = self.sf(self.encoder, self.generator, self.dataset, batch_size=batch_size)
        print(f'M: {model_name} | Fidelity score: {fidelity}')

        return fidelity

    def evaluate_model_FID2(self, model_name, batch_size=8):
        model_path = os.path.join(self.models_folder, model_name)
        checkpoint = torch.load(model_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])

        print(f'Starting evaluation for {model_name}.')
        fid2 = self.fid2(self.encoder, self.generator, self.dataset, batch_size=batch_size)
        print(f'M: {model_name} | FID2 score: {fid2}')

        return fid2


