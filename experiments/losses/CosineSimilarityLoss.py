import torch
import torch.nn as nn


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity()

    def forward(self, x, y):
        return 1 - self.cosine_similarity(x, y).mean()
