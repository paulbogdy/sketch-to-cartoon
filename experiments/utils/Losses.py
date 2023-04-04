import torchvision.models
from torchvision import transforms
import torch
import torch.nn as nn


class InceptionLoss(nn.Module):
    def __init__(self, device):
        super(InceptionLoss, self).__init__()
        self.inception_v3 = torchvision.models.inception_v3(pretrained=True)
        self.inception_v3.to(device)
        self.inception_v3.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.device = device
        self.criterion = nn.L1Loss()

    def results(self, x):
        result = []
        x = self.inception_v3.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception_v3.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception_v3.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception_v3.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception_v3.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception_v3.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception_v3.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception_v3.Mixed_5b(x)
        result.append(x)
        # N x 256 x 35 x 35
        x = self.inception_v3.Mixed_5c(x)
        result.append(x)
        # N x 288 x 35 x 35
        x = self.inception_v3.Mixed_5d(x)
        result.append(x)
        # N x 288 x 35 x 35
        x = self.inception_v3.Mixed_6a(x)
        result.append(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6b(x)
        result.append(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6c(x)
        result.append(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6d(x)
        result.append(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6e(x)
        result.append(x)

        return result

    def forward(self, x, y):
        res_x = self.results(x)
        res_y = self.results(y)

        loss = 0

        for i in range(0, len(res_x)):
            loss = loss + self.criterion(res_x[i], res_y[i])

        return loss
