import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F


class UpSampleLayer(nn.Module):
    def __init__(self, scale_factor, mode='nearest', align_corners=None):
        super(UpSampleLayer, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockDown, self).__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.residue = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.residue(x) + self.block(x)


class ResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockUp, self).__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            UpSampleLayer(scale_factor=2, mode='bilinear', align_corners=True),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
        )
        self.residue = nn.Sequential(
            UpSampleLayer(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.residue(x) + self.block(x)


class Encoder(nn.Module):
    def __init__(self, z_dim, model_size=16):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(

            ResBlockDown(1, model_size),
            ResBlockDown(model_size, model_size * 2),
            ResBlockDown(model_size * 2, model_size * 4),
            ResBlockDown(model_size * 4, model_size * 8),
            ResBlockDown(model_size * 8, model_size * 16),

            nn.Conv2d(model_size * 16, model_size * 16, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(4 * 4 * 16 * model_size, z_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class Sketcher(nn.Module):
    def __init__(self, model_size=16):
        super(Sketcher, self).__init__()

        self.d1 = ResBlockDown(3, model_size)
        self.d2 = ResBlockDown(model_size, model_size)
        self.d3 = ResBlockDown(model_size, model_size * 2)
        self.d4 = ResBlockDown(model_size * 2, model_size * 2)
        self.d5 = ResBlockDown(model_size * 2, model_size * 4)
        self.d6 = ResBlockDown(model_size * 4, model_size * 4)

        self.u1 = ResBlockUp(model_size * 4, model_size * 4)
        self.u2 = ResBlockUp(model_size * 4 * 2, model_size * 2)
        self.u3 = ResBlockUp(model_size * 2 * 2, model_size * 2)
        self.u4 = ResBlockUp(model_size * 2 * 2, model_size)
        self.u5 = ResBlockUp(model_size * 2, model_size)
        self.u6 = ResBlockUp(model_size * 2, model_size // 2)

        self.conv = nn.Conv2d(model_size // 2, 1, kernel_size=1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)

        u1 = self.u1(d6)
        u1 = torch.cat([u1, d5], dim=1)

        u2 = self.u2(u1)
        u2 = torch.cat([u2, d4], dim=1)

        u3 = self.u3(u2)
        u3 = torch.cat([u3, d3], dim=1)

        u4 = self.u4(u3)
        u4 = torch.cat([u4, d2], dim=1)

        u5 = self.u5(u4)
        u5 = torch.cat([u5, d1], dim=1)

        u6 = self.u6(u5)

        return self.conv(u6)
