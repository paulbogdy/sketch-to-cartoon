import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DCGANGenerator(nn.Module):
    def __init__(self, z_dim, out_channels, base_channels=64):
        super(DCGANGenerator, self).__init__()
        self.z_dim = z_dim

        self.deconv1 = nn.ConvTranspose2d(z_dim, base_channels * 8, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(base_channels * 8)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels * 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.deconv3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(base_channels * 2)
        self.relu3 = nn.ReLU(inplace=True)

        self.deconv4 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(base_channels)
        self.relu4 = nn.ReLU(inplace=True)

        self.deconv5 = nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=1, padding=2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu1(self.bn1(self.deconv1(x)))
        x = self.relu2(self.bn2(self.deconv2(x)))
        x = self.relu3(self.bn3(self.deconv3(x)))
        x = self.relu4(self.bn4(self.deconv4(x)))
        x = self.tanh(self.deconv5(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        x = self.lrelu3(self.bn3(self.conv3(x)))
        x = self.lrelu4(self.bn4(self.conv4(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.sigmoid(x)

    def loss(self, loss, x, y):
        x = self.lrelu1(self.conv1(x))
        y = self.lrelu1(self.conv1(y))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        y = self.lrelu2(self.bn2(self.conv2(y)))
        x = self.lrelu3(self.bn3(self.conv3(x)))
        y = self.lrelu3(self.bn3(self.conv3(y)))
        val_loss = loss(x, y)
        x = self.lrelu4(self.bn4(self.conv4(x)))
        y = self.lrelu4(self.bn4(self.conv4(y)))
        val_loss += loss(x, y)
        x = self.avg_pool(x)
        y = self.avg_pool(y)

        return val_loss / 2
