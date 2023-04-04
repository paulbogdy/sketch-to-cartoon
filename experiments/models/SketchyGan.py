import torch
import torch.nn as nn


class MRU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MRU, self).__init__()
        self.conv1 = nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, stride=1, padding=1)

        self.activation = nn.ReLU(inplace=True)

    def resized_sketch(self, x, sketch):
        return nn.functional.interpolate(sketch, size=x.shape[2:], mode='bilinear', align_corners=True)

    def resized_x(self, x):
        return x

    def forward(self, x, sketch):
        # Resize image (I) to match the dimensions of input feature maps (x)
        sketch = self.resized_sketch(x, sketch)
        x_sketch = torch.cat([x, sketch], dim=1)

        mi = self.sigmoid(self.conv1(x_sketch))
        zi = self.activation(self.conv2(torch.cat([mi * x, sketch], dim=1)))

        ni = self.sigmoid(self.conv3(x_sketch))
        yi = (1 - ni) * zi + ni * self.resized_x(x)

        return yi


class DownMRU(MRU):
    def __init__(self, in_channels, out_channels):
        super(DownMRU, self).__init__(in_channels, out_channels)
        self.conv2 = nn.Conv2d(in_channels + 1, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels + 1, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def resized_x(self, x):
        return self.conv4(x)


class DownMRU2(MRU):
    def __init__(self, in_channels, out_channels):
        super(DownMRU2, self).__init__(in_channels, out_channels)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, sketch):
        y = super().forward(x, sketch)
        y = self.sigmoid(self.conv4(y))

        return y


class UpMRU(MRU):
    def __init__(self, in_channels, out_channels):
        super(UpMRU, self).__init__(in_channels, out_channels)
        self.conv2 = nn.ConvTranspose2d(in_channels + 1, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(in_channels + 1, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def resized_x(self, x):
        return self.conv4(x)


class UpMRU2(MRU):
    def __init__(self, in_channels, out_channels):
        super(UpMRU2, self).__init__(in_channels, out_channels)
        self.conv4 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x, sketch):
        y = super().forward(x, sketch)
        y = self.conv4(y)

        return y


class SketchyGenerator(nn.Module):
    def __init__(self, input_channels=3, initial_factor=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mru64_32 = DownMRU2(input_channels, initial_factor)
        self.norm64_32 = nn.BatchNorm2d(initial_factor)
        self.mru32_16 = DownMRU2(initial_factor, initial_factor * 2)
        self.norm32_16 = nn.BatchNorm2d(initial_factor * 2)
        self.mru16_8 = DownMRU2(initial_factor * 2, initial_factor * 4)
        self.norm16_8 = nn.BatchNorm2d(initial_factor * 4)
        self.mru8_8 = MRU(initial_factor * 4, initial_factor * 4)
        self.norm8_8 = nn.BatchNorm2d(initial_factor * 4)
        self.mru8_16 = UpMRU2(initial_factor * 4, initial_factor * 2)
        self.norm8_16 = nn.BatchNorm2d(initial_factor * 2)
        self.mru16_32 = UpMRU2(initial_factor * 2, initial_factor)
        self.norm16_32 = nn.BatchNorm2d(initial_factor)
        self.mru32_64 = UpMRU2(initial_factor, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, I):
        # Encoder
        x64_32 = self.mru64_32(x, I)
        x64_32 = self.norm64_32(x64_32)
        x32_16 = self.mru32_16(x64_32, I)
        x32_16 = self.norm32_16(x32_16)
        x16_8 = self.mru16_8(x32_16, I)
        x16_8 = self.norm16_8(x16_8)

        # Encoded Space
        x8_8 = self.mru8_8(x16_8, I)
        x8_8 = self.norm8_8(x8_8)

        # Decoder
        x8_16 = self.mru8_16(x8_8 + x16_8, I)
        x8_16 = self.norm8_16(x8_16)
        x16_32 = self.mru16_32(x8_16 + x32_16, I)
        x16_32 = self.norm16_32(x16_32)
        x32_64 = self.mru32_64(x16_32 + x64_32, I)

        return self.sigmoid(x32_64)


class SketchyDiscriminator(nn.Module):
    def __init__(self, input_channels=3, initial_factor=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mru64_32 = DownMRU2(input_channels, initial_factor)
        self.norm64_32 = nn.BatchNorm2d(initial_factor)
        self.mru32_16 = DownMRU2(initial_factor, initial_factor * 2)
        self.norm32_16 = nn.BatchNorm2d(initial_factor * 2)
        self.mru16_8 = DownMRU2(initial_factor * 2, initial_factor * 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, I):
        x64_32 = self.mru64_32(x, I)
        x64_32 = self.norm64_32(x64_32)
        x32_16 = self.mru32_16(x64_32, I)
        x32_16 = self.norm32_16(x32_16)
        x16_8 = self.mru16_8(x32_16, I)

        return self.sigmoid(x16_8)


class SketchyGeneratorInstaNorm(SketchyGenerator):
    def __init__(self, input_channels=3, initial_factor=16, *args, **kwargs):
        super().__init__(input_channels, initial_factor, *args, **kwargs)
        self.norm64_32 = nn.InstanceNorm2d(initial_factor)
        self.norm32_16 = nn.InstanceNorm2d(initial_factor * 2)
        self.norm16_8 = nn.InstanceNorm2d(initial_factor * 4)
        self.norm8_8 = nn.InstanceNorm2d(initial_factor * 4)
        self.norm8_16 = nn.InstanceNorm2d(initial_factor * 2)
        self.norm16_32 = nn.InstanceNorm2d(initial_factor)


class SketchyDiscriminatorInstaNorm(SketchyDiscriminator):
    def __init__(self, input_channels=3, initial_factor=16, *args, **kwargs):
        super().__init__(input_channels, initial_factor, *args, **kwargs)
        self.norm64_32 = nn.InstanceNorm2d(initial_factor)
        self.norm32_16 = nn.InstanceNorm2d(initial_factor * 2)