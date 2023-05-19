import torch
import torch.nn as nn
import torch.functional as F


class MappingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=8, activation_slope=0.2):
        super(MappingNetwork, self).__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(activation_slope)]

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(activation_slope))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mapping_network = nn.Sequential(*layers)

    def forward(self, z):
        w = self.mapping_network(z)
        return w


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, upsample=False, downsample=False):
        super(ModulatedConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.downsample = downsample

        # Weight modulation
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.style_scale_transform = nn.Linear(style_dim, in_channels)

        # Padding
        self.pad = (kernel_size - 1) // 2

        self.eps = 1e-8

    def forward(self, x, style):
        b, c, h, w = x.shape
        style_scale = self.style_scale_transform(style).unsqueeze(-1).unsqueeze(-1)

        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            h *= 2
            w *= 2

        weight = self.weight * style_scale.view(b, 1, self.in_channels, 1, 1)
        weight = weight.view(b * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        weight = weight / torch.sqrt(torch.sum(weight ** 2, dim=[2], keepdim=True) + self.eps)

        x = x.view(1, b * c, h, w)
        x = nn.functional.conv2d(x, weight, padding=self.pad, groups=b)
        x = x.view(b, self.out_channels, h, w)

        return x


class ToRGB(nn.Module):
    def __init__(self, in_channels):
        super(ToRGB, self).__init__()

        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

        self.activate = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x) + self.bias
        x = self.activate(x)
        return x

class FromRGB(nn.Module):
    def __init__(self, out_channels):
        super(FromRGB, self).__init__()

        self.conv = nn.Conv2d(3, out_channels, kernel_size=1)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        self.activate = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x) + self.bias
        x = self.activate(x)
        return x


class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, upsample=False):
        super(SynthesisBlock, self).__init__()

        self.conv2 = ModulatedConv2d(in_channels, out_channels, kernel_size=3, style_dim=style_dim, upsample=upsample)
        self.activate = nn.LeakyReLU(0.2)

        self.noise_scaling_factors = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x, style):
        x = self.conv2(x, style)
        x = self.activate(x)

        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        scaled_noise = self.noise_scaling_factors * noise
        x = x + scaled_noise + self.bias

        return x


class SynthesisNetwork(nn.Module):
    def __init__(self, style_dim, num_channels=3, model_size=256):
        super(SynthesisNetwork, self).__init__()

        self.constant_input = nn.Parameter(torch.randn(1, model_size, 4, 4))
        self.style_dim = style_dim
        self.num_channels = num_channels

        self.block1_up = SynthesisBlock(model_size, model_size, style_dim, upsample=True)
        self.rgb1 = ToRGB(model_size)
        self.block2 = SynthesisBlock(model_size, model_size, style_dim)
        self.block2_up = SynthesisBlock(model_size, model_size, style_dim, upsample=True)
        self.rgb2 = ToRGB(model_size)
        self.block3 = SynthesisBlock(model_size, model_size//2, style_dim)
        self.block3_up = SynthesisBlock(model_size//2, model_size//2, style_dim, upsample=True)
        self.rgb3 = ToRGB(model_size//2)
        self.block4 = SynthesisBlock(model_size//2, model_size//4, style_dim)
        self.block4_up = SynthesisBlock(model_size//4, model_size//4, style_dim, upsample=True)
        self.rgb4 = ToRGB(model_size//4)
        self.block5 = SynthesisBlock(model_size//4, model_size//8, style_dim)
        self.block5_up = SynthesisBlock(model_size//8, model_size//8, style_dim, upsample=True)
        self.rgb5 = ToRGB(model_size//8)
        self.block6 = SynthesisBlock(model_size//8, model_size//16, style_dim)
        self.block6_up = SynthesisBlock(model_size//16, model_size//16, style_dim, upsample=True)
        self.rgb6 = ToRGB(model_size//16)

    def forward(self, style):
        b = style.shape[0]

        # Duplicate the constant input for the entire batch
        x = self.constant_input.repeat(b, 1, 1, 1)

        # Apply the synthesis blocks
        x = self.block1_up(x, style)
        rgb = self.rgb1(x)

        x = self.block2(x, style)
        x = self.block2_up(x, style)
        rgb = nn.functional.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False) + self.rgb2(x)

        x = self.block3(x, style)
        x = self.block3_up(x, style)
        rgb = nn.functional.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False) + self.rgb3(x)

        x = self.block4(x, style)
        x = self.block4_up(x, style)
        rgb = nn.functional.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False) + self.rgb4(x)

        x = self.block5(x, style)
        x = self.block5_up(x, style)
        rgb = nn.functional.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False) + self.rgb5(x)

        x = self.block6(x, style)
        x = self.block6_up(x, style)
        rgb = nn.functional.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False) + self.rgb6(x)

        return rgb


class Generator(nn.Module):
    def __init__(self, input_dim, num_channels=512):
        super(Generator, self).__init__()
        self.z_dim = input_dim
        self.mapping_network = MappingNetwork(input_dim, num_channels)
        self.synthesis_network = SynthesisNetwork(num_channels)

    def forward(self, z):
        w = self.mapping_network(z)
        img = self.synthesis_network(w)
        return img


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscriminatorBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activate = nn.LeakyReLU(0.2)

    def forward(self, x):
        x1 = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x1 = self.activate(self.conv3(x1))

        x = self.activate(self.conv1(x))
        x = self.activate(self.conv2(x))

        x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x = x + x1

        return x


class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size=4):
        super(MiniBatchStdDev, self).__init__()
        self.group_size = group_size

    def forward(self, input):
        grouped = input.view(self.group_size, -1)
        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        std = std.mean().view(1, 1, 1, 1)
        b, _, h, w = input.shape
        std = std.repeat(b, 1, h, w)
        return torch.cat([input, std], dim=1)


class Discriminator(nn.Module):
    def __init__(self, num_channels=3, model_size=256):
        super(Discriminator, self).__init__()

        self.from_rgb = FromRGB(model_size//32)

        self.block1 = DiscriminatorBlock(model_size//32, model_size//16)
        self.block2 = DiscriminatorBlock(model_size//16, model_size//8)
        self.block3 = DiscriminatorBlock(model_size//8, model_size//4)
        self.block4 = DiscriminatorBlock(model_size//4, model_size//2)
        self.block5 = DiscriminatorBlock(model_size//2, model_size)
        self.block6 = DiscriminatorBlock(model_size, model_size)

        # self.mini_batch = MiniBatchStdDev()

        self.flatten = nn.Flatten()
        self.final_layer = nn.Linear(model_size * model_size//64 * model_size//64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.from_rgb(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        # x = self.mini_batch(x)
        x = self.flatten(x)
        x = self.final_layer(x)

        return x
