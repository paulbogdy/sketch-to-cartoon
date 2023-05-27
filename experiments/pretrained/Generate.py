import pickle
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from models.StyleGan2 import Generator
from pretrained.PretrainedGenerator import CartoonGenerator, CartoonGeneratorStyleLatent

# g_ema = Generator(
#         256, 512, 8, channel_multiplier=2
#     ).cuda()
# checkpoint = torch.load('NaverWebtoon-040000.pt')
#
# g_ema.load_state_dict(checkpoint['g_ema'])

from pathlib import Path
root_path = Path(__file__).parent.parent

G = CartoonGenerator(root_path)

# with open('ffhq.pkl', 'rb') as f:
#     G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
#
# num_params = sum(p.numel() for p in G.parameters())
# print("Number of parameters:", num_params)

z = torch.randn([1, 512]).cuda()  # latent codes
c = None  # class labels (not used in this example)
epsilon = 0.1


arguments_strModel = 'bsds500' # only 'bsds500' for now

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-hed/network-' + arguments_strModel + '.pytorch', file_name='hed-' + arguments_strModel).items()})

    def forward(self, tenInput):
        tenInput = tenInput * 255.0
        tenInput = tenInput - torch.tensor(data=[104.00698793, 116.66876762, 122.67891434], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

        return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))
    # end
# end

netNetwork = None

##########################################################

def estimate(tenInput):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cuda().eval()
    # end
    intWidth = tenInput.shape[2]
    intHeight = tenInput.shape[3]

    # assert(intWidth == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert(intHeight == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    return netNetwork(tenInput)
# end

def binarize_image(img_tensor, threshold=0.5):
    img_tensor = torch.where(img_tensor > threshold, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
    return img_tensor


def display_image(img, color_space='RGB'):
    original_img = img.clone()
    sketch_img = estimate(img)
    sketch_img.clamp_(-1, 1)

    # Binarize the sketch
    binarized_sketch = binarize_image(sketch_img)

    # Convert tensors to numpy arrays for displaying
    original_img_np = original_img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    sketch_img_np = sketch_img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    binarized_sketch_np = binarized_sketch.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)

    # Scale images to [0, 255] if necessary
    original_img_np = scale_image(original_img_np)
    sketch_img_np = scale_image(sketch_img_np)
    binarized_sketch_np = scale_image(binarized_sketch_np)

    # Convert color space if necessary
    if color_space == 'BGR':
        original_img_np = cv2.cvtColor(original_img_np, cv2.COLOR_BGR2RGB)
        sketch_img_np = cv2.cvtColor(sketch_img_np, cv2.COLOR_BGR2RGB)
        binarized_sketch_np = cv2.cvtColor(binarized_sketch_np, cv2.COLOR_BGR2RGB)

    # Display images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_img_np, cmap='gray')
    plt.title('Original Image')
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(sketch_img_np, cmap='gray')
    plt.title('Sketch')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(binarized_sketch_np, cmap='gray')
    plt.title('Binarized Sketch')
    plt.axis("off")

    plt.draw()
    plt.pause(1)  # Pause for 1 second
    plt.cla()


def scale_image(img_np):
    # Check value range and scale to [0, 255] if necessary
    min_val = np.min(img_np)
    max_val = np.max(img_np)
    if min_val >= -1.0 and max_val <= 1.0:
        img_np = ((img_np + 1) * 127.5).astype(np.uint8)
    elif min_val >= 0.0 and max_val <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    return img_np


try:
    img = G(z)
    display_image(img)

    noise = torch.randn_like(z) * epsilon
    z = z + noise
    z = torch.clamp(z, min=-1, max=1)

except KeyboardInterrupt:
    print("Image generation stopped.")
    plt.close()                # NCHW, f, Noneloat32, dynamic range [-1, +1]

