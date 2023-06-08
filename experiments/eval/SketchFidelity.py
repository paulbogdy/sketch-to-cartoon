import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
import torch.nn.functional as F


class SketchFidelity:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.resnet = InceptionResnetV1(pretrained='vggface2')

    def __call__(self, encoder, pre_generator, dataset, batch_size=8):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.resnet = self.resnet.to(device).eval()

        encoder = encoder.to(device).eval()
        pre_generator = pre_generator.to(device).eval()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        score = 0
        for data in tqdm(dataloader, total=len(dataloader)):
            with torch.no_grad():
                sketch, src, point = data

                sketch = sketch.to(device)
                fake_z = encoder(sketch)
                fake = pre_generator(fake_z)
                # Convert sketch to 3-channel
                sketch_rgb = sketch.expand(-1, 3, -1, -1)

                # Resize tensors to 160x160
                sketch_resized = F.interpolate(sketch_rgb, size=(160, 160))
                fake_resized = F.interpolate(fake, size=(160, 160))

                # Normalize tensors
                mean = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
                std = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
                sketch_normalized = (sketch_resized - mean) / std
                fake_normalized = (fake_resized - mean) / std

                # Generate embeddings
                sketch_embeddings = self.resnet(sketch_normalized)
                fake_embeddings = self.resnet(fake_normalized)

                score += torch.pairwise_distance(sketch_embeddings, fake_embeddings).mean()

        return score / len(dataloader)
