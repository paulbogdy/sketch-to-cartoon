import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import linalg
from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F


class FID2:
    def __init__(self):
        self.model = InceptionResnetV1(pretrained='vggface2')

    def __call__(self, encoder, pre_generator, dataset, batch_size=8):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(device).eval()

        encoder.to(device).eval()
        pre_generator.to(device).eval()

        all_real_features = []
        all_fake_features = []

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        for data in tqdm(dataloader, total=len(dataloader)):
            with torch.no_grad():
                sketch, src, point = data
                sketch = sketch.to(device)
                src = src.to(device)

                fake_z = encoder(sketch)
                fake = pre_generator(fake_z)

                # Resize tensors to 160x160
                real_resized = F.interpolate(src, size=(160, 160))
                fake_resized = F.interpolate(fake, size=(160, 160))

                # Normalize tensors
                mean = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
                std = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
                real_normalized = (real_resized - mean) / std
                fake_normalized = (fake_resized - mean) / std

                real_features = self.model(src)
                fake_features = self.model(fake)

                all_real_features.append(real_features)
                all_fake_features.append(fake_features)

        all_real_features = torch.cat(all_real_features, dim=0)
        all_fake_features = torch.cat(all_fake_features, dim=0)

        real_mean = all_real_features.mean(dim=0)
        real_cov = torch.matmul((all_real_features - real_mean).T, (all_real_features - real_mean)) / (len(all_real_features) - 1)

        fake_mean = all_fake_features.mean(dim=0)
        fake_cov = torch.matmul((all_fake_features - fake_mean).T, (all_fake_features - fake_mean)) / (len(all_fake_features) - 1)

        fid_score = self.calculate_frechet_distance(real_mean, real_cov, fake_mean, fake_cov)
        return fid_score

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
                The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
                and X_2 ~ N(mu_2, C_2) is
                        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

                Stable version by Dougal J. Sutherland.

                Params:
                -- mu1   : Numpy array containing the activations of a layer of the
                           inception net (like returned by the function 'get_predictions')
                           for generated samples.
                -- mu2   : The sample mean over activations, precalculated on an
                           representative data set.
                -- sigma1: The covariance matrix over activations for generated samples.
                -- sigma2: The covariance matrix over activations, precalculated on an
                           representative data set.

                Returns:
                --   : The Frechet Distance.
                """

        # convert pytorch tensors to numpy arrays
        mu1 = mu1.cpu().numpy()
        mu2 = mu2.cpu().numpy()
        sigma1 = sigma1.cpu().numpy()
        sigma2 = sigma2.cpu().numpy()

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)
