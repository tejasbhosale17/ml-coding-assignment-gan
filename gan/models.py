import torch
import torch.nn as nn


class Generator(nn.Module):
    """Maps a latent vector to a 1x28x28 image in [-1, 1] (fully connected)."""

    def __init__(self, latent_dim=100, batchnorm=True):
        super().__init__()
        self.latent_dim = latent_dim

        def block(n_in, n_out):
            layers = [nn.Linear(n_in, n_out)]
            if batchnorm:
                layers.append(nn.BatchNorm1d(n_out))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(latent_dim, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    """Maps a 1x28x28 image to the probability that it is real (fully connected)."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, image):
        return self.net(image)


if __name__ == "__main__":
    g, d = Generator(), Discriminator()
    out = g(torch.randn(4, 100))
    print("G out:", tuple(out.shape), float(out.min()), float(out.max()))
    print("D out:", tuple(d(out).shape))
