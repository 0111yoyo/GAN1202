import torch
import torch.nn as nn

class WaveGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, output_size=32000):
        """
        WaveGAN Generator.
        Args:
            latent_dim (int): Dimension of the latent space.
            output_size (int): Output size of the generated audio (e.g., 32000 samples for 2 seconds at 16 kHz).
        """
        super(WaveGANGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.fc(z)