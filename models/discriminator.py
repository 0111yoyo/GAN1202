import torch
import torch.nn as nn

class WaveGANDiscriminator(nn.Module):
    def __init__(self, input_size=32000):
        """
        WaveGAN Discriminator.
        Args:
            input_size (int): Input size for audio waveform (e.g., 32000 samples).
        """
        super(WaveGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input to [batch_size, input_size]
        return self.model(x)