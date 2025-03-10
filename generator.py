from torch import nn
from attention_modules import ChannelAttention, SpatialAttention  # Import attention modules

def weights_init(m): # New weight initialization function
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02) # Example: Normal initialization for Conv layers
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """Generator model for Alpha-GAN-GP."""
    def __init__(self, z_dim=100, img_channels=1, base_channels=64):
        super(Generator, self).__init__()

        self.init_size = 8
        self.fc = nn.Linear(z_dim, base_channels * self.init_size * self.init_size)

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(),
            ChannelAttention(base_channels // 2),  # Channel Attention

            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 4),
            nn.ReLU(),
            SpatialAttention(),  # Spatial Attention

            nn.ConvTranspose2d(base_channels // 4, img_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.fc(z).view(z.size(0), -1, self.init_size, self.init_size)
        img = self.conv_layers(z)
        return img

    def apply(self, fn): # Apply weight initialization
        return super().apply(weights_init)

# Example of applying initialization after creating the generator in train.py (see train.py edits below) 