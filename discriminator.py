from torch import nn
from attention_modules import ChannelAttention, SpatialAttention # Import attention modules

def weights_init(m): # Weight initialization function (same as in generator.py, can be in a utils.py file if shared)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02) # Example: Normal initialization for Conv layers
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    """Discriminator model for Alpha-GAN-GP."""
    def __init__(self, img_channels=1, base_channels=64):
        super(Discriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(img_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            ChannelAttention(base_channels), # Channel Attention

            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2),
            SpatialAttention(), # Spatial Attention

            nn.Conv2d(base_channels * 2, 1, 4, stride=2, padding=1),
        )

    def forward(self, img):
        return self.conv_layers(img).view(img.size(0), -1)

    def apply(self, fn): # Apply weight initialization
        return super().apply(weights_init)

# Example of applying initialization after creating the discriminator in train.py (see train.py edits below) 