import torch
import torch.nn as nn
from attention_modules import SpatialAttention, ChannelAttention

class SingleScaleGenerator(nn.Module):
    """
    Generator for a single scale in the SinGAN pyramid.
    Each scale adds details to the upsampled output from the previous scale.
    """
    def __init__(self, in_channels, out_channels, base_channels=64, with_attention=True):
        super(SingleScaleGenerator, self).__init__()
        
        # Main convolutional blocks
        self.main = nn.Sequential(
            # First conv layer
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Middle conv layers with attention
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Optional attention mechanism
        self.with_attention = with_attention
        if with_attention:
            self.channel_attention = ChannelAttention(base_channels)
            self.spatial_attention = SpatialAttention()
        
        # Output convolutional layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x, prev_output=None):
        """
        Forward pass with optional previous scale output
        
        Args:
            x: Input noise or features
            prev_output: Output from previous scale generator (upsampled)
        
        Returns:
            Generated output for this scale
        """
        # Process input
        out = self.main(x)
        
        # Apply attention if enabled
        if self.with_attention:
            out = self.channel_attention(out)
            out = self.spatial_attention(out)
        
        # Generate output
        out = self.output_layer(out)
        
        # Add previous output if provided (residual connection)
        if prev_output is not None:
            out = out + prev_output
            
        return out


class MultiScaleSinGANGenerator(nn.Module):
    """
    Multi-scale pyramid generator based on SinGAN-Seg.
    Generates both MRI images and segmentation masks simultaneously.
    """
    def __init__(self, num_scales=5, img_channels=1, mask_channels=1, base_channels=64, z_dim=100):
        super(MultiScaleSinGANGenerator, self).__init__()
        
        self.num_scales = num_scales
        self.img_channels = img_channels
        self.mask_channels = mask_channels
        self.z_dim = z_dim
        
        # Initial size at the coarsest scale
        self.init_size = 8
        
        # Linear projection for the random noise
        self.noise_projection = nn.Linear(z_dim, base_channels * self.init_size * self.init_size)
        
        # Create pyramid of generators for different scales
        self.generators = nn.ModuleList()
        
        # Total output channels (image + mask)
        total_channels = img_channels + mask_channels
        
        # Add generators for each scale
        for i in range(num_scales):
            # First scale takes noise as input, subsequent scales take previous output + noise
            in_channels = base_channels if i == 0 else total_channels + base_channels
            
            # Create generator for this scale
            self.generators.append(
                SingleScaleGenerator(
                    in_channels=in_channels,
                    out_channels=total_channels,
                    base_channels=base_channels * (2 ** (num_scales - i - 1)),
                    with_attention=True
                )
            )
        
        # Upsampling layers between scales
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, z):
        """
        Forward pass through the multi-scale generator
        
        Args:
            z: Random noise vector [batch_size, z_dim]
            
        Returns:
            tuple: (generated images, generated masks)
        """
        batch_size = z.size(0)
        
        # Project and reshape noise for initial generator
        x = self.noise_projection(z).view(batch_size, -1, self.init_size, self.init_size)
        
        # Output from current scale
        current_output = None
        
        # Generate at each scale
        for i in range(self.num_scales):
            # For first scale, use projected noise
            if i == 0:
                current_output = self.generators[i](x)
            else:
                # For subsequent scales:
                # 1. Upsample previous output
                upsampled_output = self.upsample(current_output)
                
                # 2. Generate new noise for this scale
                scale_size = self.init_size * (2 ** i)
                noise = torch.randn(batch_size, self.generators[i].main[0].in_channels - self.img_channels - self.mask_channels, 
                                   scale_size, scale_size, device=z.device)
                
                # 3. Concatenate upsampled output with noise
                generator_input = torch.cat([upsampled_output, noise], dim=1)
                
                # 4. Generate output for this scale
                current_output = self.generators[i](generator_input, upsampled_output)
        
        # Split output into image and mask
        # Assuming output channels are ordered as [image_channels, mask_channels]
        images = current_output[:, :self.img_channels]
        masks = current_output[:, self.img_channels:]
        
        return images, masks

    def generate_from_coarse(self, coarse_input, start_scale=0):
        """
        Generate image starting from a coarse input at a specific scale
        Useful for manipulations or conditional generation
        
        Args:
            coarse_input: Input image/mask at a coarse scale
            start_scale: Scale to start generation from
            
        Returns:
            tuple: (generated images, generated masks)
        """
        if start_scale >= self.num_scales:
            raise ValueError(f"Start scale must be less than number of scales ({self.num_scales})")
        
        current_output = coarse_input
        
        # Generate from specified scale
        for i in range(start_scale, self.num_scales):
            # Upsample previous output
            upsampled_output = self.upsample(current_output)
            
            # Generate new noise for this scale
            batch_size = coarse_input.size(0)
            scale_size = self.init_size * (2 ** i)
            noise = torch.randn(batch_size, self.generators[i].main[0].in_channels - self.img_channels - self.mask_channels, 
                               scale_size, scale_size, device=coarse_input.device)
            
            # Concatenate upsampled output with noise
            generator_input = torch.cat([upsampled_output, noise], dim=1)
            
            # Generate output for this scale
            current_output = self.generators[i](generator_input, upsampled_output)
        
        # Split output into image and mask
        images = current_output[:, :self.img_channels]
        masks = current_output[:, self.img_channels:]
        
        return images, masks 