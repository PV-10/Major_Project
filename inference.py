#!/usr/bin/env python3
"""
Inference script for generating MRI images using a trained SinGAN-Seg model.
"""

import torch
import argparse
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm

from singan_generator import MultiScaleSinGANGenerator
from style_transfer import StyleTransfer

def load_model(model_path, num_scales, img_channels, mask_channels, base_channels, z_dim, device):
    """
    Load a trained SinGAN-Seg model
    
    Args:
        model_path: Path to the saved model file
        num_scales: Number of scales in the pyramid
        img_channels: Number of image channels
        mask_channels: Number of mask channels
        base_channels: Number of base channels
        z_dim: Dimension of random noise vector
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Initialize model
    model = MultiScaleSinGANGenerator(
        num_scales=num_scales,
        img_channels=img_channels,
        mask_channels=mask_channels,
        base_channels=base_channels,
        z_dim=z_dim
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.eval()
    
    return model

def generate_samples(model, num_samples, z_dim, device, output_dir, use_style_transfer=False, style_img_path=None):
    """
    Generate samples using the trained model
    
    Args:
        model: Trained SinGAN-Seg model
        num_samples: Number of samples to generate
        z_dim: Dimension of random noise vector
        device: Device to use for generation
        output_dir: Directory to save generated samples
        use_style_transfer: Whether to use style transfer for refinement
        style_img_path: Path to the style reference image (if style transfer is used)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize style transfer if needed
    if use_style_transfer:
        from PIL import Image
        import torchvision.transforms as transforms
        
        style_transfer = StyleTransfer(device=device)
        
        # Load style image
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        style_img = Image.open(style_img_path).convert('L')  # Convert to grayscale
        style_img = transform(style_img).unsqueeze(0).to(device)
    
    # Generate samples
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Generating samples"):
            # Generate random noise
            z = torch.randn(1, z_dim).to(device)
            
            # Generate image and mask
            imgs, masks = model(z)
            
            # Apply style transfer if enabled
            if use_style_transfer:
                imgs = style_transfer.transfer_style(imgs, style_img, num_steps=100, lr=0.01)
            
            # Save image
            save_image(imgs, os.path.join(output_dir, f"sample_image_{i}.png"), normalize=True)
            
            # Save mask if present
            if masks.size(1) > 0:
                save_image(masks, os.path.join(output_dir, f"sample_mask_{i}.png"), normalize=True)
                
                # Save combined visualization
                plt.figure(figsize=(10, 5))
                
                # Image
                plt.subplot(1, 2, 1)
                img = imgs[0].cpu().permute(1, 2, 0).numpy()
                img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
                plt.imshow(img.squeeze(), cmap='gray')
                plt.title("Generated MRI")
                plt.axis('off')
                
                # Mask
                plt.subplot(1, 2, 2)
                mask = masks[0].cpu().permute(1, 2, 0).numpy()
                mask = (mask + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
                plt.imshow(mask.squeeze(), cmap='jet')
                plt.title("Generated Segmentation")
                plt.axis('off')
                
                plt.savefig(os.path.join(output_dir, f"combined_{i}.png"), bbox_inches='tight')
                plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate MRI images using a trained SinGAN-Seg model")
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--num_scales', type=int, default=5, help='Number of scales in the model')
    parser.add_argument('--img_channels', type=int, default=1, help='Number of image channels')
    parser.add_argument('--mask_channels', type=int, default=1, help='Number of mask channels (0 for no mask)')
    parser.add_argument('--base_channels', type=int, default=64, help='Number of base channels')
    parser.add_argument('--z_dim', type=int, default=100, help='Dimension of random noise vector')
    
    # Generation parameters
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='generated_samples', help='Output directory')
    
    # Style transfer parameters
    parser.add_argument('--use_style_transfer', action='store_true', help='Use style transfer for refinement')
    parser.add_argument('--style_img_path', type=str, help='Path to the style reference image')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(
        args.model_path,
        args.num_scales,
        args.img_channels,
        args.mask_channels,
        args.base_channels,
        args.z_dim,
        device
    )
    
    # Check if style transfer should be used but no style image is provided
    if args.use_style_transfer and args.style_img_path is None:
        parser.error("--style_img_path is required when --use_style_transfer is set")
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    generate_samples(
        model,
        args.num_samples,
        args.z_dim,
        device,
        args.output_dir,
        args.use_style_transfer,
        args.style_img_path
    )
    
    print(f"Generation complete. Samples saved to {args.output_dir}")

if __name__ == "__main__":
    main() 