import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import torchvision.transforms as transforms
import os
import numpy as np
import argparse
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR  # Added for learning rate scheduling

from singan_generator import MultiScaleSinGANGenerator
from discriminator import Discriminator
from gradient_penalty import compute_gradient_penalty
from style_transfer import StyleTransfer

class MRIDataset(Dataset):
    """
    Dataset for loading MRI images and segmentation masks.
    Replace this with your actual dataset loading logic.
    """
    def __init__(self, image_dir, mask_dir=None, transform=None):
        """
        Initialize the dataset
        
        Args:
            image_dir: Directory containing MRI images
            mask_dir: Directory containing segmentation masks (optional)
            transform: Image transformations to apply
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get image file list - replace with your actual logic
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Check if masks are available
        self.has_masks = mask_dir is not None
        if self.has_masks:
            self.mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            assert len(self.image_files) == len(self.mask_files), "Number of images and masks must match"
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = torch.from_numpy(np.load(image_path)).float()  # Replace with your actual loading logic
        
        # Normalize to [-1, 1] range
        if self.transform:
            image = self.transform(image)
        
        # Load mask if available
        if self.has_masks:
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
            mask = torch.from_numpy(np.load(mask_path)).float()  # Replace with your actual loading logic
            
            if self.transform:
                mask = self.transform(mask)
                
            return image, mask
        
        return image

# Added function for feature matching loss
def feature_matching_loss(real_features, fake_features):
    """
    Calculates feature matching loss between real and fake feature maps
    
    Args:
        real_features: Features from real images
        fake_features: Features from fake images
        
    Returns:
        Loss value
    """
    loss = 0
    for r, f in zip(real_features, fake_features):
        if r is not None and f is not None:
            loss += torch.mean(torch.abs(r - f))
    return loss

def train_singan(args):
    """
    Train the SinGAN-Seg model
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create directories for saving results
    os.makedirs(args.output_dir, exist_ok=True)
    sample_dir = os.path.join(args.output_dir, "samples")
    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Example size, adjust as needed
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    # Create dataset - replace with your actual dataset
    if args.mask_dir:
        # Dataset with masks
        dataset = MRIDataset(args.image_dir, args.mask_dir, transform=transform)
    else:
        # Dataset without masks
        dataset = MRIDataset(args.image_dir, transform=transform)
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # Initialize models
    if args.mask_dir:
        # Model that generates both image and mask
        generator = MultiScaleSinGANGenerator(
            num_scales=args.num_scales, 
            img_channels=1,  # Assuming grayscale MRI
            mask_channels=1,  # Binary mask
            base_channels=args.base_channels,
            z_dim=args.z_dim
        ).to(device)
    else:
        # Model that only generates images
        generator = MultiScaleSinGANGenerator(
            num_scales=args.num_scales, 
            img_channels=1,
            mask_channels=0,  # No mask generation
            base_channels=args.base_channels,
            z_dim=args.z_dim
        ).to(device)
    
    # Multiple discriminators, one for each scale
    discriminators = nn.ModuleList([
        Discriminator(img_channels=1 + (1 if args.mask_dir else 0), base_channels=args.base_channels).to(device)
        for _ in range(args.num_scales)
    ])
    
    # Optimizers
    # Changed learning rates and beta parameters for stability
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D = optim.Adam(sum([list(disc.parameters()) for disc in discriminators], []), 
                            lr=args.lr * 2, betas=(args.beta1, args.beta2))  # Higher LR for discriminator
    
    # Add learning rate schedulers
    scheduler_G = StepLR(optimizer_G, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    scheduler_D = StepLR(optimizer_D, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    
    # Style transfer module for refinement
    style_transfer = StyleTransfer(
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        device=device
    )
    
    # Training loop
    for epoch in range(args.epochs):
        generator.train()
        for disc in discriminators:
            disc.train()
            
        total_loss_G = 0
        total_loss_D = 0
        
        # Calculate instance noise level (decreasing over time)
        instance_noise = args.initial_noise_level * max(0, 1 - epoch / args.noise_decay_epochs)
        
        # Progressive training: start with fewer scales and add more as training progresses
        active_scales = min(args.num_scales, 
                          1 + int(epoch / args.epochs * args.num_scales * args.progressive_factor))
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader)):
            # Get real images and masks
            if args.mask_dir:
                real_imgs, real_masks = batch_data
                real_imgs = real_imgs.to(device)
                real_masks = real_masks.to(device)
                
                # Add instance noise to real images (for stability)
                if instance_noise > 0:
                    real_imgs = real_imgs + instance_noise * torch.randn_like(real_imgs)
                    real_masks = real_masks + instance_noise * torch.randn_like(real_masks)
                    
                    # Clamp to valid range after adding noise
                    real_imgs = torch.clamp(real_imgs, -1, 1)
                    real_masks = torch.clamp(real_masks, -1, 1)
                
                # Combine image and mask for discriminator
                real_combined = torch.cat([real_imgs, real_masks], dim=1)
            else:
                real_imgs = batch_data.to(device)
                
                # Add instance noise to real images (for stability)
                if instance_noise > 0:
                    real_imgs = real_imgs + instance_noise * torch.randn_like(real_imgs)
                    real_imgs = torch.clamp(real_imgs, -1, 1)
                    
                real_combined = real_imgs
                
            batch_size = real_imgs.size(0)
            
            # ===============================
            # Train Discriminator
            # ===============================
            for _ in range(args.critic_iterations):
                optimizer_D.zero_grad()
                
                # Generate fake images and masks
                z = torch.randn(batch_size, args.z_dim).to(device)
                if args.mask_dir:
                    fake_imgs, fake_masks = generator(z)
                    
                    # Add instance noise to fake images (same level as real images)
                    if instance_noise > 0:
                        fake_imgs = fake_imgs + instance_noise * torch.randn_like(fake_imgs)
                        fake_masks = fake_masks + instance_noise * torch.randn_like(fake_masks)
                        
                        # Clamp to valid range after adding noise
                        fake_imgs = torch.clamp(fake_imgs, -1, 1)
                        fake_masks = torch.clamp(fake_masks, -1, 1)
                        
                    fake_combined = torch.cat([fake_imgs, fake_masks], dim=1)
                else:
                    fake_imgs, _ = generator(z)  # If no mask, _ is empty
                    
                    # Add instance noise to fake images (same level as real images)
                    if instance_noise > 0:
                        fake_imgs = fake_imgs + instance_noise * torch.randn_like(fake_imgs)
                        fake_imgs = torch.clamp(fake_imgs, -1, 1)
                        
                    fake_combined = fake_imgs
                
                # Store intermediate features for feature matching loss
                disc_real_features = []
                disc_fake_features = []
                
                # Calculate discriminator losses at each scale
                loss_D = 0
                for i, disc in enumerate(discriminators):
                    # Only train active scales (progressive training)
                    if i >= active_scales:
                        continue
                        
                    # Scale down real and fake images to match this scale
                    scale_factor = 2 ** (args.num_scales - i - 1)
                    if scale_factor > 1:
                        real_scaled = nn.functional.interpolate(
                            real_combined, scale_factor=1/scale_factor, mode='bilinear', align_corners=True
                        )
                        fake_scaled = nn.functional.interpolate(
                            fake_combined.detach(), scale_factor=1/scale_factor, mode='bilinear', align_corners=True
                        )
                    else:
                        real_scaled = real_combined
                        fake_scaled = fake_combined.detach()
                    
                    # Calculate loss for this scale
                    loss_D_real = -torch.mean(disc(real_scaled))
                    loss_D_fake = torch.mean(disc(fake_scaled))
                    gp = compute_gradient_penalty(disc, real_scaled, fake_scaled, device)
                    
                    # Increased lambda_gp for better stability
                    loss_D_scale = loss_D_real + loss_D_fake + args.lambda_gp * gp
                    
                    # Add to total discriminator loss
                    loss_D += loss_D_scale
                
                # Backward and optimize
                loss_D.backward()
                
                # Apply gradient clipping to discriminator
                torch.nn.utils.clip_grad_norm_(
                    sum([list(disc.parameters()) for disc in discriminators], []), 
                    max_norm=args.clip_value
                )
                
                optimizer_D.step()
                
                total_loss_D += loss_D.item()
            
            # ===============================
            # Train Generator
            # ===============================
            optimizer_G.zero_grad()
            
            # Generate fake images and masks again
            z = torch.randn(batch_size, args.z_dim).to(device)
            if args.mask_dir:
                fake_imgs, fake_masks = generator(z)
                fake_combined = torch.cat([fake_imgs, fake_masks], dim=1)
            else:
                fake_imgs, _ = generator(z)
                fake_combined = fake_imgs
            
            # Calculate generator losses at each scale
            loss_G = 0
            for i, disc in enumerate(discriminators):
                # Only train active scales (progressive training)
                if i >= active_scales:
                    continue
                    
                # Scale down fake images to match this scale
                scale_factor = 2 ** (args.num_scales - i - 1)
                if scale_factor > 1:
                    fake_scaled = nn.functional.interpolate(
                        fake_combined, scale_factor=1/scale_factor, mode='bilinear', align_corners=True
                    )
                    
                    # For feature matching, also scale real images
                    real_scaled = nn.functional.interpolate(
                        real_combined, scale_factor=1/scale_factor, mode='bilinear', align_corners=True
                    )
                else:
                    fake_scaled = fake_combined
                    real_scaled = real_combined
                
                # Calculate adversarial loss for this scale
                loss_G_scale = -torch.mean(disc(fake_scaled))
                
                # Add feature matching loss if enabled
                if args.use_feature_matching:
                    # This is a simplified example - in a real implementation,
                    # you would extract intermediate features from the discriminator
                    # For simplicity, we'll use the final features
                    real_features = [disc(real_scaled)]
                    fake_features = [disc(fake_scaled)]
                    fm_loss = feature_matching_loss(real_features, fake_features)
                    loss_G_scale += args.feature_matching_weight * fm_loss
                
                # Add to total generator loss
                loss_G += loss_G_scale
            
            # Backward and optimize
            loss_G.backward()
            
            # Apply gradient clipping to generator
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=args.clip_value)
            
            optimizer_G.step()
            
            total_loss_G += loss_G.item()
            
            # Print progress
            if batch_idx % args.log_interval == 0:
                print(f"Epoch {epoch}/{args.epochs}, Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}, "
                      f"Active Scales: {active_scales}, Noise Level: {instance_noise:.4f}")
        
        # Step the learning rate schedulers
        scheduler_G.step()
        scheduler_D.step()
        
        # Save sample images
        if (epoch + 1) % args.sample_interval == 0:
            generator.eval()
            with torch.no_grad():
                # Generate samples
                z = torch.randn(16, args.z_dim).to(device)
                if args.mask_dir:
                    fake_imgs, fake_masks = generator(z)
                    # Save images and masks
                    save_image(fake_imgs, f"{sample_dir}/images_epoch_{epoch+1}.png", normalize=True, nrow=4)
                    save_image(fake_masks, f"{sample_dir}/masks_epoch_{epoch+1}.png", normalize=True, nrow=4)
                    
                    # Apply style transfer for refinement if enabled
                    if args.use_style_transfer:
                        # Get style reference images from the dataset
                        style_imgs = next(iter(dataloader))[0][:16].to(device)
                        # Apply style transfer
                        refined_imgs = style_transfer.batch_style_transfer(
                            fake_imgs, style_imgs, num_steps=args.style_steps, lr=args.style_lr
                        )
                        # Save refined images
                        save_image(refined_imgs, f"{sample_dir}/refined_images_epoch_{epoch+1}.png", normalize=True, nrow=4)
                else:
                    fake_imgs, _ = generator(z)
                    # Save images
                    save_image(fake_imgs, f"{sample_dir}/images_epoch_{epoch+1}.png", normalize=True, nrow=4)
                    
                    # Apply style transfer for refinement if enabled
                    if args.use_style_transfer:
                        # Get style reference images from the dataset
                        style_imgs = next(iter(dataloader))[:16].to(device)
                        # Apply style transfer
                        refined_imgs = style_transfer.batch_style_transfer(
                            fake_imgs, style_imgs, num_steps=args.style_steps, lr=args.style_lr
                        )
                        # Save refined images
                        save_image(refined_imgs, f"{sample_dir}/refined_images_epoch_{epoch+1}.png", normalize=True, nrow=4)
        
        # Save model
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': [disc.state_dict() for disc in discriminators],
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'epoch': epoch,
                'active_scales': active_scales,
            }, f"{model_dir}/model_epoch_{epoch+1}.pt")
            
        # Print epoch summary
        avg_loss_G = total_loss_G / len(dataloader)
        avg_loss_D = total_loss_D / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} - Avg Loss G: {avg_loss_G:.4f}, Avg Loss D: {avg_loss_D:.4f}, "
              f"LR_G: {scheduler_G.get_last_lr()[0]:.6f}, LR_D: {scheduler_D.get_last_lr()[0]:.6f}")
    
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SinGAN-Seg model for MRI synthesis")
    
    # Dataset parameters
    parser.add_argument('--image_dir', type=str, required=True, help='Directory with MRI images')
    parser.add_argument('--mask_dir', type=str, default=None, help='Directory with segmentation masks')
    
    # Model parameters
    parser.add_argument('--num_scales', type=int, default=3, help='Number of scales in the pyramid (reduced for faster results)')
    parser.add_argument('--base_channels', type=int, default=64, help='Number of base channels in the model')
    parser.add_argument('--z_dim', type=int, default=100, help='Dimension of random noise vector')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate (reduced for stability)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer (increased for stability)')
    parser.add_argument('--lambda_gp', type=float, default=10, help='Gradient penalty lambda')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--critic_iterations', type=int, default=5, help='Critic iterations per generator update')
    parser.add_argument('--log_interval', type=int, default=5, help='Interval between logging')
    parser.add_argument('--sample_interval', type=int, default=1, help='Interval between sampling images')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval between saving models')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    # New stability parameters
    parser.add_argument('--clip_value', type=float, default=1.0, help='Value for gradient clipping')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='Step size for learning rate decay')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5, help='Gamma for learning rate decay')
    parser.add_argument('--initial_noise_level', type=float, default=0.05, help='Initial level of instance noise')
    parser.add_argument('--noise_decay_epochs', type=int, default=20, help='Epochs to decay noise to zero')
    parser.add_argument('--progressive_factor', type=float, default=1.5, help='Factor for progressive training (>1 speeds up, <1 slows down)')
    parser.add_argument('--use_feature_matching', action='store_true', help='Use feature matching loss')
    parser.add_argument('--feature_matching_weight', type=float, default=10.0, help='Weight for feature matching loss')
    
    # Style transfer parameters
    parser.add_argument('--use_style_transfer', action='store_true', help='Use style transfer for refinement')
    parser.add_argument('--content_weight', type=float, default=1.0, help='Content loss weight for style transfer')
    parser.add_argument('--style_weight', type=float, default=1000.0, help='Style loss weight for style transfer')
    parser.add_argument('--style_steps', type=int, default=50, help='Style transfer optimization steps')
    parser.add_argument('--style_lr', type=float, default=0.01, help='Style transfer learning rate')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Train the model
    train_singan(args) 