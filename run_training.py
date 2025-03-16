#!/usr/bin/env python3
"""
Run script to start training with the SinGAN-based model for MRI synthesis.
This script provides optimized defaults for stable training.
"""

import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run SinGAN-Seg training for MRI synthesis with optimized settings")
    
    # Basic parameters
    parser.add_argument('--image_dir', type=str, required=True, help='Directory with MRI images')
    parser.add_argument('--mask_dir', type=str, default=None, help='Directory with segmentation masks (optional)')
    parser.add_argument('--use_style_transfer', action='store_true', help='Enable style transfer refinement')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--fast_mode', action='store_true', help='Use faster but potentially less accurate training')
    
    args = parser.parse_args()
    
    # Construct the command for the training script
    cmd = [
        'python', 'train_singan.py',
        '--image_dir', args.image_dir,
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--output_dir', args.output_dir,
        # Optimized parameters for stable training
        '--num_scales', '3',                # Reduced from 5 for faster results
        '--z_dim', '100',
        '--base_channels', '64',
        '--lr', '0.00005',                  # Reduced for stability
        '--beta1', '0.5',
        '--beta2', '0.999',                 # Increased for better momentum
        '--critic_iterations', '8',         # Increased to train discriminator more
        '--sample_interval', '1',           # Save samples every epoch
        '--save_interval', '10',            # Save model every 10 epochs
        '--log_interval', '5',
        # New stability parameters
        '--clip_value', '1.0',              # Gradient clipping
        '--lr_decay_step', '10',            # Learning rate decay every 10 epochs
        '--lr_decay_gamma', '0.5',          # Halve learning rate when decaying
        '--initial_noise_level', '0.05',    # Instance noise for stability
        '--noise_decay_epochs', '20',       # Decay noise over 20 epochs
        '--progressive_factor', '1.5',      # Progressive scale training
        '--use_feature_matching',           # Use feature matching loss
        '--feature_matching_weight', '10.0'
    ]
    
    # Add mask directory if provided
    if args.mask_dir:
        cmd.extend(['--mask_dir', args.mask_dir])
    
    # Add style transfer arguments if enabled
    if args.use_style_transfer:
        cmd.extend([
            '--use_style_transfer',
            '--content_weight', '1.0',
            '--style_weight', '1000.0',
            '--style_steps', '50',          # Reduced for faster processing
            '--style_lr', '0.01'
        ])
    
    # Fast mode settings
    if args.fast_mode:
        # Override with even faster settings
        for i, arg in enumerate(cmd):
            if arg == '--num_scales':
                cmd[i+1] = '2'              # Even fewer scales
            elif arg == '--critic_iterations':
                cmd[i+1] = '5'              # Fewer critic iterations
            elif arg == '--epochs':
                cmd[i+1] = str(min(int(args.epochs), 30))  # Cap at 30 epochs
    
    # Print the command
    print("Running command:")
    print(" ".join(cmd))
    print("\n" + "="*80)
    print("Starting training with optimized parameters for stable results...")
    print("You should start seeing structure in generated images within 10 epochs")
    print("Full training will complete in around 50 epochs")
    print("="*80 + "\n")
    
    # Run the command
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 