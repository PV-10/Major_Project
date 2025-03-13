#!/usr/bin/env python3
"""
Run script to start training with the SinGAN-based model for MRI synthesis.
This script provides sensible defaults but can be modified as needed.
"""

import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run SinGAN-Seg training for MRI synthesis")
    
    # Basic parameters
    parser.add_argument('--image_dir', type=str, required=True, help='Directory with MRI images')
    parser.add_argument('--mask_dir', type=str, default=None, help='Directory with segmentation masks (optional)')
    parser.add_argument('--use_style_transfer', action='store_true', help='Enable style transfer refinement')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Construct the command for the training script
    cmd = [
        'python', 'train_singan.py',
        '--image_dir', args.image_dir,
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--output_dir', args.output_dir,
        '--num_scales', '5',
        '--z_dim', '100',
        '--base_channels', '64',
        '--lr', '0.0001',
        '--critic_iterations', '5',
        '--sample_interval', '10',
        '--save_interval', '50',
        '--log_interval', '5',
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
            '--style_steps', '100',
            '--style_lr', '0.01'
        ])
    
    # Print the command
    print("Running command:")
    print(" ".join(cmd))
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    # Run the command
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 