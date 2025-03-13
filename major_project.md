# Alpha-GAN-GP with SinGAN-Seg Integration

## Project Overview

This project implements an advanced GAN architecture for generating synthetic brain MRI images for Parkinson's disease research. The model, named Alpha-GAN-GP, integrates aspects from:

1. **WGAN-GP (Wasserstein GAN with Gradient Penalty)** - Provides stable training through gradient penalty
2. **Attention Mechanisms** - Spatial and channel attention to focus on disease-specific regions
3. **SinGAN-Seg** - Multi-scale pyramid structure and joint image-mask generation

## Architecture Components

### 1. Core GAN Architecture
- **Generator**: Produces synthetic MRI images from random noise
- **Discriminator**: Distinguishes between real and synthetic MRIs
- **Gradient Penalty**: Ensures stable training by penalizing large gradients

### 2. Attention Mechanisms
- **Spatial Attention**: Focuses on important spatial regions within images
- **Channel Attention**: Emphasizes relevant feature channels

### 3. SinGAN-Seg Integration
- **Multi-Scale Generation**: Pyramid structure generating from coarse to fine details
- **Joint Image-Mask Generation**: Capable of generating both MRI images and segmentation masks
- **Style Transfer**: Refinement step to enhance realism of synthetic images

## File Structure

### Original Files (Alpha-GAN-GP)
- `attention_modules.py`: Implementations of Spatial and Channel Attention modules
- `generator.py`: Original generator model with attention mechanisms
- `discriminator.py`: Discriminator model with attention mechanisms
- `gradient_penalty.py`: Implementation of WGAN's gradient penalty
- `model.py`: Training, evaluation, and inference code for the original model

### New Files (SinGAN-Seg Integration)
- `singan_generator.py`: Multi-scale pyramid generator based on SinGAN architecture
- `style_transfer.py`: Style transfer module for image refinement
- `train_singan.py`: Training script for the SinGAN-based model
- `run_training.py`: Simple script to start training with sensible defaults
- `inference.py`: Script for generating samples using a trained model
- `major_project.md`: This documentation file

## SinGAN-Seg Integration Details

### Multi-Scale Pyramid Structure (`singan_generator.py`)
The SinGAN architecture uses a coarse-to-fine pyramid of generators, where each level adds finer details to the output of the previous level. This approach allows the model to capture both global structure and fine details.

Key components:
- `SingleScaleGenerator`: Generator for a single scale in the pyramid
- `MultiScaleSinGANGenerator`: Multi-scale generator that manages the pyramid structure
- Each scale incorporates spatial and channel attention to focus on important regions

### Joint Image-Mask Generation
The model can generate pairs of MRI images and corresponding segmentation masks simultaneously, ensuring alignment between generated images and their segmentation.

Implementation details:
- Output channels are split between image channels and mask channels
- Both image and mask are generated at each scale, maintaining consistency
- Discriminator evaluates the combined image-mask pair

### Style Transfer Refinement (`style_transfer.py`)
After generation, a style transfer step is applied to make synthetic images more realistic by transferring texture patterns from real MRI scans while preserving the structural content of the generated images.

Key components:
- `VGG16FeatureExtractor`: Extracts features from specific VGG16 layers
- `StyleTransfer`: Performs style transfer from a reference image to a generated image
- Optimization process balances content preservation with style application

## Training Process (`train_singan.py`)

The training workflow involves:

1. **Data loading**: Load MRI images and segmentation masks (if available)
2. **Multiple discriminators**: One discriminator for each scale in the pyramid
3. **Multi-scale adversarial training**: Train each scale in the pyramid separately
4. **Style transfer refinement**: Optionally apply style transfer after generation

Training parameters:
- Number of scales in the pyramid
- Learning rate, batch size, optimizer settings
- Gradient penalty coefficient
- Style transfer weights and settings

## Inference and Generation (`inference.py`)

The inference process for generating synthetic MRI images:

1. **Load trained model**: Load a saved checkpoint of the multi-scale generator
2. **Generate random noise**: Sample random vectors as input to the generator
3. **Multi-scale generation**: Pass through all scales in the pyramid
4. **Optional refinement**: Apply style transfer to enhance realism
5. **Save output**: Save generated images, masks, and visualizations

## Running the Model

### Training
```bash
# Training with basic settings:
python run_training.py --image_dir path/to/mri/images

# Training with segmentation masks:
python run_training.py --image_dir path/to/mri/images --mask_dir path/to/masks

# Training with style transfer refinement:
python run_training.py --image_dir path/to/mri/images --use_style_transfer
```

### Inference
```bash
# Generate samples from a trained model:
python inference.py --model_path path/to/model/checkpoint.pt --num_samples 10

# Generate samples with style transfer:
python inference.py --model_path path/to/model/checkpoint.pt --use_style_transfer --style_img_path path/to/style/image.png
```

## Research Novelty

The integration of attention mechanisms with SinGAN's multi-scale approach creates a novel architecture specifically designed for medical imaging that:

1. Focuses on disease-relevant regions through attention
2. Captures hierarchical structure through multi-scale generation
3. Maintains anatomical consistency through joint image-mask generation
4. Enhances realism through style transfer

This hybrid approach addresses the specific challenges of Parkinson's disease MRI synthesis better than either approach alone.

## Requirements and Dependencies

- PyTorch >= 1.7.0
- torchvision
- PIL (Pillow)
- matplotlib
- numpy
- tqdm

## Future Work

Potential extensions of this research:

1. **Conditional Generation**: Generate MRIs based on specific disease parameters
2. **3D Extension**: Extend the model to generate 3D MRI volumes
3. **Transfer Learning**: Use pre-trained models for faster adaptation to new datasets
4. **Evaluation Framework**: Develop comprehensive metrics for evaluating synthetic MRI quality and clinical utility 