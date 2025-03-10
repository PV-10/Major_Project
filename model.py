import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from generator import Generator  # Import Generator from generator.py
from discriminator import Discriminator  # Import Discriminator from discriminator.py
from gradient_penalty import compute_gradient_penalty  # Import gradient_penalty.py
from attention_modules import ChannelAttention, SpatialAttention # Import both Attention modules from attention_modules.py

# --- Training Function ---
def train_model(generator, discriminator, data_loader, optimizer_G, optimizer_D, epochs, device, critic_iterations=5, log_interval=100, z_dim=100, lambda_gp=10):
    """Trains the Alpha-GAN-GP model."""
    generator.train()
    discriminator.train()
    for epoch in range(epochs):
        for batch_idx, real_imgs in enumerate(data_loader): # Assuming data_loader yields batches of real images
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # --- Discriminator Training ---
            for _ in range(critic_iterations):
                optimizer_D.zero_grad()
                z = torch.randn(batch_size, z_dim).to(device)
                fake_imgs = generator(z).detach()

                loss_D_real = -torch.mean(discriminator(real_imgs))
                loss_D_fake = torch.mean(discriminator(fake_imgs))
                gp = compute_gradient_penalty(discriminator, real_imgs, fake_imgs, device)
                loss_D = loss_D_real + loss_D_fake + lambda_gp * gp
                loss_D.backward()
                optimizer_D.step()

            # --- Generator Training ---
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, z_dim).to(device)
            fake_imgs_G = generator(z)
            loss_G = -torch.mean(discriminator(fake_imgs_G))
            loss_G.backward()
            optimizer_G.step()

            # --- Logging ---
            if batch_idx % log_interval == 0:
                print(f'Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(data_loader)}]: Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}')

        # --- Save Sample Images after each epoch ---
        evaluate_model(generator, epoch, device, z_dim) # Evaluate and save images after each epoch
    print("Training finished!")

# --- Evaluation Function (Saving Sample Images) ---
def evaluate_model(generator, epoch, device, z_dim, num_samples=16, output_dir='sample_images'):
    """Generates and saves sample images from the generator."""
    generator.eval() # Set generator to evaluation mode
    with torch.no_grad(): # Disable gradient calculation during evaluation
        sample_z = torch.randn(num_samples, z_dim).to(device)
        sample_imgs = generator(sample_z)
        save_image(sample_imgs, f'{output_dir}/sample_epoch_{epoch}.png', normalize=True) # Save images
    generator.train() # Set back to training mode

# --- Inference Function (Image Generation) ---
def generate_images(generator, num_images, z_dim, device, output_dir='generated_images'):
    """Generates and saves a specified number of images from the trained generator."""
    generator.eval() # Set generator to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        for i in range(num_images // 16): # Generate in batches of 16 for example, adjust as needed
            z = torch.randn(16, z_dim).to(device)
            fake_imgs = generator(z)
            for j in range(16):
                save_image(fake_imgs[j], f'{output_dir}/generated_image_{i*16 + j}.png', normalize=True)
    generator.train() # Set back to training mode
    print(f"Generated {num_images} images and saved to '{output_dir}' directory.")


# --- Main Block (if __name__ == "__main__") ---
if __name__ == "__main__":
    # --- Hyperparameters ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_dim = 100
    img_channels = 1  # Assuming grayscale MRI
    base_channels = 64
    batch_size = 64
    learning_rate = 0.0001
    beta1 = 0.5
    beta2 = 0.9
    epochs = 1000
    lambda_gp = 10
    critic_iterations = 5
    log_interval = 50
    output_sample_dir = 'sample_images' # Directory to save sample images during training
    output_generated_dir = 'generated_images' # Directory to save generated images after training

    # --- Create Directories ---
    import os
    os.makedirs(output_sample_dir, exist_ok=True) # Create sample images directory
    os.makedirs(output_generated_dir, exist_ok=True) # Create generated images directory

    # --- Initialize Generator and Discriminator ---
    generator = Generator(z_dim, img_channels, base_channels).to(device)
    discriminator = Discriminator(img_channels, base_channels).to(device)

    # --- Initialize Optimizers ---
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

    # --- Data Loader (Replace with your actual data loader) ---
    def real_data_loader(): # Dummy data loader - REPLACE THIS!
        # IMPORTANT: Replace with your actual MRI data loading logic
        return torch.randn(batch_size, img_channels, 64, 64)

    data_loader = [real_data_loader() for _ in range(100)] # Example dummy data loader list - REPLACE with your actual DataLoader

    # --- Train the Model ---
    train_model(generator, discriminator, data_loader, optimizer_G, optimizer_D, epochs, device, critic_iterations, log_interval, z_dim, lambda_gp)

    # --- Generate Images after Training ---
    num_generated_images = 256 # Example number of images to generate
    generate_images(generator, num_generated_images, z_dim, device, output_generated_dir)

    print("Model training, evaluation, and image generation completed.") 