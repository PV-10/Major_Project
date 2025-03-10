import torch
import torch.optim as optim
from generator import Generator  # Import Generator
from discriminator import Discriminator # Import Discriminator
from gradient_penalty import compute_gradient_penalty # Import gradient penalty

# --- Hyperparameters --- # Group hyperparameters at the top
device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim = 100
img_channels = 1 # Assuming grayscale MRI images
base_channels = 64
batch_size = 64 # Example batch size, adjust as needed
learning_rate = 0.0001
beta1 = 0.5
beta2 = 0.9
epochs = 10000
lambda_gp = 10
critic_iterations = 5 # Discriminator updates per Generator update, make it a hyperparameter
log_interval = 100 # Log losses every 100 iterations (example)

# --- Initialize Models and Optimizers ---
generator = Generator(z_dim, img_channels, base_channels).to(device)
discriminator = Discriminator(img_channels, base_channels).to(device)

# Apply weight initialization
generator.apply(generator.apply) # Apply weight initialization to generator
discriminator.apply(discriminator.apply) # Apply weight initialization to discriminator

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

# --- Dummy Data Loader (Replace with your actual data loader) ---
def real_data_loader():
    # IMPORTANT: Replace this with your actual data loading logic for MRI images
    # This is a dummy loader for demonstration purposes
    return torch.randn(batch_size, img_channels, 64, 64) # Example: batch of 64x64 grayscale images

# --- Training Loop ---
for epoch in range(epochs):
    for i in range(critic_iterations): # Use hyperparameter for critic iterations
        real_imgs = real_data_loader().to(device) # Load real images
        z = torch.randn(real_imgs.size(0), z_dim).to(device) # Generate noise vector
        fake_imgs = generator(z).detach() # Generate fake images

        # --- Discriminator Loss ---
        loss_D_real = -torch.mean(discriminator(real_imgs)) # Loss on real images
        loss_D_fake = torch.mean(discriminator(fake_imgs)) # Loss on fake images
        gp = compute_gradient_penalty(discriminator, real_imgs, fake_imgs, device)
        loss_D = loss_D_real + loss_D_fake + lambda_gp * gp # Total discriminator loss

        # --- Discriminator Optimization ---
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

    # --- Generator Loss ---
    z = torch.randn(batch_size, z_dim).to(device)
    fake_imgs_G = generator(z) # Generate fake images for Generator update
    loss_G = -torch.mean(discriminator(fake_imgs_G)) # Generator loss (negative discriminator output for fake images)

    # --- Generator Optimization ---
    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    # --- Print Epoch Summary and Logging ---
    if epoch % log_interval == 0: # Log periodically
        print(f'Epoch {epoch}/{epochs}: Loss D={loss_D.item():.4f}, Loss G={loss_G.item():.4f}')
        # You can add more logging here, like saving sample images, etc.

print("Training finished!") # Indicate training completion 