# 🚀 **Major Project Instructions: Alpha-GAN-GP for Brain MRI Synthesis**
### 📌 **Overview**
This guide details the implementation of **Alpha-GAN-GP**, which enhances **Wasserstein GAN with Gradient Penalty (WGAN-GP)** by integrating **spatial attention & channel attention** to **improve MRI synthesis** for Parkinson’s Disease dataset creation.

---

## **1️⃣. Introduction**
### ✅ Contributions of Alpha-GAN-GP:
- **Step 1**: Implements WGAN-GP for stable GAN training.
- **Step 2**: Introduces **Spatial Attention Module** (what regions to focus on).
- **Step 3**: Introduces **Channel Attention Module** (which features to emphasize).
- **Step 4**: Improves MRI dataset augmentation for medical applications.

---

## **2️⃣. Installing Dependencies**
Before running the project, install the required libraries:

```bash
pip install torch torchvision numpy tqdm matplotlib
import torch
import torch.nn.functional as F
from torch import nn, autograd

def compute_gradient_penalty(D, real_samples, fake_samples, device="cuda"):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = D(interpolates)
    
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
    
    
    class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(attention)
        
    class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = x.mean(dim=[2, 3])  
        max_out, _ = x.max(dim=[2, 3])  

        avg_out = self.fc2(self.relu(self.fc1(avg_out)))
        max_out = self.fc2(self.relu(self.fc1(max_out)))
        
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention
        
    class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=1, base_channels=64):
        super(Generator, self).__init__()

        self.init_size = 8  
        self.fc = nn.Linear(z_dim, base_channels * self.init_size * self.init_size)

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, stride=2, padding=1),  
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(),
            ChannelAttention(base_channels // 2),  

            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, 4, stride=2, padding=1),  
            nn.BatchNorm2d(base_channels // 4),
            nn.ReLU(),
            SpatialAttention(),  

            nn.ConvTranspose2d(base_channels // 4, img_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.fc(z).view(z.size(0), -1, self.init_size, self.init_size)
        img = self.conv_layers(z)
        return img
        
        
    class Discriminator(nn.Module):
    def __init__(self, img_channels=1, base_channels=64):
        super(Discriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(img_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            ChannelAttention(base_channels),

            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2),
            SpatialAttention(),

            nn.Conv2d(base_channels * 2, 1, 4, stride=2, padding=1),
        )

    def forward(self, img):
        return self.conv_layers(img).view(img.size(0), -1)
        
//Training Loop with WGAN-GP
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))

epochs = 10000
lambda_gp = 10  

for epoch in range(epochs):
    for _ in range(5):
        real_imgs = real_data_loader().to(device)
        z = torch.randn(real_imgs.size(0), 100).to(device)
        fake_imgs = generator(z).detach()

        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
        gp = compute_gradient_penalty(discriminator, real_imgs, fake_imgs, device)
        loss_D += lambda_gp * gp  

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

    z = torch.randn(batch_size, 100).to(device)
    loss_G = -torch.mean(discriminator(generator(z)))
    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    print(f'Epoch {epoch}: Loss D={loss_D.item():.4f}, Loss G={loss_G.item():.4f}')
    
    
    
