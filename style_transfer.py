import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG16FeatureExtractor(nn.Module):
    """
    VGG16 feature extractor for style transfer.
    Extracts features from specific layers of a pre-trained VGG16 model.
    """
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg = models.vgg16(pretrained=True)
        for param in vgg.parameters():
            param.requires_grad = False
            
        # Define feature layers for content and style
        self.content_layers = ['21']  # conv4_2 for content
        self.style_layers = ['0', '5', '10', '17', '24']  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 for style
        
        # Create a model to extract features
        self.vgg_features = vgg.features[:25]  # Up to conv5_1
        
    def forward(self, x):
        """
        Forward pass through the VGG16 feature extractor
        
        Args:
            x: Input image tensor [batch_size, channels, height, width]
            
        Returns:
            tuple: (content_features, style_features)
        """
        # Convert grayscale to 3-channel if needed
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # Normalize input for VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        
        # Extract features
        content_features = {}
        style_features = {}
        
        for name, layer in enumerate(self.vgg_features):
            x = layer(x)
            name = str(name)
            
            if name in self.content_layers:
                content_features[name] = x
                
            if name in self.style_layers:
                style_features[name] = x
                
        return content_features, style_features
    
    @staticmethod
    def gram_matrix(x):
        """
        Calculate Gram matrix for style representation
        
        Args:
            x: Feature map [batch_size, channels, height, width]
            
        Returns:
            Gram matrix [batch_size, channels, channels]
        """
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)


class StyleTransfer:
    """
    Style transfer for refining generated MRI images.
    Transfers style from real MRIs to synthetic MRIs while preserving content.
    """
    def __init__(self, content_weight=1.0, style_weight=1000.0, device='cuda'):
        self.device = device
        self.feature_extractor = VGG16FeatureExtractor().to(device)
        self.content_weight = content_weight
        self.style_weight = style_weight
        
    def transfer_style(self, content_img, style_img, num_steps=300, lr=0.01):
        """
        Apply style transfer to refine a generated image
        
        Args:
            content_img: Generated image to preserve content from
            style_img: Real image to take style from
            num_steps: Number of optimization steps
            lr: Learning rate for optimization
            
        Returns:
            Refined image with style transferred
        """
        # Clone content image to create the target
        target = content_img.clone().requires_grad_(True).to(self.device)
        
        # Move images to device
        content_img = content_img.to(self.device)
        style_img = style_img.to(self.device)
        
        # Extract style features from style image
        _, style_features = self.feature_extractor(style_img)
        style_grams = {layer: self.feature_extractor.gram_matrix(feature) 
                       for layer, feature in style_features.items()}
        
        # Setup optimizer
        optimizer = torch.optim.Adam([target], lr=lr)
        
        # Optimization loop
        for step in range(num_steps):
            # Extract features from target
            content_features, target_style_features = self.feature_extractor(target)
            
            # Calculate content loss
            content_loss = 0
            for layer in self.feature_extractor.content_layers:
                target_feature = content_features[layer]
                content_feature, _ = self.feature_extractor(content_img)
                content_feature = content_feature[layer]
                content_loss += F.mse_loss(target_feature, content_feature)
                
            # Calculate style loss
            style_loss = 0
            for layer in self.feature_extractor.style_layers:
                target_feature = target_style_features[layer]
                target_gram = self.feature_extractor.gram_matrix(target_feature)
                style_gram = style_grams[layer]
                style_loss += F.mse_loss(target_gram, style_gram)
                
            # Combined loss
            loss = self.content_weight * content_loss + self.style_weight * style_loss
            
            # Update target
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Ensure pixel values stay in valid range
            with torch.no_grad():
                target.clamp_(-1, 1)
                
            # Print progress
            if (step + 1) % 50 == 0:
                print(f"Step {step+1}/{num_steps}, Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss.item():.4f}")
                
        return target
    
    def batch_style_transfer(self, content_batch, style_batch, num_steps=100, lr=0.01):
        """
        Apply style transfer to a batch of images
        
        Args:
            content_batch: Batch of generated images
            style_batch: Batch of real images for style reference
            num_steps: Number of optimization steps
            lr: Learning rate
            
        Returns:
            Batch of refined images
        """
        refined_batch = []
        batch_size = content_batch.size(0)
        
        for i in range(batch_size):
            content_img = content_batch[i:i+1]
            style_img = style_batch[i:i+1] if i < style_batch.size(0) else style_batch[0:1]
            
            refined_img = self.transfer_style(content_img, style_img, num_steps, lr)
            refined_batch.append(refined_img)
            
        return torch.cat(refined_batch, dim=0) 