import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from datetime import datetime


class Encoder(nn.Module):
    """Encoder network for CVAE"""
    
    def __init__(self, condition_channels=4, latent_dim=128):
        super(Encoder, self).__init__()
        
        # Condition: 4 channels (RGB + coarse mask)
        # Target: 1 channel (refined mask)
        # Total input: 5 channels
        
        self.conv1 = nn.Conv2d(condition_channels + 1, 64, 4, 2, 1)  # 128x128
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)  # 64x64
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)  # 32x32
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)  # 16x16
        self.conv5 = nn.Conv2d(512, 512, 4, 2, 1)  # 8x8
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Calculate flattened size
        self.flat_size = 512 * 8 * 8
        
        # Latent space
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
    def forward(self, x, y):
        # x: condition (RGB + coarse mask)
        # y: target (refined mask)
        inp = torch.cat([x, y], dim=1)
        
        h = F.leaky_relu(self.bn1(self.conv1(inp)), 0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), 0.2)
        h = F.leaky_relu(self.bn3(self.conv3(h)), 0.2)
        h = F.leaky_relu(self.bn4(self.conv4(h)), 0.2)
        h = F.leaky_relu(self.bn5(self.conv5(h)), 0.2)
        
        h = h.view(-1, self.flat_size)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class Decoder(nn.Module):
    """Decoder network for CVAE"""
    
    def __init__(self, condition_channels=4, latent_dim=128):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.condition_channels = condition_channels
        
        # Project latent vector
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        
        # Upsampling layers
        self.deconv1 = nn.ConvTranspose2d(512 + condition_channels, 512, 4, 2, 1)  # 16x16
        self.deconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)  # 32x32
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 64x64
        self.deconv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # 128x128
        self.deconv5 = nn.ConvTranspose2d(64, 1, 4, 2, 1)  # 256x256
        
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        
    def forward(self, z, condition):
        # z: latent vector
        # condition: RGB + coarse mask
        
        h = self.fc(z)
        h = h.view(-1, 512, 8, 8)
        
        # Downsample condition to match
        c = F.interpolate(condition, size=(8, 8), mode='bilinear', align_corners=False)
        h = torch.cat([h, c], dim=1)
        
        h = F.relu(self.bn1(self.deconv1(h)))
        h = F.relu(self.bn2(self.deconv2(h)))
        h = F.relu(self.bn3(self.deconv3(h)))
        h = F.relu(self.bn4(self.deconv4(h)))
        h = torch.sigmoid(self.deconv5(h))
        
        return h


class CVAE(nn.Module):
    """Conditional Variational Autoencoder"""
    
    def __init__(self, condition_channels=4, latent_dim=128):
        super(CVAE, self).__init__()
        self.encoder = Encoder(condition_channels, latent_dim)
        self.decoder = Decoder(condition_channels, latent_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, condition, target):
        # Encode
        mu, logvar = self.encoder(condition, target)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon = self.decoder(z, condition)
        
        return recon, mu, logvar
    
    def generate(self, condition, z=None):
        """Generate refined mask from condition"""
        if z is None:
            # Sample from standard normal
            batch_size = condition.size(0)
            z = torch.randn(batch_size, self.decoder.latent_dim).to(condition.device)
        
        return self.decoder(z, condition)


def cvae_loss(recon, target, mu, logvar, kl_weight=1.0):
    """
    CVAE loss = Reconstruction loss + KL divergence
    """
    # Reconstruction loss (Binary Cross Entropy)
    recon_loss = F.binary_cross_entropy(recon, target, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss