"""
Conditional GAN
"""

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

class Generator(nn.Module):
    """U-Net style Generator for mask refinement"""
    
    def __init__(self, input_channels=4, output_channels=1, ngf=64):
        super(Generator, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(input_channels, ngf, normalize=False)  # 256x256
        self.enc2 = self.conv_block(ngf, ngf * 2)  # 128x128
        self.enc3 = self.conv_block(ngf * 2, ngf * 4)  # 64x64
        self.enc4 = self.conv_block(ngf * 4, ngf * 8)  # 32x32
        self.enc5 = self.conv_block(ngf * 8, ngf * 8)  # 16x16
        self.enc6 = self.conv_block(ngf * 8, ngf * 8)  # 8x8
        
        # Decoder (upsampling)
        self.dec1 = self.deconv_block(ngf * 8, ngf * 8, dropout=True)  # 16x16
        self.dec2 = self.deconv_block(ngf * 16, ngf * 8, dropout=True)  # 32x32
        self.dec3 = self.deconv_block(ngf * 16, ngf * 4)  # 64x64
        self.dec4 = self.deconv_block(ngf * 8, ngf * 2)  # 128x128
        self.dec5 = self.deconv_block(ngf * 4, ngf)  # 256x256
        
        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_channels, 4, 2, 1),
            nn.Sigmoid()
        )
        
    def conv_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def deconv_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        
        # Decoder with skip connections
        d1 = self.dec1(e6)
        d2 = self.dec2(torch.cat([d1, e5], dim=1))
        d3 = self.dec3(torch.cat([d2, e4], dim=1))
        d4 = self.dec4(torch.cat([d3, e3], dim=1))
        d5 = self.dec5(torch.cat([d4, e2], dim=1))
        
        # Final output
        out = self.final(torch.cat([d5, e1], dim=1))
        
        return out


class Discriminator(nn.Module):
    """PatchGAN Discriminator"""
    
    def __init__(self, input_channels=5, ndf=64):
        super(Discriminator, self).__init__()
        
        # input is (condition + refined_mask) = 5 channels
        self.model = nn.Sequential(
            # 256x256
            nn.Conv2d(input_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 31x31
            nn.Conv2d(ndf * 8, 1, 4, 1, 1),
            # Output: 30x30 patch
        )
    
    def forward(self, condition, mask):
        x = torch.cat([condition, mask], dim=1)
        return self.model(x)