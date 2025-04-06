#!/usr/bin/env python
import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set default datatype to 32-bit float
torch.set_default_dtype(torch.float32)

# Define paths - update these paths to match your directory structure
L_DIR = "./L"  # Directory for L* channel images
A_DIR = "./a"  # Directory for a* channel images (must be raw, not visualized)
B_DIR = "./b"  # Directory for b* channel images (must be raw, not visualized)
COLORIZED_DIR = "./colorization"  # Directory for colorized output images

# Create output directory if it doesn't exist
os.makedirs(COLORIZED_DIR, exist_ok=True)

# Print the current working directory and check if the directories exist
print(f"Current working directory: {os.getcwd()}")
print(f"L directory exists: {os.path.exists(L_DIR)}")
print(f"a directory exists: {os.path.exists(A_DIR)}")
print(f"b directory exists: {os.path.exists(B_DIR)}")

# If directories don't exist, try to find them
if not os.path.exists(L_DIR) or not os.path.exists(A_DIR) or not os.path.exists(B_DIR):
    print("Warning: L, a, or b directory not found, searching in parent directories...")
    # Try to find L, a, and b directories in the current directory or parent directories
    for root, dirs, _ in os.walk('.'):
        if 'L' in dirs and 'a' in dirs and 'b' in dirs:
            L_DIR = os.path.join(root, 'L')
            A_DIR = os.path.join(root, 'a')
            B_DIR = os.path.join(root, 'b')
            print(f"Found directories - L: {L_DIR}, a: {A_DIR}, b: {B_DIR}")
            break

# Create output directory if it doesn't exist
os.makedirs(COLORIZED_DIR, exist_ok=True)

# Define the colorization model
class ColorizationModel(nn.Module):
    def __init__(self, n_downsample_layers=5):
        super(ColorizationModel, self).__init__()
        
        # Initial feature extraction
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Downsample layers - reduce spatial resolution
        self.down_layers = nn.ModuleList()
        
        # Define resolution progression
        resolutions = [64, 128, 256, 512, 512]
        
        for i in range(n_downsample_layers - 1):
            in_channels = resolutions[i]
            out_channels = resolutions[i+1]
            
            down_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.down_layers.append(down_block)
        
        # Upsampling layers - increase spatial resolution
        self.up_layers = nn.ModuleList()
        
        for i in range(n_downsample_layers - 1, 0, -1):
            in_channels = resolutions[i]
            out_channels = resolutions[i-1]
            
            up_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.up_layers.append(up_block)
        
        # Final output layer - produces a* and b* channels
        self.output_conv = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        self.output_tanh = nn.Tanh()  # Tanh to constrain output values
        
    def forward(self, x):
        # Initial convolution
        x = self.relu1(self.bn1(self.conv1(x)))
        
        # Store intermediate outputs for skip connections
        skip_connections = [x]
        
        # Downsample path
        for down_layer in self.down_layers:
            x = down_layer(x)
            skip_connections.append(x)
        
        # Remove the last skip connection (bottleneck)
        skip_connections.pop()
        
        # Upsample path with skip connections
        for up_layer in self.up_layers:
            x = up_layer(x)
            skip = skip_connections.pop()
            x = x + skip  # Skip connection (element-wise addition)
        
        # Final output layer
        x = self.output_conv(x)
        x = self.output_tanh(x)
        
        # Scale the output to match the expected a*b* range
        # a* and b* typically range from -128 to 127
        # Tanh gives values from -1 to 1, so we scale by 127
        x = x * 127
        
        return x

# Custom dataset for colorization with separate a and b directories
class ColorizationDataset(Dataset):
    def __init__(self, l_dir, a_dir, b_dir, transform=None):
        # Note: Ensure that the images in a_dir and b_dir are the raw LAB channels,
        # not the color-coded visualizations.
        self.l_paths = []
        self.a_paths = []
        self.b_paths = []
        
        # Support multiple image extensions
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
            self.l_paths.extend(sorted(glob.glob(os.path.join(l_dir, ext))))
            self.a_paths.extend(sorted(glob.glob(os.path.join(a_dir, ext))))
            self.b_paths.extend(sorted(glob.glob(os.path.join(b_dir, ext))))
        
        # Sort to ensure matching pairs
        self.l_paths = sorted(self.l_paths)
        self.a_paths = sorted(self.a_paths)
        self.b_paths = sorted(self.b_paths)
        
        self.transform = transform
        
        print(f"Found {len(self.l_paths)} images in {l_dir}")
        print(f"Found {len(self.a_paths)} images in {a_dir}")
        print(f"Found {len(self.b_paths)} images in {b_dir}")
        
        if len(self.l_paths) == 0 or len(self.a_paths) == 0 or len(self.b_paths) == 0:
            raise ValueError(f"No images found in directories. Please check the paths.")
        
        # Use the smallest number of images across all directories
        self.dataset_size = min(len(self.l_paths), len(self.a_paths), len(self.b_paths))
        
        # Trim paths to ensure equal number of images across all directories
        self.l_paths = self.l_paths[:self.dataset_size]
        self.a_paths = self.a_paths[:self.dataset_size]
        self.b_paths = self.b_paths[:self.dataset_size]
        
        print(f"Using {self.dataset_size} images for training and testing")
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        try:
            # Load the images in

