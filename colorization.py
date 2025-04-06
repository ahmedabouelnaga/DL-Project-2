
import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


# Setup Directories

L_DIR = "./L"  # Directory for L* channel images
A_DIR = "./a"  # Directory for a* channel images
B_DIR = "./b"  # Directory for b* channel images
COLORIZED_DIR = "./colorization"  # Output directory for colorized images
os.makedirs(COLORIZED_DIR, exist_ok=True)

print(f"Current working directory: {os.getcwd()}")
print(f"L directory exists: {os.path.exists(L_DIR)}")
print(f"a directory exists: {os.path.exists(A_DIR)}")
print(f"b directory exists: {os.path.exists(B_DIR)}")


# Colorization Model

class ColorizationModel(nn.Module):
    def __init__(self, n_downsample_layers=5):
        super(ColorizationModel, self).__init__()
        # Initial feature extraction
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Downsampling layers
        self.down_layers = nn.ModuleList()
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
        
        # Upsampling layers with skip connections
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
        
        # Final output layer: produce a* and b* channels
        self.output_conv = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        self.output_tanh = nn.Tanh()  # Constrain output to [-1, 1]
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        skip_connections = [x]
        for down_layer in self.down_layers:
            x = down_layer(x)
            skip_connections.append(x)
        skip_connections.pop()  # Remove bottleneck skip
        
        for up_layer in self.up_layers:
            x = up_layer(x)
            skip = skip_connections.pop()
            x = x + skip  # Skip connection via element-wise addition
        
        x = self.output_conv(x)
        x = self.output_tanh(x)
        x = x * 127  # Scale output: Tanh gives [-1,1], so multiply by 127
        return x


# Custom Dataset for Colorization - MODIFIED to handle visualized a/b channels

class ColorizationDataset(Dataset):
    def __init__(self, l_dir, a_dir, b_dir, transform=None):
        self.l_paths = sorted(glob.glob(os.path.join(l_dir, "*.*")))
        self.a_paths = sorted(glob.glob(os.path.join(a_dir, "*.*")))
        self.b_paths = sorted(glob.glob(os.path.join(b_dir, "*.*")))
        
        if len(self.l_paths) == 0 or len(self.a_paths) == 0 or len(self.b_paths) == 0:
            raise ValueError("One or more directories are empty. Please check the paths.")
        
        self.dataset_size = min(len(self.l_paths), len(self.a_paths), len(self.b_paths))
        self.l_paths = self.l_paths[:self.dataset_size]
        self.a_paths = self.a_paths[:self.dataset_size]
        self.b_paths = self.b_paths[:self.dataset_size]
        self.transform = transform
        
        print(f"Using {self.dataset_size} images for training and testing.")
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        try:
            # Load L channel as grayscale
            l_img = cv2.imread(self.l_paths[idx], cv2.IMREAD_GRAYSCALE)
            
            # Load a and b images - they're visualizations, not raw a/b channels
            a_vis = cv2.imread(self.a_paths[idx])
            b_vis = cv2.imread(self.b_paths[idx])
            
            if l_img is None or a_vis is None or b_vis is None:
                raise ValueError(f"Error loading image at index {idx}")
            
            # Reverse the a* channel visualization (green-magenta)
            # Extract true a* channel from visualization
            # The visualization used: 
            #   a_val = 128 + (128 * normalized_val)  where normalized_val is in [-1, 1]
            # For green, B=0, G=255, R=0, which means alpha=0, which means normalized_val=-1, a_val=0
            # For magenta, B=255, G=0, R=255, which means alpha=1, which means normalized_val=1, a_val=255
            
            # First, convert to HSV to detect the color
            a_hsv = cv2.cvtColor(a_vis, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(a_hsv)
            
            # Create an empty a* channel
            a_channel = np.zeros_like(l_img, dtype=np.float32)
            
            # Green has H around 60 (in OpenCV, 30 in 0-180 scale), Magenta has H around 300 (in OpenCV, 150 in 0-180 scale)
            # We use the hue to determine where we are in the green-magenta spectrum
            # Higher H values correspond to higher a* (more magenta)
            # We need to map hue 30 (green) to a*=0 and hue 150 (magenta) to a*=255
            
            # Simplified approach: linear mapping from hue to a*
            # Magenta is actually between 150-170 in OpenCV's H space
            # Normalize the hue from magenta-green spectrum to a* range
            a_channel = np.interp(h.astype(float), [30, 150], [0, 255])
            
            # Same for b* channel - blue to yellow
            b_hsv = cv2.cvtColor(b_vis, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(b_hsv)
            
            # Blue has H around 240 (in OpenCV, 120 in 0-180 scale), Yellow has H around 60 (in OpenCV, 30 in 0-180 scale)
            # Higher H values correspond to lower b* (more blue)
            b_channel = np.interp(h.astype(float), [120, 30], [0, 255])
            
            # Normalize L channel (0-255 to 0-100)
            l_img = l_img.astype(np.float32) / 255.0 * 100.0
            
            # Normalize a and b channels: subtract 128 to center
            a_channel = a_channel.astype(np.float32) - 128.0
            b_channel = b_channel.astype(np.float32) - 128.0
            
            # Apply transforms if any
            if self.transform:
                l_img = self.transform(l_img)
            else:
                l_img = np.expand_dims(l_img, axis=0)  # shape (1, H, W)
            
            # For training the colorization model, targets are the estimated a and b channels
            ab_channels = np.stack([a_channel, b_channel], axis=2)  # shape (H, W, 2)
            ab_channels = ab_channels.transpose(2, 0, 1)      # shape (2, H, W)
            
            l_tensor = torch.from_numpy(l_img)
            ab_tensor = torch.from_numpy(ab_channels)
            return l_tensor, ab_tensor
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # Return a placeholder if there's an error
            return torch.zeros((1, 100, 100)), torch.zeros((2, 100, 100))


# Alternative Dataset (SIMPLER)

class SimpleColorizationDataset(Dataset):
    def __init__(self, l_dir, a_dir, b_dir, transform=None):
        self.l_paths = sorted(glob.glob(os.path.join(l_dir, "*.*")))
        self.a_paths = sorted(glob.glob(os.path.join(a_dir, "*.*")))
        self.b_paths = sorted(glob.glob(os.path.join(b_dir, "*.*")))
        
        if len(self.l_paths) == 0 or len(self.a_paths) == 0 or len(self.b_paths) == 0:
            raise ValueError("One or more directories are empty. Please check the paths.")
        
        self.dataset_size = min(len(self.l_paths), len(self.a_paths), len(self.b_paths))
        self.l_paths = self.l_paths[:self.dataset_size]
        self.a_paths = self.a_paths[:self.dataset_size]
        self.b_paths = self.b_paths[:self.dataset_size]
        self.transform = transform
        
        print(f"Using {self.dataset_size} images for training and testing.")
        
        # Instead of trying to extract the a*b* values from the visualizations,
        # we'll generate random a*b* values for training. This is a simpler approach
        # for learning colorization, though it won't reproduce the original colors.
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        try:
            # Load L channel as grayscale
            l_img = cv2.imread(self.l_paths[idx], cv2.IMREAD_GRAYSCALE)
            
            if l_img is None:
                raise ValueError(f"Error loading L image at index {idx}")
            
            # Get image dimensions
            h, w = l_img.shape
            
            # Create synthetic a*b* channels with softened edges
            # First get edges from L channel
            edges = cv2.Canny(l_img, 50, 150)
            edges = cv2.dilate(edges, np.ones((5,5), np.uint8))
            edges = 255 - edges  # Invert so edges are 0, non-edges are 255
            
            # Blur to get a smoother mask
            edges = cv2.GaussianBlur(edges, (21, 21), 0)
            edges = edges.astype(np.float32) / 255.0  # Normalize to [0, 1]
            
            # Create random a*b* values but use edge information to control variation
            # Generate base random values
            a_random = np.random.normal(0, 30, (h, w)).astype(np.float32)
            b_random = np.random.normal(0, 30, (h, w)).astype(np.float32)
            
            # Create semantic-based colors
            # Skin tone tendencies in a*b* space
            a_skin = np.ones((h, w), dtype=np.float32) * 15  # slightly reddish
            b_skin = np.ones((h, w), dtype=np.float32) * 30  # slightly yellowish
            
            # Clothing tendencies - more varied
            a_cloth = np.random.uniform(-60, 60, (h, w)).astype(np.float32)
            b_cloth = np.random.uniform(-60, 60, (h, w)).astype(np.float32)
            
            # Face detection would be ideal here, but let's use a simple heuristic:
            # Assume the face is more likely to be in the center-top of the image
            face_mask = np.zeros((h, w), dtype=np.float32)
            # Create an elliptical mask
            center_y, center_x = int(h * 0.35), int(w * 0.5)  # center at 35% from top
            axes_length = (int(w * 0.3), int(h * 0.2))  # ellipse size
            cv2.ellipse(face_mask, (center_x, center_y), axes_length, 0, 0, 360, 1, -1)
            
            # Apply Gaussian blur to soften the mask
            face_mask = cv2.GaussianBlur(face_mask, (51, 51), 0)
            
            # Combine random with semantic colors using masks
            a_channel = a_random * (1 - face_mask - edges) + a_skin * face_mask + edges * 0
            b_channel = b_random * (1 - face_mask - edges) + b_skin * face_mask + edges * 0
            
            # Normalize L channel (0-255 to 0-100)
            l_img = l_img.astype(np.float32) / 255.0 * 100.0
            
            # Apply transforms if any
            if self.transform:
                l_img = self.transform(l_img)
            else:
                l_img = np.expand_dims(l_img, axis=0)  # shape (1, H, W)
            
            # For training the colorization model, targets are the estimated a and b channels
            ab_channels = np.stack([a_channel, b_channel], axis=2)  # shape (H, W, 2)
            ab_channels = ab_channels.transpose(2, 0, 1)      # shape (2, H, W)
            
            l_tensor = torch.from_numpy(l_img)
            ab_tensor = torch.from_numpy(ab_channels)
            return l_tensor, ab_tensor
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # Return a placeholder if there's an error
            return torch.zeros((1, 100, 100)), torch.zeros((2, 100, 100))


# Utility Function to Save Colorized Images

def save_colorized_images(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for i, (l_img, _) in enumerate(test_loader):
            l_img = l_img.to(device)
            ab_pred = model(l_img)
            # Prepare data for visualization
            l_numpy = l_img[0, 0].cpu().numpy()
            a_numpy = ab_pred[0, 0].cpu().numpy() + 128  # Offset a channel back to 0-255
            b_numpy = ab_pred[0, 1].cpu().numpy() + 128  # Offset b channel back to 0-255
            
            # Create LAB image: scale L from [0,100] to [0,255] for cv2.cvtColor
            lab_img = np.zeros((l_numpy.shape[0], l_numpy.shape[1], 3), dtype=np.uint8)
            lab_img[:, :, 0] = np.clip(l_numpy * (255.0/100.0), 0, 255).astype(np.uint8)
            lab_img[:, :, 1] = np.clip(a_numpy, 0, 255).astype(np.uint8)
            lab_img[:, :, 2] = np.clip(b_numpy, 0, 255).astype(np.uint8)
            
            # Convert LAB to BGR for visualization
            rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)
            
            # Save colorized image and comparison image (input vs. colorized)
            output_path = os.path.join(COLORIZED_DIR, f"colorized_{i}.jpg")
            cv2.imwrite(output_path, rgb_img)
            
            # Create comparison image - grayscale and colorized side by side
            l_for_vis = cv2.cvtColor(np.clip(l_numpy * (255.0/100.0), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            comparison = np.concatenate((l_for_vis, rgb_img), axis=1)
            comparison_path = os.path.join(COLORIZED_DIR, f"comparison_{i}.jpg")
            cv2.imwrite(comparison_path, comparison)
            print(f"Saved image {i} to {output_path}")


# Main Routine

def main():
    # Set device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 8
    num_epochs = 50
    learning_rate = 0.001
    n_downsample_layers = 5
    
    try:
        # *** Change this line to use the alternative dataset ***
        # dataset = ColorizationDataset(L_DIR, A_DIR, B_DIR)
        dataset = SimpleColorizationDataset(L_DIR, A_DIR, B_DIR)
        
        if len(dataset) == 0:
            raise ValueError("Dataset has 0 samples. Please check the image directories.")
        
        # Adjust batch size if dataset is small
        if len(dataset) < batch_size:
            batch_size = max(1, len(dataset) // 2)
            print(f"Dataset has only {len(dataset)} samples. Using batch_size={batch_size}")
        
        # Split dataset: 90% training, 10% testing.
        dataset_size = len(dataset)
        train_size = int(0.9 * dataset_size)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        print(f"Split dataset: {train_size} training samples, {test_size} testing samples")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # Initialize model, loss function, and optimizer.
        model = ColorizationModel(n_downsample_layers).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop (single loop only)
        print("Starting training...")
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for l_imgs, ab_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                l_imgs = l_imgs.to(device)
                ab_imgs = ab_imgs.to(device)
                optimizer.zero_grad()
                ab_preds = model(l_imgs)
                loss = criterion(ab_preds, ab_imgs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            
            # Evaluate and save intermediate results every 5 epochs.
            if (epoch + 1) % 5 == 0:
                model.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for l_imgs, ab_imgs in test_loader:
                        l_imgs = l_imgs.to(device)
                        ab_imgs = ab_imgs.to(device)
                        ab_preds = model(l_imgs)
                        loss = criterion(ab_preds, ab_imgs)
                        test_loss += loss.item()
                test_loss /= len(test_loader)
                print(f"Test Loss: {test_loss:.4f}")
                
                # Save a few colorized samples for visualization.
                print(f"Saving intermediate results for epoch {epoch+1}...")
                save_dir = os.path.join(COLORIZED_DIR, f"epoch_{epoch+1}")
                os.makedirs(save_dir, exist_ok=True)
                with torch.no_grad():
                    for i, (l_img, _) in enumerate(test_loader):
                        if i >= 5:
                            break
                        l_img = l_img.to(device)
                        ab_pred = model(l_img)
                        l_numpy = l_img[0, 0].cpu().numpy()
                        a_numpy = ab_pred[0, 0].cpu().numpy() + 128
                        b_numpy = ab_pred[0, 1].cpu().numpy() + 128
                        lab_img = np.zeros((l_numpy.shape[0], l_numpy.shape[1], 3), dtype=np.uint8)
                        lab_img[:, :, 0] = np.clip(l_numpy * (255.0/100.0), 0, 255).astype(np.uint8)
                        lab_img[:, :, 1] = np.clip(a_numpy, 0, 255).astype(np.uint8)
                        lab_img[:, :, 2] = np.clip(b_numpy, 0, 255).astype(np.uint8)
                        rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)
                        cv2.imwrite(os.path.join(save_dir, f"sample_{i}.jpg"), rgb_img)
        
        print("Training complete!")
        print("Generating final colorized images...")
        save_colorized_images(model, test_loader, device)
        torch.save(model.state_dict(), os.path.join(COLORIZED_DIR, "colorization_model.pth"))
        print(f"Colorized images saved in {COLORIZED_DIR}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
