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
A_DIR = "./a"  # Directory for a* channel images
B_DIR = "./b"  # Directory for b* channel images
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
        # Search for image files with multiple extensions
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
            # Load the images
            l_img = cv2.imread(self.l_paths[idx], cv2.IMREAD_GRAYSCALE)
            a_img = cv2.imread(self.a_paths[idx], cv2.IMREAD_GRAYSCALE)
            b_img = cv2.imread(self.b_paths[idx], cv2.IMREAD_GRAYSCALE)
            
            if l_img is None:
                raise ValueError(f"Failed to load L image: {self.l_paths[idx]}")
            if a_img is None:
                raise ValueError(f"Failed to load a image: {self.a_paths[idx]}")
            if b_img is None:
                raise ValueError(f"Failed to load b image: {self.b_paths[idx]}")
            
            # Normalize L channel (0-255 to 0-100)
            l_img = l_img.astype(np.float32) / 255.0 * 100.0
            
            # Normalize a* and b* channels (0-255 to -128 to 127)
            a_img = a_img.astype(np.float32) - 128
            b_img = b_img.astype(np.float32) - 128
            
            # Combine a* and b* channels
            ab_channels = np.stack([a_img, b_img], axis=2)
            
            # Apply transforms if any
            if self.transform:
                l_img = self.transform(l_img)
                ab_channels = self.transform(ab_channels)
            
            # Convert to tensor and reshape
            l_tensor = torch.from_numpy(l_img).unsqueeze(0)  # Add channel dimension
            ab_tensor = torch.from_numpy(ab_channels).permute(2, 0, 1)  # HWC to CHW
            
            return l_tensor, ab_tensor
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # Return a placeholder if there's an error
            return torch.zeros((1, 100, 100)), torch.zeros((2, 100, 100))

# Function to save images
def save_colorized_images(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for i, (l_img, _) in enumerate(test_loader):
            l_img = l_img.to(device)
            
            # Get predictions
            ab_pred = model(l_img)
            
            # Convert to numpy and prepare for visualization
            l_numpy = l_img[0, 0].cpu().numpy()
            a_numpy = ab_pred[0, 0].cpu().numpy() + 128  # Add offset
            b_numpy = ab_pred[0, 1].cpu().numpy() + 128  # Add offset
            
            # Create LAB image
            lab_img = np.zeros((l_numpy.shape[0], l_numpy.shape[1], 3), dtype=np.uint8)
            lab_img[:, :, 0] = np.clip(l_numpy, 0, 100).astype(np.uint8)
            lab_img[:, :, 1] = np.clip(a_numpy, 0, 255).astype(np.uint8)
            lab_img[:, :, 2] = np.clip(b_numpy, 0, 255).astype(np.uint8)
            
            # Convert from LAB to RGB
            rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)
            
            # Save colorized image
            output_path = os.path.join(COLORIZED_DIR, f"colorized_{i}.jpg")
            cv2.imwrite(output_path, rgb_img)
            
            # Also save comparison image for visualization
            l_for_vis = cv2.cvtColor(np.clip(l_numpy, 0, 100).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            comparison = np.concatenate((l_for_vis, rgb_img), axis=1)
            comparison_path = os.path.join(COLORIZED_DIR, f"comparison_{i}.jpg")
            cv2.imwrite(comparison_path, comparison)
            
            print(f"Saved image {i} to {output_path}")

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 8  # Reduced batch size
    num_epochs = 50
    learning_rate = 0.001
    n_downsample_layers = 5
    
    try:
        # Load the dataset
        dataset = ColorizationDataset(L_DIR, A_DIR, B_DIR)
        
        # Check if dataset has any samples
        if len(dataset) == 0:
            raise ValueError("Dataset has 0 samples. Please check the image directories.")
        
        # Use a smaller batch for the first test run
        if len(dataset) < batch_size:
            batch_size = max(1, len(dataset) // 2)
            print(f"Dataset has only {len(dataset)} samples. Using batch_size={batch_size}")
        
        # Split into train and test sets (90% train, 10% test)
        dataset_size = len(dataset)
        train_size = int(0.9 * dataset_size)
        test_size = dataset_size - train_size
        
        # Ensure we have at least 1 sample in each split
        if train_size < 1 or test_size < 1:
            train_size = max(1, dataset_size - 1)
            test_size = max(1, dataset_size - train_size)
        
        print(f"Splitting dataset: {train_size} training samples, {test_size} testing samples")
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        # Create data loaders - use num_workers=0 to avoid multiprocessing issues during debugging
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # Initialize the model
        model = ColorizationModel(n_downsample_layers).to(device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        print("Starting training...")
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for l_imgs, ab_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                l_imgs = l_imgs.to(device)
                ab_imgs = ab_imgs.to(device)
                
                # Forward pass
                ab_preds = model(l_imgs)
                loss = criterion(ab_preds, ab_imgs)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Print statistics
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            
            # Evaluate and save images every 5 epochs
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
                
                # Save intermediate colorization results
                print(f"Saving intermediate results for epoch {epoch+1}...")
                save_dir = os.path.join(COLORIZED_DIR, f"epoch_{epoch+1}")
                os.makedirs(save_dir, exist_ok=True)
                
                # Save a sample colorized image
                with torch.no_grad():
                    for i, (l_img, _) in enumerate(test_loader):
                        if i >= 5:  # Save just a few samples
                            break
                            
                        l_img = l_img.to(device)
                        ab_pred = model(l_img)
                        
                        # Convert to numpy and prepare for visualization
                        l_numpy = l_img[0, 0].cpu().numpy()
                        a_numpy = ab_pred[0, 0].cpu().numpy() + 128
                        b_numpy = ab_pred[0, 1].cpu().numpy() + 128
                        
                        # Create LAB image
                        lab_img = np.zeros((l_numpy.shape[0], l_numpy.shape[1], 3), dtype=np.uint8)
                        lab_img[:, :, 0] = np.clip(l_numpy, 0, 100).astype(np.uint8)
                        lab_img[:, :, 1] = np.clip(a_numpy, 0, 255).astype(np.uint8)
                        lab_img[:, :, 2] = np.clip(b_numpy, 0, 255).astype(np.uint8)
                        
                        # Convert from LAB to RGB
                        rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)
                        
                        # Save the image
                        cv2.imwrite(os.path.join(save_dir, f"sample_{i}.jpg"), rgb_img)
        
        print("Training complete!")
        
        # Save the model
        torch.save(model.state_dict(), os.path.join(COLORIZED_DIR, "colorization_model.pth"))
        
        # Generate and save colorized images
        print("Generating colorized images...")
        save_colorized_images(model, test_loader, device)
        print(f"Colorized images saved to {COLORIZED_DIR} directory")
    
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    # Initialize the model
    model = ColorizationModel(n_downsample_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for l_imgs, ab_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            l_imgs = l_imgs.to(device)
            ab_imgs = ab_imgs.to(device)
            
            # Forward pass
            ab_preds = model(l_imgs)
            loss = criterion(ab_preds, ab_imgs)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print statistics
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        # Evaluate every 5 epochs
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
    
    print("Training complete!")
    
    # Save the model
    torch.save(model.state_dict(), os.path.join(COLORIZED_DIR, "colorization_model.pth"))
    
    # Generate and save colorized images
    print("Generating colorized images...")
    save_colorized_images(model, test_loader, device)
    print(f"Colorized images saved to {COLORIZED_DIR} directory")

if __name__ == "__main__":
    main()
