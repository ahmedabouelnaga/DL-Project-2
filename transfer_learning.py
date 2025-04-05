#!/usr/bin/env python
import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


# 1. Define a Dataset for the NCD (New Color Dataset)

class NCDDataset(Dataset):
    """
    Loads images from a new dataset (NCD) for transfer learning.
    Each image is:
      - Resized to 128x128
      - Converted from BGR to LAB
      - L channel is scaled from [0,255] to [0,100]
      - a and b channels are shifted to center around zero
    """
    def __init__(self, img_dir, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        if img is None:
            raise ValueError(f"Failed to load image at index {idx}")
        # Resize to 128x128
        img = cv2.resize(img, (128, 128))
        # Convert to LAB color space
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab_img)
        # Scale L from [0,255] to [0,100]
        L = L.astype(np.float32) / 255.0 * 100.0
        # Shift a and b channels: center around zero
        a = a.astype(np.float32) - 128.0
        b = b.astype(np.float32) - 128.0
        # Prepare tensors: L shape becomes (1,128,128); ab becomes (2,128,128)
        L = np.expand_dims(L, axis=0)
        ab = np.stack([a, b], axis=0)
        if self.transform:
            L = self.transform(L)
            ab = self.transform(ab)
        return torch.from_numpy(L), torch.from_numpy(ab)


# 2. Define the Colorization Model (Same as before)

class ColorizationModel(nn.Module):
    def __init__(self, n_downsample_layers=5):
        super(ColorizationModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Downsampling layers
        self.down_layers = nn.ModuleList()
        resolutions = [64, 128, 256, 512, 512]
        for i in range(n_downsample_layers - 1):
            in_channels = resolutions[i]
            out_channels = resolutions[i+1]
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.down_layers.append(block)
        
        # Upsampling layers with skip connections
        self.up_layers = nn.ModuleList()
        for i in range(n_downsample_layers - 1, 0, -1):
            in_channels = resolutions[i]
            out_channels = resolutions[i-1]
            block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.up_layers.append(block)
        
        # Final output: produce a* and b* channels
        self.output_conv = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        skip_connections = [x]
        for block in self.down_layers:
            x = block(x)
            skip_connections.append(x)
        skip_connections.pop()  # Remove bottleneck skip
        for block in self.up_layers:
            x = block(x)
            skip = skip_connections.pop()
            x = x + skip  # Skip connection
        x = self.output_conv(x)
        x = self.tanh(x) * 127  # Scale output to approximately match LAB range for a and b
        return x


# 3. Transfer Learning Procedure

def main():
    # Device configuration: use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Instantiate the model and load pretrained weights if available.
    model = ColorizationModel().to(device)
    pretrained_path = "./colorization/colorization_model.pth"
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print("Loaded pretrained weights from", pretrained_path)
    else:
        print("Pretrained weights not found. Training from scratch.")
    
    # Freeze early layers to retain previously learned features.
    for param in model.conv1.parameters():
        param.requires_grad = False
    for block in model.down_layers:
        for param in block.parameters():
            param.requires_grad = False


    # 4. Hyperparameter Tuning Considerations
    # -------------------------
    # Here you might experiment with:
    #   - Different learning rates (we use 0.0005 for fine-tuning)
    #   - Changing the number of feature maps in hidden layers
    #   - Varying the batch size
    # For this example, we set a learning rate and batch size that worked best in your experiments.
    learning_rate = 0.0005
    batch_size = 8
    num_epochs = 10
    
    # Create the new dataset (assume the NCD dataset is in "./NCD_dataset")
    ncd_dataset = NCDDataset("./NCD_dataset")
    dataset_size = len(ncd_dataset)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(ncd_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Loss and optimizer (only parameters that require gradients are optimized)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # Training loop for fine-tuning on the new dataset.
    print("Starting transfer learning fine-tuning...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for L, ab in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            L = L.to(device)
            ab = ab.to(device)
            optimizer.zero_grad()
            outputs = model(L)
            loss = criterion(outputs, ab)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * L.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}")
    
    # Evaluate on the test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for L, ab in test_loader:
            L = L.to(device)
            ab = ab.to(device)
            output = model(L)
            loss = criterion(output, ab)
            test_loss += loss.item()
    test_loss /= len(test_dataset)
    print(f"Test Loss on NCD dataset: {test_loss:.4f}")
    
    # Optionally save the fine-tuned model
    torch.save(model.state_dict(), "./colorization/fine_tuned_colorization_model.pth")
    print("Fine-tuned model saved.")

if __name__ == "__main__":
    main()
