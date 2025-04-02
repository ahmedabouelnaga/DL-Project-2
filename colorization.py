import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =======================
# 1. Dataset Definition
# =======================
class ColorizationDataset(Dataset):
    def __init__(self, l_dir, a_dir, b_dir, transform=None):
        """
        Args:
            l_dir (str): Directory containing L* channel images (PNG).
            a_dir (str): Directory containing a* channel images (PNG).
            b_dir (str): Directory containing b* channel images (PNG).
            transform (callable, optional): Any additional transform to apply to the L* image.
        """
        self.l_paths = sorted([os.path.join(l_dir, f) for f in os.listdir(l_dir) if f.endswith('.png')])
        self.a_paths = sorted([os.path.join(a_dir, f) for f in os.listdir(a_dir) if f.endswith('.png')])
        self.b_paths = sorted([os.path.join(b_dir, f) for f in os.listdir(b_dir) if f.endswith('.png')])
        
        # Check that all folders contain the same number of images
        assert len(self.l_paths) == len(self.a_paths) == len(self.b_paths), "Mismatch in folder image counts!"
        self.transform = transform

    def __len__(self):
        return len(self.l_paths)

    def __getitem__(self, idx):
        # Read L, a, and b images in grayscale (each as 2D arrays)
        L_img = cv2.imread(self.l_paths[idx], cv2.IMREAD_UNCHANGED)
        a_img = cv2.imread(self.a_paths[idx], cv2.IMREAD_UNCHANGED)
        b_img = cv2.imread(self.b_paths[idx], cv2.IMREAD_UNCHANGED)
        
        if L_img is None or a_img is None or b_img is None:
            raise ValueError(f"Error reading one or more images at index {idx}.")
        
        # Convert images to float32
        L_img = L_img.astype(np.float32)
        a_img = a_img.astype(np.float32)
        b_img = b_img.astype(np.float32)
        
        # Scale L* from [0, 100] to [0, 1] (adjust if your saved range is different)
        L_img = L_img / 100.0
        
        # Stack a and b channels to create a 2-channel output image
        ab_img = np.stack((a_img, b_img), axis=0)  # shape: (2, H, W)
        
        # Expand dims for L to have shape (1, H, W)
        L_img = np.expand_dims(L_img, axis=0)
        
        if self.transform:
            L_img = self.transform(L_img)
        
        # Convert to torch tensors
        L_tensor = torch.from_numpy(L_img)
        ab_tensor = torch.from_numpy(ab_img)
        return L_tensor, ab_tensor

# =======================
# 2. Colorization Network
# =======================
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        
        # Encoder: Downsample input L channel
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder: Upsample to predict a and b channels
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=4, stride=2, padding=1)
            # No BatchNorm on final layer. Optionally, you can apply an activation (e.g., Tanh) if normalizing ab.
        )

    def forward(self, x):
        # x: (B, 1, H, W)
        # Encoder
        x = self.enc1(x)  # -> (B, 64, H/2, W/2)
        x = self.enc2(x)  # -> (B, 128, H/4, W/4)
        x = self.enc3(x)  # -> (B, 256, H/8, W/8)
        # Decoder
        x = self.dec1(x)  # -> (B, 128, H/4, W/4)
        x = self.dec2(x)  # -> (B, 64, H/2, W/2)
        x = self.dec3(x)  # -> (B, 2, H, W)
        return x

# =======================
# 3. Training Loop
# =======================
def train_colorization(model, train_loader, val_loader=None, epochs=10, lr=1e-3, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_L, batch_ab in train_loader:
            batch_L = batch_L.to(device)      # (B, 1, H, W)
            batch_ab = batch_ab.to(device)    # (B, 2, H, W)
            
            optimizer.zero_grad()
            pred_ab = model(batch_L)          # (B, 2, H, W)
            loss = criterion(pred_ab, batch_ab)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}")
        
        # Optional validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_L, val_ab in val_loader:
                    val_L = val_L.to(device)
                    val_ab = val_ab.to(device)
                    pred_ab = model(val_L)
                    loss_val = criterion(pred_ab, val_ab)
                    val_loss += loss_val.item()
            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")

# =======================
# 4. Prediction Function
# =======================
def predict_colorization(model, L_img_tensor, device='cpu'):
    """
    Given a single L* image tensor (shape: (1, H, W)), predict full-resolution a and b channels.
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        # Add batch dimension: (1, 1, H, W)
        L_img_tensor = L_img_tensor.unsqueeze(0).to(device)
        pred_ab = model(L_img_tensor)  # (1, 2, H, W)
    return pred_ab.squeeze(0)  # (2, H, W)

# =======================
# 5. Main Routine
# =======================
def main():
    # Define directories containing your preprocessed images.
    # These should be the folders where you saved your L, a, and b channel images.
    l_dir = "L/"
    a_dir = "a/"
    b_dir = "b/"
    
    # Create the dataset and dataloader.
    dataset = ColorizationDataset(l_dir, a_dir, b_dir)
    # Optionally split dataset into train and validation subsets.
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    # For simplicity, here we use only training loader.
    
    # Instantiate the model.
    model = ColorizationNet()
    
    # Determine device: use GPU if available.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train the model.
    epochs = 10  # Adjust as needed
    train_colorization(model, train_loader, epochs=epochs, lr=1e-3, device=device)
    
    # Test: Predict on a single sample from the dataset.
    sample_L, sample_ab = dataset[0]
    pred_ab = predict_colorization(model, sample_L, device=device)
    
    print("Sample Ground Truth ab shape:", sample_ab.shape)
    print("Sample Predicted ab shape:", pred_ab.shape)
    
    # Optionally, you can combine L and predicted ab to form a LAB image,
    # convert it to RGB (using cv2.cvtColor), and then save or display the result.
    
if __name__ == "__main__":
    main()