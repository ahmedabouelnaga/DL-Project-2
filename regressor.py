import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# =======================
# 1. Dataset Definition
# =======================
class MeanChrominanceDataset(Dataset):
    def __init__(self, l_dir, a_dir, b_dir, transform=None):
        """
        Args:
            l_dir (str): Directory with L* channel images (PNG).
            a_dir (str): Directory with a* channel images (PNG).
            b_dir (str): Directory with b* channel images (PNG).
            transform (callable, optional): Optional transform to be applied on the L* image.
        """
        self.l_paths = sorted([os.path.join(l_dir, f) for f in os.listdir(l_dir) if f.endswith('.png')])
        self.a_paths = sorted([os.path.join(a_dir, f) for f in os.listdir(a_dir) if f.endswith('.png')])
        self.b_paths = sorted([os.path.join(b_dir, f) for f in os.listdir(b_dir) if f.endswith('.png')])
        
        # Consistency check
        assert len(self.l_paths) == len(self.a_paths) == len(self.b_paths), "Folder sizes do not match!"
        self.transform = transform

    def __len__(self):
        return len(self.l_paths)

    def __getitem__(self, idx):
        # Read images
        L_img = cv2.imread(self.l_paths[idx], cv2.IMREAD_UNCHANGED).astype(np.float32)
        a_img = cv2.imread(self.a_paths[idx], cv2.IMREAD_UNCHANGED).astype(np.float32)
        b_img = cv2.imread(self.b_paths[idx], cv2.IMREAD_UNCHANGED).astype(np.float32)
        
        # Scale L* from [0,100] -> [0,1]
        L_img /= 100.0
        
        # Compute mean a*, b* across entire image
        mean_a = a_img.mean()
        mean_b = b_img.mean()
        label = np.array([mean_a, mean_b], dtype=np.float32)
        
        # Reshape L_img to (1,H,W)
        L_img = np.expand_dims(L_img, axis=0)
        
        if self.transform:
            L_img = self.transform(L_img)
        
        return torch.from_numpy(L_img), torch.from_numpy(label)

# =======================
# 2. CNN Regressor Definition
# =======================
class MeanChrominanceRegressor(nn.Module):
    def __init__(self):
        super(MeanChrominanceRegressor, self).__init__()
        # Three conv layers with stride=2 to downsample the spatial dims
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        
        # FC layer for final 2 outputs: mean a*, mean b*
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        # x: (B, 1, H, W)
        x = F.relu(self.conv1(x))  # -> (B, 8,  H/2, W/2)
        x = F.relu(self.conv2(x))  # -> (B, 16, H/4, W/4)
        x = F.relu(self.conv3(x))  # -> (B, 32, H/8, W/8)
        
        # Global average pooling across spatial dimensions
        x = x.mean(dim=[2, 3])     # -> (B, 32)
        
        # Predict 2 values
        x = self.fc(x)            # -> (B, 2)
        return x

# =======================
# 3. Training Loop
# =======================
def train_model(train_loader, model, criterion, optimizer, num_epochs=5, device='cpu'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_L, batch_meanAB in train_loader:
            batch_L = batch_L.to(device)
            batch_meanAB = batch_meanAB.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_L)  # (B, 2)
            loss = criterion(predictions, batch_meanAB)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# =======================
# 4. Prediction
# =======================
def predict_mean_chrominance(model, L_img_tensor, device='cpu'):
    """
    Given a single L* image tensor (shape: (1, H, W)), predict mean a* and b*.
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        # Add a batch dimension: (1, 1, H, W)
        L_img_tensor = L_img_tensor.unsqueeze(0).to(device)
        prediction = model(L_img_tensor)  # shape (1, 2)
    return prediction.squeeze(0)  # shape (2,)

# =======================
# 5. Main Function
# =======================
def main():
    # Create dataset & dataloader
    l_dir = "L/"
    a_dir = "a/"
    b_dir = "b/"
    dataset = MeanChrominanceDataset(l_dir, a_dir, b_dir)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create model, loss, optimizer
    model = MeanChrominanceRegressor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train
    num_epochs = 5
    train_model(train_loader, model, criterion, optimizer, num_epochs=num_epochs)
    
    # Example: predict on a single sample
    sample_L, sample_label = dataset[0]
    prediction = predict_mean_chrominance(model, sample_L)
    print("Ground truth mean a*, b*:", sample_label.numpy())
    print("Predicted mean a*, b*   :", prediction.numpy())

if __name__ == "__main__":
    main()