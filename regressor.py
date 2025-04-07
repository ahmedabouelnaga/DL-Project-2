import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# 1. Dataset Definition

class RegressorDataset(Dataset):
    """
    Dataset that loads corresponding L, a, and b channel images.
    - L_dir: Directory containing L channel images (grayscale). These images are assumed to have values in [0, 100].
    - a_dir and b_dir: Directories containing the a and b channel images.
      For a and b, we convert the 8-bit images to float32 and subtract 128, then compute the mean value.
    The input is the scaled L channel (divided by 100 to map to [0,1]).
    The target is a vector [mean_a, mean_b] computed across the entire image.
    """
    def __init__(self, L_dir, a_dir, b_dir, transform=None):
        self.L_files = sorted(glob.glob(os.path.join(L_dir, "*.*")))
        self.a_files = sorted(glob.glob(os.path.join(a_dir, "*.*")))
        self.b_files = sorted(glob.glob(os.path.join(b_dir, "*.*")))
        
        if not self.L_files or not self.a_files or not self.b_files:
            raise ValueError("One or more directories do not contain image files.")
        
        # Use the smallest count to ensure matching samples
        self.dataset_size = min(len(self.L_files), len(self.a_files), len(self.b_files))
        self.L_files = self.L_files[:self.dataset_size]
        self.a_files = self.a_files[:self.dataset_size]
        self.b_files = self.b_files[:self.dataset_size]
        self.transform = transform
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        # Load the L channel image in grayscale
        L_img = cv2.imread(self.L_files[idx], cv2.IMREAD_GRAYSCALE)
        a_img = cv2.imread(self.a_files[idx], cv2.IMREAD_GRAYSCALE)
        b_img = cv2.imread(self.b_files[idx], cv2.IMREAD_GRAYSCALE)
        
        if L_img is None or a_img is None or b_img is None:
            raise ValueError(f"Failed to load images at index {idx}.")
        
        # The L channel originally ranges from 0 to 100.
        # Scale it to [0,1] by dividing by 100.
        L_img = L_img.astype(np.float32) / 100.0
        
        # For a and b channels, convert to float and subtract 128 to center the values.
        a_img = a_img.astype(np.float32)
        b_img = b_img.astype(np.float32) 
        
        # Compute mean chrominance values over all pixels.
        mean_a = np.mean(a_img)
        mean_b = np.mean(b_img)
        target = np.array([mean_a, mean_b], dtype=np.float32)
        
        # Optionally apply a transformation to L_img.
        if self.transform:
            L_img = self.transform(L_img)
        else:
            # Ensure the image has a channel dimension: (1, H, W)
            L_img = np.expand_dims(L_img, axis=0)
        
        # Convert to PyTorch tensors.
        input_tensor = torch.from_numpy(L_img)
        target_tensor = torch.from_numpy(target)
        return input_tensor, target_tensor


# 2. Regressor Model Definition

class RegressorCNN(nn.Module):
    def __init__(self):
        super(RegressorCNN, self).__init__()
        self.module1 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.module2 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.module3 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.module4 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.module5 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.module6 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # Module 7: final convolution to output 2 channels. The spatial size becomes 1x1.
        self.module7 = nn.Conv2d(3, 2, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        # Pass through the 7 modules sequentially.
        x = self.module1(x)  # (batch, 3, 64, 64)
        x = self.module2(x)  # (batch, 3, 32, 32)
        x = self.module3(x)  # (batch, 3, 16, 16)
        x = self.module4(x)  # (batch, 3, 8, 8)
        x = self.module5(x)  # (batch, 3, 4, 4)
        x = self.module6(x)  # (batch, 3, 2, 2)
        x = self.module7(x)  # (batch, 2, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 2)
        return x


# 3. Training and Evaluation Functions
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)


# 4. Main Routine

def main():
    # Directories for the L, a, and b images.
    # Update these paths as needed.
    L_dir = "./L"
    a_dir = "./a"
    b_dir = "./b"
    
    # Hyperparameters.
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.001
    
    # Set the default Torch data type to 32-bit float.
    torch.set_default_dtype(torch.float32)
    
    # Device configuration.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create the dataset and split into training (90%) and testing (10%).
    full_dataset = RegressorDataset(L_dir, a_dir, b_dir)
    dataset_size = len(full_dataset)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize the model, criterion, and optimizer.
    model = RegressorCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop.
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")
    
    # Save the trained model.
    model_path = "regressor_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()

