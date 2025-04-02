import os
import glob
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

###############################################################################
# 1) Dataset: Load Precomputed LAB Channel Images from Folders "L", "a", "b"
###############################################################################
class ColorizationLABDataset(Dataset):
    """
    Assumes that the folders 'L', 'a', and 'b' contain images with matching filenames.
    Loads the L-channel image (grayscale) as input and the corresponding a and b channel images
    (visualized as grayscale; if they are RGB representations with identical channels, a single channel is sufficient)
    as the target.
    """
    def __init__(self, l_dir, a_dir, b_dir, transform=None):
        self.l_files = sorted(glob.glob(os.path.join(l_dir, "*.png")))
        self.a_files = sorted(glob.glob(os.path.join(a_dir, "*.png")))
        self.b_files = sorted(glob.glob(os.path.join(b_dir, "*.png")))
        assert len(self.l_files) == len(self.a_files) == len(self.b_files), "Mismatch in number of files"
        self.transform = transform

    def __len__(self):
        return len(self.l_files)

    def __getitem__(self, idx):
        # Load L-channel in grayscale mode
        l_img = cv2.imread(self.l_files[idx], cv2.IMREAD_GRAYSCALE)
        # Load a and b images in grayscale mode (assuming they are stored as 8-bit images)
        a_img = cv2.imread(self.a_files[idx], cv2.IMREAD_GRAYSCALE)
        b_img = cv2.imread(self.b_files[idx], cv2.IMREAD_GRAYSCALE)

        # Optional transform/augmentation (if any)
        if self.transform:
            l_img = self.transform(l_img)
            a_img = self.transform(a_img)
            b_img = self.transform(b_img)

        # Use .copy() to ensure positive strides
        l_tensor = torch.from_numpy(l_img.copy()).unsqueeze(0).float()    # shape: (1, H, W)
        # Stack a and b into a (2, H, W) tensor
        ab_tensor = torch.from_numpy(np.stack([a_img.copy(), b_img.copy()], axis=0)).float()
        return l_tensor, ab_tensor

###############################################################################
# 2) (Optional) You can add a simple transform function if you wish, e.g., random flip.
###############################################################################
def simple_flip(img):
    if random.random() > 0.5:
        return np.fliplr(img)
    return img

###############################################################################
# 3) The Colorization Network (Encoder-Decoder with 5 Downsampling & 5 Upsampling Layers)
###############################################################################
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        # Downsampling layers
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )  # (B,64,H,W)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )  # (B,128,H/2,W/2)
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )  # (B,256,H/4,W/4)
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )  # (B,512,H/8,W/8)
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )  # (B,512,H/16,W/16)

        # Upsampling layers
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )  # (B,512,H/8,W/8)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )  # (B,256,H/4,W/4)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )  # (B,128,H/2,W/2)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )  # (B,64,H,W)
        self.up5 = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        )  # (B,2,H,W)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5)
        u2 = self.up2(u1)
        u3 = self.up3(u2)
        u4 = self.up4(u3)
        out = self.up5(u4)
        return out

###############################################################################
# 4) Utility: Convert LAB (L channel and predicted ab) to RGB for Visualization
###############################################################################
def lab_to_rgb(L, ab):
    """
    Expects:
      L:  (B, 1, H, W)
      ab: (B, 2, H, W)
    Returns a list of RGB images (uint8).
    """
    L_np = L.cpu().numpy()
    ab_np = ab.cpu().numpy()
    B = L.shape[0]
    rgb_list = []
    for i in range(B):
        L_i = L_np[i, 0, :, :]
        a_i = ab_np[i, 0, :, :]
        b_i = ab_np[i, 1, :, :]
        lab_img = np.zeros((L_i.shape[0], L_i.shape[1], 3), dtype=np.uint8)
        lab_img[:, :, 0] = np.clip(L_i, 0, 255).astype(np.uint8)
        lab_img[:, :, 1] = np.clip(a_i, 0, 255).astype(np.uint8)
        lab_img[:, :, 2] = np.clip(b_i, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_list.append(rgb)
    return rgb_list

###############################################################################
# 5) Training Routine
###############################################################################
def train_model(model, train_loader, device, epochs=10, lr=1e-3):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        count = 0
        for L_batch, ab_batch in train_loader:
            L_batch, ab_batch = L_batch.to(device), ab_batch.to(device)
            optimizer.zero_grad()
            ab_pred = model(L_batch)
            loss = criterion(ab_pred, ab_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * L_batch.size(0)
            count += L_batch.size(0)
        epoch_loss = total_loss / count
        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f}")

###############################################################################
# 6) Testing Routine: Save All Test Predictions in Folder "colization"
###############################################################################
def test_and_save_predictions(model, test_loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    count = 0
    os.makedirs("colization", exist_ok=True)
    global_idx = 0
    with torch.no_grad():
        for L_batch, ab_batch in test_loader:
            L_batch, ab_batch = L_batch.to(device), ab_batch.to(device)
            ab_pred = model(L_batch)
            loss = criterion(ab_pred, ab_batch)
            total_loss += loss.item() * L_batch.size(0)
            count += L_batch.size(0)
            pred_rgb_list = lab_to_rgb(L_batch, ab_pred)
            for pred_rgb in pred_rgb_list:
                save_path = os.path.join("colization", f"pred_{global_idx:04d}.png")
                cv2.imwrite(save_path, cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
                global_idx += 1
    test_loss = total_loss / count
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Saved {global_idx} prediction images in folder 'colization'.")

###############################################################################
# 7) Main Script
###############################################################################
if __name__ == "__main__":
    # For reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set paths to your L, a, b folders (adjust these paths as needed)
    l_dir = "/Users/ahmed/CU(Tech)/Deep Learning/Project 2/L"
    a_dir = "/Users/ahmed/CU(Tech)/Deep Learning/Project 2/a"
    b_dir = "/Users/ahmed/CU(Tech)/Deep Learning/Project 2/b"

    # Create the dataset from precomputed LAB visualization images
    dataset = ColorizationLABDataset(l_dir=l_dir, a_dir=a_dir, b_dir=b_dir, transform=None)
    n_total = len(dataset)
    print("Total images found in L folder:", n_total)

    # Split into 90% training and 10% testing
    n_train = int(0.9 * n_total)
    n_test = n_total - n_train
    train_set, test_set = random_split(dataset, [n_train, n_test])
    print(f"Train size: {len(train_set)} | Test size: {len(test_set)}")

    # Create DataLoaders
    batch_size = 8
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Instantiate the colorization model and move to device
    model = ColorizationNet().to(device)

    # Train the model
    train_model(model, train_loader, device, epochs=10, lr=1e-3)

    # Evaluate on the test set and save all predictions in the folder "colization"
    test_and_save_predictions(model, test_loader, device)

    print("Colorization pipeline complete.")