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
# 1) Dataset with LAB Conversion + Data Augmentation
###############################################################################
class ColorizationDataset(Dataset):
    """
    Reads images from 'image_dir', converts them to LAB, and returns:
      - L channel as input  (shape: 1 x H x W)
      - ab channels as target (shape: 2 x H x W).
    Applies optional data augmentations (random scaling, random flips, etc.).
    """
    def __init__(self, image_dir, transform=None, final_size=128):
        super().__init__()
        # Collect all image files with given extensions
        exts = ["*.jpg", "*.jpeg", "*.png"]
        self.files = []
        for e in exts:
            self.files.extend(glob.glob(os.path.join(image_dir, e)))
        self.transform = transform
        self.final_size = final_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        bgr = cv2.imread(f)  # shape: (H, W, 3) in BGR
        if bgr is None:
            # If file can't be read, pick a fallback
            return self.__getitem__((idx + 1) % len(self.files))

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Apply data augmentation transform if provided
        if self.transform:
            rgb = self.transform(rgb, self.final_size)
        else:
            rgb = cv2.resize(rgb, (self.final_size, self.final_size))

        # Convert to LAB
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        # Split channels: L, a, b
        L = lab[:, :, 0]   # [0..255]
        a = lab[:, :, 1]   # [0..255] with 128 as neutral
        b = lab[:, :, 2]   # [0..255] with 128 as neutral

        # Use .copy() to avoid negative strides
        L_t = torch.from_numpy(L.copy()).unsqueeze(0).float()  # shape: (1, H, W)
        ab_t = torch.from_numpy(np.stack([a.copy(), b.copy()], axis=0)).float()  # shape: (2, H, W)
        return L_t, ab_t

###############################################################################
# 2) Data Augmentation: Random Scaling + Horizontal Flip
###############################################################################
def random_scale_and_flip(rgb, final_size=128, scale_min=0.8, scale_max=1.2):
    """
    1) Randomly scales the input image by a factor in [scale_min, scale_max].
    2) Randomly flips horizontally (50% chance).
    3) Resizes to (final_size, final_size).
    """
    h, w, c = rgb.shape

    # 1) Random scaling
    scale = random.uniform(scale_min, scale_max)
    new_h = int(h * scale)
    new_w = int(w * scale)
    scaled = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 2) Random horizontal flip
    if random.random() > 0.5:
        scaled = np.fliplr(scaled)

    # 3) Resize to final dimensions (can be a center crop or simple resize)
    out = cv2.resize(scaled, (final_size, final_size), interpolation=cv2.INTER_AREA)
    return out

###############################################################################
# 3) The Colorization Network with 5 Downsampling & 5 Upsampling Layers
###############################################################################
class ColorizationNet(nn.Module):
    """
    An encoder-decoder network with 5 downsampling layers and 5 upsampling layers.
    Input:  1-channel L
    Output: 2-channel ab
    Uses Batch Normalization after every conv/deconv.
    """
    def __init__(self):
        super(ColorizationNet, self).__init__()

        # Downsampling layers
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )  # -> (B, 64, H, W)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )  # -> (B, 128, H/2, W/2)
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )  # -> (B, 256, H/4, W/4)
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )  # -> (B, 512, H/8, W/8)
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )  # -> (B, 512, H/16, W/16)

        # Upsampling layers
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )  # -> (B, 512, H/8, W/8)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )  # -> (B, 256, H/4, W/4)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )  # -> (B, 128, H/2, W/2)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )  # -> (B, 64, H, W)
        self.up5 = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        )  # -> (B, 2, H, W)

    def forward(self, x):
        d1 = self.down1(x)   # (B, 64, H, W)
        d2 = self.down2(d1)  # (B, 128, H/2, W/2)
        d3 = self.down3(d2)  # (B, 256, H/4, W/4)
        d4 = self.down4(d3)  # (B, 512, H/8, W/8)
        d5 = self.down5(d4)  # (B, 512, H/16, W/16)

        u1 = self.up1(d5)    # (B, 512, H/8, W/8)
        u2 = self.up2(u1)    # (B, 256, H/4, W/4)
        u3 = self.up3(u2)    # (B, 128, H/2, W/2)
        u4 = self.up4(u3)    # (B, 64, H, W)
        out = self.up5(u4)   # (B, 2, H, W)
        return out

###############################################################################
# 4) Utility: Convert LAB (L, ab) to RGB for Visualization
###############################################################################
def lab_to_rgb(L, ab):
    """
    Converts L and ab tensors to a list of RGB images (uint8).
    L:  (B, 1, H, W)
    ab: (B, 2, H, W)
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

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        count = 0
        for L_batch, ab_batch in train_loader:
            L_batch, ab_batch = L_batch.to(device), ab_batch.to(device)
            optimizer.zero_grad()
            ab_pred = model(L_batch)  # (B,2,H,W)
            loss = criterion(ab_pred, ab_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * L_batch.size(0)
            count += L_batch.size(0)
        epoch_loss = total_loss / count
        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f}")

###############################################################################
# 6) Testing Routine: Save All Predictions in Folder "colization"
###############################################################################
def test_and_save_predictions(model, test_loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    count = 0

    # Create folder for saving predictions
    os.makedirs("colization", exist_ok=True)
    global_idx = 0

    with torch.no_grad():
        for L_batch, ab_batch in test_loader:
            L_batch, ab_batch = L_batch.to(device), ab_batch.to(device)
            ab_pred = model(L_batch)
            loss = criterion(ab_pred, ab_batch)
            total_loss += loss.item() * L_batch.size(0)
            count += L_batch.size(0)

            # Convert predicted LAB to RGB for visualization
            pred_rgb_list = lab_to_rgb(L_batch, ab_pred)
            for pred_rgb in pred_rgb_list:
                save_path = os.path.join("colization", f"pred_{global_idx:04d}.png")
                # Save as PNG (convert RGB -> BGR for OpenCV)
                cv2.imwrite(save_path, cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
                global_idx += 1

    test_loss = total_loss / count
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Saved {global_idx} prediction images in folder 'colization'.")

###############################################################################
# 7) Main Script: Train and Test the Model, Save Predictions
###############################################################################
if __name__ == "__main__":
    # For reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path to your dataset directory (adjust as needed)
    image_dir = "/Users/ahmed/CU(Tech)/Deep Learning/Project 2/face_images"

    # Create dataset with data augmentation
    dataset = ColorizationDataset(
        image_dir=image_dir,
        transform=random_scale_and_flip,  # Data augmentation
        final_size=128
    )
    n_total = len(dataset)
    print("Total images found:", n_total)

    # Split: 90% training, 10% testing
    n_train = int(0.9 * n_total)
    n_test  = n_total - n_train
    train_set, test_set = random_split(dataset, [n_train, n_test])
    print(f"Train size: {len(train_set)} | Test size: {len(test_set)}")

    # DataLoaders with mini-batches
    batch_size = 8
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Instantiate the colorization model and move to device
    model = ColorizationNet().to(device)

    # Train the model
    train_model(model, train_loader, device, epochs=10, lr=1e-3)

    # Evaluate and save all predictions as PNG in folder "colization"
    test_and_save_predictions(model, test_loader, device)

    print("Colorization pipeline complete.")