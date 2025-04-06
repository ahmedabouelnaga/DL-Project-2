import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os

# -- Configuration --

# Set batch size, learning rate, and number of epochs
BATCH_SIZE = 10
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20

# Part 4: GPU Computing code
# Use CUDA if it is available (otherwise we're running on our PCs)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}") # check if CUDA is being used
os.makedirs("colorized_outputs", exist_ok=True) # directory for output

# -- Dataset wrapper class for preparation and easy retrival --

class LabDataset(Dataset):
    def __init__(self, data_tensor):
        # Normalize L* from [0, 100] to [0, 1], ab from [-128, 127] to [-1, 1]
        self.L = data_tensor[:, 0:1] / 100.0
        self.ab = data_tensor[:, 1:3] / 128.0
    
    def __len__(self):
        # returns number of items in dataset
        return self.L.shape[0]
    
    def __getitem__(self, idx):
        # gets a specific sample given the index
        return self.L[idx], self.ab[idx]

# -- Model class --

class ColorizationNet(nn.Module):
    def __init__(self):
        # define network layers
        super(ColorizationNet, self).__init__()
        
        def down_block(in_ch, out_ch):
            # convolution block that downsamples (halves resolution)
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch), # normalizes the output of the previous layer
                nn.ReLU(inplace=True), # introduce non-linearity
                nn.MaxPool2d(2) # reduces spatial resolution by half
            )
        
        def up_block(in_ch, out_ch):
            # convolution block that upsamples (doubles resolution)
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        # Encoder: reduces spatial resolution
        self.encoder = nn.Sequential(
            down_block(1, 64),    # 128 to 64
            down_block(64, 128),  # 64 to 32
            down_block(128, 256), # 32 to 16
            down_block(256, 512), # 16 to 8
            down_block(512, 512)  # 8 to 4
        )
        
        # Decoder: restores resolution while dropping channels
        self.decoder = nn.Sequential(
            up_block(512, 512),  # spatial resolution going from 4 to 8
            up_block(512, 256),  # 8 to 16
            up_block(256, 128),  # 16 to 32
            up_block(128, 64),   # 32 to 64
            up_block(64, 32),    # 64 to 128
            # (batch_size, 32, 128, 128) to (batch_size, 2, 128, 128) for a* and b* channels
            nn.Conv2d(32, 2, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        # Returns the output of encoding and decoding
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # Output: (batch_size, 2, 128, 128)

# -- Helper function to convert LAB to BGR for saving images --
def lab_to_bgr(L, ab):
    L = (L.squeeze().cpu().numpy() * 100).astype(np.uint8)
    ab = (ab.squeeze().cpu().numpy() * 128).astype(np.int8)
    lab = np.zeros((128, 128, 3), dtype=np.uint8)
    lab[:, :, 0] = L
    lab[:, :, 1:] = ab.transpose(1, 2, 0)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

# -- Main function to run the training and testing --
def main():
    
    # Load lab tensor from file
    # Shape: (N, 3, 128, 128) where channel 0 = L*, 1 = a*, 2 = b*
    lab_tensor = torch.load("lab_tensor.pt")
    
    # 90% training, 10% testing
    N = lab_tensor.shape[0]
    n_train = int(0.9 * N) # counts how much data = 90% of it
    perm = torch.randperm(N) # randomize data order
    train_tensor = lab_tensor[perm[:n_train]] # first 90% training
    test_tensor = lab_tensor[perm[n_train:]] # last 10 % testing
    
    # Loaders to yield data in mini batches, shuffles training data
    # pin_memory=True allows faster data transfer to GPU
    train_loader = DataLoader(LabDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(LabDataset(test_tensor), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    # Initialize model, loss function, and optimizer
    model = ColorizationNet().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...\n")
    # pass over the entire dataset num_epochs times
    for epoch in range(NUM_EPOCHS):
        model.train() # Set to training mode
        total_loss = 0
        
        # go over each batch of data to train
        for L_batch, ab_batch in train_loader:
            L_batch, ab_batch = L_batch.to(DEVICE), ab_batch.to(DEVICE) # Use GPU/CPU (whichever we're using)
            optimizer.zero_grad() # Clear gradients
            ab_pred = model(L_batch) # Use greyscale to predict a* and b*
            loss = criterion(ab_pred, ab_batch) # Calculate loss using MSE
            loss.backward() # Compute gradients using model parameters
            optimizer.step() # Update weights
            total_loss += loss.item() # Add loss to total
            avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}\n")
    
    # Save the model
    torch.save(model.state_dict(), "colorization_model.pt")
    print("Model saved as 'colorization_model.pt'")
    
    # Inference
    model.eval() # Set to evaluation mode
    mse = nn.MSELoss() # loss function
    total_mse = 0
    
    print("\nTesting and saving colorized images...")
    
    # Turn off gradients for inference mode, no backpropagation needed and saves memory
    num_images = 0 # total number of images
    with torch.no_grad():
        # Loops through each batch from test set
        for i, (L_batch, ab_true) in enumerate(test_loader):
            L_batch, ab_true = L_batch.to(DEVICE), ab_true.to(DEVICE)
            ab_pred = model(L_batch) # Predict a* and b* channels
            
            # Calculate MSE for this batch
            batch_mse = mse(ab_pred, ab_true).item()
            total_mse += batch_mse * L_batch.size(0) # Multiply by number of images in batch
            num_images += L_batch.size(0)
            
            # Save the images
            for j in range(L_batch.shape[0]):
                img_bgr = lab_to_bgr(L_batch[j], ab_pred[j])
                idx = i * BATCH_SIZE + j
                cv2.imwrite(f"colorized_outputs/img_{idx}.png", img_bgr)
    
    # Report test MSE over all test images
    avg_mse = total_mse / num_images
    print(f"\nTest Set Mean Squared Error: {avg_mse:.4f}")

if __name__ == "__main__":
    main()
