import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random

def fix_randomness(seed_val=42):
    """Set random seeds for reproducibility"""
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ColorPredictor(nn.Module):
    """Neural network to predict color values from grayscale input"""
    
    def __init__(self):
        super(ColorPredictor, self).__init__()
        
        # Encoder network that progressively reduces spatial dimensions
        self.encoder = nn.Sequential(
            # Layer 1: 128x128 -> 64x64
            nn.Conv2d(1, 3, 3, stride=2, padding=1),
            nn.ReLU(),
            
            # Layer 2: 64x64 -> 32x32
            nn.Conv2d(3, 3, 3, stride=2, padding=1),
            nn.ReLU(),
            
            # Layer 3: 32x32 -> 16x16
            nn.Conv2d(3, 3, 3, stride=2, padding=1),
            nn.ReLU(),
            
            # Layer 4: 16x16 -> 8x8
            nn.Conv2d(3, 3, 3, stride=2, padding=1),
            nn.ReLU(),
            
            # Layer 5: 8x8 -> 4x4
            nn.Conv2d(3, 3, 3, stride=2, padding=1),
            nn.ReLU(),
            
            # Layer 6: 4x4 -> 2x2
            nn.Conv2d(3, 3, 3, stride=2, padding=1),
            nn.ReLU(),
            
            # Layer 7: 2x2 -> 1x1
            nn.Conv2d(3, 3, 3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Final prediction layer
        self.predictor = nn.Linear(3, 2)
    
    def forward(self, x):
        features = self.encoder(x)
        flattened = features.view(features.size(0), -1)
        predictions = self.predictor(flattened)
        return predictions

def run_training():
    # Load the previously prepared LAB tensor data
    color_data = torch.load("lab_tensor.pt")
    
    # Fix random seed for reproducibility
    fix_randomness()
    
    # Extract and normalize luminance channel (L*)
    luminance = color_data[:, 0:1, :, :] / 100.0
    
    # Calculate average a* and b* values for each image
    color_targets = color_data[:, 1:, :, :].mean(dim=[2, 3])
    
    # Create dataset
    training_set = TensorDataset(luminance, color_targets)
    
    # Configure mini-batch processing
    batch_size = 16
    data_loader = DataLoader(
        training_set, 
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    # Determine hardware device
    hw_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {hw_device}")
    
    # Create model instance
    net = ColorPredictor().to(hw_device)
    
    # Setup optimizer and loss function
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # Training loop
    epochs = 10
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        
        for inputs, targets in data_loader:
            # Move data to appropriate device
            inputs = inputs.to(hw_device)
            targets = targets.to(hw_device)
            
            # Zero gradients
            opt.zero_grad()
            
            # Forward pass
            outputs = net(inputs)
            
            # Calculate loss
            loss = loss_fn(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            opt.step()
            
            # Accumulate loss
            running_loss += loss.item() * inputs.shape[0]
        
        # Calculate epoch loss
        epoch_loss = running_loss / len(training_set)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
    
    # Save trained model
    torch.save(net.state_dict(), "regressor_model.pt")
    print("Model saved successfully!")

if __name__ == "__main__":
    run_training()
