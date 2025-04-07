import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os

BS = 10
LR = 1e-3
MAX_ITER = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
os.makedirs("colorization_output", exist_ok=True)

class ColorDataset(Dataset):
    def __init__(self, data_tensor):
        self.gray = data_tensor[:, 0:1] / 100.0
        self.color = data_tensor[:, 1:3] / 128.0
    
    def __len__(self):
        return self.gray.shape[0]
    
    def __getitem__(self, idx):
        return self.gray[idx], self.color[idx]

class ImgColorizer(nn.Module):
    def __init__(self):
        super(ImgColorizer, self).__init__()
        
        def down_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
        
        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        self.encoder = nn.Sequential(
            down_block(1, 64),
            down_block(64, 128),
            down_block(128, 256),
            down_block(256, 512),
            down_block(512, 512)
        )
        
        self.decoder = nn.Sequential(
            up_block(512, 512),
            up_block(512, 256),
            up_block(256, 128),
            up_block(128, 64),
            up_block(64, 32),
            nn.Conv2d(32, 2, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def convert_to_image(lumi, chroma):
    lumi = (lumi.squeeze().cpu().numpy() * 100).astype(np.uint8)
    chroma = (chroma.squeeze().cpu().numpy() * 128).astype(np.int8)
    lab_img = np.zeros((128, 128, 3), dtype=np.uint8)
    lab_img[:, :, 0] = lumi
    lab_img[:, :, 1:] = chroma.transpose(1, 2, 0)
    image = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    return image

def main():
    data = torch.load("lab_tensor.pt")
    
    total_samples = data.shape[0]
    train_size = int(0.9 * total_samples)
    shuffle_idx = torch.randperm(total_samples)
    train_data = data[shuffle_idx[:train_size]]
    eval_data = data[shuffle_idx[train_size:]]
    
    train_loader = DataLoader(ColorDataset(train_data), batch_size=BS, shuffle=True, pin_memory=True)
    eval_loader = DataLoader(ColorDataset(eval_data), batch_size=BS, shuffle=False, pin_memory=True)
    
    net = ImgColorizer().to(DEVICE)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    
    print("Beginning training process...\n")
    for epoch in range(MAX_ITER):
        net.train()
        epoch_loss = 0
        
        for gray_in, color_gt in train_loader:
            gray_in, color_gt = gray_in.to(DEVICE), color_gt.to(DEVICE)
            opt.zero_grad()
            color_pred = net(gray_in)
            err = loss_fn(color_pred, color_gt)
            err.backward()
            opt.step()
            epoch_loss += err.item()
        
        mean_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{MAX_ITER}], Error: {mean_loss:.4f}\n")
    
    torch.save(net.state_dict(), "colorization.pt")
    print("Neural network saved as 'colorization.pt'")
    
    net.eval()
    error_calc = nn.MSELoss()
    cumulative_error = 0
    
    print("\nEvaluating model and generating colorized results...")
    
    image_count = 0
    with torch.no_grad():
        for batch_idx, (grayscale, true_color) in enumerate(eval_loader):
            grayscale, true_color = grayscale.to(DEVICE), true_color.to(DEVICE)
            pred_color = net(grayscale)
            
            batch_error = error_calc(pred_color, true_color).item()
            cumulative_error += batch_error * grayscale.size(0)
            image_count += grayscale.size(0)
            
            for img_idx in range(grayscale.shape[0]):
                output_img = convert_to_image(grayscale[img_idx], pred_color[img_idx])
                result_idx = batch_idx * BS + img_idx
                cv2.imwrite(f"colorized_Output/result_{result_idx}.png", output_img)
    # Make sure image_count is not zero to avoid division by zero error
    if image_count > 0:
        final_error = cumulative_error / image_count
        print(f"\nValidation Set Average Error: {final_error:.4f}")
    else:
        print("\nNo validation images were processed. Check your dataset.")

if __name__ == "__main__":
    main()
