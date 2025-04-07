from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob

class GrayscaleImageDataset(Dataset):
   def __init__(self, root_directory):
       # Find all JPEG images in the directory and subdirectories
       self.image_paths = glob.glob(os.path.join(root_directory, "**/*.jpg"), recursive=True)
       
       # Define transformations: convert to grayscale, resize, and convert to tensor
       self.transform = transforms.Compose([
           transforms.Grayscale(num_output_channels=1),  # Convert to single-channel grayscale
           transforms.Resize((128, 128)),                # Resize all images to 128x128
           transforms.ToTensor(),                        # Convert to PyTorch tensor (scales to [0, 1])
       ])
   def __len__(self):
       return len(self.image_paths)
   def __getitem__(self, idx):
       # Load image
       image_path = self.image_paths[idx]
       image = Image.open(image_path).convert("RGB")
       # Apply transformations
       grayscale_tensor = self.transform(image)
       # Extract class name from parent folder
       class_name = os.path.basename(os.path.dirname(image_path))
       return grayscale_tensor, class_name
