import torch
import cv2
import os
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from colorization import ImgColorizer, convert_to_image

class NCDGrayDataset(Dataset):
    def __init__(self, root_folder):
        # Recursively collect all JPG images in subfolders
        self.image_paths = []
        
        # List of fruits and vegetables from the folder structure
        categories = [
            "Apple", "Banana", "Brinjal", "Broccoli", "CapsicumGreen", 
            "Carrot", "Cherry", "ChilliGreen", "Corn", "Cucumber", 
            "LadyFinger", "Lemon", "Orange", "Peach", "Pear", 
            "Plum", "Pomegranate", "Potato", "Strawberry", "Tomato"
        ]
        
        # Build the image paths manually
        for category in categories:
            category_path = os.path.join(root_folder, category)
            if os.path.exists(category_path):
                # Get all jpg files in this category folder
                jpg_files = glob.glob(os.path.join(category_path, "*.jpg"))
                self.image_paths.extend(jpg_files)
                print(f"Found {len(jpg_files)} images in {category}")
            else:
                print(f"Warning: Category folder not found: {category_path}")
        
        print(f"Total images found: {len(self.image_paths)}")
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),  # Scales to [0, 1]
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            L_tensor = self.transform(img)
            
            # Get the class name from the folder name
            class_name = os.path.basename(os.path.dirname(img_path))
            
            return L_tensor, class_name, img_path
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            # Return a placeholder if there's an error
            placeholder = torch.zeros(1, 128, 128)
            return placeholder, "error", img_path

def get_datasets_for_transfer(root_folder, batch_size=16):
    """
    Create and return a DataLoader for the NCD grayscale dataset
    """
    print(f"Creating dataset from root folder: {root_folder}")
    dataset = NCDGrayDataset(root_folder)
    
    if len(dataset) == 0:
        print("WARNING: Dataset is empty! Check the path and image files.")
        return None
        
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    return dataloader

def transfer(model, dataloader, device):
    if dataloader is None:
        print("Error: No dataloader provided. Cannot perform transfer.")
        return
        
    model.eval()
    count = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            L_batch, class_names, img_paths = batch
            L_batch = L_batch.to(device)
            
            # Skip any error batches
            if "error" in class_names:
                print("Skipping batch with errors...")
                continue
                
            ab_pred = model(L_batch)
            
            print(f"Processing batch {i+1} with {L_batch.size(0)} images")
            
            for j in range(L_batch.size(0)):
                L_cpu = L_batch[j].cpu()
                ab_cpu = ab_pred[j].cpu()
                pred_bgr = convert_to_image(L_cpu, ab_cpu)
                
                class_folder = class_names[j]
                orig_filename = os.path.basename(img_paths[j])  # e.g., 'image_021.jpg'
                new_filename = os.path.splitext(orig_filename)[0] + '_colorized.png'  # 'image_021_colorized.png'
                
                save_path = f"NCDDataset/{class_folder}"
                os.makedirs(save_path, exist_ok=True)
                
                output_path = os.path.join(save_path, new_filename)
                cv2.imwrite(output_path, pred_bgr)
                print(f"Saved image to {output_path}")
                
                count += 1
    
    print(f"\nFinished inference on NCD grayscale dataset. Saved {count} colorized images to NCDDataset directory.")

if __name__ == '__main__':
    # Define the root directory - try to use an absolute path if possible
    root_dir = "Gray"  # Simplified path - adjust as needed
    
    # Check if the directory exists
    if not os.path.exists(root_dir):
        print(f"ERROR: Root directory '{root_dir}' not found!")
        alternative_paths = [
            "DATA/ahmedabouelnaga/DL-Project-2/Gray",
            "./Gray",
            "../Gray",
            "./DATA/ahmedabouelnaga/DL-Project-2/Gray"
        ]
        
        # Try alternative paths
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                root_dir = alt_path
                print(f"Using alternative path: {root_dir}")
                break
        else:
            print("Could not find a valid path for the Gray directory!")
            print("Please make sure the directory exists or specify the full path.")
            exit(1)
    
    # Create the output directory
    os.makedirs("NCDDataset", exist_ok=True)
    
    # Set device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    try:
        model = ImgColorizer().to(device)
        model.load_state_dict(torch.load("colorization_model.pt"))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    # Create dataloader and run inference
    ncd_loader = get_datasets_for_transfer(root_dir)
    transfer(model, ncd_loader, device)
