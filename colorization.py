import cv2
import os
import glob
import torch
import numpy as np
import random
import shutil

# torch.set_default_tensor_type('torch.FloatTensor') # Deprecated
torch.set_default_dtype(torch.float32)

def create_tensor_img(img):
    """
    Prepares image to be put into tensor
    """
    img = cv2.resize(img, (128, 128)) # resize to 128 x 128
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB
    img = img.astype(np.float32) / 255.0 # Normalize to [0, 1]
    img = torch.from_numpy(img).permute(2, 0, 1) # convert to tensor from numpy, change the order of the dimensions from from (H, W, C) to (C, H, W)
    return img

def augment_img(tensor_img, index, output_dir = "AUG_pics"):
    """
    Augments an image
    
    Random cropping is done as follows:
    Randomly choosing a crop window of 80%-100% height/width, and placing it randomly within the full height/width. The top/left value is constrained so that the crop fits inside the image
    
    """
    
    os.makedirs(os.path.join(output_dir, "augmented_pics"), exist_ok=True)
    
    img = tensor_img.permute(1, 2, 0).numpy() # tensor to numpy with columns ordered as H, W, C
    
    # Random horizontal flip (50/50)
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    
    # Random cropping
    h, w, _ = img.shape
    crop_scale = random.uniform(0.8, 1.0)
    new_h, new_w = int(h * crop_scale), int(w * crop_scale)
    top = random.randint(0, h - new_h) # picks new starting point from top for crop
    left = random.randint(0, w - new_w) # picks new starting point from left for crop
    img = img[top:top+new_h, left:left+new_w] # crop the data
    img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA) #resize back to input size
    
    # RGB value scaling
    rgb_scale = random.uniform(0.6, 1.0) # scale factor used to darken
    img = np.clip(img * rgb_scale, 0, 1)
    
    # For saving only
    img_save = np.clip(img * rgb_scale * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{output_dir}/augmented_pics/image_{index}.png", cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))
    
    return torch.from_numpy(img).permute(2, 0, 1) #reorder back to C, H, W

def rgb_to_lab(rgb_tensor, index, output_dir="LAB_pics"):
    """
    Convert RGB to LAB
    """
    
    os.makedirs(os.path.join(output_dir, "L"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "a"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "b"), exist_ok=True)
    
    rgb = rgb_tensor.permute(1, 2, 0).numpy() # shape: [H, W, 3]
    rgb = (rgb * 255).astype(np.uint8)
    
    # Convert BGR to LAB
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    
    # Split the channels
    L, A, B = cv2.split(lab)
    
    # Save L
    cv2.imwrite(f"{output_dir}/L/image_{index}.png", L)
    
    # Create green-magenta a
    lab_a_only = cv2.merge([np.full_like(L, 128), A, np.full_like(B, 128)]) # set L and B to neural constant 128 and keep A
    a_color = cv2.cvtColor(lab_a_only, cv2.COLOR_LAB2RGB)
    cv2.imwrite(f"{output_dir}/a/image_{index}.png", cv2.cvtColor(a_color, cv2.COLOR_RGB2BGR))
    
    # Create blue-yellow b
    lab_b_only = cv2.merge([np.full_like(L, 128), np.full_like(A, 128), B])
    b_color = cv2.cvtColor(lab_b_only, cv2.COLOR_LAB2RGB) # true blue→yellow
    cv2.imwrite(f"{output_dir}/b/image_{index}.png", cv2.cvtColor(b_color, cv2.COLOR_RGB2BGR))
    
    # convert LAB back to store it
    lab_float = lab.astype(np.float32)
    lab_tensor = torch.from_numpy(lab_float).permute(2, 0, 1)
    
    return lab_tensor

def reset_output_dir(path):
    """
    Clears the output folder
    """
    if os.path.exists(path):
        shutil.rmtree(path) # Delete the folder and all contents
    os.makedirs(path, exist_ok=True)  # Recreate it empty

def main():
    """
    Main function to run the script
    """
    
    # -- Clear data from previous directories --
    reset_output_dir("LAB_pics")
    reset_output_dir("LAB_pics/L")
    reset_output_dir("LAB_pics/a")
    reset_output_dir("LAB_pics/b")
    reset_output_dir("AUG_pics")
    reset_output_dir("AUG_pics/augmented_pics")
    
    # -- Loading data --
    img_dir = "/Users/ahmed/CU(Tech)/Deep Learning/Project 2/DL-Project-2/face_images" #path to images DIR
    
    # Check if directory exists, if not try an alternative path
    if not os.path.exists(img_dir):
        print(f"Warning: Directory {img_dir} not found.")
        # Try looking in current directory or parent directory
        alternative_paths = [
            "face_images",
            "DL-Project-2/face_images",
            "../datasets/face_images/face_images",
            "."
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                img_dir = path
                print(f"Using alternative directory: {img_dir}")
                break
    
    # Search for image files with multiple extensions
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        files.extend(glob.glob(os.path.join(img_dir, ext)))
    
    if not files:
        raise RuntimeError(f"No image files found in {img_dir}. Please check the path and file extensions.")
    
    print(f"Found {len(files)} image files")
    
    data = []
    tensor_images = []
    
    # Process images one by one and handle errors
    for fl in files:
        try:
            img = cv2.imread(fl)
            if img is None:
                print(f"Warning: Could not load image {fl}")
                continue
                
            tensor_img = create_tensor_img(img)
            tensor_images.append(tensor_img)
            data.append(img)  # Only append if successful
        except Exception as e:
            print(f"Error processing {fl}: {e}")
    
    if not tensor_images:
        raise RuntimeError("No images could be processed. Check file formats and permissions.")
    
    print(f"Successfully processed {len(tensor_images)} images")
    
    # Turn into pytorch tensor
    data_tensor = torch.stack(tensor_images)
    
    # Randomly shuffle data
    rand = torch.randperm(data_tensor.size(0))
    data_tensor = data_tensor[rand]
    
    # -- Augmenting dataset --
    augmented_tensor = []
    for i, img in enumerate(data_tensor):
        # create 10 augmentations for each original image
        for j in range(10):
            aug_img = augment_img(img, i * 10 + j)
            augmented_tensor.append(aug_img)
    
    # Turn into pytorch tensor
    augmented_tensor = torch.stack(augmented_tensor)
    
    # -- Converting to L* a* b* Color Space --
    lab_tensor = []
    for i, img in enumerate(augmented_tensor):
        lab_img = rgb_to_lab(img, i)
        lab_tensor.append(lab_img)
    
    lab_tensor_stacked = torch.stack(lab_tensor)
    torch.save(lab_tensor_stacked, "lab_tensor.pt")
    
    print("no. images: " + str(len(lab_tensor)))
    print("DONE!")

if __name__ == "__main__":
    main()
