# -- Initial Setup --
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

def augment_img(tensor_img, index, output_dir="augmented"):
    """
    Augments an image
    
    Random cropping is done as follows:
    Randomly choosing a crop window of 80%-100% height/width, and placing it randomly within the full height/width. 
    The top/left value is constrained so that the crop fits inside the image
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
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
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA) #resize back to input size
    
    # RGB value scaling
    rgb_scale = random.uniform(0.6, 1.0) # scale factor used to darken
    img = np.clip(img * rgb_scale, 0, 1)
    
    # For saving only
    img_save = (img * 255).astype(np.uint8)
    cv2.imwrite(f"{output_dir}/aug_{index:04d}.jpg", cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))
    
    return torch.from_numpy(img).permute(2, 0, 1) #reorder back to C, H, W

def rgb_to_lab(rgb_tensor, index):
    """
    Convert RGB to LAB
    """
    
    os.makedirs("augmented_lab", exist_ok=True)
    os.makedirs("L", exist_ok=True)
    os.makedirs("a", exist_ok=True)
    os.makedirs("b", exist_ok=True)
    
    rgb = rgb_tensor.permute(1, 2, 0).numpy() # shape: [H, W, 3]
    rgb = (rgb * 255).astype(np.uint8)
    
    # Convert RGB to LAB
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    
    # Save the full LAB image
    lab_filename = f"augmented_lab/lab_{index:04d}.png"
    cv2.imwrite(lab_filename, lab)
    
    # Split the channels
    L, A, B = cv2.split(lab)
    
    # Save L
    cv2.imwrite(f"L/L_{index:04d}.png", L)
    
    # Create green-magenta a
    lab_a_only = cv2.merge([np.full_like(L, 128), A, np.full_like(B, 128)]) # set L and B to neural constant 128 and keep A
    a_color = cv2.cvtColor(lab_a_only, cv2.COLOR_LAB2RGB)
    cv2.imwrite(f"a/a_{index:04d}.png", cv2.cvtColor(a_color, cv2.COLOR_RGB2BGR))
    
    # Create blue-yellow b
    lab_b_only = cv2.merge([np.full_like(L, 128), np.full_like(A, 128), B])
    b_color = cv2.cvtColor(lab_b_only, cv2.COLOR_LAB2RGB) # true blue→yellow
    cv2.imwrite(f"b/b_{index:04d}.png", cv2.cvtColor(b_color, cv2.COLOR_RGB2BGR))
    
    # convert LAB back to store it
    lab_float = lab.astype(np.float32)
    lab_tensor = torch.from_numpy(lab_float).permute(2, 0, 1)
    
    return lab_tensor

def reset_output_dirs():
    """
    Clears output directories
    """
    dirs = ["augmented", "augmented_lab", "L", "a", "b"]
    for dir_path in dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

def main():
    """
    Main function to run the script
    """
    
    # -- Clear data from previous directories --
    reset_output_dirs()
    
    # -- Loading data --
    base_dir = "/DATA/ahmedabouelnaga/DL-Project-2/"
    img_dir_pattern = os.path.join(base_dir, "face_images/*.jpg")
    files = glob.glob(img_dir_pattern)
    print(f"Found {len(files)} images.")
    
    data = []
    
    # Adds all the pictures to data array
    for fl in files:
        img = cv2.imread(fl)
        if img is None:
            print(f"Warning: Could not read image {fl}")
            continue
        data.append(img)
    
    print(f"Successfully loaded {len(data)} images.")
    
    # -- Creating Tensor --
    tensor_images = []
    for img in data:
        tensor_img = create_tensor_img(img)
        tensor_images.append(tensor_img)
    
    # Turn into pytorch tensor
    data_tensor = torch.stack(tensor_images)
    
    # Randomly shuffle data
    rand = torch.randperm(data_tensor.size(0))
    data_tensor = data_tensor[rand]
    
    print(f"Created tensor with shape: {data_tensor.shape}")
    
    # -- Augmenting dataset --
    augmentation_factor = 10
    augmented_tensor = []
    for i, img in enumerate(data_tensor):
        # create 10 augmentations for each original image
        for j in range(augmentation_factor):
            aug_img = augment_img(img, i * augmentation_factor + j)
            augmented_tensor.append(aug_img)
    
    # Turn into pytorch tensor
    augmented_tensor = torch.stack(augmented_tensor)
    print(f"Created augmented tensor with shape: {augmented_tensor.shape}")
    
    # -- Converting to L* a* b* Color Space --
    lab_tensor = []
    for i, img in enumerate(augmented_tensor):
        lab_img = rgb_to_lab(img, i)
        lab_tensor.append(lab_img)
    
    lab_tensor_stacked = torch.stack(lab_tensor)
    torch.save(lab_tensor_stacked, "lab_tensor.pt")
    
    print(f"Number of images: {len(lab_tensor)}")
    print("All done!")
    print("1) Augmented (RGB) images in 'augmented/'")
    print("2) LAB images in 'augmented_lab/'")
    print("3) L*, a*, b* channel images in 'L/', 'a/', 'b/' respectively.")

if __name__ == "__main__":
    main()
