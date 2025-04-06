import os
import glob
import cv2
import torch
import numpy as np


torch.set_default_dtype(torch.float32)

# 1. Load and Shuffle the Original Dataset
img_dir_pattern = "/Users/ahmed/CU(Tech)/Deep Learning/Project 2/face_images/*.jpg"
files = glob.glob(img_dir_pattern)
print(f"Found {len(files)} images.")

target_size = (128, 128)  # Resize all images to 128x128

data_list = []
for f in files:
    img = cv2.imread(f)  # OpenCV loads BGR by default
    if img is None:
        continue
    img_resized = cv2.resize(img, target_size)
    data_list.append(img_resized)

# Convert to numpy array => shape: (N, 128, 128, 3)
data_arr = np.array(data_list)
print("Loaded images shape:", data_arr.shape)

# Convert to torch => shape: (N, 3, 128, 128)
data_tensor = torch.from_numpy(data_arr).permute(0, 3, 1, 2).float()

# Shuffle
n = data_tensor.size(0)
perm = torch.randperm(n)
data_tensor = data_tensor[perm]


# 2) Augment the Dataset
#Horizontal flip (50% chance)
#Random crop -> resize
#Scale RGB by [0.6, 1.0]

augmentation_factor = 10
augmented_images = []  # will hold the augmented images as BGR uint8

def random_crop_and_resize(img_bgr, out_size=(128, 128), min_scale=0.8):
    """
    Randomly crops the image by a scale factor between [min_scale,1.0],
    then resizes back to out_size.
    """
    h, w, _ = img_bgr.shape
    scale = np.random.uniform(min_scale, 1.0)
    new_h = int(h * scale)
    new_w = int(w * scale)
    # random top-left
    max_y = h - new_h
    max_x = w - new_w
    start_y = np.random.randint(0, max_y+1) if max_y > 0 else 0
    start_x = np.random.randint(0, max_x+1) if max_x > 0 else 0
    crop = img_bgr[start_y:start_y+new_h, start_x:start_x+new_w]
    return cv2.resize(crop, out_size)

for i in range(data_tensor.size(0)):
    # Convert to (128,128,3) uint8 for OpenCV
    img_bgr = data_tensor[i].permute(1, 2, 0).numpy().astype(np.uint8)
    
    for _ in range(augmentation_factor):
        aug = img_bgr.copy()
        
        # 2a) Horizontal flip (50% chance)
        if np.random.rand() > 0.5:
            aug = cv2.flip(aug, 1)
        
        # 2b) Random crop -> resize
        aug = random_crop_and_resize(aug, out_size=target_size, min_scale=0.8)
        
        # 2c) Scale RGB by [0.6, 1.0]
        scale_factor = np.random.uniform(0.6, 1.0)
        aug = np.clip(aug * scale_factor, 0, 255).astype(np.uint8)
        
        augmented_images.append(aug)

augmented_images = np.array(augmented_images)  # shape: (N*10, 128, 128, 3)
print("Augmented images shape:", augmented_images.shape)


#3.Save the Augmented Dataset (RGB) BEFORE LAB Conversion
#The instructions want them in a folder named "augmented/"
os.makedirs("augmented", exist_ok=True)
for idx, img_bgr in enumerate(augmented_images):
    cv2.imwrite(f"augmented/aug_{idx:04d}.jpg", img_bgr)

# 4) Convert to L*a*b* and Save the Augmented LAB Dataset
#We also need to create an intensity image for L*, and color mappings for a* (green<->magenta) and b* (blue<->yellow).

os.makedirs("augmented_lab", exist_ok=True)  # Full LAB images
os.makedirs("L", exist_ok=True)
os.makedirs("a", exist_ok=True)
os.makedirs("b", exist_ok=True)

def map_a_channel_to_greens_magenta(a_channel):
    """
    Given the a_channel in [0..255], where 128 is neutral.
    We map -1 => green, +1 => magenta (in a gradient).
    
    Formula:
      val_norm = (a - 128)/128  in [-1..1]
      alpha = (val_norm + 1)/2  in [0..1]
      B = 255 * alpha
      G = 255 * (1 - alpha)
      R = 255 * alpha
    This yields a gradient from green (0,255,0) to magenta (255,0,255).
    """
    a_float = a_channel.astype(np.float32)
    val_norm = (a_float - 128.0) / 128.0  # [-1..1]
    alpha = (val_norm + 1.0) / 2.0        # [0..1]
    
    # Build B, G, R in float [0..255]
    B = 255.0 * alpha
    G = 255.0 * (1.0 - alpha)
    R = 255.0 * alpha
    
    # Stack into BGR image
    out_bgr = np.dstack([B, G, R]).astype(np.uint8)
    return out_bgr

def map_b_channel_to_blue_yellow(b_channel):
    """
    Given the b_channel in [0..255], where 128 is neutral.
    We map -1 => blue, +1 => yellow (in a gradient).
    
    Formula:
      val_norm = (b - 128)/128  in [-1..1]
      alpha = (val_norm + 1)/2  in [0..1]
      B = 255*(1 - alpha)
      G = 255*alpha
      R = 255*alpha
    This yields a gradient from blue (255,0,0) to yellow (0,255,255).
    """
    b_float = b_channel.astype(np.float32)
    val_norm = (b_float - 128.0) / 128.0  # [-1..1]
    alpha = (val_norm + 1.0) / 2.0        # [0..1]
    
    B = 255.0 * (1.0 - alpha)
    G = 255.0 * alpha
    R = 255.0 * alpha
    
    out_bgr = np.dstack([B, G, R]).astype(np.uint8)
    return out_bgr

for idx, img_bgr in enumerate(augmented_images):
    # Convert BGR -> LAB
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    # (Optional) Save the entire LAB image so we have the "augmented dataset after conversion"
    lab_filename = f"augmented_lab/lab_{idx:04d}.png"
    cv2.imwrite(lab_filename, img_lab)
    
    # Split L*, a*, b*
    L_channel, a_channel, b_channel = cv2.split(img_lab)
    
    # --- L* channel as grayscale intensity ---
    # L* is already 0..255 in OpenCV's 8-bit representation
    L_filename = f"L/L_{idx:04d}.png"
    cv2.imwrite(L_filename, L_channel)
    
    # --- a* channel in green <-> magenta ---
    a_vis = map_a_channel_to_greens_magenta(a_channel)
    a_filename = f"a/a_{idx:04d}.png"
    cv2.imwrite(a_filename, a_vis)
    
    # --- b* channel in blue <-> yellow ---
    b_vis = map_b_channel_to_blue_yellow(b_channel)
    b_filename = f"b/b_{idx:04d}.png"
    cv2.imwrite(b_filename, b_vis)
#Just for us to make sure that this worked and fully ran
print("All done!")
print("1) Augmented (RGB) images in 'augmented/'")
print("2) LAB images in 'augmented_lab/'")
print("3) L*, a*, b* channel images in 'L/', 'a/', 'b/' respectively.")
