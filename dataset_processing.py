import os
import glob
import cv2
import torch
import numpy as np
from pathlib import Path

torch.set_default_dtype(torch.float32)

# 1) Import and Randomize the Source Dataset
image_pattern = "/DATA/ahmedabouelnaga/DL-Project-2/face_images/*.jpg"
image_files = glob.glob(image_pattern)
print(f"Located {len(image_files)} facial images.")

resize_dimensions = (128, 128)  # Standard size for all images

image_collection = []
for image_path in image_files:
    image = cv2.imread(image_path)  # CV2 uses BGR format
    if image is None:
        print(f"Warning: Could not load {image_path}")
        continue
    normalized_image = cv2.resize(image, resize_dimensions)
    image_collection.append(normalized_image)

# Transform to numpy format => dimensions: (N, 128, 128, 3)
image_array = np.array(image_collection)
print("Dataset dimensions:", image_array.shape)

# Convert to PyTorch tensor => dimensions: (N, 3, 128, 128)
image_tensor = torch.from_numpy(image_array).permute(0, 3, 1, 2).float()

# Randomize order
sample_count = image_tensor.size(0)
random_indices = torch.randperm(sample_count)
image_tensor = image_tensor[random_indices]

# 2) Expand the Dataset with Augmentation

expansion_multiplier = 10
enhanced_images = []  # will store the augmented images as BGR uint8

def crop_randomly_and_resize(image_bgr, target_dimensions=(128, 128), scale_min=0.8):
    """
    Creates a random crop at a scale between [scale_min,1.0],
    then resizes to target_dimensions.
    """
    height, width, _ = image_bgr.shape
    scale_factor = np.random.uniform(scale_min, 1.0)
    crop_height = int(height * scale_factor)
    crop_width = int(width * scale_factor)
    # Calculate valid position ranges
    y_range = height - crop_height
    x_range = width - crop_width
    top = np.random.randint(0, y_range+1) if y_range > 0 else 0
    left = np.random.randint(0, x_range+1) if x_range > 0 else 0
    cropped = image_bgr[top:top+crop_height, left:left+crop_width]
    return cv2.resize(cropped, target_dimensions)

for idx in range(image_tensor.size(0)):
    # Transform to (128,128,3) uint8 for CV2 operations
    source_image = image_tensor[idx].permute(1, 2, 0).numpy().astype(np.uint8)
    
    for i in range(expansion_multiplier):
        modified = source_image.copy()
        
        # 2a) Mirror flip (50% probability)
        if np.random.random() > 0.5:
            modified = cv2.flip(modified, 1)
        
        # 2b) Dynamic cropping with resize
        modified = crop_randomly_and_resize(modified, target_dimensions=resize_dimensions, scale_min=0.8)
        
        # 2c) Adjust RGB intensity by [0.6, 1.0]
        intensity_factor = np.random.uniform(0.6, 1.0)
        modified = np.clip(modified * intensity_factor, 0, 255).astype(np.uint8)
        
        enhanced_images.append(modified)

enhanced_images = np.array(enhanced_images)  # dimensions: (N*10, 128, 128, 3)
print("Augmented dataset dimensions:", enhanced_images.shape)

# 3) Store the Enhanced Dataset (RGB) BEFORE LAB Transformation
#    Save to directory named "augmented/"
Path("augmented").mkdir(exist_ok=True)
for i, img_bgr in enumerate(enhanced_images):
    cv2.imwrite(f"augmented/enhanced_{i:04d}.jpg", img_bgr)

# 4) Transform to L*a*b* and Store the Enhanced LAB Dataset
#    Create separate channels for L* (intensity), 
#    a* (green-magenta spectrum) and b* (blue-yellow spectrum)
Path("augmented_lab").mkdir(exist_ok=True)  # Complete LAB images
Path("L").mkdir(exist_ok=True)
Path("a").mkdir(exist_ok=True)
Path("b").mkdir(exist_ok=True)

def visualize_a_channel(a_channel):
    """
    Converts a_channel [0..255] to a visual representation
    where 128 is neutral, <128 is green, >128 is magenta.
    
    Calculation:
      normalized = (a - 128)/128  range [-1..1]
      alpha = (normalized + 1)/2  range [0..1]
      Blue = 255 * alpha
      Green = 255 * (1 - alpha)
      Red = 255 * alpha
    Creates gradient: green (0,255,0) -> magenta (255,0,255)
    """
    a_normalized = (a_channel.astype(np.float32) - 128.0) / 128.0  # [-1..1]
    alpha = (a_normalized + 1.0) / 2.0  # [0..1]
    
    # Create BGR channels [0..255]
    blue = 255.0 * alpha
    green = 255.0 * (1.0 - alpha)
    red = 255.0 * alpha
    
    # Combine into BGR image
    visualization = np.dstack([blue, green, red]).astype(np.uint8)
    return visualization

def visualize_b_channel(b_channel):
    """
    Converts b_channel [0..255] to a visual representation
    where 128 is neutral, <128 is blue, >128 is yellow.
    
    Calculation:
      normalized = (b - 128)/128  range [-1..1]
      alpha = (normalized + 1)/2  range [0..1]
      Blue = 255 * (1 - alpha)
      Green = 255 * alpha
      Red = 255 * alpha
    Creates gradient: blue (255,0,0) -> yellow (0,255,255)
    """
    b_normalized = (b_channel.astype(np.float32) - 128.0) / 128.0  # [-1..1]
    alpha = (b_normalized + 1.0) / 2.0  # [0..1]
    
    blue = 255.0 * (1.0 - alpha)
    green = 255.0 * alpha
    red = 255.0 * alpha
    
    visualization = np.dstack([blue, green, red]).astype(np.uint8)
    return visualization

# Create a list to store LAB tensors
lab_tensor_list = []

for i, img_bgr in enumerate(enhanced_images):
    # Transform BGR -> LAB
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    # Save complete LAB image
    lab_filename = f"augmented_lab/lab_{i:04d}.png"
    cv2.imwrite(lab_filename, img_lab)
    
    # Extract L*, a*, b* channels
    L_channel, a_channel, b_channel = cv2.split(img_lab)
    
    # --- Save L* channel as grayscale ---
    # L* already in 0..255 range in OpenCV
    L_filename = f"L/L_{i:04d}.png"
    cv2.imwrite(L_filename, L_channel)
    
    # --- Save a* channel with green-magenta visualization ---
    a_visual = visualize_a_channel(a_channel)
    a_filename = f"a/a_{i:04d}.png"
    cv2.imwrite(a_filename, a_visual)
    
    # --- Save b* channel with blue-yellow visualization ---
    b_visual = visualize_b_channel(b_channel)
    b_filename = f"b/b_{i:04d}.png"
    cv2.imwrite(b_filename, b_visual)
    
    # Convert LAB to tensor and add to list
    lab_float = img_lab.astype(np.float32)
    lab_tensor = torch.from_numpy(lab_float).permute(2, 0, 1)
    lab_tensor_list.append(lab_tensor)

# Stack all LAB tensors into a single tensor and save to file
lab_tensor_stacked = torch.stack(lab_tensor_list)
torch.save(lab_tensor_stacked, "lab_tensor.pt")
print(f"Created lab_tensor.pt with shape: {lab_tensor_stacked.shape}")

print("Processing complete!")
print("Enhanced RGB images saved to 'augmented/'")
print("LAB converted images saved to 'augmented_lab/'") 
print("Channel visualizations saved to 'L/', 'a/', and 'b/' folders.")
print("LAB tensor saved as 'lab_tensor.pt'")
