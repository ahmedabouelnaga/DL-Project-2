import os
import glob
import cv2
import torch
import numpy as np
import random

base_path = "/content/drive/MyDrive/Deep Learning Project 2/"

torch.set_default_tensor_type(torch.FloatTensor)

IMG_DIR = os.path.join(base_path, "face_images/*.jpg")
files = glob.glob(IMG_DIR)

images_np = []
for fname in files:
    bgr_img = cv2.imread(fname)
    if bgr_img is None:
        continue
    
    bgr_img_128 = cv2.resize(bgr_img, (128, 128), interpolation=cv2.INTER_AREA)
    
    bgr_img_128 = np.transpose(bgr_img_128, (2, 0, 1))
    images_np.append(bgr_img_128)

if not images_np:
    raise ValueError("No images found. Check your IMG_DIR path.")

images_np = np.stack(images_np, axis=0)
images_torch = torch.from_numpy(images_np)
n_images = images_torch.shape[0]

perm_idx = torch.randperm(n_images)
images_torch = images_torch[perm_idx]

factor = 10
aug_images = torch.zeros((n_images * factor, 3, 128, 128), dtype=torch.float32)

def random_horizontal_flip(img_tensor):
    if random.random() < 0.5:
        return torch.flip(img_tensor, dims=[2])
    return img_tensor

def random_crop_and_resize(img_tensor, crop_percent=0.9):
    _, h, w = img_tensor.shape
    new_h = int(crop_percent * h)
    new_w = int(crop_percent * w)
    
    y0 = random.randint(0, h - new_h)
    x0 = random.randint(0, w - new_w)
    
    cropped = img_tensor[:, y0:y0+new_h, x0:x0+new_w]
    cropped_np = cropped.permute(1, 2, 0).numpy()
    resized_np = cv2.resize(cropped_np, (128, 128), interpolation=cv2.INTER_AREA)
    resized_torch = torch.from_numpy(np.transpose(resized_np, (2, 0, 1)))
    return resized_torch

def random_brightness_scale(img_tensor, min_scale=0.6, max_scale=1.0):
    scale = random.uniform(min_scale, max_scale)
    return img_tensor * scale

idx = 0
for i in range(n_images):
    original = images_torch[i]
    for _ in range(factor):
        aug = random_horizontal_flip(original)
        aug = random_crop_and_resize(aug, crop_percent=0.9)
        aug = random_brightness_scale(aug, 0.6, 1.0)
        aug = torch.clamp(aug, 0.0, 255.0)
        aug_images[idx] = aug
        idx += 1

print("Augmented dataset shape:", aug_images.shape)

aug_lab = torch.zeros_like(aug_images)

def bgr_to_lab(img_tensor):
    bgr_np = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
    lab_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2LAB)
    lab_tensor = torch.from_numpy(np.transpose(lab_np, (2, 0, 1))).float()
    return lab_tensor

for i in range(aug_images.shape[0]):
    lab_img = bgr_to_lab(aug_images[i])
    aug_lab[i] = lab_img

orig_dir = os.path.join(base_path, "augmented_original")
L_dir = os.path.join(base_path, "augmented_L")
a_dir = os.path.join(base_path, "augmented_a")
b_dir = os.path.join(base_path, "augmented_b")

os.makedirs(orig_dir, exist_ok=True)
os.makedirs(L_dir, exist_ok=True)
os.makedirs(a_dir, exist_ok=True)
os.makedirs(b_dir, exist_ok=True)

def save_rgb_image_from_bgr_tensor(bgr_tensor, path):
    bgr_np = bgr_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, rgb_np)

def save_grayscale_image(channel_tensor, path):
    channel_np = channel_tensor.cpu().numpy().astype(np.uint8)
    cv2.imwrite(path, channel_np)

def save_channel_with_colormap(channel_tensor, path):
    channel_np = channel_tensor.cpu().numpy().astype(np.uint8)
    colored = cv2.applyColorMap(channel_np, cv2.COLORMAP_JET)
    cv2.imwrite(path, colored)

for i in range(aug_images.shape[0]):
    orig_path = os.path.join(orig_dir, f"aug_{i:05d}.png")
    save_rgb_image_from_bgr_tensor(aug_images[i], orig_path)
    
    L_channel = aug_lab[i, 0, :, :]
    a_channel = aug_lab[i, 1, :, :]
    b_channel = aug_lab[i, 2, :, :]
    
    L_path = os.path.join(L_dir, f"L_{i:05d}.png")
    save_grayscale_image(L_channel, L_path)
    
    a_path = os.path.join(a_dir, f"a_{i:05d}.png")
    b_path = os.path.join(b_dir, f"b_{i:05d}.png")
    save_channel_with_colormap(a_channel, a_path)
    save_channel_with_colormap(b_channel, b_path)

print("All done!")
print("Augmented images in RGB are in:", orig_dir)
print("L channel images are in:", L_dir)
print("a channel images are in:", a_dir)
print("b channel images are in:", b_dir)