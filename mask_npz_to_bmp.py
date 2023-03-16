import cv2
import numpy as np
import os
import random
import glob
# Used for converting .npz-stored data to bmp images
# Not necessary for M2U-Net or segmentation!

# convert masks into bmp for easy viewing
data_dir = "path/to/data"
save_dir = "pat/to/destination"
n_im = 10 # set to None to run on all available .npz
randomize = True
center_crop_sz = 0
random.seed(1)

os.makedirs(save_dir, exist_ok=True)

# Get image paths
images = glob.glob(os.path.join(data_dir, "**/**.npz"), recursive=True)
if randomize:
    random.shuffle(images)
if n_im != None:
    images = images[:n_im]

# assume all images are the same shape - get shape info from first image
data = np.load(images[0])
im = data["mask"]
sh = im.shape

if center_crop_sz > 0:
    x = sh[1]/2 - center_crop_sz/2
    y = sh[0]/2 - center_crop_sz/2
for image_path in images:
    data = np.load(image_path)
    mask = data["mask"]
    if center_crop_sz > 0:
        mask = mask[int(y):int(y+center_crop_sz), int(x):int(x+center_crop_sz)]
    mask = (255*mask).astype(np.uint8)
    fname = "_".join(image_path.rsplit('.', 1)[0].split('/')[-2:])
    path = os.path.join(save_dir, f"{fname}.png") 
    cv2.imwrite(path, mask)
