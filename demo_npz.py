import cv2
import numpy as np
from run_on_image import inference_on_image_stack
import os
import random
import glob

data_dir = "/home/prakashlab/Documents/kmarx/train_m2unet_cellpose_cloud/data"
model_root = "/home/prakashlab/Documents/kmarx/train_m2unet_cellpose_cloud/m2unet_model_8"
model_name = "model_70_13.pth"
save_dir = "results_erode"
n_im = 10 # set to None to run on all available .npz
randomize = True
center_crop_sz = 512
random.seed(1)

os.makedirs(save_dir, exist_ok=True)

# Get image paths
images = glob.glob(os.path.join(data_dir, "**.npz"), recursive=True)
if randomize:
    random.shuffle(images)
if n_im != None:
    images = images[:n_im]

# assume all images are the same shape - get shape info from first image
data = np.load(images[0])
im = data["img"]
sh = im.shape

image_stack = np.zeros((len(images), sh[0], sh[1]))
mask_stack = np.zeros((len(images), sh[0], sh[1]))
if center_crop_sz > 0:
    x = sh[1]/2 - center_crop_sz/2
    y = sh[0]/2 - center_crop_sz/2
for i, image_path in enumerate(images):
    data = np.load(image_path)
    im = data["img"]
    mask = data["mask"]
    image_stack[i,:,:] = im
    mask_stack[i,:,:] = mask

    # save data as images
    fname = image_path.rsplit('.', 1)[0]
    fname = fname.split('/')[-1]
    print(fname)
    if center_crop_sz > 0:
        im = im[int(y):int(y+center_crop_sz), int(x):int(x+center_crop_sz)]
        mask = mask[int(y):int(y+center_crop_sz), int(x):int(x+center_crop_sz)]

    cv2.imwrite(os.path.join(save_dir, f"{fname}_original.png"), im)
    cv2.imwrite(os.path.join(save_dir,(f"{fname}_mask.png")), mask)

result = inference_on_image_stack(image_stack, model_root, model_name)
result = result.astype(np.uint8)
result = 225 * result / (np.max(result))

for i, segmented in enumerate(result):
    # save data as images
    fname = images[i].rsplit('.', 1)[0]
    fname = fname.split('/')[-1]
    diff = (segmented/np.max(segmented)) - (mask_stack[i,:,:]/np.max(mask_stack))
    color_diff = np.zeros((sh[0], sh[1], 3))
    color_diff[:,:,0] = (255 * (diff < 0)).astype(np.uint8)
    color_diff[:,:,2] = (255 * (diff > 0)).astype(np.uint8)

    if center_crop_sz > 0:
        segmented = segmented[int(y):int(y+center_crop_sz), int(x):int(x+center_crop_sz)]
        color_diff = color_diff[int(y):int(y+center_crop_sz), int(x):int(x+center_crop_sz),:]
    cv2.imwrite(os.path.join(save_dir, f"{fname}_diff.png"), color_diff)
    cv2.imwrite(os.path.join(save_dir, f"{fname}_pred.png"), segmented)