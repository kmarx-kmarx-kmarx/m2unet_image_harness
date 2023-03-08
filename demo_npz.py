import cv2
import numpy as np
from run_on_image import inference_on_image_stack
import os
import random
import glob
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F

def main():
    data_dir = "/home/prakashlab/Documents/kmarx/train_m2unet_cellpose_cloud/data_sbc_validation"
    model_root = "/home/prakashlab/Documents/kmarx/train_m2unet_cellpose_cloud/m2unet_model_flat_less_erode_2_rotfirst"
    model_name = "model_24_9.pth"
    save_dir = "results_flat_erode_rotfirst_with_eroded_masks_final"
    n_im = 50 # set to None to run on all available .npz
    randomize = True
    erode_mask = 1
    center_crop_sz = 512
    random.seed(3)

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
    t0 = time.time()
    for i, image_path in enumerate(tqdm(images)):
        data = np.load(image_path)
        im = data["img"]
        mask = data["mask"]
        image_stack[i,:,:] = im
        mask_stack[i,:,:] = mask

        # save data as images
        fname = image_path.rsplit('.', 1)[0]
        fname = fname.split('/')[-1]

        if center_crop_sz > 0:
            im = im[int(y):int(y+center_crop_sz), int(x):int(x+center_crop_sz)]
            mask = mask[int(y):int(y+center_crop_sz), int(x):int(x+center_crop_sz)]

        if erode_mask > 0:
            shape = cv2.MORPH_ELLIPSE
            element = cv2.getStructuringElement(shape, (2 * erode_mask + 1, 2 * erode_mask + 1), (erode_mask, erode_mask))
            mask = np.array(cv2.erode(mask, element))

        # cv2.imwrite(os.path.join(save_dir, f"{fname}_original.png"), im)
        # cv2.imwrite(os.path.join(save_dir,(f"{fname}_mask.png")), mask)
    print(image_stack.shape)
    t0 = time.time()
    result, __, times = inference_on_image_stack(image_stack, model_root, model_name)
    print(np.median(times))
    print(time.time()-t0)
    result=result.astype(np.uint8)


    for i, segmented in enumerate(tqdm(result)):
        # save data as images
        if center_crop_sz > 0:
            segmented = segmented[int(y):int(y+center_crop_sz), int(x):int(x+center_crop_sz)]
            mask = mask_stack[i, int(y):int(y+center_crop_sz), int(x):int(x+center_crop_sz)]
            sh = segmented.shape
        if erode_mask > 0:
            shape = cv2.MORPH_ELLIPSE
            element = cv2.getStructuringElement(shape, (2 * erode_mask + 1, 2 * erode_mask + 1), (erode_mask, erode_mask))
            mask = np.array(cv2.erode(mask, element))
        segmented = segmented * 255
        fname = images[i].rsplit('.', 1)[0]
        fname = fname.split('/')[-1]
        diff = (segmented/np.max(segmented)) - (mask/np.max(mask))
        color_diff = np.zeros((sh[0], sh[1], 3), dtype=np.uint8)
        color_diff[:,:,0] = (255 * (diff < 0)).astype(np.uint8)
        color_diff[:,:,2] = (255 * (diff > 0)).astype(np.uint8)
        # cv2.imwrite(os.path.join(save_dir, f"{fname}_diff.png"), color_diff)
        # cv2.imwrite(os.path.join(save_dir, f"{fname}_pred.png"), segmented)


def jaccard_similarity_bin_mask(A, B):
    # Calculate Jaccard similarity of two masks
    intersection = np.sum(np.logical_and(A, B))
    union = np.sum(np.logical_or(A, B))
    res = intersection/union
    return res

def binary_cross_entropy(y_true, y_pred):
    # Convert to tensors
    y_true = torch.tensor(y_true).float()
    y_pred = torch.tensor(y_pred).float()

    # Calculate binary cross-entropy
    bce_loss = F.binary_cross_entropy(y_pred, y_true)

    return bce_loss.item()

if __name__ == "__main__":
    main()