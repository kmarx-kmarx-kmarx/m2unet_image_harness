from utils import *
import numpy as np
from tqdm import tqdm, trange
from run_on_image import inference_on_image_stack
import glob
import pandas as pd
import torch
import torch.nn.functional as F

# Given greyscale images, masks in .npz format, get the Jaccard similarity and BCE loss from a given model
def main():
    debug_n = 0
    data_dir = "/home/prakashlab/Documents/kmarx/train_m2unet_cellpose_cloud/data_uganda_hemaprep_mid_third_subset"
    model_path = "/home/prakashlab/Documents/kmarx/train_m2unet_cellpose_cloud/m2unet_model_flat_less_erode_2_rotfirst"
    model_name = "model_24_9.pth"
    save_dir = "results_flat_erode_rotfirst_with_eroded_masks_final"
    save_file = "uganda_mid_third"
    save_best = True  # save the image with the lowest loss
    save_worst = True # save the image with the highest loss
    n_batch = 50
    erode_mask = 1 # set negative to ignore

    os.makedirs(save_dir, exist_ok=True)

    # Get image paths
    images = glob.glob(os.path.join(data_dir, "**.npz"), recursive=True)
    n_images = len(images)
    print(f"{n_images} images to process")
    w, h = np.load(images[0])["img"].shape
    # preallocate memory for the images, masks, and predictions
    image_stack = np.zeros((n_batch, w, h), dtype=float)
    mask_stack = np.zeros((n_batch, w, h), dtype=bool)
    inference_stack = np.zeros((n_batch, w, h), dtype=bool)
    # store results: paths,jaccard,bce
    data_list = []
    # loop in batches
    index = 0
    if save_best:
        best_loss = np.inf
        best_pred = np.zeros((w, h), dtype=np.uint8)
    if save_worst:
        worst_loss = 0
        worst_pred = np.zeros((w, h), dtype=np.uint8)
    pbar = tqdm(total=n_images)
    while index < n_images:
        end_idx = min(index + n_batch, n_images)
        # load batch of images
        for i in range(index,end_idx):
            data = np.load(images[i])
            # Load the image
            image_stack[(i-index), :, :] = data["img"]
            # Load the mask
            if erode_mask >= 0:
                shape = cv2.MORPH_ELLIPSE
                element = cv2.getStructuringElement(shape, (2 * erode_mask + 1, 2 * erode_mask + 1), (erode_mask, erode_mask))
                mask_stack[(i-index), :, :] = np.array(cv2.erode(data["mask"], element))
            else:
                mask_stack[(i-index), :, :] = data["mask"]
        inference_stack, __, times = inference_on_image_stack(image_stack, model_path, model_name)
        # compare results
        for i in range(index,end_idx):
            jaccard = jaccard_similarity_bin_mask(mask_stack[(i-index), :, :], inference_stack[(i-index), :, :])
            bce = binary_cross_entropy(mask_stack[(i-index), :, :], inference_stack[(i-index), :, :])
            path = images[i]
            data_list.append([path, jaccard, bce, times[i-index]])
            if save_best:
                if bce < best_loss:
                    best_loss = bce
                    best_pred = 225*inference_stack[(i-index), :, :]
            if save_worst:
                if bce > worst_loss:
                    worst_loss = bce
                    worst_pred = 225*inference_stack[(i-index), :, :]
                if jaccard < 0.7:
                    cv2.imwrite(os.path.join(save_dir, f"{i}_{save_file}_worst_pred.png"), worst_pred)
                    cv2.imwrite(os.path.join(save_dir, f"{i}_{save_file}_worst_im.png"), image_stack[(i-index), :, :])
        pbar.update(end_idx - index)
        index = end_idx
        if debug_n > 0 and index > debug_n:
            break
    pbar.close()
    results_df = pd.DataFrame(columns=['path', 'jaccard', 'bce', 'time'], data=data_list)
    results_df.to_csv(os.path.join(save_dir, f"{save_file}_{n_images}.csv"))
    if save_best:
        cv2.imwrite(os.path.join(save_dir, f"{save_file}_best.png"), best_pred)
    if save_worst:
        cv2.imwrite(os.path.join(save_dir, f"{save_file}_worst.png"), worst_pred)


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