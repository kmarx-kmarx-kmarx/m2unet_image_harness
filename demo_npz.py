import glob
import random
import os
import time
from natsort import natsorted
from m2unet import m2unet
import cv2
from run_on_image import inference_on_image_stack
import numpy as np

def main():
    data_dir = "path/to/data"
    model_root = "m2unet_model_greyscale"
    n_channels = 1 # set to 1 for greyscale, 3 for RGB
    model_name = "model_70_11.pth"
    save_dir = "path/to/save"
    batch_sz = 10 # segment this many images per batch. Should be set depending on image size, hardware.
    n_im = 5   # total number of images to segment. Set to 0 to segment all images in data_dir
    run_size = 1024 # tile the images to 1024x1024 sections. Should be set depending on image size, hardware
    overlap = 16    # Amount of overlap between adjacent tiles
    randomize = False # Segment the images in random order
    save_cp_mask = True
    save_image = True
    save_diff = True
    save_pred_mask = True
    random.seed(3)

    # Load M2Unet
    model, device = m2unet(model_root, model_name, upsamplemode='bilinear',expand_ratio=0.15, output_channels=1, activation="linear", N_CHAN=n_channels)

    os.makedirs(save_dir, exist_ok=True)

    # get image paths
    images = natsorted(glob.glob(os.path.join(data_dir, f"**_seg.npz")))
    if randomize:
        random.shuffle(images)
    if n_im > 0:
        images = images[:n_im]
    total_images = len(images)

    # Load and evaluate batch_sz images at a time
    idx = 0
    t_read = []
    t_segm = []
    t_writ = []
    t_init = time.time()
    while idx < total_images:
        end_idx = min(idx + batch_sz, total_images)
        # Load the images
        imgs = []
        masks = []
        for i in range(idx, end_idx):
            t0 = time.time()
            data = np.load(images[i])
            im = data["img"]
            if im.ndim == 2:
                im = np.expand_dims(im, axis=2)
            mask = data["mask"]
            imgs.append(im)
            masks.append(mask)
            t_read.append(time.time()-t0)
        imgs = np.array(imgs)

        # get results
        results, times = inference_on_image_stack(imgs, model, device=device, sz=run_size, overlap=overlap)
        t_segm = t_segm + times
        # save results
        for i, result in enumerate(results):
            t0 = time.time()
            fname = images[i+idx].split('/')[-1]
            fname = fname.rsplit('.', 1)[0] + ".png"
            if save_pred_mask:
                savepath = os.path.join(save_dir, f"seg_{fname}")
                cv2.imwrite(savepath, 255*result)
            if save_cp_mask:
                savepath = os.path.join(save_dir, f"cp_{fname}")
                cv2.imwrite(savepath, masks[i])
            if save_image:
                savepath = os.path.join(save_dir, f"im_{fname}")
                cv2.imwrite(savepath, imgs[i])
            if save_diff:
                diff = (result/np.max(result)) - (masks[i]/np.max(masks[i]))
                color_diff = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
                color_diff[:,:,0] = (255 * (diff < 0)).astype(np.uint8)
                color_diff[:,:,2] = (255 * (diff > 0)).astype(np.uint8)
                savepath = os.path.join(save_dir, f"diff_{fname}")
                cv2.imwrite(savepath, color_diff)
            t_writ.append(time.time()-t0)
        
        idx = end_idx
    
    # print timing statistics
    print(f"Reading:      {np.sum(t_read):.3f}, avg {np.mean(t_read):.3f}")
    print(f"Segmentation: {np.sum(t_segm):.3f}, avg {np.mean(t_segm):.3f}")
    print(f"Writing:      {np.sum(t_writ):.3f}, avg {np.mean(t_writ):.3f}")
    print(f"Total:        {(time.time()-t_init):.3f}, avg {(time.time()-t_init)/total_images:.3f}")
    
if __name__ == "__main__":
    main()