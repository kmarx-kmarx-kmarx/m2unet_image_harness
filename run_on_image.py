import os
from m2unet import M2UNet, Encoder
import numpy as np
from skimage.filters import threshold_otsu
import time
# specify the gpu number
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
from tqdm import tqdm

def inference_on_image_stack(images, model, sz=1024, device=None, threshold_set=128, overlap=16):
    '''
    Run inference on a set of images one at a time. 
    This function automatically tiles the image to sz x sz and stitches them back together

    Arguments:
        images: np array of greyscale images, 
                    n_images x img_width x img_height
        model_root: string, path to folder containing model and config.json
        model_name: string, name of model file
        theshold_scale: threshold multiplier value. 
                    default is 1.0 - use the theshold_otsu value without changing it
                    greater than 1: increase separation between masks
                    less than 1: decrease separation between masks
        sz: integer, width/height to crop the image to. Should be a power of 2
                    1024 tends to work but higher/lower values can be used based on gpu memory
        overlap: integer, pixel width shared between adjacent crops. Overlaps remove segmentation artifacts
                    along the edge of the image. overlap must be an even number.

    Returns:
        masks: np array of binary masks,
                    n_images x img_width x img_height
    '''
    # make overlap an even number
    overlap += overlap%2

    n_im, im_w, im_h, im_c = images.shape
    nx = int(np.ceil((im_w-overlap)/(sz-overlap)))
    ny = int(np.ceil((im_h-overlap)/(sz-overlap)))
    n_slices = int(n_im * nx * ny)
    # slice images down to size and put them in the stack
    image_stack = np.zeros((n_slices, sz, sz, im_c), dtype=np.uint8)
    dx, dy = (0, 0)
    for i in range(n_slices):
        x = i % nx
        y = int(np.floor(i/nx))
        z = int(np.floor(y/ny))
        y = y % ny

        # handle case x=(nx-1), y=(ny-1) separately 
        if x == (nx-1):
            x_0 = im_w-sz
            x_1 = im_w
            dx = ((sz-overlap)*x)-x_0
        else:
            x_0 = (sz-overlap)*x 
            x_1 = x_0 + sz
        
        if y == (ny-1):
            y_0 = im_h-sz
            y_1 = im_h
            dy = ((sz-overlap)*y)-y_0
        else:
            y_0 = (sz-overlap)*y 
            y_1 = y_0 + sz

        image_stack[i, :, :, :] = images[z, x_0:x_1, y_0:y_1, :]
    
    # run inference
    mask_stack, times = inference_on_sized_image_stack(image_stack, model, device=device, threshold_set=threshold_set)

    # convert back to images
    # preallocate memory for image stack
    output = np.zeros((images.shape[0],images.shape[1],images.shape[2]) , dtype=bool)
    t_full = [0] * n_im
    for i in range(n_slices):
        x = i % nx
        y = int(np.floor(i/nx))
        z = int(np.floor(y/ny))
        y = y % ny

        d = int(overlap/2)

        # slice mask_stack: don't get overlaps unless we are at the end
        if x == 0:
            x_0m = 0
            x_1m = sz-d
        elif x==(nx-1):
            x_0m = dx-2*overlap
            x_1m = sz
        else:
            x_0m = d
            x_1m = sz-d

        # slice images
        if x == 0:
            x_0 = 0
        elif x == (nx-1):
            x_0 = im_w - dx + d
        else:
            x_0 = x*sz - (2*x-1)*d
        if y == 0:
            y_0 = 0
        elif y == (nx-1):
            y_0 = im_h - dy + d
        else:
            y_0 = y*sz - (2*y-1)*d

        if y == 0:
            y_0m = 0
            y_1m = sz-d
        elif y==(nx-1):
            y_0m = dy-2*overlap
            y_1m = sz
        else:
            y_0m = d
            y_1m = sz-d

        # finish slicing images
        if x == (nx-1):
            x_1 = im_w
            x_0 = x_1-x_1m+x_0m
        else:
            x_1 = x_0+x_1m-x_0m
        if y == (ny-1):
            y_1 = im_h
            y_0 = y_1-y_1m+y_0m
        else:
            y_1 = y_0+y_1m-y_0m
        
        output[z, x_0:x_1, y_0:y_1] = mask_stack[i, x_0m:x_1m, y_0m:y_1m]
        t_full[z] = np.sum(times[z*nx*ny:(z+1)*nx*ny]) # sum of the nx * ny next slices

    return output, t_full


def inference_on_sized_image_stack(images, model, device=None, threshold_set=128):
    '''
    Run inference on a set of images one at a time

    Arguments:
        images: np array of greyscale images, 
                    n_images x img_width x img_height
                    img_width and img_height must be a power of 2 for this to work
        model_root: string, path to folder containing model and config.json
        model_name: string, name of model file
        theshold_scale: threshold multiplier value. 
                    default is 1.0 - use the theshold_otsu value without changing it
                    greater than 1: increase separation between masks
                    less than 1: decrease separation between masks

    Returns:
        masks: np array of binary masks,
                    n_images x img_width x img_height
    '''
    # init. device
    if device == None:
        device = torch.device("cpu")
    # initialize result array
    outputs = np.zeros((images.shape[0], images.shape[1], images.shape[2]), dtype=bool)
    times = [0]*images.shape[0]
    # loop through images
    for i, img in enumerate(tqdm(images)):
        t0 = time.time()
        # normalize the image
        img = (img - np.mean(img)) /np.std(img)
        # no batching - add an axis but don't put additional data there
        # for batching, stack the images along axis 0
        inputs = np.expand_dims(img, axis=0)

        # run inferece
        assert inputs.ndim == 4
        X = torch.from_numpy(inputs.transpose(0, 3, 1, 2)).to(device=device, dtype=torch.float32)
        results = model(X).detach().cpu().numpy().transpose(0, 2, 3, 1)
        # get the results - look at the first output. 
        # if batching, keep all the data instead of just index 0
        output = np.clip(results[0,:,:,0] * 255, 0, 255).astype('uint8')
        mask = (output > threshold_set)
        # save result
        outputs[i,:,:] = mask
        times[i] = time.time()-t0
    return outputs.astype(bool), times