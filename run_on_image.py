import os
from interactive_m2unet import M2UnetInteractiveModel
import numpy as np
from skimage.filters import threshold_otsu
# specify the gpu number
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch

def inference_on_image_stack(images, model_root, model_name, sz=1024, threshold_scale=1.0, overlap=16):
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
    # make overlap even if it isn't already
    overlap += overlap%2

    n_im, im_w, im_h = images.shape
    # preallocate memory for image stack
    nx = int(np.ceil((im_w-overlap)/(sz-overlap)))
    ny = int(np.ceil((im_h-overlap)/(sz-overlap)))
    n_slices = int(n_im * nx * ny)
    # pad border crops with zeros to make images nx*sz x ny*sz
    image_padded = np.zeros((n_im, sz*nx, sz*ny))
    image_padded[:, :im_w, :im_h] = images
    # slice images down to size and put them in the stack
    image_stack = np.zeros((n_slices, sz, sz))
    for i in range(n_slices):
        x = i % nx
        y = int(np.floor(i/nx))
        z = int(np.floor(y/ny))
        y = y % ny
        # shift x_0, y_0 over by overlap for x > 0, y > 0
        x_0 = (sz-overlap)*x 
        y_0 = (sz-overlap)*y
        x_1 = x_0 + sz
        y_1 = y_0 + sz

        image_stack[i, :, :] = image_padded[z, x_0:x_1, y_0:y_1]
    
    # run inference
    mask_stack = inference_on_sized_image_stack(image_stack, model_root, model_name, threshold_scale=threshold_scale)

    # convert back to images
    for i in range(n_slices):
        x = i % nx
        y = int(np.floor(i/nx))
        z = int(np.floor(y/ny))
        y = y % ny

        d = int(overlap/2)
        
        # slice mask_stack: don't get overlaps unless we are at the end
        x_0m = 0 if x==0 else d
        y_0m = 0 if y==0 else d
        x_1m = sz if x==(nx-1) else sz-d
        y_1m = sz if y==(ny-1) else sz-d
        # slice image_padded
        x_0 = 0 if x==0 else x*sz - (2*x-1)*d
        y_0 = 0 if y==0 else y*sz - (2*y-1)*d
        x_1 = x_0+x_1m-x_0m
        y_1 = y_0+y_1m-y_0m

        image_padded[z, x_0:x_1, y_0:y_1] = mask_stack[i, x_0m:x_1m, y_0m:y_1m]
        
    output = image_padded[:, :im_w, :im_h]

    return output


def inference_on_sized_image_stack(images, model_root, model_name, threshold_scale=1.0):
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
    # setup
    torch.cuda.empty_cache()

    # Load the model
    model = M2UnetInteractiveModel(
        model_dir=model_root,
        default_save_path=os.path.join(model_root, model_name),
        pretrained_model=os.path.join(model_root, model_name)
    )

    # initialize result array
    outputs = np.zeros(images.shape)

    # loop through images
    for i, img in enumerate(images):
        # normalize the image
        img = (img - np.mean(img)) /np.std(img)
        # format data
        inputs = np.stack([img,]*3, axis=2)
        # no batching - add an axis but don't put additional data there
        # for batching, stack the images along axis 0
        inputs = np.expand_dims(inputs, axis=0)

        # run inferece
        results = model.predict(inputs)
        # get the results - look at the first output. 
        # if batching, keep all the data instead of just index 0
        output = np.clip(results[0] * 255, 0, 255)[:, :, 0].astype('uint8')
        # get threshold 
        threshold = threshold_otsu(output) * threshold_scale
        mask = ((output > threshold) * 255).astype('uint8')

        # save result
        outputs[i,:,:] = mask
    
    return outputs

# if __name__ == "__main__":
#     im = cv2.imread("0_1_0_BF_LED_matrix_left_half.bmp")[:,:,0]
#     images = np.stack([im]*5, axis=0)
#     images[3,:,:] = np.transpose(images[3,:,:])

#     outs = inference_on_image_stack(images, ".", "200_model_14.pth", sz=1024, threshold_scale=1.0)

#     for i, out in enumerate(outs):
#         cv2.imwrite(f"{i}.bmp", out)