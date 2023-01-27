import os
from interactive_m2unet import M2UnetInteractiveModel
import numpy as np
from skimage.filters import threshold_otsu
# specify the gpu number
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch

def inference_on_image(images, model_root, model_name, threshold_scale=1.0):
    '''
    Run inference on a set of images one at a time

    Arguments:
        images: np array of greyscale images, 
                    n_images x img_width x img_height
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