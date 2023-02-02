import imageio
import cv2
# import cupy as cp # conda install -c conda-forge cupy==10.2
# import cupyx.scipy.ndimage
import numpy as np
from scipy import signal
import pandas as pd
import xarray as xr
import gcsfs
import os

def imread_gcsfs(fs,file_path):
	img_bytes = fs.cat(file_path)
	I = imageio.core.asarray(imageio.imread(img_bytes, "bmp"))
	return I

def generate_dpc(I1,I2,use_gpu=False):
	if use_gpu:
		# img_dpc = cp.divide(img_left_gpu - img_right_gpu, img_left_gpu + img_right_gpu)
		# to add
		I_dpc = 0
	else:
		I_dpc = np.divide(I1-I2,I1+I2)
		I_dpc = I_dpc + 0.5
	I_dpc[I_dpc<0] = 0
	I_dpc[I_dpc>1] = 1
	return I_dpc

def get_im_mask(i, indices, dataset, fs, bucket_source, flatfield_left, flatfield_right, model_cp, local=None):
    idx = indices[dataset][i]
    x = int(idx/indices[dataset+'Nx'])
    y = idx % indices[dataset+'Ny']
    k = 0
    # check if file exists
    file_id = str(x) + '_' + str(y) + '_' + str(k)
    filepath = os.path.join(local, f"{dataset}_{file_id}_seg.npz")
    if os.path.exists(filepath):
        im, mask = load_from_file(filepath)
    else:
        os.makedirs(local, exist_ok=True)
        im, mask = load_from_gcs(fs, bucket_source, dataset, file_id, flatfield_left, flatfield_right, model_cp, local=local)  
    
    return im, mask

def load_from_file(filepath):
    items = np.load(filepath, allow_pickle=True)
    return items['img'], items['mask']

def load_from_gcs(fs, bucket_source, dataset, file_id, flatfield_left, flatfield_right, model_cp, local=None):
    # generate DPC
    I_BF_left = imread_gcsfs(fs,bucket_source + '/' + dataset + '/0/' + file_id + '_' + 'BF_LED_matrix_left_half.bmp')
    I_BF_right = imread_gcsfs(fs,bucket_source + '/' + dataset + '/0/' + file_id + '_' + 'BF_LED_matrix_right_half.bmp')
    if len(I_BF_left.shape)==3: # convert to mono if color
        I_BF_left = I_BF_left[:,:,1]
        I_BF_right = I_BF_right[:,:,1]
    I_BF_left = I_BF_left.astype('float')/255
    I_BF_right = I_BF_right.astype('float')/255
    # flatfield correction
    I_BF_left = I_BF_left/flatfield_left
    I_BF_right = I_BF_right/flatfield_right
    I_DPC = generate_dpc(I_BF_left,I_BF_right)

    # cellpose preprocessing
    im = I_DPC - np.min(I_DPC)
    im = np.uint8(255 * np.array(im, dtype=np.float64)/float(np.max(im)))
    # run segmentation
    mask, flows, styles = model_cp.eval(im, diameter=None)

    outlines = mask * utils.masks_to_outlines(mask)
    mask = (mask  > 0) * 1.0
    outlines = (outlines  > 0) * 1.0
    mask = (mask * (1.0 - outlines) * 255).astype(np.uint8)

    I_DPC = (255*I_DPC).astype(np.uint8)

    if local != None:
        savepath = os.path.join(local, f"{dataset}_{file_id}_seg.npz")
        # io.masks_flows_to_seg(I_DPC, mask, flows, 0, savepath, [0,0])
        np.savez(savepath, mask=mask, img=I_DPC)

    return I_DPC, mask

def get_dpc(data_path, dataset, file_id, flatfield_left, flatfield_right):
    # generate DPC
    I_BF_left = cv2.imread(os.path.join(data_path, dataset, '0', file_id + '_' + 'BF_LED_matrix_left_half.bmp'))
    I_BF_right = cv2.imread(os.path.join(data_path, dataset, '0', file_id + '_' + 'BF_LED_matrix_right_half.bmp'))
    if len(I_BF_left.shape)==3: # convert to mono if color
        I_BF_left = I_BF_left[:,:,1]
        I_BF_right = I_BF_right[:,:,1]
    I_BF_left = I_BF_left.astype('float')/255
    I_BF_right = I_BF_right.astype('float')/255
    # flatfield correction
    I_BF_left = I_BF_left/flatfield_left
    I_BF_right = I_BF_right/flatfield_right
    I_DPC = generate_dpc(I_BF_left,I_BF_right)

    return (255*I_DPC).astype(np.uint8)